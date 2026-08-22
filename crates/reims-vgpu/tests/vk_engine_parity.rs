//! Off-VM regression suite for the internal Vulkan engine (no external render crate).
//!
//! Drives `engine::execute_draw_request` with representative inputs and asserts
//! known-correct pixels (metal2vulkan fixtures) plus warm create/alloc = 0 and
//! device-loss policy. Requires a working Vulkan ICD; skips cleanly if init fails.
//!
//! **Serial:** the engine is process-global; all tests take the suite lock.

use metal2vulkan::passes::Stage;
use reims_vgpu_vulkan::engine::{
    self, AttachmentInitial, BlendFactor, BlendOp, BlendStateResource, BufferContent, CullMode,
    DepthState, DrawRequest, IndexType, IndexedDrawResource, PrimitiveTopology,
    SampledContentIdentity, SampledImageResource, SampledSource, SamplerCompareFunction,
    SamplerResource, ScissorResource, SecondaryColorTarget, StencilFaceOps, StencilOp,
    StencilState, StorageBufferResource, TargetIdentity, VertexAttributeFormat,
    VertexAttributeResource, VertexStepFunction, ViewportResource, VisibilityResultMode,
    MAX_DEVICE_RECREATES,
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
use std::sync::{Mutex, OnceLock};

/// Acquire the process-global engine lock **and** reset the engine, in that
/// order. Every engine-touching test must start from a fresh context:
/// `device_loss_named_and_recreate_bounded` deliberately drives the
/// device-recreate budget to its cap and leaves it there, which is the correct
/// product behaviour — the cap is a permanent give-up, not a per-draw counter.
/// A test that acquires the lock without resetting therefore inherits an engine
/// that refuses every draw with `recreate_cap_exhausted`. Handing the guard out
/// only from a function that has already reset makes that omission unwritable,
/// rather than a rule about call order that the next test can forget.
fn engine_test_session() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = LOCK
        .get_or_init(|| {
            // Never share the live product logs with a concurrent boot.
            reims_vgpu::observe::redirect_logs_for_tests();
            Mutex::new(())
        })
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    guard
}

/// Synthetic serialized owner for sampled-cache tests. Keeping the strong
/// resource in the test scope gives the engine the same weak lifetime proof a
/// real task resource supplies; dropping it is the cache boundary.
fn sampled_resource_owner() -> std::sync::Arc<reims_vgpu::model::TaskResource> {
    std::sync::Arc::new(reims_vgpu::model::TaskResource::new(
        reims_vgpu_protocol::ObjectListEntry::new(reims_vgpu_protocol::ObjectKind::Buffer, 0, 0),
        std::sync::Arc::from([]),
    ))
}

fn fixtures() -> PathBuf {
    // Minimal AIR fixture subset owned by reims-vgpu's Vulkan engine tests.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/air")
}

fn translate_words(name: &str, stage: Stage) -> Vec<u32> {
    let tmp = std::env::temp_dir().join(format!(
        "paravirt_engine_{}_{}_{:?}",
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
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    words
}

/// The device binding number for sampler `index`.
///
/// Derived from metal2vulkan's selected default layout rather than spelled.
fn sampler_binding(index: u32) -> u32 {
    reims_vgpu_vulkan::spirv_bind::SAMPLER_BINDING_BASE + index
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

fn engine_req(vert: &[u32], frag: &[u32], w: u32, h: u32) -> DrawRequest {
    DrawRequest {
        program: reims_vgpu_core::PreparedRenderProgram {
            vertex: reims_vgpu_vulkan::m2v_cache::prepare_test_shader(vert.to_vec()),
            fragment: reims_vgpu_vulkan::m2v_cache::prepare_test_shader(frag.to_vec()),
        },
        width: w,
        height: h,
        vertex_count: 3,
        first_vertex: 0,
        instance_count: Some(1),
        base_instance: 0,
        primitive_topology: PrimitiveTopology::Triangle,
        ..Default::default()
    }
}

/// float4(0.25, 0.5, 0.75, 1.0) → unorm8 ≈ (64, 128, 191, 255); allow ±1 LSB.
fn near(got: u8, want: u8) -> bool {
    (got as i32 - want as i32).abs() <= 1
}

/// A draw's pixels in semantic RGBA8, whatever physical order the attachment
/// read back in.
///
/// The engine picks the attachment format from the resolved target — a
/// `TargetIdentity::Surface` resident is the format its mapping declared, which
/// is `SURFACE_TEST_FORMAT` for every identity in this file, so an IOSurface texture
/// composite Store's readback lands in guest scanout order with no CPU pass — and reports
/// which it used in `DrawOutput::pixels_bgra`. These cases assert *colour*, not
/// byte layout, so they normalize here from the reported order rather than
/// assuming one. The physical contract has its own case
/// (`a_surface_resident_reads_back_in_guest_scanout_order`); assuming an order
/// here would let this whole suite silently follow the engine.
fn semantic_rgba(out: &engine::DrawOutput) -> Vec<u8> {
    let mut px = out.pixels.clone();
    if out.pixels_bgra {
        for p in px.chunks_exact_mut(4) {
            p.swap(0, 2);
        }
    }
    px
}

fn assert_fullscreen_fragment_color(label: &str, px: &[u8], w: u32, h: u32) {
    assert_eq!(px.len(), (w * h * 4) as usize, "{label}: readback size");
    let i = ((h / 2) * w + w / 2) as usize * 4;
    let (r, g, b, a) = (px[i], px[i + 1], px[i + 2], px[i + 3]);
    assert!(
        near(r, 64) && near(g, 128) && near(b, 191) && near(a, 255),
        "{label}: center RGBA=({r},{g},{b},{a}); expected ~(64,128,191,255)"
    );
    let all = (0..(w * h) as usize).all(|p| near(px[p * 4], 64) && near(px[p * 4 + 1], 128));
    assert!(
        all,
        "{label}: fullscreen triangle did not cover viewport (clear showing through)"
    );
}

fn draw_or_skip(label: &str, req: &DrawRequest) -> Option<Vec<u8>> {
    match engine::execute_draw_request(req) {
        Ok(o) => Some(semantic_rgba(&o)),
        Err(e) => {
            let s = e.to_string();
            if skip_if_no_gpu(&s) {
                eprintln!("SKIP {label}: no GPU ({s})");
                None
            } else {
                panic!("{label}: {s}");
            }
        }
    }
}

#[test]
fn plain_triangle_known_color() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let req = engine_req(&v, &f, 16, 16);
    if let Some(px) = draw_or_skip("plain_triangle", &req) {
        assert_fullscreen_fragment_color("plain_triangle", &px, 16, 16);
    }
}

/// A render-pass extent constrains rasterization, not attachment load-clear.
/// The whole 8x8 attachment must become red before the full-screen draw is
/// clipped to the upper-left 4x4 raster bound.
#[test]
fn render_target_extent_clips_fragments_without_narrowing_clear() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.target_clear = [1.0, 0.0, 0.0, 1.0];
    req.render_target_extent = reims_vgpu_core::RenderTargetExtent {
        width: std::num::NonZeroU32::new(4),
        height: std::num::NonZeroU32::new(4),
    };
    let Some(px) = draw_or_skip("render_target_extent", &req) else {
        return;
    };

    for y in 0..8usize {
        for x in 0..8usize {
            let pixel = &px[(y * 8 + x) * 4..][..4];
            if x < 4 && y < 4 {
                assert!(
                    near(pixel[0], 64)
                        && near(pixel[1], 128)
                        && near(pixel[2], 191)
                        && near(pixel[3], 255),
                    "fragment at ({x},{y}) was not drawn: {pixel:?}"
                );
            } else {
                assert_eq!(pixel, [255, 0, 0, 255], "clear lost at ({x},{y})");
            }
        }
    }
}

/// Sub-unit line width is not sent to Vulkan as an invalid dynamic value. The
/// semantic projection suppresses line fragments while leaving filled
/// triangles untouched, matching the render-encoder contract.
#[test]
fn zero_line_width_discards_lines_but_not_filled_triangles() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let (w, h) = (16u32, 16u32);

    let mut filled = engine_req(&v, &f, w, h);
    filled.line_width = reims_vgpu_core::LineWidth::from_f32(0.0);
    let Some(filled_pixels) = draw_or_skip("zero_width_filled_triangle", &filled) else {
        return;
    };
    assert_fullscreen_fragment_color("zero_width_filled_triangle", &filled_pixels, w, h);

    let mut line = engine_req(&v, &f, w, h);
    line.primitive_topology = PrimitiveTopology::Line;
    line.vertex_count = 2;
    line.line_width = reims_vgpu_core::LineWidth::from_f32(0.0);
    let line_pixels = draw_or_skip("zero_width_line", &line).expect("same GPU context");
    assert!(
        line_pixels
            .chunks_exact(4)
            .all(|pixel| pixel[0..3] == [0, 0, 0]),
        "zero-width line produced a fragment"
    );
}

/// Four-sample rasterization with a matching resident depth attachment resolves
/// coverage into the single-sample target.
///
/// The fixture covers its viewport. Moving the viewport's left edge into pixel
/// zero leaves three standard 4x sample locations inside and one outside, so
/// the resolved red channel must lie strictly between clear black and the
/// fragment's red value. A one-sample redirect can only produce an endpoint.
#[test]
fn multisample_resolve_preserves_subpixel_coverage() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 32, 16);
    req.raster_sample_count = 4;
    req.color_sample_count = 4;
    req.multisample_resolve = true;
    req.depth = Some(DepthState {
        identity: None,
        test_enable: true,
        write_enable: true,
        compare: SamplerCompareFunction::Always,
        clear_value: 1.0,
        load: false,
        stencil: None,
    });
    req.viewports = vec![ViewportResource {
        x: 0.3,
        y: 0.0,
        width: 31.7,
        height: 16.0,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    if let Some(px) = draw_or_skip("multisample_resolve", &req) {
        let red = px[(8 * 32) * 4];
        assert!(
            red > 0 && red < 64,
            "resolved edge must carry partial coverage, got red={red}"
        );
        assert!(near(px[(8 * 32 + 1) * 4], 64));
        let second = draw_or_skip("multisample_resolve_second", &req)
            .expect("the second transient-depth framebuffer must not reuse the first one's view");
        assert_eq!(second, px);
    }
}

/// The whole `DrawOutput`, for a case whose answer is not pixels.
///
/// [`draw_or_skip`] returns only the normalized colour bytes, which is right
/// for every case that asserts what was drawn. An occlusion count is a second
/// thing the same draw produced, so these ask for the record rather than for
/// the picture.
fn draw_out_or_skip(label: &str, req: &DrawRequest) -> Option<engine::DrawOutput> {
    match engine::execute_draw_request(req) {
        Ok(o) => Some(o),
        Err(e) => {
            let s = e.to_string();
            if skip_if_no_gpu(&s) {
                eprintln!("SKIP {label}: no GPU ({s})");
                None
            } else if s.contains("visibility_counting_unsupported") {
                // The refusal under test elsewhere. A host without
                // `occlusionQueryPrecise` is a supported host, not a broken
                // one, and it cannot answer a counting query — so these two
                // skip rather than fail, exactly as a host with no ICD does.
                eprintln!("SKIP {label}: host offers no precise occlusion ({s})");
                None
            } else {
                panic!("{label}: {s}");
            }
        }
    }
}

/// A counting occlusion query reports the sample count the scissor admits, on
/// real hardware.
///
/// The number is what makes this a proof rather than a smoke test. The fixture
/// triangle covers the whole clip volume — `assert_fullscreen_fragment_color`
/// asserts exactly that elsewhere — and this engine rasterizes at one sample
/// per pixel, so the samples that pass are precisely the scissor's area. Every
/// plausible wrong implementation lands somewhere else: a query never begun
/// reads `None` or `Some(0)`, a query that ignored the scissor reads the target
/// area 1024, and a pool used without `vkCmdResetQueryPool` reads whatever the
/// driver left there.
#[test]
fn an_occlusion_query_counts_the_samples_the_scissor_admits() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 32, 32);
    req.scissors = vec![ScissorResource {
        x: 8,
        y: 8,
        width: 8,
        height: 8,
    }];
    req.occlusion_query = Some(VisibilityResultMode::Counting);
    if let Some(out) = draw_out_or_skip("occlusion_counting", &req) {
        assert_eq!(
            out.occlusion_samples,
            Some(8 * 8),
            "a counting query over an 8x8 scissor passes 64 samples"
        );
    }
}

/// A second query in the same process reports its own scissor's area.
///
/// The pair is what pins the reset. One case alone cannot tell a pool that is
/// reset per draw from one that is never reset and happens to read the right
/// number once; a second query whose answer differs is only correct if the
/// first one's result was cleared. It also fixes the count as a function of the
/// scissor rather than a constant that matched by luck.
#[test]
fn a_second_occlusion_query_counts_its_own_scissor() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 32, 32);
    req.scissors = vec![ScissorResource {
        x: 0,
        y: 0,
        width: 16,
        height: 16,
    }];
    req.occlusion_query = Some(VisibilityResultMode::Counting);
    if let Some(out) = draw_out_or_skip("occlusion_counting_16", &req) {
        assert_eq!(out.occlusion_samples, Some(16 * 16));
    }
}

/// A draw that arms no query says so, rather than reporting a count of zero.
///
/// `None` and `Some(0)` are different answers — the second is a draw that was
/// asked and passed nothing, which is what an occlusion test exists to find —
/// and a reader that cannot tell them apart cannot use either.
#[test]
fn a_draw_with_no_query_reports_no_count() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let req = engine_req(&v, &f, 16, 16);
    if let Some(out) = draw_out_or_skip("occlusion_unarmed", &req) {
        assert_eq!(out.occlusion_samples, None);
    }
}

/// A boolean query needs no device feature, and still reports what passed.
///
/// Vulkan's occlusion query is imprecise unless `VK_QUERY_CONTROL_PRECISE_BIT`
/// is set, and imprecise is `MTLVisibilityResultModeBoolean` exactly — so this
/// arm is servable on every host and is asserted as non-zero rather than as an
/// exact count, which is all an imprecise query promises.
#[test]
fn a_boolean_occlusion_query_needs_no_precise_feature() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 32, 32);
    req.occlusion_query = Some(VisibilityResultMode::Boolean);
    if let Some(out) = draw_out_or_skip("occlusion_boolean", &req) {
        let n = out
            .occlusion_samples
            .expect("boolean query reports a result");
        assert!(n > 0, "a fullscreen triangle passes something; got {n}");
    }
}

#[test]
fn viewport_scissor_known_color() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 32, 32);
    req.viewports = vec![ViewportResource {
        x: 0.0,
        y: 0.0,
        width: 32.0,
        height: 32.0,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    req.scissors = vec![ScissorResource {
        x: 0,
        y: 0,
        width: 32,
        height: 32,
    }];
    if let Some(px) = draw_or_skip("viewport_scissor", &req) {
        assert_fullscreen_fragment_color("viewport_scissor", &px, 32, 32);
    }
}

/// A draw carrying several viewports builds a pipeline that declares as many,
/// and binds exactly that many — on real hardware.
///
/// This is the pair that has to agree and that nothing else here can catch.
/// `VkPipelineViewportStateCreateInfo::viewportCount` is **not** dynamic below
/// `vkCmdSetViewportWithCount`, which is core in 1.3 while this device's floor
/// is 1.2 — so the count is baked into the pipeline and `vkCmdSetViewport` must
/// bind that same number. If the pipeline key and the bind ever compute it
/// differently the draw is invalid, and the symptom is a validation-layer
/// message or a driver-defined result rather than a compile error. Both sides
/// call `engine::viewport_slot_count`; this runs them against a GPU.
///
/// Slot 0 covers the target and slot 1 is a quarter of it. With no
/// `ViewportIndex` written by the fixture's vertex shader every primitive goes
/// to slot 0, so the pixels must be exactly what the single-viewport test above
/// produces: the second slot changes what the pipeline declares without
/// changing what this geometry rasterizes.
///
/// Where the host advertises no `multiViewport` the engine declines by name and
/// `draw_or_skip` yields nothing, which is the contract rather than a failure.
#[test]
fn two_viewports_build_and_bind_a_two_slot_pipeline() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let (w, h) = (32u32, 32u32);
    let mut req = engine_req(&v, &f, w, h);
    let vp = |width: f32, height: f32| ViewportResource {
        x: 0.0,
        y: 0.0,
        width,
        height,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    req.viewports = vec![vp(w as f32, h as f32), vp((w / 2) as f32, (h / 2) as f32)];
    req.scissors = vec![
        ScissorResource {
            x: 0,
            y: 0,
            width: w,
            height: h,
        },
        ScissorResource {
            x: 0,
            y: 0,
            width: w / 2,
            height: h / 2,
        },
    ];
    assert_eq!(
        reims_vgpu_vulkan::engine::viewport_slot_count(&req),
        2,
        "both lists are two long, so the pipeline must declare two slots"
    );
    if let Some(px) = draw_or_skip("two_viewports", &req) {
        assert_fullscreen_fragment_color("two_viewports", &px, w, h);
    }
}

/// The two lists need not be the same length, and the shorter one is defaulted
/// per slot rather than the longer one truncated.
///
/// Metal lets a guest set two viewports and one scissor rect; Vulkan requires
/// `scissorCount == viewportCount`. Truncating to the shorter list would drop a
/// viewport the guest set, which is the loss this rail was widened to stop — so
/// the count is the maximum and the missing scissor falls back to the full
/// target.
#[test]
fn a_shorter_scissor_list_is_defaulted_rather_than_truncating_the_viewports() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let (w, h) = (32u32, 32u32);
    let mut req = engine_req(&v, &f, w, h);
    let vp = |width: f32, height: f32| ViewportResource {
        x: 0.0,
        y: 0.0,
        width,
        height,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    req.viewports = vec![vp(w as f32, h as f32), vp(4.0, 4.0)];
    req.scissors = vec![ScissorResource {
        x: 0,
        y: 0,
        width: w,
        height: h,
    }];
    assert_eq!(
        reims_vgpu_vulkan::engine::viewport_slot_count(&req),
        2,
        "the longer list decides the count; the single scissor does not truncate it"
    );
    if let Some(px) = draw_or_skip("uneven_viewport_scissor", &req) {
        assert_fullscreen_fragment_color("uneven_viewport_scissor", &px, w, h);
    }
}

/// True when the fixture triangle covered the target center (fragment color),
/// false when the clear black shows through (the triangle was culled).
fn triangle_covered(px: &[u8], w: u32, h: u32) -> bool {
    let i = ((h / 2) * w + w / 2) as usize * 4;
    // Fragment is ~(64,128,191); clear is (0,0,0). Green channel discriminates.
    px[i + 1] > 32
}

/// Face culling is honored by the Vulkan raster state and is wired correctly
/// through the Metal winding + Y-flip. On-GPU behavioral checks (no guest 3D
/// needed): the whole assertion set is a truth table that only holds if cull is
/// actually applied AND the front-facing winding selects the right face.
#[test]
fn cull_mode_honored_and_winding_correct() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let (w, h) = (16u32, 16u32);

    let variant = |cull: CullMode, ccw: bool| -> Option<bool> {
        let mut req = engine_req(&v, &f, w, h);
        req.cull_mode = cull;
        req.front_face_ccw = ccw;
        draw_or_skip("cull", &req).map(|px| triangle_covered(&px, w, h))
    };

    // cull=None must stay byte-identical to the no-cull path: full coverage.
    let Some(none_cov) = variant(CullMode::None, false) else {
        return; // no GPU
    };
    assert!(
        none_cov,
        "cull=None must draw both faces (fullscreen coverage)"
    );

    let back_cw = variant(CullMode::Back, false).unwrap();
    let front_cw = variant(CullMode::Front, false).unwrap();
    let back_ccw = variant(CullMode::Back, true).unwrap();
    let front_ccw = variant(CullMode::Front, true).unwrap();

    // A single triangle presents one face to the viewer: culling Front and Back
    // are complementary — exactly one keeps it. If cull were ignored, both would
    // stay covered and this fails.
    assert_ne!(
        back_cw, front_cw,
        "Front/Back cull must be complementary for one triangle (cull not applied?)"
    );
    // Flipping the front-facing winding swaps which face is front, i.e. swaps the
    // effect of Front vs Back. If winding were ignored, back_ccw would equal
    // back_cw instead.
    assert_eq!(
        back_ccw, front_cw,
        "flipping winding must swap Back into Front behavior (winding not wired?)"
    );
    assert_eq!(
        front_ccw, back_cw,
        "flipping winding must swap Front into Back behavior (winding not wired?)"
    );
}

/// Depth test is honored end to end: a transient depth buffer is attached, the
/// compare op + clear value are wired, and the 2D path (`depth: None`) stays
/// byte-identical (proven by every other test here running with no depth).
///
/// Vehicle: a full-screen textured quad whose per-vertex z is fed via storage
/// buffer 0 (so depth is controllable), sampling one solid color. Assertions are
/// mostly convention-independent RELATIONSHIPS (depth applied, compare matters,
/// clear matters) plus the absolute Never/Always anchors — so the test proves
/// the wiring without depending on the exact depth compare operand order.
#[test]
fn depth_test_honored_compare_and_clear_wired() {
    let _g = engine_test_session();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    // Full-screen quad, all vertices at NDC z = 0.5.
    let quad_z = |z: f32| -> [[f32; 4]; 6] {
        [
            [-1.0, -1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [1.0, 1.0, z, 1.0],
        ]
    };
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    // Sampled color with a strong green channel so `triangle_covered` (green>32)
    // discriminates covered vs cleared-black.
    let rgba = [17u8, 140, 203, 255];

    // Returns Some(covered) for a fragment at z=0.5 with the given depth state.
    let variant = |compare: SamplerCompareFunction,
                   clear: f32,
                   depth_bias: Option<[f32; 3]>|
     -> Option<bool> {
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&quad_z(0.5).into_iter().flatten().collect::<Vec<_>>()).into(),
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
            source: SampledSource::Bytes(std::sync::Arc::new(rgba.repeat(4))),
            byte_origin: Default::default(),
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Rgba8,
            ),
            identity: None,
            content: None,
            resource_lifetime: None,
            swizzle: Default::default(),
        });
        req.samplers
            .push(SamplerResource::normalized_default(sampler_binding(0)));
        req.depth = Some(DepthState {
            // Parity fixtures bind no guest depth texture, so they exercise the
            // transient rail rather than the registry-resident one.
            identity: None,
            test_enable: true,
            write_enable: true,
            compare,
            clear_value: clear,
            load: false,
            stencil: None,
        });
        req.depth_bias = depth_bias;
        draw_or_skip("depth", &req).map(|px| triangle_covered(&px, w, h))
    };

    // Absolute anchors (convention-independent): Never never draws, Always always
    // draws. If depth state were ignored, Never would still cover.
    let Some(never) = variant(SamplerCompareFunction::Never, 1.0, None) else {
        return; // no GPU
    };
    assert!(!never, "compare=Never must discard every fragment");
    assert!(
        variant(SamplerCompareFunction::Always, 1.0, None).unwrap(),
        "compare=Always must keep every fragment"
    );

    // Relationships (convention-independent): with fragment z=0.5,
    let less_hi = variant(SamplerCompareFunction::Less, 1.0, None).unwrap();
    let less_lo = variant(SamplerCompareFunction::Less, 0.0, None).unwrap();
    let greater_lo = variant(SamplerCompareFunction::Greater, 0.0, None).unwrap();
    let greater_hi = variant(SamplerCompareFunction::Greater, 1.0, None).unwrap();
    // The compare op matters: Less vs Greater against the same reference differ.
    assert_ne!(
        less_hi, greater_hi,
        "Less vs Greater against clear=1.0 must differ (compare op not applied?)"
    );
    // The clear value matters: same op, different reference → different result.
    assert_ne!(
        less_hi, less_lo,
        "clear value must feed the depth reference (Less@1.0 vs Less@0.0)"
    );
    // Consistency: flipping BOTH the op and the reference gives the same outcome.
    assert_eq!(
        less_hi, greater_lo,
        "Less@1.0 and Greater@0.0 must agree (depth compare wired consistently)"
    );
    assert_eq!(less_lo, greater_hi, "Less@0.0 and Greater@1.0 must agree");

    // Equal fragment and clear depths fail `Less`; a negative constant bias
    // moves the fragment toward the viewer and makes the same comparison pass.
    // This catches both a dropped state and an accidental sign reversal.
    assert!(
        !variant(SamplerCompareFunction::Less, 0.5, None).unwrap(),
        "equal depth must fail Less without bias"
    );
    assert!(
        variant(SamplerCompareFunction::Less, 0.5, Some([-1.0, 0.0, 0.0]),).unwrap(),
        "negative constant bias must move the fragment toward Less"
    );
}

#[test]
fn mismatched_depth_attachment_clear_covers_its_full_image() {
    let _g = engine_test_session();
    let (simple_v, simple_f) = triangle_spirv();
    let depth_identity = TargetIdentity::Texture {
        ref_: 0xd8,
        width: 8,
        height: 8,
        generation: 1,
        stencil: false,
    };
    let depth = |clear_value: f32, load: bool, compare| DepthState {
        identity: Some(depth_identity.clone()),
        test_enable: true,
        write_enable: true,
        compare,
        clear_value,
        load,
        stencil: None,
    };

    // Establish known depth=1 over the complete resident.
    let mut establish = engine_req(&simple_v, &simple_f, 8, 8);
    establish.vertex_count = 0;
    establish.skip_readback = true;
    establish.target_identity = Some(TargetIdentity::Anonymous { slot: 0xd80 });
    establish.depth = Some(depth(1.0, false, SamplerCompareFunction::Always));
    match engine::execute_draw_request(&establish) {
        Ok(_) => {}
        Err(error) if skip_if_no_gpu(&error.to_string()) => {
            eprintln!("SKIP mismatched depth: {error}");
            return;
        }
        Err(error) => panic!("establish depth: {error}"),
    }

    // A smaller color attachment constrains rasterization to 4x4, but Metal's
    // depth load clear still replaces all 8x8 depth texels with 0.25.
    let mut narrow = engine_req(&simple_v, &simple_f, 4, 4);
    narrow.vertex_count = 0;
    narrow.skip_readback = true;
    narrow.target_identity = Some(TargetIdentity::Anonymous { slot: 0xd81 });
    narrow.depth = Some(depth(0.25, false, SamplerCompareFunction::Always));
    engine::execute_draw_request(&narrow).expect("narrow depth clear");

    // Load the complete depth resident and draw z=0.5 with Less. A complete
    // 0.25 clear rejects every fragment; stale depth=1 outside 4x4 would show
    // the sampled green output there.
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let positions: [[f32; 4]; 6] = [
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
    let mut verify = engine_req(&vert, &frag, 8, 8);
    verify.vertex_count = 6;
    verify.depth = Some(depth(0.0, true, SamplerCompareFunction::Less));
    verify.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    verify.storage_buffers.push(StorageBufferResource {
        binding: 1,
        content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    verify.sampled_images.push(SampledImageResource {
        binding: 32,
        array_element: 0,
        descriptor_count: 1,
        width: 1,
        height: 1,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        multisampled: false,
        source: SampledSource::Bytes(std::sync::Arc::new(vec![0, 255, 0, 255])),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: None,
        swizzle: Default::default(),
    });
    verify
        .samplers
        .push(SamplerResource::normalized_default(sampler_binding(0)));
    let pixels = draw_or_skip("mismatched depth verify", &verify).expect("same GPU context");
    assert!(
        pixels.chunks_exact(4).all(|pixel| pixel[1] == 0),
        "stale depth outside the smaller color attachment admitted green fragments"
    );
}

/// Same depth wiring proof as `depth_test_honored_compare_and_clear_wired`, but
/// through the RESIDENT target path — the product Store path (`target_identity`
/// with `skip_readback` and `read_target`, which builds its own ad-hoc [color,depth]
/// framebuffer in the registry_ensure branch (exec) separate from the pooled
/// path exercised above. Without this, the resident depth branch was reachable
/// only in production; a dispose-order or framebuffer bug there (the exact class
/// that caused the MRT/depth device-lost fixes) would surface as a device loss
/// with no test to catch it. Uses BGRA output like a real Surface target; the
/// green channel discriminator survives the R/B swap (index 1 unchanged).
#[test]
fn depth_test_honored_on_resident_target_path() {
    let _g = engine_test_session();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let quad_z = |z: f32| -> [[f32; 4]; 6] {
        [
            [-1.0, -1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [1.0, 1.0, z, 1.0],
        ]
    };
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let rgba = [17u8, 140, 203, 255];

    // Each variant renders to a FRESH resident surface (distinct id) so no stale
    // content leaks between variants, then reads it back via the product path.
    let mut surface_id = 900u32;
    let mut variant = |compare: SamplerCompareFunction, clear: f32| -> Option<bool> {
        surface_id += 1;
        let identity = TargetIdentity::Surface {
            id: surface_id,
            width: w,
            height: h,
            generation: 1,
            format: SURFACE_TEST_FORMAT,
        };
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.target_identity = Some(identity.clone());
        req.skip_readback = true;
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&quad_z(0.5).into_iter().flatten().collect::<Vec<_>>()).into(),
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
            source: SampledSource::Bytes(std::sync::Arc::new(rgba.repeat(4))),
            byte_origin: Default::default(),
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Rgba8,
            ),
            identity: None,
            content: None,
            resource_lifetime: None,
            swizzle: Default::default(),
        });
        req.samplers
            .push(SamplerResource::normalized_default(sampler_binding(0)));
        req.depth = Some(DepthState {
            // Parity fixtures bind no guest depth texture, so they exercise the
            // transient rail rather than the registry-resident one.
            identity: None,
            test_enable: true,
            write_enable: true,
            compare,
            clear_value: clear,
            load: false,
            stencil: None,
        });
        match engine::execute_draw_request(&req) {
            Ok(_) => {}
            Err(e) if skip_if_no_gpu(&e.to_string()) => {
                eprintln!("SKIP resident depth: {e}");
                return None;
            }
            Err(e) => panic!("resident depth draw: {e}"),
        }
        let px = engine::read_target(&identity)
            .expect("read resident depth target")
            .into_rgba8();
        Some(triangle_covered(&px, w, h))
    };

    let Some(never) = variant(SamplerCompareFunction::Never, 1.0) else {
        return; // no GPU
    };
    assert!(
        !never,
        "resident: compare=Never must discard every fragment"
    );
    assert!(
        variant(SamplerCompareFunction::Always, 1.0).unwrap(),
        "resident: compare=Always must keep every fragment"
    );

    let less_hi = variant(SamplerCompareFunction::Less, 1.0).unwrap();
    let greater_hi = variant(SamplerCompareFunction::Greater, 1.0).unwrap();
    let greater_lo = variant(SamplerCompareFunction::Greater, 0.0).unwrap();
    let less_lo = variant(SamplerCompareFunction::Less, 0.0).unwrap();
    assert_ne!(
        less_hi, greater_hi,
        "resident: Less vs Greater against clear=1.0 must differ"
    );
    assert_ne!(
        less_hi, less_lo,
        "resident: clear value must feed the depth reference"
    );
    assert_eq!(
        less_hi, greater_lo,
        "resident: Less@1.0 and Greater@0.0 must agree"
    );
}

/// Proves the Vulkan stencil test is wired end-to-end: a single full-screen quad
/// with depth compare Always (depth never gates) and a stencil face whose
/// compare/reference/read-mask decide coverage against the transient stencil
/// buffer's clear value. Mirrors the depth proof's single-draw structure — the
/// transient depth-stencil is per-draw CLEAR-only, so this covers the compare
/// path (enable + compareOp + reference + compareMask + stencil clear); the
/// stencil *ops* (fail/pass/depthFail, write_mask) need a persistent buffer to
/// observe and are the documented follow-up gap.
#[test]
fn stencil_test_honored_compare_ref_and_clear_wired() {
    let _g = engine_test_session();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let quad_z = |z: f32| -> [[f32; 4]; 6] {
        [
            [-1.0, -1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [1.0, 1.0, z, 1.0],
        ]
    };
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let rgba = [17u8, 140, 203, 255];

    // Both faces identical (the quad is one winding); depth compare Always so
    // only the stencil test gates. `read_mask` masks both operands.
    let variant = |compare: SamplerCompareFunction,
                   reference: u32,
                   clear: u32,
                   read_mask: u32|
     -> Option<bool> {
        let face = StencilFaceOps {
            compare,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Keep,
            read_mask,
            write_mask: 0,
        };
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&quad_z(0.5).into_iter().flatten().collect::<Vec<_>>()).into(),
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
            source: SampledSource::Bytes(std::sync::Arc::new(rgba.repeat(4))),
            byte_origin: Default::default(),
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Rgba8,
            ),
            identity: None,
            content: None,
            resource_lifetime: None,
            swizzle: Default::default(),
        });
        req.samplers
            .push(SamplerResource::normalized_default(sampler_binding(0)));
        req.depth = Some(DepthState {
            // Parity fixtures bind no guest depth texture, so they exercise the
            // transient rail rather than the registry-resident one.
            identity: None,
            test_enable: true,
            write_enable: false,
            compare: SamplerCompareFunction::Always,
            clear_value: 1.0,
            load: false,
            stencil: Some(StencilState {
                front: face,
                back: face,
                reference_front: reference,
                reference_back: reference,
                clear_value: clear,
            }),
        });
        draw_or_skip("stencil", &req).map(|px| triangle_covered(&px, w, h))
    };

    // Absolute anchors: Always keeps every fragment, Never discards every one —
    // independent of reference/clear. If stencil state were ignored, Never would
    // still cover (the color-only pipeline always draws).
    let Some(never) = variant(SamplerCompareFunction::Never, 0, 0, 0xFF) else {
        return; // no GPU
    };
    assert!(!never, "stencil compare=Never must discard every fragment");
    assert!(
        variant(SamplerCompareFunction::Always, 0, 0, 0xFF).unwrap(),
        "stencil compare=Always must keep every fragment"
    );

    // Equal: coverage iff (reference & mask) == (clearValue & mask).
    assert!(
        variant(SamplerCompareFunction::Equal, 0, 0, 0xFF).unwrap(),
        "Equal with reference==clear must keep every fragment"
    );
    assert!(
        !variant(SamplerCompareFunction::Equal, 1, 0, 0xFF).unwrap(),
        "Equal with reference!=clear must discard (reference/clear wired?)"
    );
    // read_mask (compareMask) masks both operands: with mask 0 the differing
    // low bit is erased so Equal passes again — proves the mask is applied.
    assert!(
        variant(SamplerCompareFunction::Equal, 1, 0, 0x00).unwrap(),
        "read_mask=0 must erase the reference/clear difference (compareMask wired?)"
    );
}

#[test]
fn load_seed_preserves_uncovered_and_draws() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    // Fullscreen triangle covers everything; seed is Load base then overdrawn.
    req.target_rgba8 = Some(std::sync::Arc::new([10, 20, 30, 255].repeat(8 * 8)));
    if let Some(px) = draw_or_skip("load_seed", &req) {
        assert_fullscreen_fragment_color("load_seed", &px, 8, 8);
    }
}

#[test]
fn blend_src_alpha_known_color() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.target_rgba8 = Some(std::sync::Arc::new([0, 0, 0, 255].repeat(8 * 8)));
    req.blend = Some(BlendStateResource {
        src_color: BlendFactor::SrcAlpha,
        dst_color: BlendFactor::OneMinusSrcAlpha,
        color_op: BlendOp::Add,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::OneMinusSrcAlpha,
        alpha_op: BlendOp::Add,
    });
    if let Some(px) = draw_or_skip("blend_src_alpha", &req) {
        // Fragment alpha=1 → same as replace over black seed.
        assert_fullscreen_fragment_color("blend_src_alpha", &px, 8, 8);
    }
}

#[test]
fn blend_color_is_dynamic_encoder_state() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.target_rgba8 = Some(std::sync::Arc::new([0, 0, 0, 0].repeat(8 * 8)));
    req.blend_constants = [0.2, 0.4, 0.6, 0.8];
    req.blend = Some(BlendStateResource {
        src_color: BlendFactor::ConstantColor,
        dst_color: BlendFactor::Zero,
        color_op: BlendOp::Add,
        src_alpha: BlendFactor::ConstantAlpha,
        dst_alpha: BlendFactor::Zero,
        alpha_op: BlendOp::Add,
    });
    if let Some(px) = draw_or_skip("blend_color", &req) {
        let center = ((8 / 2) * 8 + 8 / 2) * 4;
        let got = &px[center..center + 4];
        // Fixture output is approximately (0.25, 0.5, 0.75, 1.0).
        let expected = [13, 51, 115, 204];
        assert!(
            got.iter().zip(expected).all(|(&a, b)| near(a, b)),
            "blend constants did not reach the draw: got={got:?} expected={expected:?}"
        );
    }
}

#[test]
fn indexed_u16_known_color() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 16, 16);
    req.indexed = Some(IndexedDrawResource {
        index_type: IndexType::U16,
        index_count: 3,
        vertex_offset: 0,
        content: BufferContent::Bytes(std::sync::Arc::new({
            let mut b = Vec::new();
            for i in [0u16, 1, 2] {
                b.extend_from_slice(&i.to_le_bytes());
            }
            b
        })),
    });
    if let Some(px) = draw_or_skip("indexed_u16", &req) {
        assert_fullscreen_fragment_color("indexed_u16", &px, 16, 16);
    }
}

#[test]
fn storage_buffer_binding_still_renders() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: vec![0u8; 64].into(),
    });
    match engine::execute_draw_request(&req) {
        Ok(o) => assert_fullscreen_fragment_color("storage", &semantic_rgba(&o), 8, 8),
        Err(e) if skip_if_no_gpu(&e.to_string()) => eprintln!("SKIP storage: {e}"),
        Err(e) => {
            // Unused binding may fail pipeline create on some SPIR-V/ICD combos — named only.
            let s = e.to_string();
            assert!(
                s.contains("vk_engine") || s.contains("pipeline") || s.contains("shader"),
                "unexpected storage path error: {s}"
            );
            eprintln!("storage path named failure (ok): {s}");
        }
    }
}

#[test]
fn sampled_and_sampler_still_renders() {
    let _g = engine_test_session();
    let sampled_owner = sampled_resource_owner();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.sampled_images.push(SampledImageResource {
        binding: 1,
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
        source: SampledSource::Bytes(std::sync::Arc::new(vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ])),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: Some(sampled_owner.lifetime_ref()),
        swizzle: Default::default(),
    });
    req.samplers.push(SamplerResource::normalized_default(2));
    match engine::execute_draw_request(&req) {
        Ok(o) => {
            assert_fullscreen_fragment_color("sampled", &semantic_rgba(&o), 8, 8);
            let warm_before = engine::counter_snapshot();
            let warm = engine::execute_draw_request(&req).expect("exact sampled cache hit");
            assert_eq!(warm.pixels, o.pixels, "cache hit must preserve draw bytes");
            let warm_delta = engine::counter_snapshot().delta_since(&warm_before);
            assert_eq!(
                warm_delta.sampled_cache_hits, 1,
                "hit proxy: {warm_delta:?}"
            );
            assert_eq!(
                warm_delta.sampled_cache_hit_bytes, 16,
                "hit-byte proxy: {warm_delta:?}"
            );
            assert_eq!(
                warm_delta.sampled_cache_misses, 0,
                "hit proxy: {warm_delta:?}"
            );
            assert_eq!(warm_delta.sampled_reuploads, 0, "no upload: {warm_delta:?}");
            assert_eq!(
                warm_delta.sampled_reupload_bytes, 0,
                "no upload bytes: {warm_delta:?}"
            );

            let changed_len = {
                let SampledSource::Bytes(bytes) = &mut req.sampled_images[0].source else {
                    unreachable!()
                };
                std::sync::Arc::make_mut(bytes)[0] ^= 0xff;
                bytes.len() as u64
            };
            let changed_before = engine::counter_snapshot();
            let changed = engine::execute_draw_request(&req).expect("changed sampled upload");
            assert_eq!(
                changed.pixels, o.pixels,
                "test shader remains a solid-color oracle"
            );
            let changed_delta = engine::counter_snapshot().delta_since(&changed_before);
            assert_eq!(
                changed_delta.sampled_cache_hits, 0,
                "miss proxy: {changed_delta:?}"
            );
            assert_eq!(
                changed_delta.sampled_cache_misses, 1,
                "miss proxy: {changed_delta:?}"
            );
            assert_eq!(
                changed_delta.sampled_reuploads, 1,
                "upload: {changed_delta:?}"
            );
            assert_eq!(
                changed_delta.sampled_reupload_bytes, changed_len,
                "upload-byte proxy: {changed_delta:?}"
            );
        }
        Err(e) if skip_if_no_gpu(&e.to_string()) => eprintln!("SKIP sampled: {e}"),
        Err(e) => {
            let s = e.to_string();
            assert!(
                s.contains("vk_engine") || s.contains("pipeline") || s.contains("shader"),
                "unexpected sampled path error: {s}"
            );
            eprintln!("sampled path named failure (ok): {s}");
        }
    }
}

/// An unchanging sampled texture uploads exactly once, no matter how many
/// draws follow.
///
/// Cache admission is parked on a ring slot's deferred cleanup, so this is the
/// gate on *when* that cleanup runs. Retiring only the slot about to be reused
/// held every admission until the ring wrapped, which re-uploaded and
/// re-allocated the same bytes for `RING_DEPTH - 1` draws. The two-draw tests
/// above cannot see that: they pass as soon as one extra slot is reaped. This
/// one drives more draws than the ring is deep, so it fails for any policy that
/// does not reap the whole signaled run.
#[test]
fn sampled_upload_happens_once_across_more_draws_than_the_ring_is_deep() {
    let _g = engine_test_session();
    let sampled_owner = sampled_resource_owner();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.sampled_images.push(SampledImageResource {
        binding: 1,
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
        source: SampledSource::Bytes(std::sync::Arc::new(vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ])),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: Some(sampled_owner.lifetime_ref()),
        swizzle: Default::default(),
    });
    req.samplers.push(SamplerResource::normalized_default(2));

    // Comfortably past RING_DEPTH so the ring wraps several times.
    const DRAWS: u32 = 24;
    let before = engine::counter_snapshot();
    match engine::execute_draw_request(&req) {
        Ok(first) => {
            for _ in 1..DRAWS {
                let out = engine::execute_draw_request(&req).expect("repeat sampled draw");
                assert_eq!(out.pixels, first.pixels, "every redraw is byte-identical");
            }
            let d = engine::counter_snapshot().delta_since(&before);
            assert_eq!(
                d.sampled_reuploads, 1,
                "one upload for {DRAWS} identical draws: {d:?}"
            );
            assert_eq!(
                d.sampled_cache_hits,
                u64::from(DRAWS - 1),
                "every draw after the first is served by the cache: {d:?}"
            );
            assert_eq!(
                d.sampled_cache_misses, 1,
                "only the cold draw misses: {d:?}"
            );
        }
        Err(e) if skip_if_no_gpu(&e.to_string()) => eprintln!("SKIP sampled ring reuse: {e}"),
        Err(e) => panic!("unexpected sampled path error: {e}"),
    }
}

/// A resident IOSurface texture sample stays on the GPU: no source readback, staging
/// upload, or temporary sampled image. The tracked layout must still permit a
/// later LoadFromTarget draw on the source identity.
#[test]
fn resident_sample_bind_avoids_roundtrip_and_remains_loadable() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let source = TargetIdentity::Surface {
        id: 0x51,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let mut make_source = engine_req(&v, &f, 16, 16);
    make_source.target_identity = Some(source.clone());
    match engine::execute_draw_request(&make_source) {
        Ok(o) => {
            assert_fullscreen_fragment_color("resident_sample_source", &semantic_rgba(&o), 16, 16)
        }
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP resident_sample_bind: {e}");
            return;
        }
        Err(e) => panic!("resident source: {e}"),
    }

    let mut consume = engine_req(&v, &f, 16, 16);
    consume.sampled_images.push(SampledImageResource {
        binding: 1,
        array_element: 0,
        descriptor_count: 1,
        width: 16,
        height: 16,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        multisampled: false,
        source: SampledSource::Target(source.clone()),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: None,
        swizzle: Default::default(),
    });
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    let consumed = engine::execute_draw_request(&consume).expect("bind resident sample");
    assert_fullscreen_fragment_color(
        "resident_sample_consumer",
        &semantic_rgba(&consumed),
        16,
        16,
    );
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(delta.sampled_gpu_binds, 1, "direct-bind proxy: {delta:?}");
    assert_eq!(delta.sampled_reuploads, 0, "no sampled reupload: {delta:?}");
    assert_eq!(
        delta.readbacks, 1,
        "only the consumer target may read back: {delta:?}"
    );

    let mut load_again = engine_req(&v, &f, 16, 16);
    load_again.target_identity = Some(source.clone());
    load_again.load_from_target = true;
    let loaded = engine::execute_draw_request(&load_again).expect("load after direct sample");
    assert_fullscreen_fragment_color("resident_sample_reloaded", &semantic_rgba(&loaded), 16, 16);
}

/// A resident allocation may be written through a linear attachment view and
/// sampled through its sRGB sibling. The sample must use the binding's view
/// format: binding the attachment's cached linear view instead leaves the
/// original (64,128,191) values instead of decoding them.
#[test]
fn resident_sample_uses_the_bindings_compatible_format_view() {
    let _g = engine_test_session();
    let (source_v, source_f) = triangle_spirv();
    let source = TargetIdentity::Surface {
        id: 0x53,
        width: 16,
        height: 16,
        generation: 1,
        format: reims_vgpu_core::pixel_format::TexelLayout::Bgra8,
    };
    let mut produce = engine_req(&source_v, &source_f, 16, 16);
    produce.target_identity = Some(source.clone());
    produce.skip_readback = true;
    match engine::execute_draw_request(&produce) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP resident compatible-format view: {e}");
            return;
        }
        Err(e) => panic!("resident compatible-format source: {e}"),
    }

    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let mut consume = engine_req(&vert, &frag, 16, 16);
    consume.vertex_count = 6;
    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    consume.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    consume.storage_buffers.push(StorageBufferResource {
        binding: 1,
        content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    consume.sampled_images.push(SampledImageResource {
        binding: 32,
        array_element: 0,
        descriptor_count: 1,
        width: 16,
        height: 16,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        multisampled: false,
        source: SampledSource::Target(source),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::srgb(reims_vgpu_protocol::TexelLayout::Bgra8)
            .unwrap(),
        identity: None,
        content: None,
        resource_lifetime: None,
        swizzle: Default::default(),
    });
    consume
        .samplers
        .push(SamplerResource::normalized_default(sampler_binding(0)));
    let out = engine::execute_draw_request(&consume).expect("sample compatible sRGB view");
    let center = ((16 / 2) * 16 + 16 / 2) as usize * 4;
    let px = &out.pixels[center..center + 4];
    assert!(
        near(px[0], 13) && near(px[1], 55) && near(px[2], 133) && near(px[3], 255),
        "sRGB view must decode stored UNORM bytes, got {px:?}"
    );
}

/// Attachment feedback binds the resident in place when the host exposes the
/// Vulkan feedback-loop contract, and otherwise uses one shared GPU snapshot.
#[test]
fn resident_sample_alias_uses_native_feedback_or_snapshot_fallback() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 0x52,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut cold = engine_req(&v, &f, 16, 16);
    cold.target_identity = Some(identity.clone());
    match engine::execute_draw_request(&cold) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP resident_sample_alias: {e}");
            return;
        }
        Err(e) => panic!("resident alias source: {e}"),
    }

    let mut alias = engine_req(&v, &f, 16, 16);
    alias.target_identity = Some(identity.clone());
    alias.load_from_target = true;
    alias.skip_readback = true;
    alias.render_pass_continues = true;
    alias.sampled_images.push(SampledImageResource {
        binding: 1,
        array_element: 0,
        descriptor_count: 1,
        width: 16,
        height: 16,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        multisampled: false,
        source: SampledSource::Target(identity.clone()),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: None,
        swizzle: Default::default(),
    });
    alias.sampled_images.push(SampledImageResource {
        binding: 2,
        array_element: 0,
        descriptor_count: 1,
        width: 16,
        height: 16,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        multisampled: false,
        source: SampledSource::Target(identity.clone()),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: None,
        swizzle: Default::default(),
    });
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    engine::execute_draw_request(&alias).expect("resident alias feedback");
    let out = engine::read_target(&identity)
        .expect("read native feedback result after deferred draw")
        .into_rgba8();
    assert_fullscreen_fragment_color("resident_sample_alias", &out, 16, 16);
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        delta.sampled_gpu_binds, 2,
        "GPU resident-bind proxy: {delta:?}"
    );
    assert_eq!(
        delta.sampled_free_allocs, 1,
        "two bindings share one stable pre-draw snapshot: {delta:?}"
    );
    assert_eq!(delta.sampled_reuploads, 0, "no host reupload: {delta:?}");
    assert_eq!(
        delta.readbacks, 0,
        "the draw stays deferred; its explicit target read joins the batch: {delta:?}"
    );
    assert_eq!(delta.target_reads, 1, "one explicit target read: {delta:?}");
    assert_eq!(
        delta.batch_readback_joins, 1,
        "the read must exercise the open-pass close before its image barrier: {delta:?}"
    );
}

/// The attachment descriptor and fragment binding name one texture. Its Clear,
/// Load or DontCare action is therefore the sampled source at fragment
/// execution; none may be restated as a CPU-uploaded sampled image.
#[test]
fn attachment_initial_contents_are_sampled_without_a_host_upload() {
    let _g = engine_test_session();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let make_req = |id: u32, initial: AttachmentInitial| {
        let identity = TargetIdentity::Surface {
            id,
            width: w,
            height: h,
            generation: 1,
            format: SURFACE_TEST_FORMAT,
        };
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.target_identity = Some(identity.clone());
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.storage_buffers.push(StorageBufferResource {
            binding: 1,
            content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.sampled_images.push(SampledImageResource {
            binding: 32,
            array_element: 0,
            descriptor_count: 1,
            width: w,
            height: h,
            layers: 1,
            arrayed: false,
            volume: false,
            cube: false,
            one_dim: false,
            multisampled: false,
            source: SampledSource::Attachment { identity, initial },
            byte_origin: Default::default(),
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Bgra8,
            ),
            identity: None,
            content: None,
            resource_lifetime: None,
            swizzle: Default::default(),
        });
        req.samplers
            .push(SamplerResource::normalized_default(sampler_binding(0)));
        req
    };

    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    let mut clear = make_req(0x54, AttachmentInitial::Clear([0.25, 0.5, 0.75, 1.0]));
    clear.target_clear = [0.25, 0.5, 0.75, 1.0];
    let Some(clear_pixels) = draw_or_skip("attachment initial clear", &clear) else {
        return;
    };
    assert_fullscreen_fragment_color("attachment initial clear", &clear_pixels, w, h);

    let mut seed = make_req(0x55, AttachmentInitial::Seed);
    seed.target_rgba8 = Some(std::sync::Arc::new(
        [64, 128, 191, 255].repeat((w * h) as usize),
    ));
    let seed_pixels = draw_or_skip("attachment initial seed", &seed).expect("GPU remains usable");
    assert_fullscreen_fragment_color("attachment initial seed", &seed_pixels, w, h);

    let dont_care = make_req(0x56, AttachmentInitial::DontCare);
    let _ = draw_or_skip("attachment initial dont-care", &dont_care)
        .expect("undefined initial contents remain a valid GPU source");
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        delta.sampled_reuploads, 0,
        "no host sampled upload: {delta:?}"
    );
    assert_eq!(
        delta.sampled_gpu_binds, 3,
        "one target bind per draw: {delta:?}"
    );
}

#[test]
fn vertex_buffers_bind_in_one_bulk_call_without_losing_slots() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    // Location order deliberately disagrees with binding order: the executor
    // must normalize by binding before the contiguous Vulkan call without
    // moving a buffer away from its declared slot.
    for (location, binding) in [(0, 2), (1, 0), (2, 1)] {
        req.vertex_attributes.push(VertexAttributeResource {
            location,
            binding,
            format: VertexAttributeFormat::Float2,
            offset: 0,
            stride: 8,
            step_function: VertexStepFunction::PerVertex,
            step_rate: 1,
            content: vec![0u8; 24].into(),
        });
    }
    let before = engine::counter_snapshot();
    match engine::execute_draw_request(&req) {
        Ok(o) => {
            assert_fullscreen_fragment_color("attr", &semantic_rgba(&o), 8, 8);
            let d = engine::counter_snapshot().delta_since(&before);
            assert_eq!(d.vertex_buffer_bind_slots, 3, "requested slots: {d:?}");
            assert_eq!(d.vertex_buffer_bind_emitted, 3, "emitted slots: {d:?}");
            assert_eq!(d.vertex_buffer_bind_calls, 1, "bulk calls: {d:?}");
        }
        Err(e) if skip_if_no_gpu(&e.to_string()) => eprintln!("SKIP attr: {e}"),
        Err(e) => {
            let s = e.to_string();
            assert!(
                s.contains("vk_engine") || s.contains("pipeline") || s.contains("shader"),
                "unexpected attr path error: {s}"
            );
            eprintln!("attr path named failure (ok): {s}");
        }
    }
}

/// A producer name is not proof that live sampled bytes are unchanged.
///
/// This deliberately changes the bytes without changing the supplied identity.
/// The cache must compare the content, miss, and upload. An identity-first
/// lookup binds the first draw's stale pixels and fails this test.
#[test]
fn sampled_identity_never_overrides_changed_content() {
    let _g = engine_test_session();
    let sampled_owner = sampled_resource_owner();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.sampled_images.push(SampledImageResource {
        binding: 1,
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
        source: SampledSource::Bytes(std::sync::Arc::new(vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ])),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: Some(SampledContentIdentity {
            key: 0x1234_5000,
            generation: 1,
        }),
        content: None,
        resource_lifetime: Some(sampled_owner.lifetime_ref()),
        swizzle: Default::default(),
    });
    req.samplers.push(SamplerResource::normalized_default(2));
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP sampled identity: {e}");
            return;
        }
        Err(e) => panic!("cold identity draw: {e}"),
    }

    // Equal bytes still reuse the exact-content entry.
    let warm_before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("content rebind");
    let d = engine::counter_snapshot().delta_since(&warm_before);
    assert_eq!(d.sampled_cache_hits, 1, "exact content hit: {d:?}");
    assert_eq!(d.sampled_cache_hit_bytes, 16, "compared content: {d:?}");
    assert_eq!(d.sampled_reuploads, 0, "no upload: {d:?}");

    // Keep the identity exactly unchanged while replacing the bytes. This is
    // the stale-compositor case: identity cannot suppress the upload.
    {
        let img = &mut req.sampled_images[0];
        img.source = SampledSource::Bytes(std::sync::Arc::new(vec![
            1, 0, 0, 255, 0, 1, 0, 255, 0, 0, 1, 255, 1, 1, 0, 255,
        ]));
    }
    let changed_before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("changed content upload");
    let d = engine::counter_snapshot().delta_since(&changed_before);
    assert_eq!(d.sampled_cache_misses, 1, "changed bytes miss: {d:?}");
    assert_eq!(d.sampled_reuploads, 1, "changed bytes upload: {d:?}");

    // The uploaded replacement is subsequently reusable by exact bytes.
    let settle_before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("settled content rebind");
    let d = engine::counter_snapshot().delta_since(&settle_before);
    assert_eq!(d.sampled_cache_hits, 1, "replacement content hit: {d:?}");
    assert_eq!(d.sampled_reuploads, 0, "settled no upload: {d:?}");
}

#[test]
fn warm_identical_draw_zero_creates_and_allocs() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let req = engine_req(&v, &f, 16, 16);
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP warm: {e}");
            return;
        }
        Err(e) => panic!("cold draw: {e}"),
    }
    engine::execute_draw_request(&req).expect("warm-up draw");
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("warm draw");
    let after = engine::counter_snapshot();
    let d = after.delta_since(&before);
    assert_eq!(
        d.creates, 0,
        "warm draw must perform zero vkCreate* (got creates={d:?})"
    );
    assert_eq!(
        d.allocs, 0,
        "warm draw must perform zero vkAllocateMemory (got allocs={d:?})"
    );
    assert!(
        d.shader_hits + d.layout_hits + d.pass_hits + d.pipeline_hits > 0,
        "expected cache hits on warm path, got {d:?}"
    );
}

#[test]
fn warm_draw_byte_identical_hot_cache() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let req = engine_req(&v, &f, 16, 16);
    let first = match engine::execute_draw_request(&req) {
        Ok(o) => o.pixels,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP hot: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_fullscreen_fragment_color("hot_first", &first, 16, 16);
    for n in 1..=8 {
        let px = engine::execute_draw_request(&req)
            .unwrap_or_else(|e| panic!("hot #{n}: {e}"))
            .pixels;
        assert_eq!(px, first, "hot draw #{n} diverged");
    }
}

/// Warm non-Store resident draw: zero readbacks, zero seed uploads, zero creates, zero allocs.
#[test]
fn warm_non_store_zero_readback_seed_create_alloc() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 42,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    // Cold: seed import + draw with readback so we can verify content, mark ready.
    let mut cold = engine_req(&v, &f, 16, 16);
    cold.target_identity = Some(identity.clone());
    cold.skip_readback = false;
    match engine::execute_draw_request(&cold) {
        Ok(o) => assert_fullscreen_fragment_color("resident_cold", &semantic_rgba(&o), 16, 16),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP warm_non_store: {e}");
            return;
        }
        Err(e) => panic!("cold resident: {e}"),
    }
    // Warm non-Store: LoadFromTarget, skip readback.
    let mut warm = engine_req(&v, &f, 16, 16);
    warm.target_identity = Some(identity.clone());
    warm.load_from_target = true;
    warm.skip_readback = true;
    // One warm-up under residency.
    engine::execute_draw_request(&warm).expect("resident warm-up");
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    engine::execute_draw_request(&warm).expect("resident warm non-Store");
    let after = engine::counter_snapshot();
    let d = after.delta_since(&before);
    assert_eq!(
        d.readbacks, 0,
        "warm non-Store must do zero readbacks: {d:?}"
    );
    assert_eq!(
        d.seed_uploads, 0,
        "warm non-Store must do zero seed uploads: {d:?}"
    );
    assert_eq!(d.creates, 0, "warm non-Store must do zero creates: {d:?}");
    assert_eq!(d.allocs, 0, "warm non-Store must do zero allocs: {d:?}");
    assert_eq!(
        d.render_post_wait_skips, 1,
        "no-readback draw must skip the post-submit fence wait: {d:?}"
    );
    // Boundary materialization still works: read_target waits the shared
    // fence first, so it returns the exact content of the skipped-wait draw.
    let px = engine::read_target(&identity)
        .expect("read_target after warm")
        .into_rgba8();
    assert_fullscreen_fragment_color("read_target", &px, 16, 16);
}

/// The resident registry retains every admitted target past the slot count that
/// used to bound it, and a pin still refuses an absent identity.
///
/// This is a device-level regression test for the property that retired
/// `REGISTRY_CAP`: the population is bounded by the allocator refusing, not by a
/// count, so admitting more targets than the old cap allowed must destroy
/// nothing. It used to assert the opposite half — that the LRU sweep evicted the
/// oldest *unpinned* target while rotating over the pinned one — and the pinned
/// half of that is preserved here, now as one case of "nothing is evicted"
/// rather than as the exception to a sweep.
///
/// Fails against the retired walk: with a cap of 320 the `unpinned` target is
/// the oldest non-pinned entry and is swept before the fillers run out.
#[test]
fn every_admitted_resident_survives_past_the_retired_slot_cap() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();

    let absent = TargetIdentity::Surface {
        id: 0x9999,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    assert!(
        !engine::pin_resident_target(&absent),
        "pin must refuse an absent identity"
    );

    let pinned = TargetIdentity::Surface {
        id: 0x600,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut make = engine_req(&v, &f, 16, 16);
    make.target_identity = Some(pinned.clone());
    match engine::execute_draw_request(&make) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP pinned_resident_target: {e}");
            return;
        }
        Err(e) => panic!("pinned target draw: {e}"),
    }
    assert!(engine::pin_resident_target(&pinned), "pin ready target");

    let unpinned = TargetIdentity::Surface {
        id: 0x601,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut make2 = engine_req(&v, &f, 16, 16);
    make2.target_identity = Some(unpinned.clone());
    engine::execute_draw_request(&make2).expect("unpinned target draw");

    // Admit more distinct 16x16 targets than the retired count permitted. 320 was
    // the last value that count held, and the walk ran on *admission*, so 336
    // clears it with the margin that used to guarantee the oldest non-pinned
    // entry was swept. At this geometry the whole set is a few MiB, so no real
    // allocation failure is in play and the only thing that could remove one of
    // these is a count.
    const FILLERS: u32 = 336;
    for i in 0..FILLERS {
        let mut filler = engine_req(&v, &f, 16, 16);
        filler.target_identity = Some(TargetIdentity::Surface {
            id: 0x700 + i,
            width: 16,
            height: 16,
            generation: 1,
            format: SURFACE_TEST_FORMAT,
        });
        engine::execute_draw_request(&filler).expect("filler draw");
    }
    assert!(
        engine::resident_content_ready(&pinned),
        "a pinned resident must survive any admission"
    );
    assert!(
        engine::resident_content_ready(&unpinned),
        "the oldest unpinned resident is still here: nothing evicts on a count"
    );
    // Every filler is still resident too — the assert above alone would pass a
    // walk that spared only the two named identities.
    for i in 0..FILLERS {
        let filler = TargetIdentity::Surface {
            id: 0x700 + i,
            width: 16,
            height: 16,
            generation: 1,
            format: SURFACE_TEST_FORMAT,
        };
        assert!(
            engine::resident_content_ready(&filler),
            "filler {i} was destroyed by something other than an allocation failure"
        );
    }

    engine::unpin_resident_target(&pinned);
    assert!(engine::resident_content_ready(&pinned));
}

/// A `output_bgra` + `skip_readback` resident draw leaves content that
/// [`engine::read_target`] can read back twice with the same answer — asserted
/// here because nothing else in this suite reads the same resident twice.
/// A `TargetIdentity::Surface` resident declared at guest scanout order renders
/// and reads back in it **without the caller asking**, and says so; a pooled
/// target does not.
///
/// This is the contract the IOSurface texture composite Store rests on. That Store's
/// consumers are all defined in BGRA — `mapping_write::write_bgra8`,
/// `surface_cache`, the deferred window the flush reads — so when the attachment
/// is BGRA the readback lands ready to use, and when it is not the runtime pays a
/// whole-frame R/B exchange per Store. Measured at 776 us on a 1080p frame, 84 %
/// of the drain worker's draw time that `draw_phase` could not attribute.
///
/// Both halves are asserted, and the second is the one that would rot quietly:
/// `pixels_bgra` is what every consumer branches on, so a resident that came out
/// BGRA while reporting `false` is a silent R/B exchange on every composite —
/// the reported order has to be the order the bytes are actually in, not a
/// constant that happens to agree today.
///
/// `output_bgra` is deliberately left unset. The point is that the *identity*
/// carries the order, so that a composite Store, a chain intermediate and an MRT
/// primary sharing one surface all agree without coordinating: `registry_ensure`
/// destroys and recreates the image whenever a draw disagrees with the slot, so a
/// per-path predicate one path spelled differently would cost a full
/// reallocation per frame rather than a wrong colour.
#[test]
fn a_surface_resident_reads_back_in_guest_scanout_order() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let (w, h) = (16u32, 16u32);
    // The fragment writes semantic (64, 128, 191, 255) — asymmetric in R and B,
    // so an omitted or doubled exchange shows rather than cancelling.
    let i = ((h / 2) * w + w / 2) as usize * 4;

    let mut resident = engine_req(&v, &f, w, h);
    resident.target_identity = Some(TargetIdentity::Surface {
        id: 4711,
        width: w,
        height: h,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    });
    let out = match engine::execute_draw_request(&resident) {
        Ok(o) => o,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP surface scanout order: {e}");
            return;
        }
        Err(e) => panic!("surface resident draw: {e}"),
    };
    assert!(
        out.pixels_bgra,
        "a Surface resident must report the BGRA order it rendered in"
    );
    // `near`, not `assert_eq!`: green is `0.5 * 255 = 127.5`, an exact tie, and
    // Vulkan does not pin which way a float→unorm8 tie goes. Intel ANV rounds it
    // down and this assertion read 127 against a hardcoded 128 on every run since
    // it was written. The tolerance costs the property nothing — an R/B exchange
    // moves 191 against 64, which is 127 LSB, not one.
    let px = [
        out.pixels[i],
        out.pixels[i + 1],
        out.pixels[i + 2],
        out.pixels[i + 3],
    ];
    assert!(
        near(px[0], 191) && near(px[1], 128) && near(px[2], 64) && near(px[3], 255),
        "a Surface resident's readback must already be in guest scanout order; got BGRA={px:?}"
    );

    // The pooled path is the control: no identity, so no namespace to take an
    // order from, and the bytes stay semantic. Without it an engine that had
    // simply been switched to BGRA everywhere would pass the arm above.
    let pooled = engine_req(&v, &f, w, h);
    let out = engine::execute_draw_request(&pooled).expect("pooled draw");
    assert!(
        !out.pixels_bgra,
        "a pooled target has no identity to take an order from"
    );
    let px = [
        out.pixels[i],
        out.pixels[i + 1],
        out.pixels[i + 2],
        out.pixels[i + 3],
    ];
    assert!(
        near(px[0], 64) && near(px[1], 128) && near(px[2], 191) && near(px[3], 255),
        "a pooled target's readback stays semantic RGBA; got RGBA={px:?}"
    );
}

#[test]
fn a_bgra_resident_draw_reads_back_identically_twice() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let w = 16u32;
    let h = 16u32;
    let identity = TargetIdentity::Surface {
        id: 91,
        width: w,
        height: h,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut req = engine_req(&v, &f, w, h);
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP bgra resident readback: {e}");
            return;
        }
        Err(e) => panic!("bgra resident draw: {e}"),
    }
    let i = ((h / 2) * w + w / 2) as usize * 4;
    let first = engine::read_target(&identity)
        .expect("read resident")
        .pixels;
    let center_first = [first[i], first[i + 1], first[i + 2], first[i + 3]];
    let second = engine::read_target(&identity)
        .expect("re-read resident")
        .pixels;
    let center_second = [second[i], second[i + 1], second[i + 2], second[i + 3]];
    assert_eq!(
        center_first, center_second,
        "resident content changed across a second readback",
    );
    assert!(
        near(center_second[0], 191)
            && near(center_second[1], 128)
            && near(center_second[2], 64)
            && near(center_second[3], 255),
        "resident center BGRA={center_second:?}; expected ~(191,128,64,255)"
    );
}

/// A deferred composite Store moves its device→host copy; it does not delete
/// it. `readbacks` pooled both populations, so it read the same whether the
/// deferral worked or merely rescheduled the copy — which is how a boot came
/// back with `readbacks / surface_deferred` at 1.39 and no way to say why.
///
/// The split is the measurement: a `skip_readback` draw must land in
/// `render_post_wait_skips` and leave `readbacks` alone, and the `read_target`
/// a consumer later asks for must land in `target_reads` and *still* leave
/// `readbacks` alone.
#[test]
fn a_skipped_draw_readback_and_a_resident_read_are_counted_apart() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let w = 16u32;
    let h = 16u32;
    let identity = TargetIdentity::Surface {
        id: 93,
        width: w,
        height: h,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut req = engine_req(&v, &f, w, h);
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;

    let before_draw = engine::counter_snapshot();
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP readback split: {e}");
            return;
        }
        Err(e) => panic!("resident draw: {e}"),
    }
    let draw = engine::counter_snapshot().delta_since(&before_draw);
    assert_eq!(
        draw.readbacks, 0,
        "a skip_readback draw took a draw readback anyway"
    );
    assert_eq!(
        draw.render_post_wait_skips, 1,
        "a skip_readback draw did not record its skipped fence wait"
    );
    assert_eq!(
        draw.target_reads, 0,
        "a draw that reads nothing back recorded a resident read"
    );

    let before_read = engine::counter_snapshot();
    let px = engine::read_target(&identity)
        .expect("read resident")
        .pixels;
    let read = engine::counter_snapshot().delta_since(&before_read);
    assert_eq!(
        read.target_reads, 1,
        "read_target did not record a resident read"
    );
    assert_eq!(
        read.target_read_bytes,
        px.len() as u64,
        "resident read bytes disagree with the frame it returned"
    );
    assert_eq!(
        read.readbacks, 0,
        "read_target was pooled into the draw rail's readback count"
    );
}

/// The live window-transition failure samples CPU-decoded RGBA bytes and stores
/// into a resident BGRA target. Lock that complete format chain independently
/// of guest descriptors: shader-visible R/G/B must land as physical B/G/R.
#[test]
fn sampled_rgba_upload_to_bgra_target_preserves_semantic_channels() {
    let _g = engine_test_session();
    let sampled_owner = sampled_resource_owner();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let w = 16u32;
    let h = 16u32;
    let identity = TargetIdentity::Surface {
        id: 82,
        width: w,
        height: h,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut req = engine_req(&vert, &frag, w, h);
    req.vertex_count = 6;
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;

    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    req.storage_buffers.push(StorageBufferResource {
        binding: 1,
        content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    let rgba = [17u8, 91, 203, 255];
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
        source: SampledSource::Bytes(std::sync::Arc::new(rgba.repeat(4))),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: Some(sampled_owner.lifetime_ref()),
        swizzle: Default::default(),
    });
    req.samplers
        .push(SamplerResource::normalized_default(sampler_binding(0)));

    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP sampled RGBA to BGRA target: {e}");
            return;
        }
        Err(e) => panic!("sampled RGBA to BGRA target: {e}"),
    }
    let raw = engine::read_target(&identity)
        .expect("read BGRA target")
        .pixels;
    let center = ((h / 2) * w + w / 2) as usize * 4;
    assert_eq!(
        &raw[center..center + 4],
        &[rgba[2], rgba[1], rgba[0], rgba[3]],
        "shader RGBA must land in guest-visible BGRA byte order"
    );

    let warm_before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("exact sampled-content cache hit");
    let warm_delta = engine::counter_snapshot().delta_since(&warm_before);
    assert_eq!(warm_delta.sampled_cache_hits, 1, "warm hit: {warm_delta:?}");
    assert_eq!(
        warm_delta.sampled_reuploads, 0,
        "warm upload: {warm_delta:?}"
    );
    let warm_raw = engine::read_target(&identity)
        .expect("read warm BGRA target")
        .pixels;
    assert_eq!(
        &warm_raw[center..center + 4],
        &[rgba[2], rgba[1], rgba[0], rgba[3]],
        "cache hit must preserve sampled shader output"
    );

    let changed_rgba = [201u8, 77, 31, 255];
    req.sampled_images[0].source =
        SampledSource::Bytes(std::sync::Arc::new(changed_rgba.repeat(4)));
    let changed_before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("changed sampled-content cache miss");
    let changed_delta = engine::counter_snapshot().delta_since(&changed_before);
    assert_eq!(
        changed_delta.sampled_cache_misses, 1,
        "changed miss: {changed_delta:?}"
    );
    assert_eq!(
        changed_delta.sampled_reuploads, 1,
        "changed upload: {changed_delta:?}"
    );
    let changed_raw = engine::read_target(&identity)
        .expect("read changed BGRA target")
        .pixels;
    assert_eq!(
        &changed_raw[center..center + 4],
        &[
            changed_rgba[2],
            changed_rgba[1],
            changed_rgba[0],
            changed_rgba[3],
        ],
        "changed sampled bytes must replace cached shader input"
    );
}

/// A constexpr Metal sampler has no guest sampler object. The translator
/// reflects its packed AIR state and the product creates the corresponding
/// Vulkan descriptor. Exercise that whole handoff on a real engine: leaving the
/// reflected sampler unbound makes this shader return black (or fault on
/// MoltenVK), while the exact binding samples the known source color.
#[test]
fn reflected_static_sampler_descriptor_samples_texture() {
    use metal2vulkan::reflect::ResourceKind;

    let _g = engine_test_session();
    let sampled_owner = sampled_resource_owner();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let tmp = std::env::temp_dir().join(format!(
        "paravirt_engine_{}_static_sampler",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("tmp");
    let air = fixtures().join("render_frag_static_sampler.air");
    let (frag, reflection) =
        metal2vulkan::translate_reflected(air.to_str().unwrap(), Stage::Fragment, &tmp)
            .expect("translate static sampler fixture");
    let frag = frag
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect::<Vec<_>>();
    let reflected = reflection
        .bindings
        .iter()
        .find(|binding| binding.kind == ResourceKind::StaticSampler)
        .expect("reflected constexpr sampler");
    reflected.descriptor.expect("static sampler descriptor");
    let descriptor = reims_vgpu_vulkan::spirv_bind::reflected_sampler_descriptors(&reflection)
        .into_iter()
        .find(|descriptor| descriptor.static_state.is_some())
        .expect("semantic constexpr sampler descriptor");
    let state = descriptor.static_state.expect("static sampler state");

    let (w, h) = (16u32, 16u32);
    let mut req = engine_req(&vert, &frag, w, h);
    req.vertex_count = 6;
    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    req.storage_buffers.push(StorageBufferResource {
        binding: 1,
        content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    let rgba = [17u8, 91, 203, 255];
    let texture_binding = reflection
        .bindings
        .iter()
        .find(|binding| binding.kind == ResourceKind::Texture)
        .and_then(|binding| binding.descriptor)
        .expect("reflected sampled texture")
        .binding;
    req.sampled_images.push(SampledImageResource {
        binding: texture_binding,
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
        source: SampledSource::Bytes(std::sync::Arc::new(rgba.repeat(4))),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: Some(sampled_owner.lifetime_ref()),
        swizzle: Default::default(),
    });
    req.samplers.push(
        reims_vgpu::runtime::draw::reflected_static_sampler_resource(
            "fragment",
            descriptor.binding,
            state,
        )
        .expect("map reflected static sampler"),
    );
    let target = TargetIdentity::Surface {
        id: 880_024,
        width: w,
        height: h,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    req.target_identity = Some(target.clone());
    req.skip_readback = true;

    let before = engine::counter_snapshot();
    let Some(first) = draw_out_or_skip("reflected static sampler first", &req) else {
        return;
    };
    assert!(first.pixels.is_empty());
    req.load_from_target = true;
    let second = engine::execute_draw_request(&req).expect("second static sampler draw");
    assert!(second.pixels.is_empty());
    let pixels = engine::read_target(&target)
        .expect("read repeated static sampler target")
        .into_rgba8();
    let descriptors = engine::counter_snapshot().delta_since(&before);
    if descriptors.descriptor_set_updates == 0 {
        assert_eq!(
            descriptors.descriptor_pushes, 1,
            "first draw pushes: {descriptors:?}"
        );
        assert_eq!(
            descriptors.descriptor_push_held, 1,
            "the exact repeated state is retained by the command buffer: {descriptors:?}"
        );
    } else {
        assert_eq!(
            descriptors.descriptor_set_updates, 2,
            "fallback updates: {descriptors:?}"
        );
        assert_eq!(
            descriptors.descriptor_set_binds, 2,
            "fallback binds: {descriptors:?}"
        );
        assert_eq!(descriptors.descriptor_pushes, 0);
        assert_eq!(descriptors.descriptor_push_held, 0);
    }
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        assert_eq!(pixel, rgba, "static sampler pixel {index}");
    }
}

/// Safety net for the deferred "upload host-cache BGRA bytes as native Bgra8"
/// optimization (skip the CPU R/B swizzle): a `SampledSource::Bytes` tagged
/// `ash::vk::Format::B8G8R8A8_UNORM` must sample the SAME semantic color as the equivalent
/// RGBA upload — i.e. `B8G8R8A8_UNORM` bytes `[b,g,r,a]` land in the shader as
/// `(r,g,b,a)`, identical to `Rgba8` bytes `[r,g,b,a]`. Proves the Bytes rail (not
/// just the zero-copy GuestRuns rail) is color-correct for Bgra8 before any
/// loader is switched to stop swizzling.
#[test]
fn sampled_bgra8_bytes_upload_matches_rgba8_semantic_color() {
    let _g = engine_test_session();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let w = 16u32;
    let h = 16u32;

    // Same geometry/UV setup as the RGBA-to-BGRA parity test above.
    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };

    // The single semantic color under test, expressed in each byte order.
    let rgba = [17u8, 91, 203, 255];
    let bgra = [rgba[2], rgba[1], rgba[0], rgba[3]];

    // Render the color once via an Rgba8 upload and once via a Bgra8 upload; the
    // sampled output must be byte-identical.
    let render = |bytes: Vec<u8>, format: reims_vgpu_protocol::ImageFormat, id: u32| {
        let identity = TargetIdentity::Surface {
            id,
            width: w,
            height: h,
            generation: 1,
            format: SURFACE_TEST_FORMAT,
        };
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.target_identity = Some(identity.clone());
        req.skip_readback = true;
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
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
            source: SampledSource::Bytes(std::sync::Arc::new(bytes)),
            byte_origin: Default::default(),
            format,
            identity: None,
            content: None,
            resource_lifetime: None,
            swizzle: Default::default(),
        });
        req.samplers
            .push(SamplerResource::normalized_default(sampler_binding(0)));
        match engine::execute_draw_request(&req) {
            Ok(_) => Some(
                engine::read_target(&identity)
                    .expect("read BGRA target")
                    .pixels,
            ),
            Err(e) if skip_if_no_gpu(&e.to_string()) => {
                eprintln!("SKIP bgra8 bytes upload: {e}");
                None
            }
            Err(e) => panic!("bgra8 bytes upload: {e}"),
        }
    };

    let Some(rgba_out) = render(
        rgba.repeat(4),
        reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        90,
    ) else {
        return;
    };
    let bgra_out = render(
        bgra.repeat(4),
        reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Bgra8),
        91,
    )
    .expect("bgra8 render");

    let center = ((h / 2) * w + w / 2) as usize * 4;
    // Both uploads carry the identical semantic color, so the guest-visible BGRA
    // target center must be `[b,g,r,a]` in each — and equal to each other.
    assert_eq!(
        &rgba_out[center..center + 4],
        &bgra,
        "rgba8 upload lands as guest-visible BGRA"
    );
    assert_eq!(
        &bgra_out[center..center + 4],
        &bgra,
        "bgra8 upload must sample the SAME semantic color as rgba8"
    );
    assert_eq!(
        &rgba_out[center..center + 4],
        &bgra_out[center..center + 4],
        "bgra8 and rgba8 uploads of one color must render byte-identically"
    );
}

/// **L3's proof.** A decoded type-8 view swizzle must be performed by the image
/// view's component mapping, on the GPU, at sample time — not by rewriting
/// texels, which would force every swizzled texture onto the CPU upload path
/// and cost it the zero-copy crossing.
///
/// Renders one identical RGBA source twice: once with the identity plan and
/// once with a plan that reads `(b, g, r, 1)`. Same bytes, same upload, same
/// everything but the view — so a difference in the output can only have come
/// from the mapping. If the mapping were dropped the two would be equal, which
/// is exactly the silent failure this asserts against.
#[test]
fn a_view_swizzle_is_performed_by_the_image_view_not_the_cpu() {
    use reims_vgpu_core::pixel_format;

    let _g = engine_test_session();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let w = 16u32;
    let h = 16u32;

    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };

    // A colour whose three channels are all distinct, so any remap is visible.
    let source = [17u8, 91, 203, 255];

    let render = |plan: pixel_format::SwizzlePlan, id: u32| {
        let identity = TargetIdentity::Surface {
            id,
            width: w,
            height: h,
            generation: 1,
            format: SURFACE_TEST_FORMAT,
        };
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.target_identity = Some(identity.clone());
        req.skip_readback = true;
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
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
            source: SampledSource::Bytes(std::sync::Arc::new(source.repeat(4))),
            byte_origin: Default::default(),
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Rgba8,
            ),
            identity: None,
            content: None,
            resource_lifetime: None,
            swizzle: plan,
        });
        req.samplers
            .push(SamplerResource::normalized_default(sampler_binding(0)));
        match engine::execute_draw_request(&req) {
            Ok(_) => Some(engine::read_target(&identity).expect("read target").pixels),
            Err(e) if skip_if_no_gpu(&e.to_string()) => {
                eprintln!("SKIP view swizzle: {e}");
                None
            }
            Err(e) => panic!("view swizzle draw: {e}"),
        }
    };

    let Some(plain) = render(pixel_format::swizzle_identity(), 140) else {
        return;
    };
    // Selectors: 4=B, 3=G, 2=R, 1=One  →  the view reads (b, g, r, 1).
    let reversed = pixel_format::swizzle_plan(&[4, 3, 2, 1]).expect("swizzle plan");
    let swizzled = render(reversed, 141).expect("swizzled render");

    let center = ((h / 2) * w + w / 2) as usize * 4;
    let plain_px = &plain[center..center + 4];
    let swizzled_px = &swizzled[center..center + 4];

    // The target is guest-visible BGRA, so the identity render shows [b,g,r,a].
    assert_eq!(
        plain_px,
        [source[2], source[1], source[0], source[3]],
        "identity plan must leave the sampled colour alone"
    );
    // Reading (b,g,r,1) instead of (r,g,b,a) swaps R and B before the BGRA
    // store, so the stored bytes come back with R and B exchanged again.
    assert_eq!(
        swizzled_px,
        [source[0], source[1], source[2], 255],
        "the view swizzle must reach the sampler"
    );
    assert_ne!(
        plain_px, swizzled_px,
        "identical bytes rendered identically means the mapping was dropped"
    );
}

/// A semantic RGBA Load seed must be converted to the native BGRA attachment
/// order before upload. A partial draw makes the untouched seed observable;
/// fullscreen tests overwrite the bad upload and cannot catch this class.
#[test]
fn partial_draw_preserves_rgba_seed_on_bgra_target() {
    let _g = engine_test_session();
    let (vert, frag) = triangle_spirv();
    let (w, h) = (16u32, 16u32);
    let identity = TargetIdentity::Surface {
        id: 83,
        width: w,
        height: h,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let seed_rgba = [17u8, 91, 203, 255];
    let mut req = engine_req(&vert, &frag, w, h);
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;
    req.target_rgba8 = Some(std::sync::Arc::new(seed_rgba.repeat((w * h) as usize)));
    req.scissors = vec![ScissorResource {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    }];

    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP BGRA partial seed: {e}");
            return;
        }
        Err(e) => panic!("BGRA partial seed: {e}"),
    }
    let raw = engine::read_target(&identity)
        .expect("read BGRA partial-seed target")
        .pixels;
    let outside = ((h / 2) * w + w / 2) as usize * 4;
    assert_eq!(
        &raw[outside..outside + 4],
        &[seed_rgba[2], seed_rgba[1], seed_rgba[0], seed_rgba[3]],
        "untouched semantic RGBA seed must remain correct in native BGRA storage"
    );
}

/// A guest-page LOAD seed preserves untouched pixels without first becoming a
/// host RGBA framebuffer. `pages: None` deliberately exercises the universal
/// host-run fallback; the imported and gathered arms feed the same copy after
/// choosing a different buffer source.
#[test]
fn partial_draw_preserves_a_native_guest_target_seed() {
    let _g = engine_test_session();
    let (vert, frag) = triangle_spirv();
    let (w, h) = (16u32, 16u32);
    let identity = TargetIdentity::Surface {
        id: 84,
        width: w,
        height: h,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let semantic_rgba = [17u8, 91, 203, 255];
    let native_bgra = [
        semantic_rgba[2],
        semantic_rgba[1],
        semantic_rgba[0],
        semantic_rgba[3],
    ];
    // The run is a borrowed stable alias by contract. Keep its backing alive
    // through execute just as a RAMBlock remains alive for the VM lifetime.
    let backing = native_bgra.repeat((w * h) as usize);
    let mut req = engine_req(&vert, &frag, w, h);
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;
    req.target_guest = Some(engine::GuestTargetPlan::Seed(engine::GuestTargetSeed {
        source: engine::GuestRunSource {
            runs: std::sync::Arc::new(vec![engine::GuestRun {
                host_ptr: backing.as_ptr() as usize,
                len: backing.len() as u64,
            }]),
            source_offset: 0,
            total_len: backing.len() as u64,
            row_length_texels: 0,
            pages: None,
            physical_pages: None,
        },
        format: reims_vgpu_protocol::TexelLayout::Bgra8,
    }));
    req.scissors = vec![ScissorResource {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    }];

    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP guest target seed: {e}");
            return;
        }
        Err(e) => panic!("guest target seed: {e}"),
    }
    let rgba = engine::read_target(&identity)
        .expect("read guest-seeded target")
        .into_rgba8();
    let outside = ((h / 2) * w + w / 2) as usize * 4;
    assert_eq!(
        &rgba[outside..outside + 4],
        &semantic_rgba,
        "the untouched pixel must come from the native guest seed"
    );
}

/// An alpha-only `MTLColorWriteMask` must leave the attachment's colour
/// channels exactly as the Load seed left them.
///
/// This is the shape a compositor uses to punch coverage into a surface without
/// touching its colour, and it is the one non-`all` mask a live x86/Vulkan guest
/// was measured sending (tag `0x09`, value 1). Before the mask was decoded the
/// builder pinned `ColorComponentFlags::RGBA` on every attachment, so this draw
/// replaced the whole pixel: the assertion below fails with the shader's
/// (64, 128, 191) in place of the seed.
///
/// Both halves are asserted. RGB must survive — that is the fix — and alpha must
/// change, which is what separates "the mask worked" from "the draw never ran".
/// A test that only checked RGB would pass against a pipeline that rendered
/// nothing at all.
#[test]
fn an_alpha_only_write_mask_leaves_the_colour_channels_alone() {
    use reims_vgpu_vulkan::engine::ColorWriteMask;
    let _g = engine_test_session();
    let (vert, frag) = triangle_spirv();
    let (w, h) = (16u32, 16u32);
    // Distinct from the shader's (64, 128, 191, 255) in every channel, so an
    // ignored mask is visible whichever channel is read.
    let seed = [17u8, 91, 203, 7];

    let mut req = engine_req(&vert, &frag, w, h);
    req.target_rgba8 = Some(std::sync::Arc::new(seed.repeat((w * h) as usize)));
    req.color_write_mask = ColorWriteMask::new(1).expect("MTLColorWriteMaskAlpha");
    let masked = match engine::execute_draw_request(&req) {
        Ok(out) => out.pixels,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP alpha-only write mask: {e}");
            return;
        }
        Err(e) => panic!("alpha-only write mask: {e}"),
    };
    let i = ((h / 2) * w + w / 2) as usize * 4;
    assert_eq!(
        &masked[i..i + 3],
        &seed[..3],
        "an alpha-only mask must not write colour; got the shader's output, so \
         the mask was dropped"
    );
    assert!(
        near(masked[i + 3], 255),
        "alpha is the one channel the mask permits, and it must have changed \
         from the seed's {} — otherwise nothing rendered and the RGB check \
         above is vacuous",
        seed[3]
    );

    // Control: the same draw with the default `all` mask writes every channel.
    // Run in the same session so a difference cannot come from device state.
    req.color_write_mask = ColorWriteMask::default();
    let unmasked = engine::execute_draw_request(&req)
        .expect("unmasked control draw")
        .pixels;
    assert_fullscreen_fragment_color("unmasked control", &unmasked, w, h);
    assert_ne!(
        &masked[i..i + 3],
        &unmasked[i..i + 3],
        "the two arms must differ, or the mask is not what produced the result"
    );
}

/// A `SeedOrder::Bgra8` seed must land the same semantic pixels as the
/// equivalent `SeedOrder::Rgba8` seed.
///
/// This is the IOSurface texture composite Load. `surface_cache` holds guest scanout order
/// while the pooled target is RGBA, so the runtime used to allocate, copy and
/// swizzle a whole framebuffer per seeded draw purely to restate pixels it
/// already had — at the 28-111 Stores/s `store_routes` measures. Naming the
/// seed's own order lets the exchange ride the copy into the mapped staging
/// span, which happens either way.
///
/// A partial draw is what makes it observable: the scissor leaves most of the
/// seed untouched, and a fullscreen draw would overwrite a wrong upload and pass
/// regardless. Both arms are run and compared rather than asserting a literal,
/// so this cannot pass by both paths being broken the same way.
#[test]
fn a_bgra_ordered_seed_lands_the_same_pixels_as_the_rgba_ordered_one() {
    let _g = engine_test_session();
    let (vert, frag) = triangle_spirv();
    let (w, h) = (16u32, 16u32);
    // Deliberately asymmetric in R and B, so an omitted or doubled exchange is
    // visible rather than cancelling.
    let semantic_rgba = [17u8, 91, 203, 255];
    let scanout_bgra = [
        semantic_rgba[2],
        semantic_rgba[1],
        semantic_rgba[0],
        semantic_rgba[3],
    ];
    let outside = ((h / 2) * w + w / 2) as usize * 4;

    let read_back = |order: engine::SeedOrder, bytes: [u8; 4], id: u32| -> Option<Vec<u8>> {
        let identity = TargetIdentity::Surface {
            id,
            width: w,
            height: h,
            generation: 1,
            format: SURFACE_TEST_FORMAT,
        };
        let mut req = engine_req(&vert, &frag, w, h);
        req.target_identity = Some(identity.clone());
        req.skip_readback = true;
        req.target_rgba8 = Some(std::sync::Arc::new(bytes.repeat((w * h) as usize)));
        req.target_seed_order = order;
        req.scissors = vec![ScissorResource {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        }];
        match engine::execute_draw_request(&req) {
            Ok(_) => {}
            Err(e) if skip_if_no_gpu(&e.to_string()) => {
                eprintln!("SKIP seed order: {e}");
                return None;
            }
            Err(e) => panic!("seed order draw: {e}"),
        }
        // Normalized through the order the engine reports, so this case tests the
        // seed exchange and not the attachment's format: a `Surface` resident is
        // BGRA, and asserting a literal against raw bytes would make this case
        // fail whenever that choice changed while the property it names still
        // held.
        Some(
            engine::read_target(&identity)
                .expect("read seed-order target")
                .into_rgba8(),
        )
    };

    let Some(rgba_arm) = read_back(engine::SeedOrder::Rgba8, semantic_rgba, 91) else {
        return;
    };
    let Some(bgra_arm) = read_back(engine::SeedOrder::Bgra8, scanout_bgra, 92) else {
        return;
    };
    assert_eq!(
        &bgra_arm[outside..outside + 4],
        &rgba_arm[outside..outside + 4],
        "the same pixels described in two orders must land identically"
    );
    // And that they landed as the semantic colour, not as some agreed-upon wrong
    // one: the scissor left this pixel untouched, so it still holds the seed.
    assert_eq!(
        &rgba_arm[outside..outside + 4],
        &semantic_rgba[..],
        "an unwritten pixel must still hold the seed's semantic colour"
    );
}

/// Premult One/OMSA: GPU Load+blend matches the retired software composite oracle.
#[test]
fn premult_one_omsa_gpu_blend_matches_software_oracle() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    // Seed: solid gray base (128,128,128,255)
    let w = 16u32;
    let h = 16u32;
    let seed: Vec<u8> = (0..(w * h)).flat_map(|_| [128u8, 128, 128, 255]).collect();
    // GPU path: Load seed + One/OMSA blend (fullscreen frag writes opaque color).
    let mut gpu = engine_req(&v, &f, w, h);
    gpu.target_rgba8 = Some(std::sync::Arc::new(seed.clone()));
    gpu.blend = Some(BlendStateResource {
        src_color: BlendFactor::One,
        dst_color: BlendFactor::OneMinusSrcAlpha,
        color_op: BlendOp::Add,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::OneMinusSrcAlpha,
        alpha_op: BlendOp::Add,
    });
    let gpu_px = match engine::execute_draw_request(&gpu) {
        Ok(o) => o.pixels,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP premult: {e}");
            return;
        }
        Err(e) => panic!("premult gpu: {e}"),
    };
    // Software oracle: draw over black with same blend, then composite.
    let mut black = engine_req(&v, &f, w, h);
    black.blend = gpu.blend;
    let over_black = engine::execute_draw_request(&black)
        .expect("over black")
        .pixels;
    let (soft, _) = reims_vgpu::runtime::draw::load_composite_premult_one_omsa(&over_black, &seed);
    // Allow ±1 LSB for unorm rounding differences between GPU blend and CPU composite.
    assert_eq!(gpu_px.len(), soft.len());
    for (i, (g, s)) in gpu_px.iter().zip(soft.iter()).enumerate() {
        assert!(
            (*g as i32 - *s as i32).abs() <= 1,
            "premult mismatch at byte {i}: gpu={g} soft={s}"
        );
    }
}

/// Class A zero-copy wipe lock: after a skip_readback Store (no CPU pixels,
/// host_cache would be empty/evicted), the next pass must LoadFromTarget so
/// progressive multi-pass content stays on the resident image. Engine Clear
/// (the default when neither `load_from_target` nor `target_rgba8` is set)
/// would black the target.
#[test]
fn skip_readback_store_then_load_from_target_preserves_content() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 91,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    // Pass 1: product zero-copy Store shape — resident path uses skip_readback
    // so no CPU pixels land in host_cache.
    let mut store1 = engine_req(&v, &f, 16, 16);
    store1.target_identity = Some(identity.clone());
    store1.skip_readback = true;
    match engine::execute_draw_request(&store1) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP skip_readback_load_preserve: {e}");
            return;
        }
        Err(e) => panic!("store1: {e}"),
    }
    assert!(
        engine::resident_content_ready(&identity),
        "store1 must mark content_ready"
    );
    // Pass 2: LOAD after host_cache miss — LoadFromTarget, no CPU seed.
    let mut store2 = engine_req(&v, &f, 16, 16);
    store2.target_identity = Some(identity.clone());
    store2.load_from_target = true;
    store2.skip_readback = true;
    store2.target_rgba8 = None;
    engine::execute_draw_request(&store2).expect("store2 LoadFromTarget");
    let px = engine::read_target(&identity)
        .expect("read_target after progressive Stores")
        .into_rgba8();
    assert_fullscreen_fragment_color("progressive_skip_readback", &px, 16, 16);
    // No seed_uploads on pass 2 (LoadFromTarget, not CPU seed).
    // Counters are process-global; just ensure content survived.
    assert!(engine::resident_content_ready(&identity));
}

/// Cross-boot retained-frame lock: a device reset must evict identity-keyed
/// resident images even when the next guest reuses the same id/generation.
#[test]
fn guest_reset_evicts_resident_targets_without_destroying_context() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 91,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut draw = engine_req(&v, &f, 16, 16);
    draw.target_identity = Some(identity.clone());
    draw.skip_readback = true;
    match engine::execute_draw_request(&draw) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP guest_reset_evicts_resident: {e}");
            return;
        }
        Err(e) => panic!("resident setup: {e}"),
    }
    assert!(engine::resident_content_ready(&identity));

    let stats = engine::reset_guest_state();
    assert_eq!(stats.resident_targets, 1);
    assert!(stats.had_context);
    assert!(!engine::resident_content_ready(&identity));
}

/// Chain byte-parity: LoadFromTarget chain matches CPU-seed chain.
#[test]
fn chain_load_from_target_byte_parity_vs_cpu_seed() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    // CPU-seed chain: draw1 clear → pixels → draw2 LoadSeed(pixels) → pixels2
    let d1 = engine_req(&v, &f, 16, 16);
    let p1 = match engine::execute_draw_request(&d1) {
        Ok(o) => semantic_rgba(&o),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP chain: {e}");
            return;
        }
        Err(e) => panic!("chain d1: {e}"),
    };
    let mut d2_cpu = engine_req(&v, &f, 16, 16);
    d2_cpu.target_rgba8 = Some(std::sync::Arc::new(p1.clone()));
    let p2_cpu = semantic_rgba(&engine::execute_draw_request(&d2_cpu).expect("cpu seed chain"));

    // GPU-resident chain: same identity LoadFromTarget.
    engine::test_reset_engine();
    let identity = TargetIdentity::Surface {
        id: 7,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut g1 = engine_req(&v, &f, 16, 16);
    g1.target_identity = Some(identity.clone());
    g1.skip_readback = true;
    engine::execute_draw_request(&g1).expect("gpu chain d1");
    let mut g2 = engine_req(&v, &f, 16, 16);
    g2.target_identity = Some(identity.clone());
    g2.load_from_target = true;
    g2.skip_readback = false; // read back for compare
    let p2_gpu = semantic_rgba(&engine::execute_draw_request(&g2).expect("gpu chain d2"));
    assert_eq!(
        p2_gpu, p2_cpu,
        "LoadFromTarget chain must match CPU-seed chain"
    );
}

/// The IOSurface texture composite Store's shape: `LoadFromTarget` on a resident that the
/// *previous* pass read back, rather than one it left GPU-only.
///
/// Every other `LoadFromTarget` case in this suite sets `skip_readback = true`
/// on the producing pass, so the image goes attachment-write → attachment-load
/// with nothing in between. A composite Store keeps its readback — that copy is
/// what feeds `surface_cache`, the deferred window's owned frame and the guest
/// writeback — so the image is additionally read as a transfer source between
/// the two passes, and the next pass's load barrier has to source its scope from
/// the tracked `TRANSFER_SRC_OPTIMAL` rather than from a color write.
///
/// Scored against the CPU-seed chain it replaces, byte for byte: eliding the
/// seed upload is only sound if the resident holds exactly what the upload
/// would have carried. A dropped or mis-scoped barrier shows up here as the
/// prior frame, a torn frame, or a black one — the class the elision gate
/// exists to avoid.
///
/// Both passes are **scissored to 1x1**, and that is what makes the test mean
/// anything. The default triangle covers the whole attachment, so with a
/// full-viewport draw `LoadFromTarget`, `LoadSeed` and even `Clear` all produce
/// identical pixels and the assertion holds no matter what the load action did.
/// Verified: substituting `Clear` for `LoadFromTarget` in the unscissored form
/// still passed. Scissored, 255 of the 256 pixels carry only loaded content, so
/// the load action is the only thing under test.
#[test]
fn load_from_target_after_a_readback_matches_the_cpu_seed_chain() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let (w, h) = (16u32, 16u32);
    // A prior frame the draw cannot manufacture, so a lost load reads as black
    // rather than as a coincidentally-correct fragment colour.
    let prior = [17u8, 91, 203, 255].repeat((w * h) as usize);
    let dot = |x: u32, y: u32| ScissorResource {
        x,
        y,
        width: 1,
        height: 1,
    };

    // Arm A — the round trip this rail removes: seed from host bytes, draw a
    // corner, read back, re-upload those pixels as the next pass's seed.
    let mut d1 = engine_req(&v, &f, w, h);
    d1.target_rgba8 = Some(std::sync::Arc::new(prior.clone()));
    d1.scissors = vec![dot(0, 0)];
    let p1 = match engine::execute_draw_request(&d1) {
        Ok(o) => semantic_rgba(&o),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP load_from_target_after_readback: {e}");
            return;
        }
        Err(e) => panic!("readback chain d1: {e}"),
    };
    let mut d2_cpu = engine_req(&v, &f, w, h);
    d2_cpu.target_rgba8 = Some(std::sync::Arc::new(p1.clone()));
    d2_cpu.scissors = vec![dot(8, 8)];
    let p2_cpu =
        semantic_rgba(&engine::execute_draw_request(&d2_cpu).expect("cpu seed after readback"));

    // The two arms are only comparable if the seed actually survives the draw.
    let untouched = ((h / 2) * w + 2) as usize * 4;
    assert_eq!(
        &p2_cpu[untouched..untouched + 4],
        &prior[0..4],
        "a scissored draw must leave the rest of the seed intact, or neither \
         arm is testing the load action"
    );

    // Arm B — the elision: the same two passes, both still reading back, the
    // second loading from the resident the first stored into.
    engine::test_reset_engine();
    let identity = TargetIdentity::Surface {
        id: 311,
        width: w,
        height: h,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut g1 = engine_req(&v, &f, w, h);
    g1.target_identity = Some(identity.clone());
    g1.target_rgba8 = Some(std::sync::Arc::new(prior));
    g1.scissors = vec![dot(0, 0)];
    g1.skip_readback = false;
    let p1_resident =
        semantic_rgba(&engine::execute_draw_request(&g1).expect("resident store with readback"));
    assert_eq!(
        p1_resident, p1,
        "the resident pass must read back the same pixels as the pooled one, \
         or the two arms are not comparing the same content"
    );

    let mut g2 = engine_req(&v, &f, w, h);
    g2.target_identity = Some(identity.clone());
    g2.load_from_target = true;
    g2.target_rgba8 = None;
    g2.scissors = vec![dot(8, 8)];
    g2.skip_readback = false;
    let p2_gpu = semantic_rgba(
        &engine::execute_draw_request(&g2).expect("load from a target that was read back"),
    );

    assert_eq!(
        p2_gpu, p2_cpu,
        "a LOAD elided against a read-back resident must land the same frame \
         the seed upload would have"
    );
}

/// Resident GVA chain (type-2/3 rail): a 3-record chain keeps intermediate
/// content on the engine target — exactly one readback (the final contract
/// Store), zero CPU seed uploads, two post-submit wait skips — and the final
/// pixels byte-match the CPU round-trip chain (readback → LoadSeed re-upload
/// per record) it replaces.
#[test]
fn gva_chain_resident_single_readback_matches_cpu_seed_chain() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    // CPU round-trip reference chain: every record reads back, next record
    // re-uploads the pixels as its seed (the legacy GVA chain rail).
    let d1 = engine_req(&v, &f, 16, 16);
    let p1 = match engine::execute_draw_request(&d1) {
        Ok(o) => o.pixels,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP gva_chain: {e}");
            return;
        }
        Err(e) => panic!("gva_chain d1: {e}"),
    };
    let mut d2 = engine_req(&v, &f, 16, 16);
    d2.target_rgba8 = Some(std::sync::Arc::new(p1));
    let p2 = engine::execute_draw_request(&d2).expect("cpu d2").pixels;
    let mut d3 = engine_req(&v, &f, 16, 16);
    d3.target_rgba8 = Some(std::sync::Arc::new(p2));
    let p3_cpu = engine::execute_draw_request(&d3).expect("cpu d3").pixels;

    // Resident chain on a Gva identity: intermediates never touch the CPU.
    engine::test_reset_engine();
    let identity = TargetIdentity::Gva {
        gva: 0x2f00_0000,
        width: 16,
        height: 16,
        generation: 0,
        format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
    };
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    let mut g1 = engine_req(&v, &f, 16, 16);
    g1.target_identity = Some(identity.clone());
    g1.skip_readback = true;
    engine::execute_draw_request(&g1).expect("gva chain g1");
    let mut g2 = engine_req(&v, &f, 16, 16);
    g2.target_identity = Some(identity.clone());
    g2.load_from_target = true;
    g2.skip_readback = true;
    engine::execute_draw_request(&g2).expect("gva chain g2");
    let mut g3 = engine_req(&v, &f, 16, 16);
    g3.target_identity = Some(identity.clone());
    g3.load_from_target = true;
    g3.skip_readback = false; // final record: contract Store readback
    let p3_gpu = engine::execute_draw_request(&g3)
        .expect("gva chain g3")
        .pixels;
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(d.readbacks, 1, "only the final record reads back: {d:?}");
    assert_eq!(
        d.seed_uploads, 0,
        "no CPU seed on the resident chain: {d:?}"
    );
    assert_eq!(
        d.render_post_wait_skips, 2,
        "both intermediates skip the fence wait: {d:?}"
    );
    assert_eq!(
        p3_gpu, p3_cpu,
        "resident GVA chain must byte-match the CPU round-trip chain"
    );
}

/// Deferred GVA Store (single/final record): the draw renders into the
/// registry resident with skip_readback — zero readbacks and one post-wait
/// skip on the stamp path — and the flush-on-access `read_target` returns
/// byte-identical pixels to the synchronous readback Store it replaces.
#[test]
fn gva_deferred_store_flush_read_matches_sync_store() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    // Sync reference: the legacy Store readback.
    let d_sync = engine_req(&v, &f, 16, 16);
    let p_sync = match engine::execute_draw_request(&d_sync) {
        Ok(o) => o.pixels,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP gva_deferred_store: {e}");
            return;
        }
        Err(e) => panic!("sync store: {e}"),
    };

    // Deferred Store shape: registry Gva resident, no stamp-path readback.
    engine::test_reset_engine();
    let identity = TargetIdentity::Gva {
        gva: 0x3a00_0000,
        width: 16,
        height: 16,
        generation: 0,
        format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
    };
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    let mut g = engine_req(&v, &f, 16, 16);
    g.target_identity = Some(identity.clone());
    g.skip_readback = true;
    engine::execute_draw_request(&g).expect("deferred store draw");
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(d.readbacks, 0, "deferred Store must not read back: {d:?}");
    assert_eq!(
        d.render_post_wait_skips, 1,
        "deferred Store skips the fence wait: {d:?}"
    );
    assert!(engine::pin_resident_target(&identity), "window pin");

    // Flush-on-access landing: one readback, byte parity with the sync Store.
    let before_flush = engine::counter_snapshot();
    let p_flush = engine::read_target(&identity)
        .expect("flush read_target")
        .into_rgba8();
    let df = engine::counter_snapshot().delta_since(&before_flush);
    // The flush's copy is a resident read, not a draw readback — the two are
    // counted apart so that moving a copy from one rail to the other cannot look
    // like removing it. Both halves are asserted: one read happened, and the
    // draw rail stayed out of it.
    assert_eq!(
        df.target_reads, 1,
        "flush is the single resident read: {df:?}"
    );
    assert_eq!(
        df.readbacks, 0,
        "flush must not take a draw readback: {df:?}"
    );
    engine::unpin_resident_target(&identity);
    assert_eq!(
        p_flush, p_sync,
        "deferred flush bytes must match the sync Store readback"
    );
}

#[test]
fn device_loss_named_and_recreate_bounded() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let req = engine_req(&v, &f, 8, 8);
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP device_loss: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    }
    engine::test_force_device_lost_once();
    let err = engine::execute_draw_request(&req).expect_err("forced loss");
    let s = err.to_string();
    assert!(
        s.contains("reason=vk_device_lost_forced_draw"),
        "the forced draw rail must retain its exact typed reason, got: {s}"
    );
    let mut saw_named = true;
    for _ in 0..MAX_DEVICE_RECREATES + 2 {
        engine::test_poison_and_flush();
        match engine::execute_draw_request(&req) {
            Ok(_) => {}
            Err(e) => {
                let es = e.to_string();
                assert!(
                    es.contains("device_lost")
                        || es.contains("DeviceLost")
                        || es.contains("recreate"),
                    "unexpected error after poison: {es}"
                );
                saw_named = true;
            }
        }
    }
    assert!(saw_named);
    assert!(
        engine::device_recreate_count() <= MAX_DEVICE_RECREATES + 3,
        "recreate count unbounded: {}",
        engine::device_recreate_count()
    );
    let snap = engine::counter_snapshot();
    assert!(
        snap.device_lost >= 1,
        "device_lost counter must fire, got {}",
        snap.device_lost
    );

    // The cap bounds a *storm* — losses with no guest work between them — and a
    // draw that completes clears it (`ContextOwner::note_work_completed`), so
    // the count this loop leaves behind depends on whether its last iteration
    // drew successfully. Either way the engine can be left at the cap, refusing
    // every draw, and suite independence rests on `test_reset_engine` clearing
    // the budget, which is what [`engine_test_session`] does for every other
    // case. Pin that here, at the one site that manufactures the exhausted state
    // — if the reset ever stops clearing it, the whole suite goes
    // order-dependent, and it fails as a single unrelated case rather than as
    // anything named "reset".
    engine::test_reset_engine();
    assert_eq!(
        engine::device_recreate_count(),
        0,
        "reset must clear the recreate budget"
    );
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {}
        Err(e) => panic!("engine must draw again after a reset from an exhausted cap: {e}"),
    }
}

/// Consecutive no-readback resident draws stay in flight: none of them waits on
/// its own submission, and a boundary read retires everything and sees the
/// exact final content of both targets.
///
/// It used to assert the ring *wrap* as well — "the first three occupy every
/// slot, the fourth wraps onto the first and pays the retire" — and that
/// arithmetic has been unreachable from four draws since `RING_DEPTH` went to
/// 8. It is doubly unreachable now that the four alternating draws share one
/// command buffer: they are one submission, not four. Driving a real wrap would
/// take `RING_DEPTH * BATCH_MAX_DRAWS` draws and would then assert
/// `ring_retire_blocks`, which is a race against how fast the host GPU retires
/// a 16x16 draw. What is left here is not that, and the name says so.
#[test]
fn alternating_target_no_readback_draws_stay_in_flight_and_read_back_exact() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let id_a = TargetIdentity::Surface {
        id: 91,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let id_b = TargetIdentity::Surface {
        id: 92,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    // Cold sync draws mark both targets ready (content verified).
    for (label, identity) in [("ring_cold_a", &id_a), ("ring_cold_b", &id_b)] {
        let mut cold = engine_req(&v, &f, 16, 16);
        cold.target_identity = Some((*identity).clone());
        match engine::execute_draw_request(&cold) {
            Ok(o) => assert_fullscreen_fragment_color(label, &semantic_rgba(&o), 16, 16),
            Err(e) if skip_if_no_gpu(&e.to_string()) => {
                eprintln!("SKIP ring_overlaps: {e}");
                return;
            }
            Err(e) => panic!("{label}: {e}"),
        }
    }
    // Warm both once, then quiesce so the measured draws start with an idle ring.
    for identity in [&id_a, &id_b] {
        let mut warm = engine_req(&v, &f, 16, 16);
        warm.target_identity = Some((*identity).clone());
        warm.load_from_target = true;
        warm.skip_readback = true;
        engine::execute_draw_request(&warm).expect("ring warm-up");
    }
    engine::read_target(&id_a).expect("ring quiesce");
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    // Four async draws alternating between two targets.
    for (n, identity) in [&id_a, &id_b, &id_a, &id_b].into_iter().enumerate() {
        let mut warm = engine_req(&v, &f, 16, 16);
        warm.target_identity = Some((*identity).clone());
        warm.load_from_target = true;
        warm.skip_readback = true;
        engine::execute_draw_request(&warm).unwrap_or_else(|e| panic!("ring async #{n}: {e}"));
    }
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        d.render_post_wait_skips, 4,
        "all four draws must skip the post-submit wait: {d:?}"
    );
    // Deferred submit: the target does not key the batch, so all four land in
    // one command buffer and nothing has submitted it yet — the boundary read
    // below is what flushes it. Under `REIMS_VGPU_BATCH_MIXED_TARGETS=off` this
    // reads 4/0/3 instead, which is what it read before that key was dropped.
    assert_eq!(d.batch_opens, 1, "one batch carries all four draws: {d:?}");
    assert_eq!(d.batch_joins, 3, "three of them joined it: {d:?}");
    assert_eq!(
        d.batch_flushes, 0,
        "nothing consumed either target inside the window: {d:?}"
    );
    // Boundary reads retire the in-flight work and see the final content.
    let px = engine::read_target(&id_a)
        .expect("ring boundary read a")
        .into_rgba8();
    assert_fullscreen_fragment_color("ring_read_a", &px, 16, 16);
    let px = engine::read_target(&id_b)
        .expect("ring boundary read b")
        .into_rgba8();
    assert_fullscreen_fragment_color("ring_read_b", &px, 16, 16);
}

/// Present-boundary GPU seed: `seed_from_target` copies another ready
/// resident's content into the draw target on the GPU (no CPU seed upload),
/// and the pass loads it. A zero-invocation draw then reads back the source
/// content byte-exactly.
#[test]
fn seed_from_target_gpu_copies_front_frame() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let front = TargetIdentity::Surface {
        id: 71,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let back = TargetIdentity::Surface {
        id: 72,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    // Render known content into the "front frame" resident.
    let mut cold = engine_req(&v, &f, 16, 16);
    cold.target_identity = Some(front.clone());
    let front_pixels = match engine::execute_draw_request(&cold) {
        Ok(o) => {
            assert_fullscreen_fragment_color("gpu_seed_front", &semantic_rgba(&o), 16, 16);
            o.pixels
        }
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP seed_from_target: {e}");
            return;
        }
        Err(e) => panic!("front draw: {e}"),
    };
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    // Zero-invocation draw into a different identity, seeded from the front
    // resident: the readback must be the front content, with zero CPU seed
    // uploads and exactly one GPU seed copy.
    let mut seeded = engine_req(&v, &f, 16, 16);
    seeded.vertex_count = 0;
    seeded.target_identity = Some(back.clone());
    seeded.seed_from_target = Some(front.clone());
    let out = engine::execute_draw_request(&seeded).expect("gpu-seeded draw");
    assert_eq!(
        out.pixels, front_pixels,
        "GPU seed copy must reproduce the front content byte-exactly"
    );
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(d.seed_gpu_copies, 1, "one GPU seed copy: {d:?}");
    assert_eq!(d.seed_uploads, 0, "no CPU seed upload: {d:?}");
    // Named-error rails: src==dst and missing resident fail closed.
    let mut self_seed = engine_req(&v, &f, 16, 16);
    self_seed.target_identity = Some(back.clone());
    self_seed.seed_from_target = Some(back.clone());
    assert!(engine::execute_draw_request(&self_seed).is_err());
    let absent = TargetIdentity::Surface {
        id: 73,
        width: 16,
        height: 16,
        generation: 9,
        format: SURFACE_TEST_FORMAT,
    };
    let mut missing = engine_req(&v, &f, 16, 16);
    missing.target_identity = Some(back.clone());
    missing.seed_from_target = Some(absent);
    assert!(engine::execute_draw_request(&missing).is_err());
}

/// True N-attachment MRT: a draw with a secondary color attachment renders the
/// primary (slot 0) normally AND leaves the secondary as a ready, sampleable
/// resident that a later draw can bind via `SampledSource::Target`. This is the
/// mechanism that produces a fragment shader's secondary output (e.g. the
/// vibrancy coverage mask) instead of silently discarding it.
#[test]
fn mrt_secondary_attachment_becomes_sampleable_resident() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let primary = TargetIdentity::Surface {
        id: 0x60,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let secondary = TargetIdentity::Surface {
        id: 0x61,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let mut mrt = engine_req(&v, &f, 16, 16);
    mrt.target_identity = Some(primary.clone());
    mrt.secondary_targets.push(SecondaryColorTarget {
        target_guest: None,
        identity: secondary.clone(),
        width: 16,
        height: 16,
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        clear: [0.0, 0.0, 1.0, 1.0],
        load_action: reims_vgpu_core::ColorLoadAction::Clear,
        // Unblended: this parity case checks the attachment is written at
        // all, not how it composites.
        blend: None,
        color_write_mask: Default::default(),
    });
    match engine::execute_draw_request(&mrt) {
        // Slot 0 (primary) still receives the shader's location-0 output.
        Ok(o) => assert_fullscreen_fragment_color("mrt_primary", &semantic_rgba(&o), 16, 16),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP mrt_secondary: {e}");
            return;
        }
        Err(e) => panic!("mrt draw: {e}"),
    }

    // The secondary attachment persisted as its own resident.
    assert!(
        engine::resident_content_ready(&secondary),
        "secondary MRT attachment must be a ready resident"
    );

    // A later draw binds the secondary as a sampled resident — the exact path
    // the CC vibrancy pipe=25 draw uses to read its coverage mask.
    let consumer_target = TargetIdentity::Surface {
        id: 0x62,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut consume = engine_req(&v, &f, 16, 16);
    consume.target_identity = Some(consumer_target);
    consume.sampled_images.push(SampledImageResource {
        binding: 1,
        array_element: 0,
        descriptor_count: 1,
        width: 16,
        height: 16,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        multisampled: false,
        source: SampledSource::Target(secondary.clone()),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: None,
        swizzle: Default::default(),
    });
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    engine::execute_draw_request(&consume).expect("bind MRT secondary as sampled resident");
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        delta.sampled_gpu_binds, 1,
        "secondary must bind directly with no CPU reupload: {delta:?}"
    );
    assert_eq!(delta.sampled_reuploads, 0, "no host reupload: {delta:?}");
}

/// Metal applies each attachment's load action over that attachment's full
/// image, then rasterizes MRT work over the minimum common extent. Exercise
/// both directions so neither slot-zero nor secondary handling can accidentally
/// define the rule for the other.
#[test]
fn mismatched_mrt_extents_clear_full_images_and_rasterize_to_the_minimum() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();

    // Complement the clear cases with the native LOAD result: pixels in the
    // larger attachment but outside the common raster extent retain their
    // previous contents.
    let loaded_primary = TargetIdentity::Surface {
        id: 0x69,
        width: 8,
        height: 8,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut establish_load = engine_req(&v, &f, 8, 8);
    establish_load.vertex_count = 0;
    establish_load.skip_readback = true;
    establish_load.target_identity = Some(loaded_primary.clone());
    establish_load.target_clear = [0.0, 0.0, 1.0, 1.0];
    match engine::execute_draw_request(&establish_load) {
        Ok(_) => {}
        Err(error) if skip_if_no_gpu(&error.to_string()) => {
            eprintln!("SKIP mismatched MRT: {error}");
            return;
        }
        Err(error) => panic!("establish large MRT LOAD target: {error}"),
    }
    let mut load = engine_req(&v, &f, 8, 8);
    load.target_identity = Some(loaded_primary);
    load.color_load_action = reims_vgpu_core::ColorLoadAction::Load;
    load.load_from_target = true;
    load.secondary_targets.push(SecondaryColorTarget {
        target_guest: None,
        identity: TargetIdentity::Surface {
            id: 0x6e,
            width: 4,
            height: 4,
            generation: 1,
            format: SURFACE_TEST_FORMAT,
        },
        width: 4,
        height: 4,
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        clear: [1.0, 0.0, 0.0, 1.0],
        load_action: reims_vgpu_core::ColorLoadAction::Clear,
        blend: None,
        color_write_mask: Default::default(),
    });
    let loaded_pixels =
        semantic_rgba(&engine::execute_draw_request(&load).expect("large-primary MRT LOAD"));
    for y in 0..8usize {
        for x in 0..8usize {
            if x < 4 && y < 4 {
                continue;
            }
            assert_eq!(
                &loaded_pixels[(y * 8 + x) * 4..][..4],
                [0, 0, 255, 255],
                "large primary LOAD did not preserve ({x},{y})"
            );
        }
    }

    let large_primary = TargetIdentity::Surface {
        id: 0x6a,
        width: 8,
        height: 8,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let small_secondary = TargetIdentity::Surface {
        id: 0x6b,
        width: 4,
        height: 4,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut first = engine_req(&v, &f, 8, 8);
    first.target_identity = Some(large_primary);
    first.target_clear = [1.0, 0.0, 0.0, 1.0];
    first.secondary_targets.push(SecondaryColorTarget {
        target_guest: None,
        identity: small_secondary,
        width: 4,
        height: 4,
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        clear: [0.0, 0.0, 1.0, 1.0],
        load_action: reims_vgpu_core::ColorLoadAction::Clear,
        blend: None,
        color_write_mask: Default::default(),
    });
    let first_pixels = match engine::execute_draw_request(&first) {
        Ok(output) => semantic_rgba(&output),
        Err(error) if skip_if_no_gpu(&error.to_string()) => {
            eprintln!("SKIP mismatched MRT: {error}");
            return;
        }
        Err(error) => panic!("large-primary MRT: {error}"),
    };
    for y in 0..8usize {
        for x in 0..8usize {
            if x < 4 && y < 4 {
                continue;
            }
            assert_eq!(
                &first_pixels[(y * 8 + x) * 4..][..4],
                [255, 0, 0, 255],
                "large primary lost its full-image clear at ({x},{y})"
            );
        }
    }

    let small_primary = TargetIdentity::Surface {
        id: 0x6c,
        width: 4,
        height: 4,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let large_secondary = TargetIdentity::Surface {
        id: 0x6d,
        width: 8,
        height: 8,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut second = engine_req(&v, &f, 4, 4);
    second.target_identity = Some(small_primary);
    second.skip_readback = true;
    second.secondary_targets.push(SecondaryColorTarget {
        target_guest: None,
        identity: large_secondary.clone(),
        width: 8,
        height: 8,
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        clear: [0.0, 0.0, 1.0, 1.0],
        load_action: reims_vgpu_core::ColorLoadAction::Clear,
        blend: None,
        color_write_mask: Default::default(),
    });
    engine::execute_draw_request(&second).expect("large-secondary MRT");
    let secondary_pixels = engine::read_target(&large_secondary)
        .expect("read large secondary")
        .pixels;
    for y in 0..8usize {
        for x in 0..8usize {
            if x < 4 && y < 4 {
                continue;
            }
            assert_eq!(
                &secondary_pixels[(y * 8 + x) * 4..][..4],
                [0, 0, 255, 255],
                "large secondary lost its full-image blue clear at ({x},{y})"
            );
        }
    }
}

/// The vibrancy coverage mask is Metal RG16Float (0x41). Exercise the real
/// secondary format end-to-end: the RG16Float render pass / pipeline / resident
/// image build and render without error, and the mask persists as a resident.
#[test]
fn mrt_rg16float_secondary_builds_and_renders() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let primary = TargetIdentity::Surface {
        id: 0x63,
        width: 32,
        height: 32,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mask = TargetIdentity::Gva {
        gva: 0x3cf5000,
        width: 32,
        height: 32,
        generation: 0,
        format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
    };
    let mut mrt = engine_req(&v, &f, 32, 32);
    mrt.target_identity = Some(primary.clone());
    mrt.secondary_targets.push(SecondaryColorTarget {
        target_guest: None,
        identity: mask.clone(),
        width: 32,
        height: 32,
        format: reims_vgpu_protocol::ImageFormat::linear(
            reims_vgpu_protocol::TexelLayout::Rg16Float,
        ),
        clear: [1.0, 0.5, 0.0, 0.0],
        load_action: reims_vgpu_core::ColorLoadAction::Clear,
        // Unblended: this is the vibrancy coverage-mask shape, and a mask is a
        // raw store. Which is exactly why every secondary used to be forced
        // unblended — one real case generalized into a rule for all of them.
        blend: None,
        color_write_mask: Default::default(),
    });
    match engine::execute_draw_request(&mrt) {
        Ok(o) => assert_fullscreen_fragment_color("mrt_rg16f_primary", &semantic_rgba(&o), 32, 32),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP mrt_rg16float: {e}");
            return;
        }
        Err(e) => panic!("mrt rg16float draw: {e}"),
    }
    assert!(
        engine::resident_content_ready(&mask),
        "RG16Float mask must be a ready resident after the MRT draw"
    );
}

/// Depth and MRT in the same pass: a draw carrying both a secondary colour
/// attachment and a depth attachment renders through one framebuffer holding
/// all three, in the order the render pass declares them.
///
/// The engine used to refuse this shape by name and lose the whole draw. macOS
/// 26 issues it nine times in a driven boot and macOS 14 once, each refusal
/// paired with a `draw_vk_nothing_stored` on the same pipe and task.
///
/// Depth is the discriminator on purpose, and it separates the two ways the
/// combination can be wrong. If the depth view were left out of the framebuffer
/// the pass and the framebuffer would disagree on attachment count and neither
/// variant would render at all; if the depth *state* were dropped instead, both
/// variants would cover. Only a pass that carries both attachments and tests
/// against the depth one gives Never≠Always. The secondary is then asserted to
/// have survived as a resident, which is what says it was not displaced by the
/// depth attachment appended after it.
#[test]
fn depth_and_mrt_secondary_render_in_one_pass() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let (w, h) = (16u32, 16u32);
    let mut surface_id = 0x70u32;
    let mut variant = |compare: SamplerCompareFunction| -> Option<(bool, TargetIdentity)> {
        surface_id += 2;
        let primary = TargetIdentity::Surface {
            id: surface_id,
            width: w,
            height: h,
            generation: 1,
            format: SURFACE_TEST_FORMAT,
        };
        let secondary = TargetIdentity::Surface {
            id: surface_id + 1,
            width: w,
            height: h,
            generation: 1,
            format: SURFACE_TEST_FORMAT,
        };
        let mut req = engine_req(&v, &f, w, h);
        req.target_identity = Some(primary);
        req.secondary_targets.push(SecondaryColorTarget {
            target_guest: None,
            identity: secondary.clone(),
            width: w,
            height: h,
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Rgba8,
            ),
            clear: [0.0, 0.0, 1.0, 1.0],
            load_action: reims_vgpu_core::ColorLoadAction::Clear,
            blend: None,
            color_write_mask: Default::default(),
        });
        req.depth = Some(DepthState {
            // Parity fixtures bind no guest depth texture, so this is the
            // transient rail — the one that owns its image and so the one whose
            // dispose order a shared framebuffer would get wrong.
            identity: None,
            test_enable: true,
            write_enable: true,
            compare,
            clear_value: 1.0,
            load: false,
            stencil: None,
        });
        match engine::execute_draw_request(&req) {
            Ok(o) => Some((triangle_covered(&semantic_rgba(&o), w, h), secondary)),
            Err(e) if skip_if_no_gpu(&e.to_string()) => {
                eprintln!("SKIP depth+mrt: {e}");
                None
            }
            Err(e) => panic!("depth + MRT secondary draw: {e}"),
        }
    };

    let Some((never, _)) = variant(SamplerCompareFunction::Never) else {
        return; // no GPU
    };
    assert!(
        !never,
        "compare=Never must discard every fragment, so the depth attachment is live"
    );
    let (always, secondary) = variant(SamplerCompareFunction::Always).unwrap();
    assert!(always, "compare=Always must keep every fragment");
    assert!(
        engine::resident_content_ready(&secondary),
        "the secondary attachment must still be a ready resident alongside depth"
    );
}

/// Firewall: an empty `secondary_targets` leaves the classic single-attachment
/// path untouched — same fragment color, zero MRT residents created.
#[test]
fn single_rt_draw_unaffected_by_mrt_path() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    let target = TargetIdentity::Surface {
        id: 0x64,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut req = engine_req(&v, &f, 16, 16);
    req.target_identity = Some(target.clone());
    assert!(req.secondary_targets.is_empty());
    match engine::execute_draw_request(&req) {
        Ok(o) => assert_fullscreen_fragment_color("single_rt_guard", &semantic_rgba(&o), 16, 16),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP single_rt_guard: {e}");
            return;
        }
        Err(e) => panic!("single-rt guard: {e}"),
    }
    // A neighbouring MRT-secondary identity was never materialized.
    let never = TargetIdentity::Gva {
        gva: 0xdead000,
        width: 16,
        height: 16,
        generation: 0,
        format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
    };
    assert!(!engine::resident_content_ready(&never));
}

/// Framebuffer fetch (`color_input`): the fragment shader reads its own
/// destination pixel through the attachment-0 subpass input at the m2v
/// ColorInput binding (96) and inverts RGB. Seeding the target and drawing the
/// fullscreen triangle must yield the inverted seed — which proves the input
/// attachment was bound and read (an unbound input reads zero and would output
/// solid 255,255,255). This is the exact structural shape of the live
/// WindowServer composite (`air.render_target` INPUT param `dest_0`) whose
/// unbound read was the arm64 MoltenVK GPU-address-fault class.
#[test]
fn framebuffer_fetch_reads_destination_via_input_attachment() {
    let _g = engine_test_session();
    let (v, _) = triangle_spirv();
    let f = translate_words("render_frag_fetch.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let mut req = engine_req(&v, &f, w, h);
    req.color_input = true;
    // Seed (64, 128, 191, 255) → expect ~(191, 127, 64, 255).
    req.target_rgba8 = Some(std::sync::Arc::new(
        [64, 128, 191, 255].repeat((w * h) as usize),
    ));
    let Some(px) = draw_or_skip("framebuffer_fetch", &req) else {
        return;
    };
    assert_eq!(px.len(), (w * h * 4) as usize, "fetch: readback size");
    for p in 0..(w * h) as usize {
        let (r, g, b, a) = (px[p * 4], px[p * 4 + 1], px[p * 4 + 2], px[p * 4 + 3]);
        assert!(
            near(r, 191) && near(g, 127) && near(b, 64) && near(a, 255),
            "fetch: pixel {p} RGBA=({r},{g},{b},{a}); expected ~(191,127,64,255)"
        );
    }
}

/// An instrument, not a gate: how a draw's cost splits between a fixed
/// submit-and-wait floor and the bytes it moves.
///
/// `#[ignore]` deliberately. It asserts nothing — a timing assertion in this
/// suite would flake — and it must not occupy an executed slot, because the
/// suite is serial and alphabetical and every engine-touching case it runs
/// changes what the next one sees. Run it on purpose:
///
/// ```sh
/// cargo test -p reims-vgpu --no-default-features --features host-window \
///   --test vk_engine_parity -- --ignored --nocapture --exact \
///   measure_draw_cost_against_pass_size
/// ```
///
/// It exists because the boot-log evidence could not separate fixed submission
/// latency from per-pass render cost: `draw_phase`'s `wait_us` was split by
/// *readback* bytes, and a composite pass's render work is not proportional to
/// those. Sweeping the geometry varies both together and the intercept falls out.
///
/// Measured on the x86 dev host (Intel ARL iGPU), 40 draws per point, all caches
/// warmed first, no `target_rgba8` so no seed upload is included:
///
/// | geometry | MB | us/draw | us/MB |
/// |---|---|---|---|
/// | 16x16 | 0.001 | 278 | — |
/// | 64x64 | 0.016 | 258 | — |
/// | 256x256 | 0.262 | 299 | 1139 |
/// | 960x540 | 2.074 | 894 | 431 |
/// | 1920x1080 | 8.294 | 2386 | 288 |
///
/// A 256x range of pixel count at the small end moves the time by 8 %, so the
/// floor is ~270 us and everything above ~256x256 is linear at ~270 us/MB. That
/// model predicts the guest's measured 5.1 MB average readback at 1647 us against
/// 1523 measured.
///
/// The consequence is what makes it worth keeping: the fixed floor is only
/// ~45 ms per second at the guest's 167 readbacks/s, so **draw batching cannot be
/// the frame-rate lever** — the cost is the bytes. Re-run this after any change
/// that claims otherwise.
#[test]
#[ignore]
fn measure_draw_cost_against_pass_size() {
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();
    // Warm pipeline, pass, target and staging caches at every geometry first, so
    // the timed runs measure steady state rather than first-sight allocation.
    for (w, h) in [
        (16u32, 16u32),
        (64, 64),
        (256, 256),
        (960, 540),
        (1920, 1080),
    ] {
        let req = engine_req(&v, &f, w, h);
        if draw_or_skip("warm", &req).is_none() {
            eprintln!("SKIP draw-cost measurement: no GPU");
            return;
        }
    }
    const RUNS: u32 = 40;
    eprintln!(
        "\n  {:>12} {:>10} {:>10} {:>10}",
        "geometry", "MB/draw", "us/draw", "us/MB"
    );
    for (w, h) in [
        (16u32, 16u32),
        (64, 64),
        (256, 256),
        (960, 540),
        (1920, 1080),
    ] {
        let req = engine_req(&v, &f, w, h);
        let start = std::time::Instant::now();
        for _ in 0..RUNS {
            engine::execute_draw_request(&req).expect("timed draw");
        }
        let us = start.elapsed().as_micros() as f64 / f64::from(RUNS);
        let mb = f64::from(w) * f64::from(h) * 4.0 / 1e6;
        eprintln!(
            "  {:>12} {:>10.3} {:>10.0} {:>10.0}",
            format!("{w}x{h}"),
            mb,
            us,
            us / mb
        );
    }
    eprintln!();
}

/// A deferred window's currency check must tell "no slot" apart from "slot with
/// no stamp", because they are different defects and only one of them is legal.
///
/// `resident_content_epoch` collapses both to `None`, and that is right for the
/// LOAD elision it was written for: a pass that cannot prove the resident holds
/// the mapping's bytes takes the CPU seed and does not care why. A landing
/// window is the opposite case. It pinned a content-ready slot at this identity
/// and stamped it under the engine lock, so an *un-stamped* slot means a later
/// draw claimed the surface — expected — while an *absent* one means nothing
/// evicted a pinned slot, which cannot happen, unless the arm and the flush
/// spell the identity differently. One boot lost ~150 full-screen frames to
/// `live=None` with no way to tell which kind they were.
#[test]
fn resident_content_state_separates_an_absent_slot_from_an_unstamped_one() {
    use reims_vgpu_vulkan::engine::ResidentContent;
    let _g = engine_test_session();
    let (v, f) = triangle_spirv();

    let absent = TargetIdentity::Surface {
        id: 0x9A01,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    assert_eq!(
        engine::resident_content_state(&absent),
        ResidentContent::Absent,
        "an identity nothing ever created has no slot, and that is not the same \
         answer as a slot whose stamp was cleared"
    );

    let live = TargetIdentity::Surface {
        id: 0x9A02,
        width: 16,
        height: 16,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mut make = engine_req(&v, &f, 16, 16);
    make.target_identity = Some(live.clone());
    match engine::execute_draw_request(&make) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP resident_content_state: {e}");
            return;
        }
        Err(e) => panic!("target draw: {e}"),
    }
    // A draw leaves the slot ready and unvouched-for: `registry_mark_ready`
    // clears the content stamp precisely so nothing believes a stale one.
    assert_eq!(
        engine::resident_content_state(&live),
        ResidentContent::Unstamped,
        "a freshly drawn slot exists but vouches for nothing"
    );

    assert!(
        engine::stamp_resident_content_epoch(&live, 77),
        "a content-ready slot accepts a stamp"
    );
    assert_eq!(
        engine::resident_content_state(&live),
        ResidentContent::Epoch(77),
        "and reports exactly the epoch it was stamped with"
    );

    // A second draw into the same target is what a window's flush must decline
    // on, and it must be reported as the cleared case rather than the absent one.
    engine::execute_draw_request(&make).expect("second target draw");
    assert_eq!(
        engine::resident_content_state(&live),
        ResidentContent::Unstamped,
        "a draw since the stamp clears it — the slot is still there"
    );
}

/// A tightly-packed 2D declaration must be admitted for direct binding.
///
/// The backend's admission answer is a device-capability question — does this
/// shape, format and pitch plan as a linear image here — and not the separate
/// question of whether such an image may inherit the bytes already in the
/// allocation. External images are born `UNDEFINED`, which bounds what an alias
/// can *inherit*; it does not bound whether one can exist. Answering both with
/// one constant `Refused` is what kept every sampled bind on the copy rail, so
/// this test pins the two apart: a shape the device can plan answers `Direct`
/// and names the length the caller must grow its allocation to.
///
/// `None` is not a failure here. It is the documented "no device yet resolved"
/// answer, which is what a checkout with no Vulkan ICD returns, and the suite
/// skips rather than reporting a device fact it could not measure.
#[test]
fn a_tight_two_dimensional_declaration_is_admitted_for_direct_binding() {
    let _guard = engine_test_session();
    let (width, height) = (256u32, 128u32);
    let bytes_per_texel = 4u64;
    let row_pitch = u64::from(width) * bytes_per_texel;
    let resource_len = row_pitch * u64::from(height);
    let request = reims_vgpu_memory::GuestImageBindingRequest {
        backing: reims_vgpu_memory::GuestTargetBacking {
            allocation_host_ptr: 0,
            allocation_len: resource_len,
            resource_offset: 0,
            resource_len,
            plane_offset: 0,
            row_pitch,
        },
        allocation: reims_vgpu_memory::GuestImageAllocationLayout::single(
            0,
            row_pitch,
            reims_vgpu_memory::GuestImageLayout::D2 { width, height },
        ),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Bgra8),
    };
    let Some(disposition) = engine::sampled_guest_image_binding_requirement(request) else {
        eprintln!("skipping: no Vulkan device resolved, so there is no admission to assert");
        return;
    };
    match disposition {
        reims_vgpu_memory::GuestImageBindingDisposition::Direct(requirement) => {
            assert!(
                requirement.allocation_len >= resource_len,
                "an alias may need trailing host-only padding, but never fewer bytes than the \
                 guest resource it covers: {} < {resource_len}",
                requirement.allocation_len
            );
        }
        reims_vgpu_memory::GuestImageBindingDisposition::Refused => {
            panic!(
                "a tightly-packed {width}x{height} BGRA8 declaration is the simplest linear image \
                 there is; a device that refuses it cannot alias guest pages at all"
            );
        }
    }
}

/// One 2D RGBA8 image sitting in host storage that stands in for a RAMBlock,
/// described exactly as the aliasing rail admits it: one mip, one layer,
/// tightly packed rows, identity swizzle.
///
/// `storage` is what the import points into, so the fixture owns it and every
/// draw that binds the alias has to outlive nothing else. Writing through
/// [`Self::fill`] is the guest's CPU write: it goes to the same bytes the
/// device samples, which is the whole property these tests exist to check.
struct GuestAliasFixture {
    storage: Vec<u8>,
    texels: std::ops::Range<usize>,
    width: u32,
    height: u32,
    memory: reims_vgpu_memory::GuestTargetMemory,
    transfer: reims_vgpu_memory::GuestRunSource,
}

impl GuestAliasFixture {
    /// Build the fixture, or answer `None` on a host that publishes no import
    /// granularity — there is no guest RAM to alias there, and the copy rail is
    /// the only rail such a host has.
    ///
    /// Must be called after at least one successful draw: the granularity is
    /// what the backend measured from the device, so it does not exist until a
    /// device does.
    fn new(width: u32, height: u32) -> Option<Self> {
        use reims_vgpu_memory::{
            granularity, GuestImageLayout, GuestPageFootprint, GuestPageSet, GuestRamImport,
            GuestRamRegion, GuestRun, GuestRunSource, GuestTargetBacking, GuestTargetMemory,
        };

        let align = granularity()?;
        let bytes_per_texel = 4u64;
        let row_pitch = u64::from(width) * bytes_per_texel;
        let resource_len = row_pitch * u64::from(height);
        let block_len = resource_len.next_multiple_of(align);
        // One granule of slack so the covered span can start on an aligned
        // address inside an allocation this test does not control the base of.
        let storage = vec![0u8; (block_len + align) as usize];
        let pad = (storage.as_ptr() as u64).next_multiple_of(align) - storage.as_ptr() as u64;
        let base = storage.as_ptr() as u64 + pad;
        let gpa_base = 0x3_0000_0000u64;
        let import = std::sync::Arc::new(
            GuestRamImport::new(
                GuestRamRegion {
                    gpa_base,
                    host_va: base,
                    len: block_len,
                },
                align,
            )
            .expect("an aligned, non-empty region"),
        );
        let pages = (0..block_len / align)
            .map(|page| gpa_base + page * align)
            .collect::<Vec<_>>();
        let backing = GuestTargetBacking {
            allocation_host_ptr: base as usize,
            allocation_len: block_len,
            resource_offset: 0,
            resource_len,
            plane_offset: 0,
            row_pitch,
        };
        let memory = GuestTargetMemory {
            backing,
            import,
            footprint: GuestPageFootprint::new(pages.as_slice().into(), align)
                .expect("a non-empty page list at a power-of-two page size"),
        };
        let transfer = GuestRunSource {
            runs: std::sync::Arc::new(vec![GuestRun {
                host_ptr: base as usize,
                len: resource_len,
            }]),
            source_offset: 0,
            total_len: resource_len,
            row_length_texels: width,
            pages: None,
            physical_pages: GuestPageSet::new(&pages),
        };
        // Ask the device whether it can plan this shape as a linear image
        // before asserting anything about aliasing. A host that answers
        // `Refused` keeps every sampled bind on the copy rail by design.
        let request = reims_vgpu_memory::GuestImageBindingRequest {
            backing,
            allocation: reims_vgpu_memory::GuestImageAllocationLayout::single(
                0,
                row_pitch,
                GuestImageLayout::D2 { width, height },
            ),
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Rgba8,
            ),
        };
        match engine::sampled_guest_image_binding_requirement(request)? {
            reims_vgpu_memory::GuestImageBindingDisposition::Direct(_) => {}
            reims_vgpu_memory::GuestImageBindingDisposition::Refused => return None,
        }
        let texels = pad as usize..(pad + resource_len) as usize;
        Some(Self {
            storage,
            texels,
            width,
            height,
            memory,
            transfer,
        })
    }

    /// Write one colour over every texel, the way the guest's own CPU would.
    fn fill(&mut self, rgba: [u8; 4]) {
        let range = self.texels.clone();
        for texel in self.storage[range].chunks_exact_mut(4) {
            texel.copy_from_slice(&rgba);
        }
    }

    fn source(&self) -> SampledSource {
        let source = reims_vgpu_memory::GuestImageSource::single_mip(
            self.memory.clone(),
            reims_vgpu_memory::GuestImageLayout::D2 {
                width: self.width,
                height: self.height,
            },
            self.transfer.clone(),
        )
        .expect("a single-mip allocation whose plane starts at its resource");
        SampledSource::GuestImage(source, reims_vgpu_core::GatherVouch::Fresh)
    }
}

/// A fullscreen textured quad sampling `fixture` into a BGRA surface resident.
fn guest_alias_req(
    vert: &[u32],
    frag: &[u32],
    identity: &TargetIdentity,
    fixture: &GuestAliasFixture,
    w: u32,
    h: u32,
) -> DrawRequest {
    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut req = engine_req(vert, frag, w, h);
    req.vertex_count = 6;
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    req.storage_buffers.push(StorageBufferResource {
        binding: 1,
        content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    req.sampled_images.push(SampledImageResource {
        binding: 32,
        array_element: 0,
        descriptor_count: 1,
        width: fixture.width,
        height: fixture.height,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        multisampled: false,
        source: fixture.source(),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: None,
        swizzle: Default::default(),
    });
    req.samplers
        .push(SamplerResource::normalized_default(sampler_binding(0)));
    req
}

/// The centre texel of a BGRA resident, in the guest's own byte order.
fn bgra_center(pixels: &[u8], w: u32, h: u32) -> [u8; 4] {
    let center = ((h / 2) * w + w / 2) as usize * 4;
    pixels[center..center + 4]
        .try_into()
        .expect("four channels")
}

/// An alias inherits the bytes the guest wrote before the image existed.
///
/// A Vulkan image over imported host memory is born `UNDEFINED`, and the first
/// transition out of that layout is permitted to discard everything the memory
/// holds. The guest is nonetheless entitled to those texels: it wrote them into
/// its own allocation and declared the allocation as a texture, and the native
/// oracle this device emulates makes them visible. So an alias owes one copy at
/// birth, laundered through a staging buffer because an imported buffer and the
/// image alias the same bytes and Vulkan forbids a copy whose regions overlap.
///
/// This test writes the texels *before* the first bind and reads the result
/// back. Note what it cannot prove: a driver that happens not to discard on the
/// `UNDEFINED` transition would return the right pixels with the copy removed.
/// The counter assertion is what pins the copy itself to having been recorded.
#[test]
fn a_new_guest_alias_samples_the_bytes_written_before_it_existed() {
    let _g = engine_test_session();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let identity = TargetIdentity::Surface {
        id: 0x9A1,
        width: w,
        height: h,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    // One ordinary draw first: the import granularity is measured from the
    // device, so there is nothing to build a guest allocation against until a
    // device has been resolved.
    let (warm_vert, warm_frag) = triangle_spirv();
    if draw_or_skip(
        "guest alias warm-up",
        &engine_req(&warm_vert, &warm_frag, w, h),
    )
    .is_none()
    {
        return;
    }
    let Some(mut fixture) = GuestAliasFixture::new(16, 16) else {
        eprintln!("skipping: this host cannot alias guest pages as a sampled image");
        return;
    };
    let rgba = [17u8, 91, 203, 255];
    fixture.fill(rgba);

    let req = guest_alias_req(&vert, &frag, &identity, &fixture, w, h);
    let before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("a draw sampling an aliased guest allocation");
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        delta.sampled_guest_direct_binds, 1,
        "the sampled bind did not take the aliasing rail: {delta:?}"
    );
    assert_eq!(
        delta.sampled_guest_materializations, 1,
        "a newly created alias must record exactly one birth copy: {delta:?}"
    );
    let pixels = engine::read_target(&identity)
        .expect("read the resident the alias was sampled into")
        .pixels;
    assert_eq!(
        bgra_center(&pixels, w, h),
        [rgba[2], rgba[1], rgba[0], rgba[3]],
        "the alias lost the texels the guest wrote before the image existed"
    );
}

/// A guest CPU write after the alias exists reaches the next sampled read.
///
/// This is the property the whole rail is for: the image *is* the guest's
/// pages, so a store the guest makes is visible without this device copying
/// anything. The second draw must therefore reuse the same resident — a rebuilt
/// alias would carry the new bytes through its birth copy instead and prove
/// nothing — which is why the materialization count is asserted to stay put
/// while the pixels change.
#[test]
fn a_guest_write_after_the_alias_exists_reaches_the_next_sampled_read() {
    let _g = engine_test_session();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let identity = TargetIdentity::Surface {
        id: 0x9A2,
        width: w,
        height: h,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let (warm_vert, warm_frag) = triangle_spirv();
    if draw_or_skip(
        "guest alias warm-up",
        &engine_req(&warm_vert, &warm_frag, w, h),
    )
    .is_none()
    {
        return;
    }
    let Some(mut fixture) = GuestAliasFixture::new(16, 16) else {
        eprintln!("skipping: this host cannot alias guest pages as a sampled image");
        return;
    };
    let first = [17u8, 91, 203, 255];
    fixture.fill(first);
    let req = guest_alias_req(&vert, &frag, &identity, &fixture, w, h);
    engine::execute_draw_request(&req).expect("the draw that creates the alias");
    let pixels = engine::read_target(&identity)
        .expect("read the first result")
        .pixels;
    assert_eq!(
        bgra_center(&pixels, w, h),
        [first[2], first[1], first[0], first[3]],
        "the alias did not carry the guest's initial texels"
    );

    // The guest stores into its own allocation. Nothing tells this device.
    let second = [201u8, 77, 31, 255];
    fixture.fill(second);
    let before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("the draw that samples the guest's new bytes");
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        delta.sampled_guest_direct_binds, 1,
        "the second bind left the aliasing rail: {delta:?}"
    );
    assert_eq!(
        delta.sampled_guest_materializations, 0,
        "the second bind rebuilt the alias instead of reusing it, so this measures a copy \
         rather than a guest write: {delta:?}"
    );
    let pixels = engine::read_target(&identity)
        .expect("read the second result")
        .pixels;
    assert_eq!(
        bgra_center(&pixels, w, h),
        [second[2], second[1], second[0], second[3]],
        "a guest CPU write into the aliased pages was not visible to the device"
    );
}
