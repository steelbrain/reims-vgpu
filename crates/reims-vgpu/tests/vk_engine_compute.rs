//! Off-VM compute regression suite for the internal Vulkan engine.
//!
//! Drives `engine::execute_compute_request` (the shipped entry) and asserts
//! **known-correct** buffer/image results (no external compute executor). Also
//! locks warm-dispatch zero create/alloc. Requires a Vulkan ICD; skips cleanly
//! if init fails.
//!
//! **Serial:** process-global engine; suite takes a lock.

#![cfg(feature = "backend-vulkan")]

use reims_vgpu::backend::vulkan::engine::{
    self, ComputeBufferResource, ComputeRequest, ComputeResidentSampleBind,
    ComputeSampledImageResource, ComputeStorageImageResource, ComputeStorageResidency,
    StorageImageFormat,
};
use reims_vgpu::model::ComputeStorageResidencyKey;
use std::path::PathBuf;
use std::process::Command;
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

fn skip_if_no_gpu(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("no vulkan")
        || lower.contains("load vulkan")
        || lower.contains("create_instance")
        || lower.contains("no graphics")
        || lower.contains("vk_engine_init")
        || lower.contains("no combined")
}

fn inc_comp_spirv() -> Option<Vec<u32>> {
    let comp = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inc.comp");
    let spv = std::env::temp_dir().join(format!("paravirt_engine_inc_{}.spv", std::process::id()));
    let status = Command::new("glslc")
        .args([
            "-fshader-stage=comp",
            comp.to_str().unwrap(),
            "-o",
            spv.to_str().unwrap(),
        ])
        .status();
    if !matches!(status, Ok(s) if s.success()) {
        eprintln!("SKIP: no glslc for inc.comp");
        return None;
    }
    let bytes = std::fs::read(&spv).ok()?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// The one storage-image kernel every write test in this file uses:
/// `imageStore(img, ivec2(gid.xy), vec4(1,0,0,1))` over a 1x1x1 local size,
/// with the image's SPIR-V storage format left to the caller.
///
/// Six tests carried byte-identical copies of this assembly that differed only
/// in that one token — `Rgba8` for the native-format cases, `Unknown` for the
/// BGRA view, which SPIR-V has no storage format for. Parameterising the token
/// is what keeps a change to the kernel from having to be made six times.
///
/// `compute_storage_image_r16float_if_supported` deliberately does not use this
/// and keeps its own copy: R16f needs `OpCapability
/// StorageImageExtendedFormats` and drops the `NonReadable` decoration, so it
/// is a different kernel rather than a different format.
/// The sampled-image fetch kernel: read texel (0,0) of a combined sampled image
/// at descriptor 0/binding 0 and write it to the storage buffer at binding 1.
///
/// Three tests used byte-identical copies of this — 2587 characters each, not
/// one token apart. It is one fixture, so it is one constant.
const SAMPLED_IMAGE_FETCH_KERNEL: &str = r#"
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid %out %image_var
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %out DescriptorSet 0
               OpDecorate %out Binding 0
               OpDecorate %image_var DescriptorSet 0
               OpDecorate %image_var Binding 32
               OpDecorate %Out Block
               OpMemberDecorate %Out 0 Offset 0
               OpDecorate %OutWords ArrayStride 4
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
     %v3uint = OpTypeVector %uint 3
     %v2uint = OpTypeVector %uint 2
     %v4uint = OpTypeVector %uint 4
    %v4float = OpTypeVector %float 4
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
     %uint_3 = OpConstant %uint 3
   %OutWords = OpTypeRuntimeArray %uint
        %Out = OpTypeStruct %OutWords
%_ptr_StorageBuffer_Out = OpTypePointer StorageBuffer %Out
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
        %out = OpVariable %_ptr_StorageBuffer_Out StorageBuffer
      %Image = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_Image = OpTypePointer UniformConstant %Image
  %image_var = OpVariable %_ptr_UniformConstant_Image UniformConstant
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
        %gid = OpVariable %_ptr_Input_v3uint Input
    %fn_type = OpTypeFunction %void
       %main = OpFunction %void None %fn_type
      %entry = OpLabel
    %gid_val = OpLoad %v3uint %gid
          %x = OpCompositeExtract %uint %gid_val 0
      %coord = OpCompositeConstruct %v2uint %x %uint_0
    %image_v = OpLoad %Image %image_var
      %texel = OpImageFetch %v4float %image_v %coord Lod %uint_0
       %bits = OpBitcast %v4uint %texel
      %lane0 = OpCompositeExtract %uint %bits 0
      %lane1 = OpCompositeExtract %uint %bits 1
      %lane2 = OpCompositeExtract %uint %bits 2
      %lane3 = OpCompositeExtract %uint %bits 3
       %ptr0 = OpAccessChain %_ptr_StorageBuffer_uint %out %uint_0 %uint_0
               OpStore %ptr0 %lane0
       %ptr1 = OpAccessChain %_ptr_StorageBuffer_uint %out %uint_0 %uint_1
               OpStore %ptr1 %lane1
       %ptr2 = OpAccessChain %_ptr_StorageBuffer_uint %out %uint_0 %uint_2
               OpStore %ptr2 %lane2
       %ptr3 = OpAccessChain %_ptr_StorageBuffer_uint %out %uint_0 %uint_3
               OpStore %ptr3 %lane3
               OpReturn
               OpFunctionEnd
"#;

fn storage_image_write_red_kernel(spirv_image_format: &str) -> String {
    KERNEL_TEMPLATE.replace("{FMT}", spirv_image_format)
}

const KERNEL_TEMPLATE: &str = r#"
               OpCapability Shader
               OpCapability StorageImageWriteWithoutFormat
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid %img
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %img DescriptorSet 0
               OpDecorate %img Binding 0
               OpDecorate %img NonReadable
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
     %v3uint = OpTypeVector %uint 3
      %v2int = OpTypeVector %int 2
    %v4float = OpTypeVector %float 4
    %float_1 = OpConstant %float 1
    %float_0 = OpConstant %float 0
     %red = OpConstantComposite %v4float %float_1 %float_0 %float_0 %float_1
%img_ty = OpTypeImage %float 2D 0 0 0 2 {FMT}
%_ptr_img = OpTypePointer UniformConstant %img_ty
        %img = OpVariable %_ptr_img UniformConstant
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
        %gid = OpVariable %_ptr_Input_v3uint Input
    %fn_type = OpTypeFunction %void
       %main = OpFunction %void None %fn_type
      %entry = OpLabel
    %gid_val = OpLoad %v3uint %gid
          %x = OpCompositeExtract %uint %gid_val 0
          %y = OpCompositeExtract %uint %gid_val 1
         %xi = OpBitcast %int %x
         %yi = OpBitcast %int %y
       %coord = OpCompositeConstruct %v2int %xi %yi
      %img_l = OpLoad %img_ty %img
               OpImageWrite %img_l %coord %red
               OpReturn
               OpFunctionEnd
"#;

fn assemble_spvasm(asm: &str, name: &str) -> Option<Vec<u32>> {
    let asm_path = std::env::temp_dir().join(format!(
        "paravirt_engine_{}_{}.spvasm",
        name,
        std::process::id()
    ));
    let spv_path = std::env::temp_dir().join(format!(
        "paravirt_engine_{}_{}.spv",
        name,
        std::process::id()
    ));
    std::fs::write(&asm_path, asm).ok()?;
    let status = Command::new("spirv-as")
        .args([asm_path.to_str().unwrap(), "-o", spv_path.to_str().unwrap()])
        .status();
    if !matches!(status, Ok(s) if s.success()) {
        eprintln!("SKIP {name}: no spirv-as");
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

fn translate_kernel(name: &str) -> Option<Vec<u32>> {
    use metal2vulkan::passes::Stage;
    let air = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/air")
        .join(format!("{name}.air"));
    if !air.exists() {
        eprintln!("SKIP {name}: fixture missing");
        return None;
    }
    let tmp =
        std::env::temp_dir().join(format!("paravirt_engine_k_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).ok()?;
    let bytes = metal2vulkan::translate(air.to_str().unwrap(), Stage::Kernel, &tmp).ok()?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn engine_or_skip(label: &str, req: &ComputeRequest) -> Option<engine::ComputeOutput> {
    match engine::execute_compute_request(req) {
        Ok(o) => Some(o),
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

/// SSBO-only: b[i] += 1 for 256 floats (known result 11.0 from seed 10.0).
#[test]
fn compute_inc_ssbo_known_result() {
    let _g = engine_test_session();
    let Some(words) = inc_comp_spirv() else {
        return;
    };
    let n = 256usize;
    let input: Vec<u8> = (0..n).flat_map(|_| 10.0f32.to_le_bytes()).collect();
    let grid = (n as u32).div_ceil(64);

    let req = ComputeRequest {
        spirv: words,
        entry: "main".into(),
        grid: [grid, 1, 1],
        storage_buffers: vec![ComputeBufferResource {
            binding: 0,
            bytes: input,
            writable: true,
        }],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![],
    };
    let Some(out) = engine_or_skip("compute_inc", &req) else {
        return;
    };
    assert_eq!(out.buffers.len(), 1);
    let r: Vec<f32> = out.buffers[0]
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(r.len(), n);
    assert!(
        r.iter().all(|&x| x == 11.0),
        "expected all 11.0, got {:?}",
        &r[..4.min(r.len())]
    );
}

/// Proven read-only SSBOs are uploaded and bound, but never mapped back to the
/// host after the fence.
#[test]
fn compute_readonly_ssbo_has_zero_readback() {
    let _g = engine_test_session();
    let spvasm = r#"
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %input
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %input DescriptorSet 0
               OpDecorate %input Binding 0
               OpDecorate %Buf Block
               OpMemberDecorate %Buf 0 Offset 0
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
        %Buf = OpTypeStruct %uint
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
      %input = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
    %fn_type = OpTypeFunction %void
       %main = OpFunction %void None %fn_type
      %entry = OpLabel
        %ptr = OpAccessChain %_ptr_StorageBuffer_uint %input %uint_0
      %value = OpLoad %uint %ptr
               OpReturn
               OpFunctionEnd
"#;
    let Some(words) = assemble_spvasm(spvasm, "readonly_ssbo") else {
        return;
    };
    let req = ComputeRequest {
        spirv: words,
        entry: "main".into(),
        grid: [1, 1, 1],
        storage_buffers: vec![ComputeBufferResource {
            binding: 0,
            bytes: 0x1234_5678u32.to_le_bytes().to_vec(),
            writable: false,
        }],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![],
    };
    let Some(out) = engine_or_skip("compute readonly ssbo", &req) else {
        return;
    };
    assert!(out.buffers.is_empty());
    let snap = engine::counter_snapshot();
    assert_eq!(snap.readbacks, 0);
    assert_eq!(snap.readback_bytes, 0);
}

/// 2D grid tiling: proves grid.y is not dropped (GlobalInvocationId.y varies).
#[test]
fn compute_2d_grid_tiles_global_invocation_xy() {
    let _g = engine_test_session();
    let spvasm = r#"
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid %out
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %out DescriptorSet 0
               OpDecorate %out Binding 0
               OpDecorate %Out Block
               OpMemberDecorate %Out 0 Offset 0
               OpDecorate %OutWords ArrayStride 4
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
     %uint_0 = OpConstant %uint 0
     %uint_8 = OpConstant %uint 8
  %uint_1000 = OpConstant %uint 1000
   %OutWords = OpTypeRuntimeArray %uint
        %Out = OpTypeStruct %OutWords
%_ptr_StorageBuffer_Out = OpTypePointer StorageBuffer %Out
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
        %out = OpVariable %_ptr_StorageBuffer_Out StorageBuffer
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
        %gid = OpVariable %_ptr_Input_v3uint Input
    %fn_type = OpTypeFunction %void
       %main = OpFunction %void None %fn_type
      %entry = OpLabel
    %gid_val = OpLoad %v3uint %gid
          %x = OpCompositeExtract %uint %gid_val 0
          %y = OpCompositeExtract %uint %gid_val 1
         %yw = OpIMul %uint %y %uint_8
        %idx = OpIAdd %uint %yw %x
      %y1000 = OpIMul %uint %y %uint_1000
        %val = OpIAdd %uint %y1000 %x
        %ptr = OpAccessChain %_ptr_StorageBuffer_uint %out %uint_0 %idx
               OpStore %ptr %val
               OpReturn
               OpFunctionEnd
"#;
    let Some(words) = assemble_spvasm(spvasm, "dispatch_2d") else {
        return;
    };
    let zeros = vec![0u8; 64 * 4];
    let req = ComputeRequest {
        spirv: words.clone(),
        entry: "main".into(),
        grid: [8, 8, 1],
        storage_buffers: vec![ComputeBufferResource {
            binding: 0,
            bytes: zeros.clone(),
            writable: true,
        }],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![],
    };
    let Some(out) = engine_or_skip("compute_2d", &req) else {
        return;
    };
    let vals: Vec<u32> = out.buffers[0]
        .bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for y in 0..8u32 {
        for x in 0..8u32 {
            let idx = (y * 8 + x) as usize;
            let want = y * 1000 + x;
            assert_eq!(vals[idx], want, "at ({x},{y})");
        }
    }
}

/// Storage image Rgba8Unorm: each invocation writes solid red (known unorm result).
#[test]
fn compute_storage_image_rgba8unorm_known_result() {
    let _g = engine_test_session();
    // Kernel: imageStore(out, ivec2(gid.xy), vec4(1,0,0,1)) for 4x4 Rgba8Unorm.
    let spvasm = &storage_image_write_red_kernel("Rgba8");
    let Some(words) = assemble_spvasm(spvasm, "simg_rgba8") else {
        return;
    };
    let w = 4u32;
    let h = 4u32;
    let seed = vec![0u8; (w * h * 4) as usize];
    let identity = ComputeStorageResidencyKey {
        mapping_id: 77,
        map_generation: 3,
        surface_offset: 0,
        surface_bpr: w * 4,
        span_end: (w * h * 4) as u64,
        width: w,
        height: h,
        pixel_format: 0x46,
        texture_ref: 0,
    };
    let req = ComputeRequest {
        spirv: words.clone(),
        entry: "main".into(),
        grid: [w, h, 1],
        storage_buffers: vec![],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![ComputeStorageImageResource {
            binding: 0,
            format: StorageImageFormat::Rgba8Unorm,
            width: w,
            height: h,
            bytes: seed.clone(),
            residency: Some(ComputeStorageResidency {
                identity,
                seed_generation: 1,
                output_generation: 2,
            }),
            seed_skipped: false,
        }],
    };
    let Some(out) = engine_or_skip("simg_rgba8", &req) else {
        return;
    };
    assert_eq!(out.images.len(), 1);
    // Every texel should be (255,0,0,255) approximately for unorm write of 1,0,0,1
    for p in out.images[0].chunks_exact(4) {
        assert!(
            p[0] >= 254 && p[1] == 0 && p[2] == 0 && p[3] >= 254,
            "unexpected texel {p:?}"
        );
    }
    let snap = engine::counter_snapshot();
    assert_eq!(snap.compute_storage_seed_uploads, 1);
    assert_eq!(snap.compute_storage_seed_upload_bytes, seed.len() as u64);
    assert_eq!(snap.compute_sampled_uploads, 0);

    engine::reset_draw_counters();
    let resident_seed = out.images[0].clone();
    let hit_req = ComputeRequest {
        spirv: words.clone(),
        entry: "main".into(),
        grid: [w, h, 1],
        storage_buffers: vec![],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![ComputeStorageImageResource {
            binding: 0,
            format: StorageImageFormat::Rgba8Unorm,
            width: w,
            height: h,
            bytes: resident_seed.clone(),
            residency: Some(ComputeStorageResidency {
                identity,
                seed_generation: 2,
                output_generation: 3,
            }),
            seed_skipped: false,
        }],
    };
    let hit = engine::execute_compute_request(&hit_req).expect("resident compute hit");
    assert_eq!(hit.images[0], resident_seed);
    let hit_counters = engine::counter_snapshot();
    assert_eq!(hit_counters.compute_storage_seed_uploads, 0);
    assert_eq!(hit_counters.compute_storage_seed_upload_bytes, 0);
    assert_eq!(hit_counters.readback_bytes, seed.len() as u64);

    // A generation mismatch must upload the supplied guest seed. Dispatch one
    // texel only so untouched texels prove the black seed replaced residency.
    engine::reset_draw_counters();
    let mismatch_req = ComputeRequest {
        spirv: words,
        entry: "main".into(),
        grid: [1, 1, 1],
        storage_buffers: vec![],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![ComputeStorageImageResource {
            binding: 0,
            format: StorageImageFormat::Rgba8Unorm,
            width: w,
            height: h,
            bytes: seed.clone(),
            residency: Some(ComputeStorageResidency {
                identity,
                seed_generation: 9,
                output_generation: 10,
            }),
            seed_skipped: false,
        }],
    };
    let mismatch = engine::execute_compute_request(&mismatch_req).expect("generation mismatch");
    assert!(mismatch.images[0][0] >= 254 && mismatch.images[0][3] >= 254);
    assert!(mismatch.images[0][4..].iter().all(|byte| *byte == 0));
    let mismatch_counters = engine::counter_snapshot();
    assert_eq!(mismatch_counters.compute_storage_seed_uploads, 1);
    assert_eq!(
        mismatch_counters.compute_storage_seed_upload_bytes,
        seed.len() as u64
    );
    let reset = engine::reset_guest_state();
    assert_eq!(reset.storage_images, 1, "resident image is guest state");
}

/// The compute-storage registry retains every admitted resident past the slot
/// count that used to bound it.
///
/// A device-level regression test for the property that retired
/// `COMPUTE_STORAGE_REGISTRY_CAP = 64`: that population is bounded by the
/// allocation refusing, not by a count, so admitting more identities than the old
/// count allowed must destroy none of them. This registry's losses were the worse
/// of the two — nothing recreates a compute-storage resident's contents, so one
/// taken here costs a refused dispatch rather than a re-upload.
///
/// The observable is `compute_storage_seed_uploads`. A dispatch whose
/// `seed_generation` matches the resident's `output_generation` skips the seed
/// upload entirely; a dispatch that finds no resident must upload. So a zero here
/// on the *first* identity admitted — the one an LRU sweep takes first — is the
/// resident still being there after 79 later admissions.
///
/// Fails against the retired walk: the sweep starts at 64 admissions and the
/// first identity is its first victim, so the re-dispatch uploads its seed.
#[test]
fn every_admitted_compute_storage_resident_survives_past_the_retired_slot_cap() {
    let _g = engine_test_session();
    let spvasm = &storage_image_write_red_kernel("Rgba8");
    let Some(words) = assemble_spvasm(spvasm, "simg_retain") else {
        return;
    };
    let (w, h) = (4u32, 4u32);
    let seed = vec![0u8; (w * h * 4) as usize];
    // 64 was the last value that count held, and it swept on *admission*, so 80
    // clears it with margin. At 4x4 the whole set is a few hundred KiB, so no
    // real allocation failure is in play and the only thing that could remove one
    // of these is a count.
    const ADMITS: u32 = 80;
    let identity = |i: u32| ComputeStorageResidencyKey {
        mapping_id: 0x900 + i,
        map_generation: 3,
        surface_offset: 0,
        surface_bpr: w * 4,
        span_end: (w * h * 4) as u64,
        width: w,
        height: h,
        pixel_format: 0x46,
        texture_ref: 0,
    };
    let request = |i: u32, seed_generation: u32| ComputeRequest {
        spirv: words.clone(),
        entry: "main".into(),
        grid: [w, h, 1],
        storage_buffers: vec![],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![ComputeStorageImageResource {
            binding: 0,
            format: StorageImageFormat::Rgba8Unorm,
            width: w,
            height: h,
            bytes: seed.clone(),
            residency: Some(ComputeStorageResidency {
                identity: identity(i),
                seed_generation,
                output_generation: 2,
            }),
            seed_skipped: false,
        }],
    };

    if engine_or_skip("compute_storage_retention", &request(0, 1)).is_none() {
        return;
    }
    // The runtime takes this edge after `writeback_texture` lands the output in
    // the guest's pages; this suite drives the engine directly, so it takes it
    // here. Without it every resident stays flagged as the only copy of its
    // contents, no reclaim path may select one, and the assertions below hold
    // against *any* policy — including a slot count — which is exactly the
    // vacuous pass this call removes.
    engine::note_resident_storage_copied_out(&identity(0));
    for i in 1..ADMITS {
        engine::execute_compute_request(&request(i, 1)).expect("filler dispatch");
        engine::note_resident_storage_copied_out(&identity(i));
    }

    // Every one of them, oldest first. The first assertion alone would pass a
    // sweep that happened to spare the identity it was asked about.
    for i in 0..ADMITS {
        engine::reset_draw_counters();
        engine::execute_compute_request(&request(i, 2)).expect("resident re-dispatch");
        assert_eq!(
            engine::counter_snapshot().compute_storage_seed_uploads,
            0,
            "resident {i} was destroyed by something other than an allocation failure"
        );
    }
}

/// Regression lock for the BGRA storage-composite R/B fix: a guest `BGRA8Unorm`
/// storage surface composites through a format-less (`Unknown`) SPIR-V storage
/// image viewed `B8G8R8A8_UNORM`, so a shader that writes logical red `(1,0,0,1)`
/// must land in guest channel order — byte0=B=0, byte1=G=0, byte2=R=255,
/// byte3=A=255 — NOT the swapped `(255,0,0,255)` a `Rgba8Unorm` view produces.
/// This is the exact mechanism that rendered the 4K desktop blue before the fix.
#[test]
fn compute_storage_image_bgra8unorm_is_not_channel_swapped() {
    let _g = engine_test_session();
    if !engine::supports_storage_image_write_without_format() {
        // The device (or its B8G8R8A8_UNORM storage support) is absent — the
        // product path degrades to the swapped Rgba8Unorm view, so this
        // BGRA-order assertion does not apply. Mirrors the runtime gate.
        eprintln!("skip: no shaderStorageImageWriteWithoutFormat / BGRA8 storage");
        return;
    }
    // Kernel: imageStore(out, gid.xy, vec4(1,0,0,1)) into an Unknown-format
    // storage image — the only SPIR-V form compatible with a B8G8R8A8_UNORM
    // view (SPIR-V has no `Bgra8` storage format).
    let spvasm = &storage_image_write_red_kernel("Unknown");
    let Some(words) = assemble_spvasm(spvasm, "simg_bgra8") else {
        return;
    };
    let w = 4u32;
    let h = 4u32;
    let seed = vec![0u8; (w * h * 4) as usize];
    let identity = ComputeStorageResidencyKey {
        mapping_id: 78,
        map_generation: 3,
        surface_offset: 0,
        surface_bpr: w * 4,
        span_end: (w * h * 4) as u64,
        width: w,
        height: h,
        // MTLPixelFormatBGRA8Unorm.
        pixel_format: 0x50,
        texture_ref: 0,
    };
    let req = ComputeRequest {
        spirv: words,
        entry: "main".into(),
        grid: [w, h, 1],
        storage_buffers: vec![],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![ComputeStorageImageResource {
            binding: 0,
            format: StorageImageFormat::Bgra8Unorm,
            width: w,
            height: h,
            bytes: seed,
            residency: Some(ComputeStorageResidency {
                identity,
                seed_generation: 1,
                output_generation: 2,
            }),
            seed_skipped: false,
        }],
    };
    let Some(out) = engine_or_skip("simg_bgra8", &req) else {
        return;
    };
    assert_eq!(out.images.len(), 1);
    // Logical red stored into BGRA memory: B=0, G=0, R=255, A=255. A swap
    // (Rgba8Unorm view) would instead give byte0=255 — the bug.
    for p in out.images[0].chunks_exact(4) {
        assert!(
            p[0] == 0 && p[1] == 0 && p[2] >= 254 && p[3] >= 254,
            "BGRA channel order wrong (R/B swap?): texel {p:?}"
        );
    }
}

/// `seed_skipped` contract: a generation-matching resident dispatch renders
/// from the GPU-resident content while `bytes` is a zero placeholder (no seed
/// upload); once the resident is gone the same request must fail with
/// `vk_compute_exec_resident_seed_generation_lost` — never silently seed the
/// placeholder.
#[test]
fn compute_storage_image_seed_skip_and_lost_resident() {
    let _g = engine_test_session();
    // Same red-fill kernel as compute_storage_image_rgba8unorm_known_result.
    let spvasm = &storage_image_write_red_kernel("Rgba8");
    let Some(words) = assemble_spvasm(spvasm, "simg_seed_skip") else {
        return;
    };
    let w = 4u32;
    let h = 4u32;
    let identity = ComputeStorageResidencyKey {
        mapping_id: 91,
        map_generation: 1,
        surface_offset: 0,
        surface_bpr: w * 4,
        span_end: (w * h * 4) as u64,
        width: w,
        height: h,
        pixel_format: 0x46,
        texture_ref: 0,
    };
    let make = |grid: [u32; 3], seed_generation: u32, output_generation: u32, skipped: bool| {
        ComputeRequest {
            spirv: words.clone(),
            entry: "main".into(),
            grid,
            storage_buffers: vec![],
            sampled_images: vec![],
            samplers: vec![],
            storage_images: vec![ComputeStorageImageResource {
                binding: 0,
                format: StorageImageFormat::Rgba8Unorm,
                width: w,
                height: h,
                bytes: vec![0u8; (w * h * 4) as usize],
                residency: Some(ComputeStorageResidency {
                    identity,
                    seed_generation,
                    output_generation,
                }),
                seed_skipped: skipped,
            }],
        }
    };
    // Full red fill establishes the resident at generation 2.
    let Some(fill) = engine_or_skip("seed_skip_fill", &make([w, h, 1], 1, 2, false)) else {
        return;
    };
    assert!(fill.images[0].chunks_exact(4).all(|p| p[0] >= 254));

    // Skip dispatch: one texel, zero-placeholder bytes, matching generation.
    // Untouched texels staying red prove the placeholder was never seeded.
    engine::reset_draw_counters();
    let skip = engine::execute_compute_request(&make([1, 1, 1], 2, 3, true))
        .expect("seed-skip resident hit");
    for p in skip.images[0].chunks_exact(4) {
        assert!(
            p[0] >= 254 && p[1] == 0 && p[2] == 0 && p[3] >= 254,
            "placeholder leaked into resident: {p:?}"
        );
    }
    let skip_counters = engine::counter_snapshot();
    assert_eq!(skip_counters.compute_storage_seed_uploads, 0);
    assert_eq!(skip_counters.compute_storage_seed_upload_bytes, 0);

    // Lost resident: evicting guest state must turn the same skip request
    // into the named failure, never a silent zero seed.
    engine::reset_guest_state();
    let err = engine::execute_compute_request(&make([1, 1, 1], 3, 4, true))
        .expect_err("lost resident must fail");
    assert!(
        err.to_string()
            .contains("vk_compute_exec_resident_seed_generation_lost"),
        "unexpected error: {err}"
    );
}


/// Copy-on-sample contract: a sampled input bound to a generation-matching
/// resident storage image is seeded by a device-local copy (zero-placeholder
/// `bytes` never uploaded, `compute_sampled_uploads == 0`) and fetches the
/// resident content; a stale generation or evicted resident fails with
/// `vk_compute_exec_resident_sample_*` — never a silent zero seed.
#[test]
fn compute_sampled_resident_copy_and_lost_resident() {
    let _g = engine_test_session();
    // Same red-fill kernel as compute_storage_image_seed_skip_and_lost_resident.
    let fill_spvasm = &storage_image_write_red_kernel("Rgba8");
    // Same fetch-to-buffer kernel shape as
    // compute_sampled_image_fetch_preserves_float_bits (binding 32 sampled,
    // binding 0 storage buffer, fetches texel (0,0)).
    let fetch_spvasm = SAMPLED_IMAGE_FETCH_KERNEL;
    let Some(fill_words) = assemble_spvasm(fill_spvasm, "resident_sample_fill") else {
        return;
    };
    let Some(fetch_words) = assemble_spvasm(fetch_spvasm, "resident_sample_fetch") else {
        return;
    };
    let w = 4u32;
    let h = 4u32;
    let identity = ComputeStorageResidencyKey {
        mapping_id: 93,
        map_generation: 1,
        surface_offset: 0,
        surface_bpr: w * 4,
        span_end: (w * h * 4) as u64,
        width: w,
        height: h,
        pixel_format: 0x46,
        texture_ref: 0,
    };
    // Full red fill establishes the resident at generation 2.
    let fill_req = ComputeRequest {
        spirv: fill_words,
        entry: "main".into(),
        grid: [w, h, 1],
        storage_buffers: vec![],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![ComputeStorageImageResource {
            binding: 0,
            format: StorageImageFormat::Rgba8Unorm,
            width: w,
            height: h,
            bytes: vec![0u8; (w * h * 4) as usize],
            residency: Some(ComputeStorageResidency {
                identity,
                seed_generation: 1,
                output_generation: 2,
            }),
            seed_skipped: false,
        }],
    };
    let Some(fill) = engine_or_skip("resident_sample_fill", &fill_req) else {
        return;
    };
    assert!(fill.images[0].chunks_exact(4).all(|p| p[0] >= 254));

    let make_fetch = |generation: u32| ComputeRequest {
        spirv: fetch_words.clone(),
        entry: "main".into(),
        grid: [1, 1, 1],
        storage_buffers: vec![ComputeBufferResource {
            binding: 0,
            bytes: vec![0; 16],
            writable: true,
        }],
        sampled_images: vec![ComputeSampledImageResource {
            binding: 32,
            format: StorageImageFormat::Rgba8Unorm,
            width: w,
            height: h,
            bytes: vec![0u8; (w * h * 4) as usize],
            resident_bind: Some(ComputeResidentSampleBind {
                identity,
                generation,
            }),
        }],
        samplers: vec![],
        storage_images: vec![],
    };

    // Resident hit: the sampled image is seeded device-locally from the red
    // resident; the zero placeholder never uploads.
    engine::reset_draw_counters();
    let hit = engine::execute_compute_request(&make_fetch(2)).expect("resident sample hit");
    let got: Vec<u32> = hit.buffers[0]
        .bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        got,
        vec![
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits()
        ],
        "sampled texel must be the resident red, not the zero placeholder"
    );
    let snap = engine::counter_snapshot();
    assert_eq!(snap.compute_sampled_uploads, 0);
    assert_eq!(snap.compute_sampled_upload_bytes, 0);
    assert_eq!(snap.compute_sampled_resident_copies, 1);
    assert_eq!(snap.compute_sampled_resident_copy_bytes, (w * h * 4) as u64);

    // Stale generation must fail visibly, never seed the placeholder.
    let err =
        engine::execute_compute_request(&make_fetch(9)).expect_err("stale generation must fail");
    assert!(
        err.to_string()
            .contains("vk_compute_exec_resident_sample_generation_mismatch"),
        "unexpected error: {err}"
    );

    // Evicted resident must fail visibly too.
    engine::reset_guest_state();
    let err = engine::execute_compute_request(&make_fetch(2)).expect_err("lost resident must fail");
    assert!(
        err.to_string()
            .contains("vk_compute_exec_resident_sample_absent"),
        "unexpected error: {err}"
    );
}


/// Sampled inputs must use SAMPLED_IMAGE descriptors and remain input-only.
#[test]
fn compute_sampled_image_fetch_preserves_float_bits() {
    let _g = engine_test_session();
    let spvasm = SAMPLED_IMAGE_FETCH_KERNEL;
    let Some(words) = assemble_spvasm(spvasm, "sampled_image") else {
        return;
    };
    let values = [0.0f32, 0.5, 1.0, 2.0];
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let mut req = ComputeRequest {
        spirv: words,
        entry: "main".into(),
        grid: [1, 1, 1],
        storage_buffers: vec![ComputeBufferResource {
            binding: 0,
            bytes: vec![0; 16],
            writable: true,
        }],
        sampled_images: vec![ComputeSampledImageResource {
            binding: 32,
            format: StorageImageFormat::Rgba32Float,
            width: 1,
            height: 1,
            bytes,
            resident_bind: None,
        }],
        samplers: vec![],
        storage_images: vec![],
    };
    let Some(out) = engine_or_skip("compute sampled image", &req) else {
        return;
    };
    let got: Vec<u32> = out.buffers[0]
        .bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let want: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
    assert_eq!(got, want);
    assert!(out.images.is_empty(), "sampled input must not be read back");
    let snap = engine::counter_snapshot();
    assert_eq!(snap.compute_sampled_uploads, 1);
    assert_eq!(snap.compute_sampled_upload_bytes, 16);
    assert_eq!(snap.compute_storage_seed_uploads, 0);

    // RGB9E5 has no writable-storage selector in the guest ABI, but it is a
    // valid sampled texture. Zero packed RGB decodes to (0, 0, 0, 1).
    req.sampled_images[0].format = StorageImageFormat::Rgb9e5Ufloat;
    req.sampled_images[0].bytes = vec![0; 4];
    let Some(out) = engine_or_skip("compute sampled RGB9E5 image", &req) else {
        return;
    };
    let got: Vec<u32> = out.buffers[0]
        .bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got, vec![0.0f32.to_bits(), 0, 0, 1.0f32.to_bits()]);

    // Integer sampled views need an integer OpTypeImage and R32_UINT backing.
    let uint_spvasm = spvasm
        .replace("%Image = OpTypeImage %float", "%Image = OpTypeImage %uint")
        .replace(
            "%texel = OpImageFetch %v4float",
            "%texel = OpImageFetch %v4uint",
        )
        .replace(
            "%bits = OpBitcast %v4uint %texel",
            "%bits = OpCopyObject %v4uint %texel",
        );
    let Some(words) = assemble_spvasm(&uint_spvasm, "sampled_r32uint") else {
        return;
    };
    req.spirv = words;
    req.sampled_images[0].format = StorageImageFormat::R32Uint;
    req.sampled_images[0].bytes = 0x1234_5678u32.to_le_bytes().to_vec();
    let Some(out) = engine_or_skip("compute sampled R32Uint image", &req) else {
        return;
    };
    let got: Vec<u32> = out.buffers[0]
        .bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got, vec![0x1234_5678, 0, 0, 1]);
}

/// m2v Kernel fixture (float_mul4_add3): known CPU formula x*4+3.
#[test]
fn compute_m2v_float_mul4_add3_known_result() {
    let _g = engine_test_session();
    let Some(words) = translate_kernel("float_mul4_add3") else {
        return;
    };
    // Same shape as metal2vulkan/tests/compute.rs float_mul4_add3_runs.
    let inp: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    let want: Vec<f32> = vec![7., 11., 15., 19., 23., 27., 31., 35.];
    let input: Vec<u8> = inp.iter().flat_map(|x| x.to_le_bytes()).collect();
    let grid = 1u32; // LocalSize 64, one group covers ≤64
    let req = ComputeRequest {
        spirv: words,
        entry: "main".into(),
        grid: [grid, 1, 1],
        storage_buffers: vec![ComputeBufferResource {
            binding: 0,
            bytes: input,
            writable: true,
        }],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![],
    };
    let Some(out) = engine_or_skip("m2v_float_mul4_add3", &req) else {
        return;
    };
    let got: Vec<f32> = out.buffers[0]
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(&got[..want.len()], want.as_slice());
}

/// Warm identical dispatch: zero creates and zero allocs.
#[test]
fn warm_identical_dispatch_zero_creates_and_allocs() {
    let _g = engine_test_session();
    let Some(words) = inc_comp_spirv() else {
        return;
    };
    let n = 64usize;
    let input: Vec<u8> = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    let make_req = || ComputeRequest {
        spirv: words.clone(),
        entry: "main".into(),
        grid: [1, 1, 1],
        storage_buffers: vec![ComputeBufferResource {
            binding: 0,
            bytes: input.clone(),
            writable: true,
        }],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![],
    };
    // Cold
    let Some(_) = engine_or_skip("warm_dispatch cold", &make_req()) else {
        return;
    };
    engine::reset_draw_counters();
    // Warm
    let Some(_) = engine_or_skip("warm_dispatch warm", &make_req()) else {
        return;
    };
    let snap = engine::counter_snapshot();
    assert_eq!(
        snap.creates, 0,
        "warm dispatch must perform zero vkCreate* (got {})",
        snap.creates
    );
    assert_eq!(
        snap.allocs, 0,
        "warm dispatch must perform zero vkAllocateMemory (got {})",
        snap.allocs
    );
    assert!(
        snap.dispatches >= 1,
        "dispatch counter should advance on warm path"
    );
}

/// 16-bit storage shape: R16Float storage image (capability-gated on device).
#[test]
fn compute_storage_image_r16float_if_supported() {
    let _g = engine_test_session();
    // imageStore half red into R16Float 2x2.
    let spvasm = r#"
               OpCapability Shader
               OpCapability StorageImageWriteWithoutFormat
               OpCapability StorageImageExtendedFormats
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid %img
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %img DescriptorSet 0
               OpDecorate %img Binding 0
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
     %v3uint = OpTypeVector %uint 3
      %v2int = OpTypeVector %int 2
    %v4float = OpTypeVector %float 4
    %float_1 = OpConstant %float 1
    %float_0 = OpConstant %float 0
      %red = OpConstantComposite %v4float %float_1 %float_0 %float_0 %float_1
%img_ty = OpTypeImage %float 2D 0 0 0 2 R16f
%_ptr_img = OpTypePointer UniformConstant %img_ty
        %img = OpVariable %_ptr_img UniformConstant
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
        %gid = OpVariable %_ptr_Input_v3uint Input
    %fn_type = OpTypeFunction %void
       %main = OpFunction %void None %fn_type
      %entry = OpLabel
    %gid_val = OpLoad %v3uint %gid
          %x = OpCompositeExtract %uint %gid_val 0
          %y = OpCompositeExtract %uint %gid_val 1
         %xi = OpBitcast %int %x
         %yi = OpBitcast %int %y
       %coord = OpCompositeConstruct %v2int %xi %yi
      %img_l = OpLoad %img_ty %img
               OpImageWrite %img_l %coord %red
               OpReturn
               OpFunctionEnd
"#;
    let Some(words) = assemble_spvasm(spvasm, "simg_r16f") else {
        return;
    };
    let seed = vec![0u8; 2 * 2 * 2]; // R16 = 2 bytes/texel
    let req = ComputeRequest {
        spirv: words,
        entry: "main".into(),
        grid: [2, 2, 1],
        storage_buffers: vec![],
        sampled_images: vec![],
        samplers: vec![],
        storage_images: vec![ComputeStorageImageResource {
            binding: 0,
            format: StorageImageFormat::R16Float,
            width: 2,
            height: 2,
            bytes: seed,
            residency: None,
            seed_skipped: false,
        }],
    };
    match engine::execute_compute_request(&req) {
        Ok(out) => {
            assert_eq!(out.images.len(), 1);
            assert_eq!(out.images[0].len(), 8);
        }
        Err(e) => {
            let s = e.to_string();
            if skip_if_no_gpu(&s) || s.contains("unsupported") || s.contains("create_") {
                eprintln!("SKIP r16f (device/format): {s}");
            } else {
                panic!("r16f: {s}");
            }
        }
    }
}
