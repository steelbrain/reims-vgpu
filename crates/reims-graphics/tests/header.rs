//! The committed wire header, `include/reims_graphics_wire.h`, as the guest's kext and option
//! ROM compile it: as C11 and as C++17, with no warning, its records laid out as `wire`'s.

// A failure is the assertion (architecture 17.4).
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;

/// What the guest's code checks of the header: each record's size and fields' offsets, as the
/// design's tables give them.
const CHECKS: &str = r#"
#include "reims_graphics_wire.h"
#include <stddef.h>
#ifdef __cplusplus
#define CHECK(e) static_assert(e, #e)
#else
#define CHECK(e) _Static_assert(e, #e)
#endif
CHECK(sizeof(RgfxRingHeader) == RGFX_RING_HEADER_SIZE);
CHECK(offsetof(RgfxRingHeader, producer) == 0x00);
CHECK(offsetof(RgfxRingHeader, completed_seqno) == 0x08);
CHECK(offsetof(RgfxRingHeader, consumer) == 0x40);
CHECK(offsetof(RgfxRingHeader, consumer_idle) == 0x80);
CHECK(offsetof(RgfxRingHeader, armed) == 0xc0);
CHECK(sizeof(RgfxMsgHeader) == 16);
CHECK(offsetof(RgfxMsgHeader, size) == 4);
CHECK(offsetof(RgfxMsgHeader, seqno) == 8);
CHECK(sizeof(RgfxSetScanout) == 40);
CHECK(offsetof(RgfxSetScanout, offset) == 32);
CHECK(sizeof(RgfxSetCursor) == 48);
CHECK(offsetof(RgfxSetCursor, offset) == 40);
CHECK(sizeof(RgfxMoveCursor) == 32);
CHECK(sizeof(RgfxSetVblank) == 24);
CHECK(sizeof(RgfxComplete) == 24);
CHECK(offsetof(RgfxDisplays, count) + 4 == RGFX_DISPLAYS_ENTRIES_OFFSET);
CHECK(sizeof(RgfxDisplayEntry) == 16);
CHECK(sizeof(RgfxModeEntry) == 12);
CHECK(sizeof(RgfxDisplayChanged) == 24);
CHECK(sizeof(RgfxVblank) == 40);
CHECK(offsetof(RgfxVblank, count) == 24);
CHECK(RGFX_MAGIC == 0x58464752);
CHECK(RGFX_CONTROL_SUBMIT_OFFSET == 0x00000);
CHECK(RGFX_CONTROL_COMPLETE_OFFSET == 0x10000);
CHECK(RGFX_ALLOCATIONS_OFFSET == 0x400000);
CHECK(RGFX_SCANOUT_SPAN == 0x8000000);
CHECK(RGFX_CURSOR_SPAN == 0x40000);
CHECK(RGFX_CONTEXT_ALLOCATIONS_OFFSET == 0x8400000);
CHECK(RGFX_CONTEXT_OBJ_ID_SHIFT == 40);
CHECK(sizeof(RgfxContextClose) == 24);
CHECK(RGFX_CMD_CONTEXT_CLOSE == 0x2f0);
CHECK(sizeof(RgfxLibraryCreate) == RGFX_LIBRARY_BYTES_OFFSET);
CHECK(offsetof(RgfxLibraryCreate, key) == 0x28);
CHECK(sizeof(RgfxLibraryFromRegion) == 112);
CHECK(offsetof(RgfxLibraryFromRegion, key) == 0x50);
CHECK(sizeof(RgfxLibraryLookup) == 64);
CHECK(offsetof(RgfxLibraryLookup, key) == 0x20);
CHECK(RGFX_CMD_LIBRARY_LOOKUP == 0x213);
CHECK(sizeof(RgfxResourceIdChunk) == 24);
CHECK(RGFX_CMD_RESOURCE_ID_CHUNK == 0x2f1);
CHECK(sizeof(RgfxCommandBufferInRegion) == 32);
CHECK(offsetof(RgfxCommandBufferInRegion, byte_length) == 0x18);
CHECK(RGFX_CMD_COMMAND_BUFFER_IN_REGION == 0x301);
CHECK(RGFX_RESOURCE_ID_LIMIT == 16381);
CHECK(RGFX_RESOURCE_ID_CHUNK == 64);
CHECK(RGFX_FEATURE_SMALL_ID_CHUNKS == (1 << 23));
CHECK(RGFX_CMD_RESOURCE_ID_SMALL_CHUNK == 0x2f8);
CHECK(RGFX_RESOURCE_ID_SMALL_CHUNK == 16);
CHECK(RGFX_RESOURCE_ID_SMALL_CHUNKS == 1020);
CHECK(RGFX_RESOURCE_ID_CHUNKS == 255);
CHECK(sizeof(RgfxTextureFromIOSurface) == RGFX_SURFACE_FACTS_OFFSET);
CHECK(sizeof(RgfxSurfaceFacts) == 40);
CHECK(offsetof(RgfxSurfaceFacts, chroma_offset) == 0x10);
CHECK(offsetof(RgfxSurfaceFacts, ycbcr_matrix) == 0x20);
CHECK(RGFX_YCBCR_MATRIX_OTHER == 7);
CHECK(sizeof(RgfxTextureView) == 72);
CHECK(offsetof(RgfxTextureView, pixel_format) == 0x28);
CHECK(offsetof(RgfxTextureView, swizzle) == 0x44);
CHECK(RGFX_CMD_TEXTURE_VIEW == 0x217);
CHECK(RGFX_TEXTURE_VIEW_SUBRANGE == 1);
CHECK(RGFX_TEXTURE_VIEW_SWIZZLE == 2);
CHECK(sizeof(RgfxArgumentLayoutQuery) == 80);
CHECK(offsetof(RgfxArgumentLayoutQuery, reply_offset) == 0x18);
CHECK(offsetof(RgfxArgumentLayoutQuery, source) == 0x30);
CHECK(offsetof(RgfxArgumentLayoutQuery, constant_count) == 0x4c);
CHECK(sizeof(RgfxArgumentDescriptor) == 24);
CHECK(sizeof(RgfxArgumentLayoutReply) == 16);
CHECK(sizeof(RgfxArgumentLayoutRecord) == 24);
CHECK(sizeof(RgfxArgumentMemberRecord) == 48);
CHECK(offsetof(RgfxArgumentMemberRecord, layout) == 0x28);
CHECK(RGFX_CMD_ARGUMENT_LAYOUT == 0x219);
CHECK(RGFX_EVT_SAMPLER_STATE == 0x404);
CHECK(RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION_OPTIONS == 0x104f);
CHECK(sizeof(RgfxOpSetDepthStoreActionOptions) == 8);
CHECK(RGFX_PASS_FLAG_SAMPLE_POSITIONS == 8);
CHECK(sizeof(RgfxRenderPassSamplePositions) == 136);
CHECK(offsetof(RgfxRenderPassSamplePositions, position_bits) == 8);
CHECK(RGFX_PASS_FLAG_VISIBILITY_RESULT == 16);
CHECK(sizeof(RgfxRenderPassVisibility) == 16);
CHECK(offsetof(RgfxRenderPassVisibility, result_type) == 8);
CHECK(sizeof(RgfxSamplerState) == 32);
CHECK(sizeof(RgfxCbScheduled) == 24);
CHECK(offsetof(RgfxCbScheduled, cb_obj_id) == 16);
CHECK(sizeof(RgfxCbSubmitted) == 24);
CHECK(offsetof(RgfxCbSubmitted, cb_obj_id) == 16);
CHECK(RGFX_EVT_CB_SUBMITTED == 0x406);
CHECK(RGFX_FEATURE_CB_SUBMITTED == 4);
CHECK(RGFX_OP_SUBMITTED == 8);
CHECK(RGFX_FEATURE_SURFACE_PAGES == 1);
CHECK(RGFX_CMD_SURFACE_PAGES == 0x2f6);
CHECK(RGFX_CMD_BUFFER_OVER_SURFACE_PAGES == 0x21c);
CHECK(sizeof(RgfxPageRun) == 16);
CHECK(sizeof(RgfxSurfacePages) == 56);
CHECK(offsetof(RgfxSurfacePages, run_count) == 0x14);
CHECK(offsetof(RgfxSurfacePages, runs_offset) == 0x28);
CHECK(offsetof(RgfxSurfacePages, length) == 0x30);
CHECK(RGFX_SURFACE_PAGES_CACHE_OFFSET == sizeof(RgfxSurfacePages));
CHECK(sizeof(RgfxSurfacePagesCache) == 8);
CHECK(RGFX_CMD_SURFACE_DOWNLOAD == 0x2f7);
CHECK(sizeof(RgfxSurfaceDownload) == 24);
CHECK(offsetof(RgfxSurfaceDownload, block_offset) == 16);
CHECK(RGFX_OP_SURFACE_USE == 7);
CHECK(sizeof(RgfxOpSurfaceUse) == 24);
CHECK(offsetof(RgfxOpSurfaceUse, cpu_gen) == 8);
CHECK(offsetof(RgfxOpSurfaceUse, flags) == 16);
CHECK(RGFX_PRESENT_SURFACE_GEN_OFFSET == sizeof(RgfxPresentSurface));
CHECK(sizeof(RgfxPresentSurfaceGen) == 8);
CHECK(sizeof(RgfxBufferOverSurfacePages) == 56);
CHECK(offsetof(RgfxBufferOverSurfacePages, length) == 0x28);
CHECK(offsetof(RgfxBufferOverSurfacePages, options) == 0x30);
CHECK(offsetof(RgfxSamplerState, status) == 16);
CHECK(offsetof(RgfxSamplerState, gpu_resource_id) == 24);
CHECK(RGFX_STATUS_TOO_SMALL == 6);
CHECK(RGFX_ARGUMENT_STRUCT == 11);
CHECK(RGFX_CONTROL_RING_ID == 0);
CHECK(RGFX_CONTROL_VECTOR == 0);
CHECK(RGFX_REG_DRIVER_FEATURES_LO == 0x024);
CHECK(RGFX_REG_DRIVER_FEATURES_HI == 0x028);
CHECK(RGFX_REG_CONSUMED == 0x02c);
CHECK(RGFX_FEATURE_QUIET_SCHEDULED == (1 << 3));
CHECK(RGFX_FEATURE_QUEUE_VECTORS == (1 << 8));
CHECK(offsetof(RgfxRingHeader, producer_waiting) == 0x10);
CHECK(offsetof(RgfxRingHeader, raise_all) == 0xc4);
CHECK(RGFX_QUEUE_CREATE_FLAGS_OFFSET == sizeof(RgfxQueueCreate));
CHECK(sizeof(RgfxQueueCreate) + sizeof(RgfxQueueCreateFlags) == 64);
CHECK(RGFX_FEATURE_UNSHARED_QUEUES == (1 << 13));
CHECK(RGFX_QUEUE_UNSHARED == (1 << 3));
CHECK(RGFX_OP_QUEUE_BARRIER == 0x000a);
CHECK(sizeof(RgfxOpHeader) + sizeof(RgfxOpQueueBarrier) == 24);
CHECK(offsetof(RgfxOpQueueBarrier, before_stages) == 8);
CHECK(RGFX_OP_SCHEDULED_WANTED == 0x000b);
CHECK(RGFX_OP_SURFACE_ORDER == 0x000c);
CHECK(sizeof(RgfxSurfaceOrderWait) == 16);
CHECK(offsetof(RgfxSurfaceOrderWait, seqno) == 8);
CHECK(RGFX_OP_COMMIT_ORDER == 0x000d);
CHECK(RGFX_FEATURE_COMMIT_ORDER == (1 << 18));
CHECK(RGFX_FEATURE_HOST_TIME == (1 << 20));
CHECK(RGFX_REG_HOST_TIME_LO == 0x038);
CHECK(RGFX_REG_HOST_TIME_HI == 0x03c);
CHECK(RGFX_FEATURE_MANAGED_TEXTURE_SLABS == (1 << 21));
CHECK(RGFX_FEATURE_MANAGED_BUFFER_SLABS == (1 << 22));
CHECK(RGFX_CMD_MANAGED_BUFFER_CREATE == 0x22d);
CHECK(sizeof(RgfxManagedBufferCreate) == 56);
CHECK(offsetof(RgfxManagedBufferCreate, gpu_address) == 24);
CHECK(offsetof(RgfxManagedBufferCreate, cpu_buffer_obj_id) == 32);
CHECK(offsetof(RgfxManagedBufferCreate, length) == 40);
CHECK(offsetof(RgfxManagedBufferCreate, options) == 48);
CHECK(RGFX_FEATURE_DISPLAY_COLOR == (1 << 24));
CHECK(RGFX_FEATURE_MESH_PIPELINES == (1 << 25));
CHECK(RGFX_CMD_MESH_RENDER_PIPELINE_CREATE == 0x22e);
CHECK(sizeof(RgfxMeshRenderPipelineCreate) == 512);
CHECK(offsetof(RgfxMeshRenderPipelineCreate, colors) == 192);
CHECK(RGFX_PIPELINE_KIND_MESH == 2);
CHECK(RGFX_PIPELINE_REFLECTION_MESH_OFFSET == sizeof(RgfxPipelineReflection));
CHECK(sizeof(RgfxPipelineReflection) + sizeof(RgfxMeshPipelineReflection) == 104);
CHECK(RGFX_OP_RENDER_SET_OBJECT_BUFFERS == 0x1051);
CHECK(RGFX_OP_RENDER_DRAW_MESH_THREADGROUPS_INDIRECT == 0x105e);
CHECK(sizeof(RgfxOpDrawMeshThreadgroups) == 72);
CHECK(sizeof(RgfxOpDrawMeshThreadgroupsIndirect) == 64);
CHECK(RGFX_CMD_DISPLAY_COLOR == 7);
CHECK(sizeof(RgfxDisplayColor) == 40);
CHECK(sizeof(RgfxColorEntry) == 16);
CHECK(offsetof(RgfxPresentSurface, flags) == 0x1c);
CHECK(RGFX_PRESENT_KEEP == (1 << 0));
CHECK(RGFX_FEATURE_SURFACE_ORDER == (1 << 11));
CHECK(RGFX_FEATURE_SUBMIT_ROOM == (1 << 16));
CHECK(RGFX_EVT_SUBMIT_ROOM == 0x408);
CHECK(RGFX_FEATURE_SHARED_EVENT_NAMES == (1 << 26));
CHECK(RGFX_FEATURE_OPENCL_MODE == (1 << 27));
CHECK(RGFX_FEATURE_PRIVATE_RENDER_STATE == (1 << 28));
CHECK(RGFX_RENDER_PIPELINE_PRIVATE_STATE == (1 << 6));
CHECK(sizeof(RgfxRenderPipelinePrivateState) == 24);
CHECK(offsetof(RgfxRenderPipelinePrivateState, sample_coverage) == 8);
CHECK(offsetof(RgfxRenderPipelinePrivateState, logic_operation) == 12);
CHECK(offsetof(RgfxRenderPipelinePrivateState, flags) == 16);
CHECK(sizeof(RgfxRenderPipelineCreate) % 8 == 0);
CHECK(RGFX_CMD_SHARED_EVENT_NAME == 0x22f);
CHECK(RGFX_CMD_SHARED_EVENT_IMPORT == 0x230);
CHECK(RGFX_CMD_SHARED_EVENT_UNNAME == 0x231);
CHECK(RGFX_EVT_SHARED_EVENT_VALUE == 0x409);
CHECK(sizeof(RgfxSharedEventName) == 40);
CHECK(offsetof(RgfxSharedEventName, obj_id) == 16);
CHECK(offsetof(RgfxSharedEventName, name) == 24);
CHECK(offsetof(RgfxSharedEventName, listen_queue_obj_id) == 32);
CHECK(sizeof(RgfxSharedEventImport) == 40);
CHECK(offsetof(RgfxSharedEventImport, name) == 24);
CHECK(offsetof(RgfxSharedEventImport, listen_queue_obj_id) == 32);
CHECK(sizeof(RgfxSharedEventValue) == 32);
CHECK(offsetof(RgfxSharedEventValue, status) == 16);
CHECK(offsetof(RgfxSharedEventValue, value) == 24);
CHECK(sizeof(RgfxSharedEventUnname) == 32);
CHECK(offsetof(RgfxSharedEventUnname, name) == 24);
CHECK(RGFX_CMD_BUFFER_OVER_PAGES == 0x21e);
CHECK(RGFX_FEATURE_MANAGED_AT_OFFSET == (1 << 14));
CHECK(RGFX_CMD_BUFFER_MANAGED_AT_OFFSET == 0x22b);
CHECK(sizeof(RgfxBufferManagedAtOffset) == 64);
CHECK(offsetof(RgfxBufferManagedAtOffset, gpu_address) == 24);
CHECK(offsetof(RgfxBufferManagedAtOffset, heap_obj_id) == 32);
CHECK(offsetof(RgfxBufferManagedAtOffset, offset) == 40);
CHECK(offsetof(RgfxBufferManagedAtOffset, length) == 48);
CHECK(offsetof(RgfxBufferManagedAtOffset, options) == 56);
CHECK(RGFX_FEATURE_HEAP_OVER_PAGES == (1 << 15));
CHECK(RGFX_CMD_HEAP_OVER_PAGES == 0x22a);
CHECK(sizeof(RgfxHeapOverPages) == 64);
CHECK(offsetof(RgfxHeapOverPages, gpu_address) == 24);
CHECK(offsetof(RgfxHeapOverPages, block_offset) == 32);
CHECK(offsetof(RgfxHeapOverPages, block_size) == 40);
CHECK(offsetof(RgfxHeapOverPages, size_bytes) == 48);
CHECK(offsetof(RgfxHeapOverPages, options) == 56);
CHECK(offsetof(RgfxHeapOverPages, heap_type) == 60);
CHECK(sizeof(RgfxBufferOverPages) == 72);
CHECK(offsetof(RgfxBufferOverPages, gpu_address) == 24);
CHECK(offsetof(RgfxBufferOverPages, length) == 56);
CHECK(offsetof(RgfxBufferOverPages, options) == 64);
CHECK(RGFX_SURFACE_PAGES_PRIVATE == (1 << 1));
CHECK(RGFX_CMD_COMMAND_BUFFER_PART == 0x302);
CHECK(sizeof(RgfxCommandBufferPart) == 56);
CHECK(offsetof(RgfxCommandBufferPart, part) == 32);
CHECK(offsetof(RgfxCommandBufferPart, flags) == 36);
CHECK(offsetof(RgfxCommandBufferPart, region_offset) == 40);
CHECK(offsetof(RgfxCommandBufferPart, byte_length) == 48);
CHECK(RGFX_CB_PART_LAST == (1 << 0) && RGFX_CB_PART_ABORT == (1 << 1));
CHECK(RGFX_CB_PARTS_MAX_PARTIAL == 64);
CHECK(RGFX_OP_RENDER_RESET_STATE == 0x1050);
CHECK(RGFX_SAMPLER_CREATE_EXT_OFFSET == sizeof(RgfxSamplerCreate));
CHECK(sizeof(RgfxSamplerCreate) + sizeof(RgfxSamplerCreateExt) == 80);
CHECK(offsetof(RgfxSamplerCreateExt, lod_bias_bits) == 4);
CHECK(offsetof(RgfxDeviceInfo, features_lo) == 0x451);
CHECK(offsetof(RgfxDeviceInfo, sampler_reduction_support) == 0x452);
CHECK(offsetof(RgfxDeviceInfo, linear_row_alignment) == 0x454);
int main(void) { return 0; }
"#;

/// Compiles [`CHECKS`] against the committed header with `cc`, in `language`, with `std`.
fn compile(language: &str, std: &str) {
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let tmp = std::env::temp_dir().join(format!("rgfx-wire-{}-{language}.c", std::process::id()));
    std::fs::write(&tmp, CHECKS).unwrap();
    let out = Command::new("cc")
        .args([
            "-x",
            language,
            std,
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fsyntax-only",
        ])
        .arg("-I")
        .arg(dir.join("include"))
        .arg(&tmp)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert!(
        out.status.success(),
        "{language}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The header compiles as C11, with each record laid out as the design says
#[test]
fn the_wire_header_compiles_as_c11() {
    compile("c", "-std=c11");
}

/// The header compiles as C++17, as the kext includes it
#[test]
fn the_wire_header_compiles_as_cxx17() {
    compile("c++", "-std=c++17");
}

/// Every `#[repr(C)]` record of `wire` and `ops` is in the committed header, but those
/// `cbindgen.toml` excludes: cbindgen writes a record no function names only where its export list
/// names it, and the guest's kext and bundle build their messages from the header
#[test]
fn every_wire_record_is_in_the_header() {
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let read = |path: PathBuf| std::fs::read_to_string(path).unwrap();
    let header = read(dir.join("include/reims_graphics_wire.h"));
    let config = read(dir.join("cbindgen.toml"));
    let excluded: Vec<&str> = config
        .lines()
        .find_map(|l| l.strip_prefix("exclude = ["))
        .unwrap()
        .split('"')
        .skip(1)
        .step_by(2)
        .collect();
    let mut sources = vec![dir.join("src/wire.rs"), dir.join("src/ops.rs")];
    for entry in std::fs::read_dir(dir.join("src/ops")).unwrap() {
        sources.push(entry.unwrap().path());
    }
    let mut missing = Vec::new();
    let mut records = 0;
    for source in sources {
        let text = read(source);
        let lines: Vec<&str> = text.lines().collect();
        for (n, line) in lines.iter().enumerate() {
            let Some(name) = line
                .strip_prefix("pub struct ")
                .and_then(|rest| rest.split([' ', '{', '<']).next())
            else {
                continue;
            };
            // Its attributes: the lines above it back to its doc comment.
            let attributes = lines[..n]
                .iter()
                .rev()
                .take_while(|l| !l.starts_with("///") && !l.trim().is_empty());
            if !attributes.into_iter().any(|l| *l == "#[repr(C)]") {
                continue;
            }
            records += 1;
            if !excluded.contains(&name) && !header.contains(&format!("}} {name};")) {
                missing.push(name.to_owned());
            }
        }
    }
    assert!(records > 200, "only {records} records found");
    assert!(missing.is_empty(), "not in the header: {missing:?}");
}
