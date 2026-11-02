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
CHECK(sizeof(RgfxSamplerState) == 32);
CHECK(offsetof(RgfxSamplerState, status) == 16);
CHECK(offsetof(RgfxSamplerState, gpu_resource_id) == 24);
CHECK(RGFX_STATUS_TOO_SMALL == 6);
CHECK(RGFX_ARGUMENT_STRUCT == 11);
CHECK(RGFX_CONTROL_RING_ID == 0);
CHECK(RGFX_CONTROL_VECTOR == 0);
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
