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
CHECK(RGFX_DEVICE_SUBMIT_OFFSET == 0x20000);
CHECK(RGFX_DEVICE_COMPLETE_OFFSET == 0x30000);
CHECK(RGFX_ALLOCATIONS_OFFSET == 0x40000);
CHECK(RGFX_CONTROL_RING_ID == 0);
CHECK(RGFX_DEVICE_RING_ID == 1);
CHECK(RGFX_CONTROL_VECTOR == 0);
CHECK(RGFX_DEVICE_VECTOR == 1);
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
