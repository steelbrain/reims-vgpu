//! The device's own scatter kernel: guest pages written by one dispatch rather
//! than by one transfer region per run.
//!
//! # Why this exists
//!
//! `runtime::render_writeback`'s module doc carries the measurement. In short:
//! the guest backs a surface in 16 KiB physically-contiguous granules, so one
//! writeback is ~200 `VkBufferCopy` regions, and this rail is bound by that
//! count rather than by the bytes in it. Quadrupling the regions for
//! byte-identical output halves the frame rate; `record_us` does not move while
//! `slot_us` nearly triples; and the host GPU sits at 86-91 percent busy on 3-4
//! percent memory utilization throughout. The cost is GPU-side per-region work,
//! so the repair has to remove regions rather than batch them.
//!
//! # The dispatch shape, and why no workgroup size is written down here
//!
//! One workgroup per run, each invocation striding its run in 4-byte words. So
//! `groupCountX` is the run count and nothing outside the shader needs to know
//! its `local_size_x` — the kernel strides by `gl_WorkGroupSize.x`, which it
//! reads from its own declaration. A constant here would be a second spelling
//! of that, and a wrong one would copy a fraction of every run with nothing to
//! report it.
//!
//! # This is our own shader, and it is checked against its source
//!
//! [`GUEST_SCATTER_SPIRV`] is `shaders/guest_scatter.comp` compiled by
//! `glslc -O`. Embedding the words rather than compiling at build time keeps a
//! shader toolchain off every machine that builds this crate; the risk that buys
//! is the two drifting apart, which is what
//! `the_embedded_scatter_spirv_matches_its_source` exists to catch. It skips
//! where `glslc` is absent rather than failing, exactly as the `inc.comp`
//! fixture test does, so a checkout without the compiler reports `ignored`
//! instead of claiming to have checked.

/// `shaders/guest_scatter.comp`, compiled with `glslc -O`.
pub(crate) const GUEST_SCATTER_SPIRV: [u32; 411] = [
    0x07230203, 0x00010000, 0x000d000b, 0x0000005a, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x0000002a, 0x00060010,
    0x00000004, 0x00000011, 0x00000100, 0x00000001, 0x00000001, 0x00040047, 0x0000000b, 0x0000000b,
    0x0000001a, 0x00030047, 0x00000011, 0x00000002, 0x00050048, 0x00000011, 0x00000000, 0x00000023,
    0x00000000, 0x00040047, 0x00000021, 0x00000006, 0x00000010, 0x00030047, 0x00000022, 0x00000003,
    0x00040048, 0x00000022, 0x00000000, 0x00000018, 0x00050048, 0x00000022, 0x00000000, 0x00000023,
    0x00000000, 0x00030047, 0x00000024, 0x00000018, 0x00040047, 0x00000024, 0x00000021, 0x00000002,
    0x00040047, 0x00000024, 0x00000022, 0x00000000, 0x00040047, 0x0000002a, 0x0000000b, 0x0000001b,
    0x00040047, 0x00000037, 0x00000006, 0x00000004, 0x00030047, 0x00000038, 0x00000003, 0x00040048,
    0x00000038, 0x00000000, 0x00000019, 0x00050048, 0x00000038, 0x00000000, 0x00000023, 0x00000000,
    0x00030047, 0x0000003a, 0x00000019, 0x00040047, 0x0000003a, 0x00000021, 0x00000001, 0x00040047,
    0x0000003a, 0x00000022, 0x00000000, 0x00040047, 0x00000040, 0x00000006, 0x00000004, 0x00030047,
    0x00000041, 0x00000003, 0x00040048, 0x00000041, 0x00000000, 0x00000018, 0x00050048, 0x00000041,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000043, 0x00000018, 0x00040047, 0x00000043,
    0x00000021, 0x00000000, 0x00040047, 0x00000043, 0x00000022, 0x00000000, 0x00040047, 0x0000004f,
    0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015,
    0x00000006, 0x00000020, 0x00000000, 0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020,
    0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b,
    0x00000006, 0x0000000c, 0x00000000, 0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x0003001e,
    0x00000011, 0x00000006, 0x00040020, 0x00000012, 0x00000009, 0x00000011, 0x0004003b, 0x00000012,
    0x00000013, 0x00000009, 0x00040015, 0x00000014, 0x00000020, 0x00000001, 0x0004002b, 0x00000014,
    0x00000015, 0x00000000, 0x00040020, 0x00000016, 0x00000009, 0x00000006, 0x00020014, 0x00000019,
    0x00040017, 0x0000001e, 0x00000006, 0x00000004, 0x0003001d, 0x00000021, 0x0000001e, 0x0003001e,
    0x00000022, 0x00000021, 0x00040020, 0x00000023, 0x00000002, 0x00000022, 0x0004003b, 0x00000023,
    0x00000024, 0x00000002, 0x00040020, 0x00000026, 0x00000002, 0x0000001e, 0x0004003b, 0x0000000a,
    0x0000002a, 0x00000001, 0x0003001d, 0x00000037, 0x00000006, 0x0003001e, 0x00000038, 0x00000037,
    0x00040020, 0x00000039, 0x00000002, 0x00000038, 0x0004003b, 0x00000039, 0x0000003a, 0x00000002,
    0x0004002b, 0x00000006, 0x0000003b, 0x00000001, 0x0003001d, 0x00000040, 0x00000006, 0x0003001e,
    0x00000041, 0x00000040, 0x00040020, 0x00000042, 0x00000002, 0x00000041, 0x0004003b, 0x00000042,
    0x00000043, 0x00000002, 0x00040020, 0x00000048, 0x00000002, 0x00000006, 0x0004002b, 0x00000006,
    0x0000004c, 0x00000100, 0x0006002c, 0x00000009, 0x0000004f, 0x0000004c, 0x0000003b, 0x0000003b,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000300f7,
    0x00000050, 0x00000000, 0x000300fb, 0x0000000c, 0x00000051, 0x000200f8, 0x00000051, 0x00050041,
    0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e,
    0x00050041, 0x00000016, 0x00000017, 0x00000013, 0x00000015, 0x0004003d, 0x00000006, 0x00000018,
    0x00000017, 0x000500ae, 0x00000019, 0x0000001a, 0x0000000f, 0x00000018, 0x000300f7, 0x0000001c,
    0x00000000, 0x000400fa, 0x0000001a, 0x0000001b, 0x0000001c, 0x000200f8, 0x0000001b, 0x000200f9,
    0x00000050, 0x000200f8, 0x0000001c, 0x00060041, 0x00000026, 0x00000027, 0x00000024, 0x00000015,
    0x0000000f, 0x0004003d, 0x0000001e, 0x00000028, 0x00000027, 0x00050041, 0x0000000d, 0x0000002b,
    0x0000002a, 0x0000000c, 0x0004003d, 0x00000006, 0x0000002c, 0x0000002b, 0x000200f9, 0x0000002d,
    0x000200f8, 0x0000002d, 0x000700f5, 0x00000006, 0x00000059, 0x0000002c, 0x0000001c, 0x0000004e,
    0x0000002e, 0x00050051, 0x00000006, 0x00000035, 0x00000028, 0x00000002, 0x000500b0, 0x00000019,
    0x00000036, 0x00000059, 0x00000035, 0x000400f6, 0x0000002f, 0x0000002e, 0x00000000, 0x000400fa,
    0x00000036, 0x0000002e, 0x0000002f, 0x000200f8, 0x0000002e, 0x00050051, 0x00000006, 0x0000003d,
    0x00000028, 0x00000001, 0x00050080, 0x00000006, 0x0000003f, 0x0000003d, 0x00000059, 0x00050051,
    0x00000006, 0x00000045, 0x00000028, 0x00000000, 0x00050080, 0x00000006, 0x00000047, 0x00000045,
    0x00000059, 0x00060041, 0x00000048, 0x00000049, 0x00000043, 0x00000015, 0x00000047, 0x0004003d,
    0x00000006, 0x0000004a, 0x00000049, 0x00060041, 0x00000048, 0x0000004b, 0x0000003a, 0x00000015,
    0x0000003f, 0x0003003e, 0x0000004b, 0x0000004a, 0x00050080, 0x00000006, 0x0000004e, 0x00000059,
    0x0000004c, 0x000200f9, 0x0000002d, 0x000200f8, 0x0000002f, 0x000200f9, 0x00000050, 0x000200f8,
    0x00000050, 0x000100fd, 0x00010038,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn compiled() -> Option<Vec<u32>> {
        let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/backend/vulkan/engine/shaders/guest_scatter.comp");
        let out = std::env::temp_dir().join("reims_guest_scatter_check.spv");
        let ok = std::process::Command::new("glslc")
            .args(["-fshader-stage=compute", "-O"])
            .arg(&src)
            .arg("-o")
            .arg(&out)
            .status()
            .ok()
            .is_some_and(|s| s.success());
        if !ok {
            eprintln!("SKIP: no glslc, cannot check the embedded SPIR-V against its source");
            return None;
        }
        let bytes = std::fs::read(&out).ok()?;
        Some(
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    /// The embedded words are a build artifact of a file in this repository and
    /// nothing in the toolchain relates the two, so recompile and compare.
    #[test]
    fn the_embedded_scatter_spirv_matches_its_source() {
        let Some(fresh) = compiled() else { return };
        assert_eq!(
            fresh.as_slice(),
            GUEST_SCATTER_SPIRV.as_slice(),
            "guest_scatter.comp no longer compiles to the embedded words - recompile \
             with `glslc -fshader-stage=compute -O` and re-embed"
        );
    }

    /// A module the driver would reject outright is worth catching without a
    /// device, and the magic is the one field that says this is SPIR-V at all.
    #[test]
    fn the_embedded_scatter_spirv_is_a_spirv_module() {
        assert_eq!(GUEST_SCATTER_SPIRV[0], 0x0723_0203, "SPIR-V magic word");
    }
}
