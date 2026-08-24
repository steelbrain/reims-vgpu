//! Off-VM reflection probe for metal2vulkan failure artifacts: prints the m2v
//! reflection for a dumped AIR blob so declared bindings can be compared
//! against the emitted SPIR-V (`spirv-dis`) and a live draw's binds.
//! No-op (skips green) when `PROBE_AIR` is unset, so it is inert in suite runs.
//! Usage: PROBE_AIR=/path/to/x.air [PROBE_STAGE=vertex] \
//!   cargo test --test reflection_probe -- --test-threads=1 --nocapture

#[test]
fn print_reflection() {
    let Some(path) = std::env::var_os("PROBE_AIR") else {
        eprintln!("PROBE_AIR unset; skipping");
        return;
    };
    let air = std::fs::read(&path).expect("read PROBE_AIR");
    let stage = match std::env::var("PROBE_STAGE").as_deref() {
        Ok("vertex") => reims_vgpu_vulkan::m2v_cache::RenderTranslationStage::Vertex,
        _ => reims_vgpu_vulkan::m2v_cache::RenderTranslationStage::Fragment,
    };
    let shader =
        reims_vgpu_vulkan::m2v_cache::translate_render_cached_reflected(&air, stage, 1, 9999)
            .expect("translate");
    println!(
        "stage={:?} entry bindings={}",
        stage,
        shader.interface.bindings.len()
    );
    for b in &shader.interface.bindings {
        println!("  {b:?}");
    }
    println!(
        "spirv_len={} words={}",
        shader.module_byte_len(),
        shader.words.len()
    );
}
