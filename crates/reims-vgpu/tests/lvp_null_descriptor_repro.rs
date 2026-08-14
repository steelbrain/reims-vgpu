//! Repro: a fragment shader that samples a texture whose descriptor binding is
//! absent from the request (and therefore from the descriptor set layout).
//!
//! `textured_quad.air` declares set 0 binding 32 (sampled image) and binding 64
//! (sampler). The engine derives both the descriptor set layout and the
//! descriptor writes from the DrawRequest, so a request that omits the sampled
//! image produces a pipeline whose shader reads a descriptor the set never
//! declared or wrote.

#![cfg(feature = "backend-vulkan")]

use metal2vulkan::passes::Stage;
use reims_vgpu::backend::vulkan::engine::{
    self, DrawRequest, PrimitiveTopology, SampledImageResource, SampledSource, SamplerResource,
    StorageBufferResource,
};
use std::path::PathBuf;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/air")
}

fn translate_words(name: &str, stage: Stage) -> Vec<u32> {
    let tmp = std::env::temp_dir().join(format!(
        "lvp_repro_{}_{}_{:?}",
        std::process::id(),
        name,
        stage
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("tmp");
    let path = fixtures().join(name);
    let spv = metal2vulkan::translate(path.to_str().unwrap(), stage, &tmp)
        .unwrap_or_else(|e| panic!("translate {name}: {e}"));
    spv.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Full-screen quad + UVs in storage buffers 0/1, which is what the vertex half
/// of `textured_quad.air` fetches.
fn base_request(with_texture: bool, with_sampler: bool) -> DrawRequest {
    base_request_px(with_texture, with_sampler, [17u8, 140, 203, 255], 2)
}

fn base_request_px(with_texture: bool, with_sampler: bool, px: [u8; 4], dim: u32) -> DrawRequest {
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
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
        vert_spirv: std::sync::Arc::new(vert),
        frag_spirv: std::sync::Arc::new(frag),
        width: 64,
        height: 64,
        vertex_count: 6,
        first_vertex: 0,
        instance_count: Some(1),
        base_instance: 0,
        primitive_topology: PrimitiveTopology::Triangle,
        // Default `target_clear` is `[0.0; 4]` and no seed/load-from-target is
        // set, so the pass clears to transparent black — the same load action
        // this request spelled explicitly before `LoadOp` was retired in favor
        // of `target_clear` + the load-source fields it now shares the struct
        // with (see `DrawRequest::target_clear`'s doc).
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
    if with_texture {
        req.sampled_images.push(SampledImageResource {
            binding: 32,
            width: dim,
            height: dim,
            layers: 1,
            arrayed: false,
            volume: false,
            cube: false,
            one_dim: false,
            source: SampledSource::Bytes(std::sync::Arc::new(px.repeat((dim * dim) as usize))),
            format: ash::vk::Format::R8G8B8A8_UNORM,
            identity: None,
            swizzle: Default::default(),
        });
    }
    if with_sampler {
        req.samplers.push(SamplerResource::normalized_default(64));
    }
    req
}

fn run(label: &str, req: &DrawRequest) {
    eprintln!("[{label}] executing draw");
    match engine::execute_draw_request(req) {
        Ok(out) => eprintln!(
            "[{label}] OK first_pixel={:?} bgra={}",
            &out.pixels[..4],
            out.pixels_bgra
        ),
        Err(e) => eprintln!("[{label}] declined: {e}"),
    }
    eprintln!("[{label}] survived");
}

/// Control: every binding the shader reads is present in the request.
#[test]
fn texture_and_sampler_present() {
    run("control", &base_request(true, true));
}

/// Repro: the sampler is bound, the sampled image is not.
#[test]
fn texture_missing_sampler_present() {
    run("no-texture", &base_request(false, true));
}

/// Repro: neither the sampled image nor the sampler is bound.
#[test]
fn texture_and_sampler_missing() {
    run("neither", &base_request(false, false));
}

/// The proposed fix: the guest bound nothing, but a 1x1 opaque-black placeholder
/// is synthesized at the reflected binding so every descriptor the shader reads
/// is declared and written.
#[test]
fn placeholder_texture_substituted() {
    run(
        "placeholder",
        &base_request_px(true, true, [0u8, 0, 0, 255], 1),
    );
}

/// Transparent-black placeholder: Metal reads an unbound texture as zero, and
/// this is what NVIDIA already returns for the unbound case.
#[test]
fn placeholder_transparent_black() {
    run("zero-placeholder", &base_request_px(true, true, [0u8, 0, 0, 0], 1));
}
