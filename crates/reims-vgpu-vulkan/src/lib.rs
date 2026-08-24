//! Vulkan implementation policy for Reims vGPU.
//!
//! This crate is the only layer allowed to interpret host Vulkan capabilities
//! as placement and transfer choices. Guest-visible resource lifetime and
//! content authority remain in `reims-vgpu-core`.

pub mod api_floor;
pub mod capabilities;
pub mod device_features;
pub mod device_select;
pub mod engine;
pub mod format;
pub mod gpu_hang_trail;
pub mod host_pointer;
pub mod m2v_cache;
pub mod memory;
pub mod policy;
pub mod preparation;
pub mod push_descriptor;
pub mod spirv_bind;
pub mod spirv_vertex_input;
pub mod srgb_census;
pub mod telemetry;
pub mod translate;
