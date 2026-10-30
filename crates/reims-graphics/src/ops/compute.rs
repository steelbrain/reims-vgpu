//! The compute-encoder ops (`tag >> 12 == `[`RGFX_OP_GROUP_COMPUTE`](super::RGFX_OP_GROUP_COMPUTE)):
//! one tag and one body struct per `MTLComputeCommandEncoder` selector the guest
//! serialises (docs/graphics-kext/design.md, "Stage 3 wire protocol").
//!
//! A body struct is the op's bytes **after** its [`RgfxOpHeader`](super::RgfxOpHeader),
//! not the whole op (see the [module doc](super)). Object references are `u64`
//! `obj_id` (0 = nil).
//!
//! This module is a stub: the compute ops land here, one tag and one body struct
//! each, their tags in the `0x2xxx` range. The host's compute decoder
//! (`reims-vgpu`'s `graphics::decode::compute`) already routes to them.

// The compute ops land here; the group's base is RGFX_OP_GROUP_COMPUTE << 12 (0x2000).
