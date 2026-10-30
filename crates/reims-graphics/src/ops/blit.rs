//! The blit-encoder ops (`tag >> 12 == `[`RGFX_OP_GROUP_BLIT`](super::RGFX_OP_GROUP_BLIT)):
//! one tag and one body struct per `MTLBlitCommandEncoder` selector the guest
//! serialises (docs/graphics-kext/design.md, "Stage 3 wire protocol").
//!
//! A body struct is the op's bytes **after** its [`RgfxOpHeader`](super::RgfxOpHeader),
//! not the whole op (see the [module doc](super)). Object references are `u64`
//! `obj_id` (0 = nil).
//!
//! This module is a stub: the blit ops land here, one tag and one body struct
//! each, their tags in the `0x3xxx` range. The host's blit decoder
//! (`reims-vgpu`'s `graphics::decode::blit`) already routes to them.

// The blit ops land here; the group's base is RGFX_OP_GROUP_BLIT << 12 (0x3000).
