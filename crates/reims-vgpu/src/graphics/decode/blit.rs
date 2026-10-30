//! The blit-encoder op decoder: one arm per [`reims_graphics::ops::blit`] op,
//! turning a wire op into a `BlitCmd` on the open `P::BlitEncoder`
//! (docs/graphics-kext/design.md, "Stage 3 wire protocol"). The command-buffer
//! walk ([`super::cmdbuf`]) routes every blit-group op
//! (`tag >> 12 == `[`RGFX_OP_GROUP_BLIT`]) here.
//!
//! This is a stub: it handles no tag yet (returns `Ok(false)` for every one, so
//! the walk faults an unrecognised blit op). The blit ops land here, one arm each,
//! matching [`super::render`]'s pattern.
//!
//! [`RGFX_OP_GROUP_BLIT`]: reims_graphics::ops::RGFX_OP_GROUP_BLIT

use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::raise::{OnFault, RaiseResult};

use super::GraphicsDecoder;

/// Decodes one blit-encoder op: `tag` its op, `payload` its body (the bytes after
/// the [`RgfxOpHeader`](reims_graphics::ops::RgfxOpHeader)). `objs` resolves object
/// references, `enc` is the open blit encoder, and `on` says how an encode's raise
/// is taken. `Ok(true)` when the op was handled, `Ok(false)` for an unknown tag
/// (the walk then faults); a platform raise propagates as `Err`.
pub(crate) fn decode_op<P: Platform>(
    tag: u16,
    payload: &[u8],
    objs: &GraphicsDecoder<P>,
    enc: &P::BlitEncoder,
    on: OnFault,
) -> RaiseResult<bool> {
    // No blit op is decoded yet; the walk faults every blit-group op.
    let _ = (tag, payload, objs, enc, on);
    Ok(false)
}
