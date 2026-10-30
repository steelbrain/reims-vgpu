//! The compute-encoder op decoder: one arm per
//! [`reims_graphics::ops::compute`] op, turning a wire op into a `ComputeCmd` on
//! the open `P::ComputeEncoder` (docs/graphics-kext/design.md, "Stage 3 wire
//! protocol"). The command-buffer walk ([`super::cmdbuf`]) routes every
//! compute-group op (`tag >> 12 == `[`RGFX_OP_GROUP_COMPUTE`]) here.
//!
//! This is a stub: it handles no tag yet (returns `Ok(false)` for every one, so
//! the walk faults an unrecognised compute op). The compute ops land here, one
//! arm each, matching [`super::render`]'s pattern.
//!
//! [`RGFX_OP_GROUP_COMPUTE`]: reims_graphics::ops::RGFX_OP_GROUP_COMPUTE

use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::raise::{OnFault, RaiseResult};

use super::GraphicsDecoder;

/// Decodes one compute-encoder op: `tag` its op, `payload` its body (the bytes
/// after the [`RgfxOpHeader`](reims_graphics::ops::RgfxOpHeader)). `objs` resolves
/// object references, `enc` is the open compute encoder, and `on` says how an
/// encode's raise is taken. `Ok(true)` when the op was handled, `Ok(false)` for an
/// unknown tag (the walk then faults); a platform raise propagates as `Err`.
pub(crate) fn decode_op<P: Platform>(
    tag: u16,
    payload: &[u8],
    objs: &GraphicsDecoder<P>,
    enc: &P::ComputeEncoder,
    on: OnFault,
) -> RaiseResult<bool> {
    // No compute op is decoded yet; the walk faults every compute-group op.
    let _ = (tag, payload, objs, enc, on);
    Ok(false)
}
