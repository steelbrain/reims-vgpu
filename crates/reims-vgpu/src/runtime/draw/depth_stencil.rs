//! The host-side depth and stencil attachment buffers of the direct-Metal rail.
//!
//! Metal is handed a pointer to host memory for each aspect, seeded from the
//! guest's CLEAR value or read back from its texture, and written back into the
//! guest's type-11 mapping when the pass stores. That is one procedure over a
//! buffer whose texel is four bytes for depth and one for stencil — and it used
//! to be written out twice, once per aspect, in the caller.
//!
//! The two copies had already drifted. Both tested `level` and
//! `resolve_texture_ref` and neither tested `slice` or `depth_plane`, while the
//! owning rule they were copies of
//! ([`attachment_subresource_is_bindable`](crate::runtime::decode::render::attachment_subresource_is_bindable))
//! tests all four; and both refused in silence, so an attachment this rail
//! could not carry vanished with nothing in the log.

use super::{degrade_log_first, load_linear_raw, DeviceState, DrawEncodeRequest};
use crate::contract::pass_action::{
    is_declared_load_action, is_declared_store_action, MTL_LOAD_ACTION_CLEAR, MTL_LOAD_ACTION_LOAD,
    MTL_STORE_ACTION_STORE,
};
use crate::runtime::decode::render::{
    attachment_subresource_is_bindable, AttachSubresource, DepthAttachment, LevelSupport,
    StencilAttachment,
};
use crate::runtime::host::HostMemory;
use crate::runtime::{mapper, mapping_write, objects, HostOps};

fn fill_depth32(buf: &mut [u8], depth: f32) {
    let bits = depth.to_bits().to_le_bytes();
    for i in 0..(buf.len() / 4) {
        buf[i * 4..i * 4 + 4].copy_from_slice(&bits);
    }
}

/// Which aspect a host-side attachment buffer carries, and its clear value.
#[derive(Clone, Copy)]
pub(super) enum DepthStencilAspect {
    /// `MTLPixelFormatDepth32Float` — one `f32` per texel.
    Depth { clear: f64 },
    /// `MTLPixelFormatStencil8` — one byte per texel.
    Stencil { clear: u32 },
}

/// The per-aspect constants of [`DepthStencilAspect`], in one place so a new
/// consumer cannot spell three of the four and derive the fourth.
pub(super) struct AspectSpec {
    bytes_per_texel: u32,
    /// Reported when the guest asked to LOAD prior contents and this device
    /// could not read them. One slug per aspect so `degrade_log_first`'s latch
    /// keeps a depth failure from silencing a stencil one.
    readback_failed: &'static str,
    /// Reported when the load/store action pair is outside what a host-side
    /// buffer can carry.
    actions_refused: &'static str,
    /// What the substituted contents are, for the degrade line's own words.
    clear_name: &'static str,
}

impl DepthStencilAspect {
    fn spec(self) -> &'static AspectSpec {
        const DEPTH: AspectSpec = AspectSpec {
            bytes_per_texel: 4,
            readback_failed: "depth_load_readback_failed",
            actions_refused: "depth_actions_unsupported",
            clear_name: "clear_depth",
        };
        const STENCIL: AspectSpec = AspectSpec {
            bytes_per_texel: 1,
            readback_failed: "stencil_load_readback_failed",
            actions_refused: "stencil_actions_unsupported",
            clear_name: "clear_stencil",
        };
        match self {
            Self::Depth { .. } => &DEPTH,
            Self::Stencil { .. } => &STENCIL,
        }
    }

    fn fill_clear(self, buf: &mut [u8]) {
        match self {
            Self::Depth { clear } => fill_depth32(buf, clear as f32),
            Self::Stencil { clear } => buf.fill(clear as u8),
        }
    }
}

/// The half of a decoded depth or stencil attachment this rail reads.
#[derive(Clone, Copy)]
pub(super) struct HostAttachment {
    texture_ref: u32,
    load_action: u16,
    store_action: u16,
    sub: AttachSubresource,
}

impl From<DepthAttachment> for HostAttachment {
    fn from(a: DepthAttachment) -> Self {
        Self {
            texture_ref: a.texture_ref,
            load_action: a.load_action,
            store_action: a.store_action,
            sub: a.into(),
        }
    }
}

impl From<StencilAttachment> for HostAttachment {
    fn from(a: StencilAttachment) -> Self {
        Self {
            texture_ref: a.texture_ref,
            load_action: a.load_action,
            store_action: a.store_action,
            sub: a.into(),
        }
    }
}

/// A host-side depth or stencil attachment buffer: the bytes Metal loads, and
/// where a STORE puts them back.
pub(super) struct HostDepthStencil {
    pub(super) data: Vec<u8>,
    /// Type-11 mapping behind the attachment's texture, or 0 when the texture
    /// resolved to none — a STORE has nowhere to go in that case.
    mapping_id: u32,
    store_action: u16,
    row_bytes: u32,
}

impl HostDepthStencil {
    pub(super) fn store_back<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        extent: (u32, u32),
    ) {
        if self.store_action != MTL_STORE_ACTION_STORE || self.mapping_id == 0 {
            return;
        }
        let (width, height) = extent;
        let _ = mapper::ensure_resolved_for_scanout(state, host, self.mapping_id);
        let _ = mapping_write::write_raw_rows(
            state,
            host,
            self.mapping_id,
            &self.data,
            self.row_bytes,
            self.row_bytes,
            width,
            height,
        );
    }
}

/// Build the host-side buffer for one depth or stencil attachment.
///
/// `None` means this rail refused the attachment, having named why. The
/// subresource half of the admission is [`attachment_subresource_is_bindable`], which
/// the stream decode already applied — it is re-asked here because nothing but
/// this call records that the two arms use one rule. It asks with
/// [`LevelSupport::LevelZeroOnly`], the same answer the stream decode's depth
/// and stencil arms give: a host-side buffer for one aspect is built from the
/// texture's base level, so a named level would be read from the wrong plane.
/// The action half is this rail's own: a host-side buffer carries the three
/// `MTLLoadAction`s and the two non-resolving `MTLStoreAction`s, and nothing
/// else.
pub(super) fn seed_host_depth_stencil<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    aspect: DepthStencilAspect,
    attach: HostAttachment,
    extent: (u32, u32),
) -> Option<HostDepthStencil> {
    let spec = aspect.spec();
    if attach.texture_ref == 0 {
        return None;
    }
    // The same admission the colour rail applies, asked through the predicates
    // beside the constants: this site used to spell the store half as
    // `== DONT_CARE || == STORE` while `draw::store_action_in_contract` spelled
    // it as `<= STORE`, one rule in two forms with nothing comparing them.
    let actions_ok = is_declared_load_action(attach.load_action)
        && is_declared_store_action(attach.store_action);
    let subresource_ok =
        attachment_subresource_is_bindable(attach.sub, LevelSupport::LevelZeroOnly);
    if !subresource_ok || !actions_ok {
        if degrade_log_first(req.pipeline_ref, spec.actions_refused) {
            crate::observe::fail(format!(
                "shader_state_degraded reason={} pipe={} task={} ds_ref={} \
                 level={} slice={} plane={} resolve={} load={} store={} \
                 (attachment dropped; the pass runs with this aspect unbound)",
                spec.actions_refused,
                req.pipeline_ref,
                req.task_id,
                attach.texture_ref,
                attach.sub.level,
                attach.sub.slice,
                attach.sub.depth_plane,
                attach.sub.resolve_texture_ref,
                attach.load_action,
                attach.store_action,
            ));
        }
        return None;
    }

    // The pass extent, same as every other attachment in it — the buffer's rows
    // and its row count have to come from one geometry.
    let (width, height) = extent;
    let row_bytes = width.saturating_mul(spec.bytes_per_texel);
    let mut data = vec![0u8; (row_bytes as usize).saturating_mul(height as usize)];
    let mapping_id =
        objects::resolve_type11_ref(state, host, req.task_id, attach.texture_ref).unwrap_or(0);
    if mapping_id != 0 {
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
    }
    match attach.load_action {
        MTL_LOAD_ACTION_CLEAR => aspect.fill_clear(&mut data),
        MTL_LOAD_ACTION_LOAD => {
            let ok = if mapping_id != 0 {
                mapping_write::read_raw_rows(
                    state, host, mapping_id, &mut data, row_bytes, row_bytes, width, height,
                )
            } else {
                load_linear_raw(
                    state,
                    host,
                    req.task_id,
                    attach.texture_ref,
                    &mut data,
                    row_bytes,
                    row_bytes,
                    width,
                    height,
                )
            };
            if !ok {
                // The guest asked to load prior contents and this device could
                // not read them, so the pass runs against clear values instead:
                // its depth and stencil tests decide against content the guest
                // never wrote. The load action deliberately stays LOAD — the
                // seeded buffer *is* what Metal loads, so switching it to CLEAR
                // would describe the same bytes twice — but the substitution is
                // a loss of guest state and says so.
                aspect.fill_clear(&mut data);
                if degrade_log_first(req.pipeline_ref, spec.readback_failed) {
                    crate::observe::fail(format!(
                        "shader_state_degraded reason={} pipe={} task={} ds_ref={} \
                         mid={mapping_id} {width}x{height} \
                         (guest contents unreadable; pass seeded with {})",
                        spec.readback_failed,
                        req.pipeline_ref,
                        req.task_id,
                        attach.texture_ref,
                        spec.clear_name,
                    ));
                }
            }
        }
        _ => {}
    }
    Some(HostDepthStencil {
        data,
        mapping_id,
        store_action: attach.store_action,
        row_bytes,
    })
}
