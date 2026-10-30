//! The command-buffer decoder: it carries out one
//! [`RGFX_CMD_COMMAND_BUFFER`](reims_graphics::wire::RGFX_CMD_COMMAND_BUFFER)
//! message (docs/graphics-kext/design.md, "Stage 3 wire protocol"). It resolves
//! the queue the buffer runs on, opens a `P::CommandBuffer`, walks the op stream
//! that follows the message, and at the end registers the completion path and
//! commits.
//!
//! **The op walk.** Each op is an [`RgfxOpHeader`] then its body. The walk reads
//! the header, checks its `len` (at least a header, a multiple of 8, and within
//! the remaining bytes), and dispatches by the op's group ([`op_group`]): the
//! framing group (encoder begin/end) it handles here; the render, compute and
//! blit groups it hands to [`super::render`], [`super::compute`] and
//! [`super::blit`], one encoder held open at a time.
//!
//! **Two kinds of failure.** A message or op that will not parse structurally — a
//! short message, a bad `len`, an op body too short for a header — is a protocol
//! error, reported as a [`DecodeError`] and the work abandoned. A structurally
//! valid stream the host cannot honour — an unknown op, an op with no matching
//! open encoder, a reference to an object the host does not hold — faults the exec
//! ([`op_fault`], [`bad_reference`](super)), since the stream is already running on
//! the GPU's queue; the ring loop's handler turns the fault into a command-buffer
//! error.

use std::sync::Arc;

use reims_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE, RGFX_OP_BODY_OFFSET,
    RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_BEGIN_COMPUTE, RGFX_OP_ENCODER_BEGIN_RENDER,
    RGFX_OP_ENCODER_END, RGFX_OP_GROUP_BLIT, RGFX_OP_GROUP_COMPUTE, RGFX_OP_GROUP_FRAMING,
    RGFX_OP_GROUP_RENDER, RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT,
    RGFX_PASS_FLAG_RT_ARRAY_LENGTH, RGFX_PASS_FLAG_RT_SIZE, RGFX_RENDER_PASS_COLOR_OFFSET,
    RgfxCommandBuffer, RgfxComputePassDesc, RgfxOpHeader, RgfxRenderPassColorAttachment,
    RgfxRenderPassDesc, op_group,
};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue, CompletedInfo};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder};
use reims_vgpu_core::platform::descriptor::RenderPassDescriptor;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder};
use reims_vgpu_core::platform::types::ClearColor;
use reims_vgpu_core::platform::values::{
    ColorAttachmentSet, ComputePassDesc, DepthAttachmentSet, PassSet, StencilAttachmentSet,
};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult};
use zerocopy::FromBytes;

use super::super::completion::cb_complete;
use super::{
    DecodeError, Drain, GraphicsDecoder, ObjHandle, blit, compute, op_fault, read, render,
};

/// The op stream runs under a live raise handler (the ring loop's), so an encode's
/// raise returns for the handler to catch.
const ON: OnFault = OnFault::Return;

/// The encoder the walk currently holds open, one at a time, between an
/// `ENCODER_BEGIN_*` and the matching `ENCODER_END`.
enum OpenEncoder<P: Platform> {
    Render(P::RenderEncoder),
    Compute(P::ComputeEncoder),
    Blit(P::BlitEncoder),
}

impl<P: Platform> GraphicsDecoder<P> {
    /// Carries out one `COMMAND_BUFFER` message (`decode`'s
    /// [`RGFX_CMD_COMMAND_BUFFER`](reims_graphics::wire::RGFX_CMD_COMMAND_BUFFER)
    /// arm): opens a command buffer on the named queue, walks the op stream, and
    /// commits with a completion handler, which holds `drain` (its context's) until it has run.
    pub(super) fn decode_command_buffer(
        &self,
        message: &[u8],
        drain: Option<Arc<Drain>>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxCommandBuffer>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        // The op stream follows the message; `op_bytes` must lie within it.
        let start = RGFX_COMMAND_BUFFER_OPS_OFFSET as usize;
        let Some(ops) = (m.op_bytes as usize)
            .checked_add(start)
            .and_then(|end| message.get(start..end))
        else {
            return Ok(Err(DecodeError::Malformed));
        };
        // The queue the buffer runs on; a missing or wrong-kind queue is a
        // message-level bad reference (the op stream has not started).
        let cb = {
            let Some(ObjHandle::CommandQueue(queue)) = self.named(m.queue_obj_id) else {
                return Ok(Err(DecodeError::BadReference));
            };
            match queue.command_buffer(ON)? {
                Some(cb) => cb,
                None => return Ok(Err(DecodeError::Refused)),
            }
        };
        // A structural malformation in the stream stops the walk and is reported;
        // everything else faults the exec and propagates.
        if let Err(e) = self.walk_ops(ops, &cb)? {
            return Ok(Err(e));
        }
        // Register the completion path and commit. The GPU start and end times are
        // 0 here; the ring loop fills them when it posts the record on the queue's
        // completion ring in a later increment.
        let cb_obj_id = m.cb_obj_id;
        let seqno = m.header.seqno;
        let sink = self.completion_sink.clone();
        cb.add_completed_handler(
            ON,
            Box::new(move |info: CompletedInfo| {
                let record = cb_complete(cb_obj_id, seqno, 0, 0, info.error.as_ref());
                if let Some(sink) = &sink {
                    sink(record);
                }
                // The buffer's work and its completion are done: a drain of its context may end.
                drop(drain);
            }),
        )?;
        cb.commit(ON)?;
        Ok(Ok(()))
    }

    /// Walks the op stream `ops` on the command buffer `cb`. `Ok(Err(..))` for a
    /// structural malformation, `Ok(Ok(()))` when the whole stream was carried out;
    /// a fault propagates as `Err`.
    fn walk_ops(
        &self,
        mut ops: &[u8],
        cb: &P::CommandBuffer,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let mut encoder: Option<OpenEncoder<P>> = None;
        while !ops.is_empty() {
            let Some(h) = read::<RgfxOpHeader>(ops) else {
                return Ok(Err(DecodeError::Malformed));
            };
            let len = h.len as usize;
            if len < RGFX_OP_BODY_OFFSET as usize || !len.is_multiple_of(8) || len > ops.len() {
                return Ok(Err(DecodeError::Malformed));
            }
            let body = &ops[RGFX_OP_BODY_OFFSET as usize..len];
            match op_group(h.tag) {
                RGFX_OP_GROUP_FRAMING => self.framing_op(h.tag, body, cb, &mut encoder)?,
                RGFX_OP_GROUP_RENDER => {
                    let Some(OpenEncoder::Render(enc)) = encoder.as_ref() else {
                        return Err(no_encoder(h.tag, "render"));
                    };
                    if !render::decode_op(h.tag, body, self, enc, ON)? {
                        return Err(unknown_op(h.tag));
                    }
                }
                RGFX_OP_GROUP_COMPUTE => {
                    let Some(OpenEncoder::Compute(enc)) = encoder.as_ref() else {
                        return Err(no_encoder(h.tag, "compute"));
                    };
                    if !compute::decode_op(h.tag, body, self, enc, ON)? {
                        return Err(unknown_op(h.tag));
                    }
                }
                RGFX_OP_GROUP_BLIT => {
                    let Some(OpenEncoder::Blit(enc)) = encoder.as_ref() else {
                        return Err(no_encoder(h.tag, "blit"));
                    };
                    if !blit::decode_op(h.tag, body, self, enc, ON)? {
                        return Err(unknown_op(h.tag));
                    }
                }
                _ => return Err(unknown_op(h.tag)),
            }
            ops = &ops[len..];
        }
        if encoder.is_some() {
            return Err(op_fault("the op stream ended with an encoder still open"));
        }
        Ok(Ok(()))
    }

    /// A framing op: open the encoder its `ENCODER_BEGIN_*` names, or end the open
    /// one. An already-open encoder at a begin, or no open encoder at an end,
    /// faults.
    fn framing_op(
        &self,
        tag: u16,
        body: &[u8],
        cb: &P::CommandBuffer,
        encoder: &mut Option<OpenEncoder<P>>,
    ) -> RaiseResult<()> {
        match tag {
            RGFX_OP_ENCODER_BEGIN_RENDER => {
                already_open(encoder, "ENCODER_BEGIN_RENDER")?;
                let pass = self.build_render_pass(body)?;
                let enc = cb
                    .render_encoder(ON, &pass)?
                    .ok_or_else(|| op_fault("the render encoder was nil"))?;
                *encoder = Some(OpenEncoder::Render(enc));
            }
            RGFX_OP_ENCODER_BEGIN_COMPUTE => {
                already_open(encoder, "ENCODER_BEGIN_COMPUTE")?;
                let desc = compute_pass(body)?;
                let enc = cb
                    .compute_encoder_for_pass(&desc)?
                    .ok_or_else(|| op_fault("the compute encoder was nil"))?;
                *encoder = Some(OpenEncoder::Compute(enc));
            }
            RGFX_OP_ENCODER_BEGIN_BLIT => {
                already_open(encoder, "ENCODER_BEGIN_BLIT")?;
                let enc = cb
                    .blit_encoder(ON)?
                    .ok_or_else(|| op_fault("the blit encoder was nil"))?;
                *encoder = Some(OpenEncoder::Blit(enc));
            }
            RGFX_OP_ENCODER_END => match encoder.take() {
                Some(OpenEncoder::Render(enc)) => enc.encode(ON, RenderCmd::EndEncoding)?,
                Some(OpenEncoder::Compute(enc)) => enc.encode(ON, ComputeCmd::EndEncoding)?,
                Some(OpenEncoder::Blit(enc)) => enc.encode(ON, BlitCmd::EndEncoding)?,
                None => return Err(op_fault("ENCODER_END with no open encoder")),
            },
            _ => return Err(unknown_op(tag)),
        }
        Ok(())
    }

    /// Builds a `P::RenderPassDescriptor` from an `ENCODER_BEGIN_RENDER` op's body
    /// (an [`RgfxRenderPassDesc`] then its `color_count`
    /// [`RgfxRenderPassColorAttachment`] records). A body too short faults; a
    /// reference to a texture the host does not hold faults through the resolution
    /// API.
    fn build_render_pass(&self, body: &[u8]) -> RaiseResult<P::RenderPassDescriptor> {
        let (desc, _) = RgfxRenderPassDesc::read_from_prefix(body).map_err(|_| {
            op_fault("ENCODER_BEGIN_RENDER: body too short for the pass descriptor")
        })?;
        let pass = self.platform.new_render_pass_descriptor();

        let mut set = PassSet::default();
        if desc.flags & RGFX_PASS_FLAG_RT_ARRAY_LENGTH != 0 {
            set.render_target_array_length = Some(desc.render_target_array_length as u64);
        }
        if desc.flags & RGFX_PASS_FLAG_RT_SIZE != 0 {
            set.render_target_width = Some(desc.render_target_width as u64);
            set.render_target_height = Some(desc.render_target_height as u64);
        }
        if desc.flags & RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT != 0 {
            set.default_raster_sample_count = Some(desc.default_raster_sample_count as u64);
        }
        pass.set_pass(&set);

        let stride = size_of::<RgfxRenderPassColorAttachment>();
        let color_start = RGFX_RENDER_PASS_COLOR_OFFSET as usize;
        for i in 0..desc.color_count as usize {
            let off = color_start + i * stride;
            let a = body
                .get(off..off + stride)
                .and_then(|s| RgfxRenderPassColorAttachment::read_from_bytes(s).ok())
                .ok_or_else(|| {
                    op_fault("ENCODER_BEGIN_RENDER: body too short for a colour attachment")
                })?;
            if a.texture_obj_id == 0 {
                continue; // an absent (nil) attachment sets nothing
            }
            let tex = self.require_texture(a.texture_obj_id)?;
            let resolve = self.get_texture(a.resolve_texture_obj_id)?;
            let aset = ColorAttachmentSet {
                texture: Some(Some(&tex)),
                resolve_texture: resolve.as_ref().map(Some),
                level: Some(a.level as u64),
                slice: Some(a.slice as u64),
                depth_plane: Some(a.depth_plane as u64),
                resolve_level: resolve.as_ref().map(|_| a.resolve_level as u64),
                resolve_slice: resolve.as_ref().map(|_| a.resolve_slice as u64),
                resolve_depth_plane: resolve.as_ref().map(|_| a.resolve_depth_plane as u64),
                load_action: Some(a.load_action as u64),
                store_action: Some(a.store_action as u64),
                store_action_options: Some(a.store_action_options as u64),
                clear_color: Some(clear_color(a.clear_color_bits)),
            };
            pass.set_color_attachment(ON, i as u64, &aset)?;
        }

        if desc.depth.texture_obj_id != 0 {
            let d = &desc.depth;
            let tex = self.require_texture(d.texture_obj_id)?;
            let resolve = self.get_texture(d.resolve_texture_obj_id)?;
            let dset = DepthAttachmentSet {
                texture: Some(Some(&tex)),
                resolve_texture: resolve.as_ref().map(Some),
                level: Some(d.level as u64),
                slice: Some(d.slice as u64),
                depth_plane: Some(d.depth_plane as u64),
                resolve_level: resolve.as_ref().map(|_| d.resolve_level as u64),
                resolve_slice: resolve.as_ref().map(|_| d.resolve_slice as u64),
                resolve_depth_plane: resolve.as_ref().map(|_| d.resolve_depth_plane as u64),
                load_action: Some(d.load_action as u64),
                store_action: Some(d.store_action as u64),
                store_action_options: Some(d.store_action_options as u64),
                clear_depth: Some(f64::from_bits(d.clear_depth_bits)),
                depth_resolve_filter: Some(d.depth_resolve_filter as u64),
            };
            pass.set_depth_attachment(&dset);
        }

        if desc.stencil.texture_obj_id != 0 {
            let s = &desc.stencil;
            let tex = self.require_texture(s.texture_obj_id)?;
            let resolve = self.get_texture(s.resolve_texture_obj_id)?;
            let sset = StencilAttachmentSet {
                texture: Some(Some(&tex)),
                resolve_texture: resolve.as_ref().map(Some),
                level: Some(s.level as u64),
                slice: Some(s.slice as u64),
                depth_plane: Some(s.depth_plane as u64),
                resolve_level: resolve.as_ref().map(|_| s.resolve_level as u64),
                resolve_slice: resolve.as_ref().map(|_| s.resolve_slice as u64),
                resolve_depth_plane: resolve.as_ref().map(|_| s.resolve_depth_plane as u64),
                load_action: Some(s.load_action as u64),
                store_action: Some(s.store_action as u64),
                store_action_options: Some(s.store_action_options as u64),
                clear_stencil: Some(s.clear_stencil),
                stencil_resolve_filter: Some(s.stencil_resolve_filter as u64),
            };
            pass.set_stencil_attachment(&sset);
        }

        Ok(pass)
    }
}

/// A fault: an encoder op whose group has no open encoder.
fn no_encoder(tag: u16, group: &str) -> Raise {
    op_fault(format_args!(
        "a {group} op {tag:#06x} with no open {group} encoder"
    ))
}

/// A fault: an op the host does not decode.
fn unknown_op(tag: u16) -> Raise {
    op_fault(format_args!("unknown op {tag:#06x}"))
}

/// A fault when an encoder is already open at an `ENCODER_BEGIN_*`.
fn already_open<P: Platform>(encoder: &Option<OpenEncoder<P>>, op: &str) -> RaiseResult<()> {
    if encoder.is_some() {
        return Err(op_fault(format_args!("{op} with an encoder already open")));
    }
    Ok(())
}

/// The compute pass descriptor from an `ENCODER_BEGIN_COMPUTE` op's body (an empty
/// body is Metal's default pass).
fn compute_pass(body: &[u8]) -> RaiseResult<ComputePassDesc> {
    if body.is_empty() {
        return Ok(ComputePassDesc::default());
    }
    let (m, _) = RgfxComputePassDesc::read_from_prefix(body)
        .map_err(|_| op_fault("ENCODER_BEGIN_COMPUTE: body too short"))?;
    Ok(ComputePassDesc {
        dispatch_type: (m.flags & RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE != 0)
            .then_some(m.dispatch_type as u64),
    })
}

/// An `MTLClearColor` from its four `f64` bits.
fn clear_color(bits: [u64; 4]) -> ClearColor {
    ClearColor {
        red: f64::from_bits(bits[0]),
        green: f64::from_bits(bits[1]),
        blue: f64::from_bits(bits[2]),
        alpha: f64::from_bits(bits[3]),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_graphics::ops::render::{
        RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES, RGFX_OP_RENDER_DRAW_PRIMITIVES,
        RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE, RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
        RGFX_OP_RENDER_SET_SCISSOR_RECT, RGFX_OP_RENDER_SET_VERTEX_BUFFER,
        RGFX_OP_RENDER_SET_VIEWPORT, RgfxOpDrawIndexedPrimitives, RgfxOpDrawPrimitives,
        RgfxOpSetFragmentTexture, RgfxOpSetRenderPipelineState, RgfxOpSetScissorRect,
        RgfxOpSetVertexBuffer, RgfxOpSetViewport,
    };
    use reims_graphics::wire::{
        RGFX_CB_OK, RGFX_CMD_BUFFER_CREATE, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_QUEUE_CREATE,
        RGFX_CMD_TEXTURE_CREATE, RgfxBufferCreate, RgfxCbComplete, RgfxMsgHeader, RgfxQueueCreate,
        RgfxTextureCreate, RgfxTextureDescriptor,
    };
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor;
    use reims_vgpu_core::platform::values::AttachmentState;
    use reims_vgpu_core::sync::Mutex;
    use reims_vgpu_core::testkit::FakePlatform;
    use reims_vgpu_core::testkit::objects::CompleteOn;
    use zerocopy::IntoBytes;

    use super::*;

    /// One op: its header then its body, `len` the whole op.
    fn op(tag: u16, body: &[u8]) -> Vec<u8> {
        let mut v = RgfxOpHeader {
            tag,
            flags: 0,
            len: (8 + body.len()) as u32,
        }
        .as_bytes()
        .to_vec();
        v.extend_from_slice(body);
        v
    }

    /// A fresh decoder with a queue (id 1), two buffers (2 vertex, 3 index) and two
    /// textures (4 colour attachment, 5 fragment) already created.
    fn setup() -> (Arc<FakePlatform>, GraphicsDecoder<FakePlatform>) {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );

        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 1,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));

        for id in [2, 3] {
            let b = RgfxBufferCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_BUFFER_CREATE,
                    size: size_of::<RgfxBufferCreate>() as u32,
                    seqno: 0,
                },
                obj_id: id,
                length: 65536,
                ..RgfxBufferCreate::default()
            };
            assert_eq!(dec.decode(&device, b.as_bytes()).unwrap(), Ok(()));
        }

        // The textures' resource ids are their object ids, in chunk 0.
        dec.hold_resource_ids(0, 0);
        for id in [4, 5] {
            let t = RgfxTextureCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_TEXTURE_CREATE,
                    size: size_of::<RgfxTextureCreate>() as u32,
                    seqno: 0,
                },
                obj_id: id,
                desc: RgfxTextureDescriptor {
                    pixel_format: 80,
                    texture_type: 2,
                    width: 256,
                    height: 256,
                    depth: 1,
                    mipmap_level_count: 1,
                    array_length: 1,
                    sample_count: 1,
                    usage: 5,
                    ..RgfxTextureDescriptor::default()
                },
                resource_id: id,
                ..RgfxTextureCreate::default()
            };
            assert_eq!(dec.decode(&device, t.as_bytes()).unwrap(), Ok(()));
        }
        (p, dec)
    }

    /// An `ENCODER_BEGIN_RENDER` body: one colour attachment (texture 4, clear then
    /// store) over a 1920×1080 target.
    fn begin_render_body() -> Vec<u8> {
        let pass = RgfxRenderPassDesc {
            color_count: 1,
            flags: RGFX_PASS_FLAG_RT_SIZE,
            render_target_width: 1920,
            render_target_height: 1080,
            ..RgfxRenderPassDesc::default()
        };
        let color = RgfxRenderPassColorAttachment {
            texture_obj_id: 4,
            load_action: 2,  // Clear
            store_action: 1, // Store
            clear_color_bits: [0, 0, 0, 1.0f64.to_bits()],
            ..RgfxRenderPassColorAttachment::default()
        };
        let mut body = pass.as_bytes().to_vec();
        body.extend_from_slice(color.as_bytes());
        body
    }

    /// The reference render ops, as the op stream carries them.
    fn render_ops() -> Vec<u8> {
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body());
        ops.extend(op(
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            RgfxOpSetRenderPipelineState { pipeline_obj_id: 0 }.as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_SET_VERTEX_BUFFER,
            RgfxOpSetVertexBuffer {
                buffer_obj_id: 2,
                offset: 0,
                index: 0,
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
            RgfxOpSetFragmentTexture {
                texture_obj_id: 5,
                index: 0,
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_SET_VIEWPORT,
            RgfxOpSetViewport {
                origin_x_bits: 0.0f64.to_bits(),
                origin_y_bits: 0.0f64.to_bits(),
                width_bits: 1920.0f64.to_bits(),
                height_bits: 1080.0f64.to_bits(),
                znear_bits: 0.0f64.to_bits(),
                zfar_bits: 1.0f64.to_bits(),
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_SET_SCISSOR_RECT,
            RgfxOpSetScissorRect {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_DRAW_PRIMITIVES,
            RgfxOpDrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES,
            RgfxOpDrawIndexedPrimitives {
                primitive_type: 3,
                index_count: 6,
                index_type: 1,
                index_buffer_obj_id: 3,
                index_buffer_offset: 0,
            }
            .as_bytes(),
        ));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        ops
    }

    /// A `COMMAND_BUFFER` message for queue 1, command buffer `cb_obj_id`, carrying
    /// `ops`.
    fn command_buffer_message(cb_obj_id: u64, seqno: u64, ops: &[u8]) -> Vec<u8> {
        let m = RgfxCommandBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMMAND_BUFFER,
                size: (RGFX_COMMAND_BUFFER_OPS_OFFSET as usize + ops.len()) as u32,
                seqno,
            },
            queue_obj_id: 1,
            cb_obj_id,
            op_bytes: ops.len() as u64,
        };
        let mut msg = m.as_bytes().to_vec();
        msg.extend_from_slice(ops);
        msg
    }

    /// A command buffer opens a render encoder for its pass, sends each reference
    /// command in order, ends the encoder, and commits with a completion handler;
    /// completing the buffer produces the completion record.
    #[test]
    fn a_command_buffer_drives_the_encoder() {
        let (p, mut dec) = setup();
        let device = p.device();

        let captured = Arc::new(Mutex::new(Vec::<RgfxCbComplete>::new()));
        let sink = Arc::clone(&captured);
        dec.set_completion_sink(Arc::new(move |record| sink.lock().push(record)));

        let message = command_buffer_message(100, 55, &render_ops());
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));

        // The command-buffer and encoder selectors, in order (retains and the
        // descriptor builder's unordered sets filtered out).
        let want = [
            "commandBuffer",
            "renderCommandEncoderWithDescriptor:",
            "setRenderPipelineState:",
            "setVertexBuffer:offset:atIndex:",
            "setFragmentTexture:atIndex:",
            "setViewport:",
            "setScissorRect:",
            "drawPrimitives:vertexStart:vertexCount:",
            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
            "endEncoding",
            "addCompletedHandler:",
            "commit",
        ];
        let got: Vec<&str> = p
            .call_log()
            .sels_since(mark)
            .into_iter()
            .filter(|s| want.contains(s))
            .collect();
        assert_eq!(got, want);

        // Completing the committed buffer runs the handler, which produces the
        // completion record for cb_obj_id 100 at the submission seqno.
        p.world().complete_all(CompleteOn::Caller);
        let recs = captured.lock();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].cb_obj_id, 100);
        assert_eq!(recs[0].header.seqno, 55);
        assert_eq!(recs[0].status, RGFX_CB_OK);
    }

    /// The render-pass descriptor is built from the wire schema: its colour
    /// attachment carries the wire load and store actions.
    #[test]
    fn a_render_pass_is_built_from_the_wire_schema() {
        let (_p, dec) = setup();
        let pass = dec.build_render_pass(&begin_render_body()).unwrap();
        assert_eq!(
            pass.color_attachment_state(0).unwrap(),
            AttachmentState {
                load_action: 2,
                store_action: 1,
            }
        );
    }

    /// An unknown op in the stream faults the exec (a raise), after the encoder has
    /// opened.
    #[test]
    fn an_unknown_op_faults() {
        let (p, mut dec) = setup();
        let device = p.device();
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body());
        // A render-group tag (0x1fff) the decoder does not implement.
        ops.extend(op(0x1fff, &[0u8; 8]));
        let message = command_buffer_message(101, 1, &ops);
        assert!(dec.decode(&device, &message).is_err());
    }

    /// A render op with no open encoder faults the exec.
    #[test]
    fn a_render_op_without_an_encoder_faults() {
        let (p, mut dec) = setup();
        let device = p.device();
        let ops = op(
            RGFX_OP_RENDER_DRAW_PRIMITIVES,
            RgfxOpDrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            }
            .as_bytes(),
        );
        let message = command_buffer_message(102, 1, &ops);
        assert!(dec.decode(&device, &message).is_err());
    }

    /// A command buffer naming a queue the host does not hold is a bad reference,
    /// not a fault; a short message is malformed.
    #[test]
    fn a_missing_queue_and_a_short_message_are_reported() {
        let (p, mut dec) = setup();
        let device = p.device();
        let mut m = RgfxCommandBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMMAND_BUFFER,
                size: RGFX_COMMAND_BUFFER_OPS_OFFSET,
                seqno: 1,
            },
            queue_obj_id: 999,
            cb_obj_id: 1,
            op_bytes: 0,
        }
        .as_bytes()
        .to_vec();
        assert_eq!(
            dec.decode(&device, &m).unwrap(),
            Err(DecodeError::BadReference)
        );
        m.truncate(8);
        assert_eq!(
            dec.decode(&device, &m).unwrap(),
            Err(DecodeError::Malformed)
        );
    }
}
