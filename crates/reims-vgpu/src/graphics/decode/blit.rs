//! The blit-encoder op decoder: one arm per [`reims_graphics::ops::blit`] op,
//! turning a wire op into a `BlitCmd` on the open `P::BlitEncoder`
//! (docs/graphics-kext/design.md, "Stage 3 wire protocol"). The command-buffer
//! walk ([`super::cmdbuf`]) routes every blit-group op
//! (`tag >> 12 == `[`RGFX_OP_GROUP_BLIT`]) here.
//!
//! Each arm reads its op's body with zerocopy, resolves its object references
//! through the [object-resolution API](super) (`require_*`/`get_*`), and sends the
//! matching [`BlitCmd`] with [`BlitEncoder::encode`]. A body too short for its op
//! faults the exec ([`super::op_fault`]); a bad reference faults through the
//! resolution API.
//!
//! The blit encoder is opened by the framing `ENCODER_BEGIN_BLIT` op and closed by
//! `ENCODER_END` (`endEncoding`, [`super::cmdbuf`]), so `BlitCmd::EndEncoding` has no
//! arm here; every other `MTLBlitCommandEncoder` selector does.
//!
//! [`RGFX_OP_GROUP_BLIT`]: reims_graphics::ops::RGFX_OP_GROUP_BLIT

use reims_graphics::ops::blit::*;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder, ResourceRef};
use reims_vgpu_core::platform::host_range::HostRange;
use reims_vgpu_core::platform::types::{ClearColor, MtlOrigin, MtlRegion, MtlSize, NsRange};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult};
use zerocopy::FromBytes;

use super::{GraphicsDecoder, op_fault};

/// Reads op body `T` from `payload`, or an exec fault when it is too short.
#[inline]
fn body<T: FromBytes>(tag: u16, payload: &[u8]) -> RaiseResult<T> {
    match T::read_from_prefix(payload) {
        Ok((t, _)) => Ok(t),
        Err(_) => Err(short_body(tag)),
    }
}

/// The exec fault of blit op `tag`, whose body is too short: out of line, as no well-formed
/// op meets it.
#[cold]
#[inline(never)]
fn short_body(tag: u16) -> Raise {
    op_fault(format_args!("blit op {tag:#06x}: body too short"))
}

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
    match tag {
        RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER => {
            let m: RgfxBlitOpCopyBufferToBuffer = body(tag, payload)?;
            let source = objs.get_buffer(m.source_buffer_obj_id)?;
            let destination = objs.get_buffer(m.destination_buffer_obj_id)?;
            enc.encode(
                on,
                BlitCmd::CopyBufferToBuffer {
                    source_buffer: source,
                    source_offset: m.source_offset,
                    destination_buffer: destination,
                    destination_offset: m.destination_offset,
                    size: m.size,
                },
            )?;
        }
        RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE => {
            let m: RgfxBlitOpCopyBufferToTexture = body(tag, payload)?;
            let source = objs.require_buffer(m.source_buffer_obj_id)?;
            let destination = objs.require_texture(m.destination_texture_obj_id)?;
            enc.encode(
                on,
                BlitCmd::CopyBufferToTexture {
                    source_buffer: source,
                    source_offset: m.source_offset,
                    source_bytes_per_row: m.source_bytes_per_row,
                    source_bytes_per_image: m.source_bytes_per_image,
                    source_size: MtlSize::new(
                        m.source_size_width,
                        m.source_size_height,
                        m.source_size_depth,
                    ),
                    destination_texture: destination,
                    destination_slice: m.destination_slice,
                    destination_level: m.destination_level,
                    destination_origin: MtlOrigin::new(
                        m.destination_origin_x,
                        m.destination_origin_y,
                        m.destination_origin_z,
                    ),
                },
            )?;
        }
        RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE_OPTIONS => {
            let m: RgfxBlitOpCopyBufferToTextureOptions = body(tag, payload)?;
            let source = objs.get_buffer(m.source_buffer_obj_id)?;
            let destination = objs.get_texture(m.destination_texture_obj_id)?;
            enc.encode(
                on,
                BlitCmd::CopyBufferToTextureOptions {
                    source_buffer: source,
                    source_offset: m.source_offset,
                    source_bytes_per_row: m.source_bytes_per_row,
                    source_bytes_per_image: m.source_bytes_per_image,
                    source_size: MtlSize::new(
                        m.source_size_width,
                        m.source_size_height,
                        m.source_size_depth,
                    ),
                    destination_texture: destination,
                    destination_slice: m.destination_slice,
                    destination_level: m.destination_level,
                    destination_origin: MtlOrigin::new(
                        m.destination_origin_x,
                        m.destination_origin_y,
                        m.destination_origin_z,
                    ),
                    options: m.options,
                },
            )?;
        }
        RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER => {
            let m: RgfxBlitOpCopyTextureToBuffer = body(tag, payload)?;
            let source = objs.require_texture(m.source_texture_obj_id)?;
            let destination = objs.require_buffer(m.destination_buffer_obj_id)?;
            enc.encode(
                on,
                BlitCmd::CopyTextureToBuffer {
                    source_texture: source,
                    source_slice: m.source_slice,
                    source_level: m.source_level,
                    source_origin: MtlOrigin::new(
                        m.source_origin_x,
                        m.source_origin_y,
                        m.source_origin_z,
                    ),
                    source_size: MtlSize::new(
                        m.source_size_width,
                        m.source_size_height,
                        m.source_size_depth,
                    ),
                    destination_buffer: destination,
                    destination_offset: m.destination_offset,
                    destination_bytes_per_row: m.destination_bytes_per_row,
                    destination_bytes_per_image: m.destination_bytes_per_image,
                },
            )?;
        }
        RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER_OPTIONS => {
            let m: RgfxBlitOpCopyTextureToBufferOptions = body(tag, payload)?;
            let source = objs.get_texture(m.source_texture_obj_id)?;
            let destination = objs.get_buffer(m.destination_buffer_obj_id)?;
            enc.encode(
                on,
                BlitCmd::CopyTextureToBufferOptions {
                    source_texture: source,
                    source_slice: m.source_slice,
                    source_level: m.source_level,
                    source_origin: MtlOrigin::new(
                        m.source_origin_x,
                        m.source_origin_y,
                        m.source_origin_z,
                    ),
                    source_size: MtlSize::new(
                        m.source_size_width,
                        m.source_size_height,
                        m.source_size_depth,
                    ),
                    destination_buffer: destination,
                    destination_offset: m.destination_offset,
                    destination_bytes_per_row: m.destination_bytes_per_row,
                    destination_bytes_per_image: m.destination_bytes_per_image,
                    options: m.options,
                },
            )?;
        }
        RGFX_OP_BLIT_COPY_TEXTURE_REGION => {
            let m: RgfxBlitOpCopyTextureRegion = body(tag, payload)?;
            let source = objs.require_texture(m.source_texture_obj_id)?;
            let destination = objs.require_texture(m.destination_texture_obj_id)?;
            enc.encode(
                on,
                BlitCmd::CopyTextureRegion {
                    source_texture: source,
                    source_slice: m.source_slice,
                    source_level: m.source_level,
                    source_origin: MtlOrigin::new(
                        m.source_origin_x,
                        m.source_origin_y,
                        m.source_origin_z,
                    ),
                    source_size: MtlSize::new(
                        m.source_size_width,
                        m.source_size_height,
                        m.source_size_depth,
                    ),
                    destination_texture: destination,
                    destination_slice: m.destination_slice,
                    destination_level: m.destination_level,
                    destination_origin: MtlOrigin::new(
                        m.destination_origin_x,
                        m.destination_origin_y,
                        m.destination_origin_z,
                    ),
                },
            )?;
        }
        RGFX_OP_BLIT_COPY_TEXTURE_REGION_OPTIONS => {
            let m: RgfxBlitOpCopyTextureRegionOptions = body(tag, payload)?;
            let source = objs.require_texture(m.source_texture_obj_id)?;
            let destination = objs.require_texture(m.destination_texture_obj_id)?;
            enc.encode(
                on,
                BlitCmd::CopyTextureRegionOptions {
                    source_texture: source,
                    source_slice: m.source_slice,
                    source_level: m.source_level,
                    source_origin: MtlOrigin::new(
                        m.source_origin_x,
                        m.source_origin_y,
                        m.source_origin_z,
                    ),
                    source_size: MtlSize::new(
                        m.source_size_width,
                        m.source_size_height,
                        m.source_size_depth,
                    ),
                    destination_texture: destination,
                    destination_slice: m.destination_slice,
                    destination_level: m.destination_level,
                    destination_origin: MtlOrigin::new(
                        m.destination_origin_x,
                        m.destination_origin_y,
                        m.destination_origin_z,
                    ),
                    options: m.options,
                },
            )?;
        }
        RGFX_OP_BLIT_COPY_TEXTURE_SLICES => {
            let m: RgfxBlitOpCopyTextureSlices = body(tag, payload)?;
            let source = objs.require_texture(m.source_texture_obj_id)?;
            let destination = objs.require_texture(m.destination_texture_obj_id)?;
            enc.encode(
                on,
                BlitCmd::CopyTextureSlices {
                    source_texture: source,
                    source_slice: m.source_slice,
                    source_level: m.source_level,
                    destination_texture: destination,
                    destination_slice: m.destination_slice,
                    destination_level: m.destination_level,
                    slice_count: m.slice_count,
                    level_count: m.level_count,
                },
            )?;
        }
        RGFX_OP_BLIT_COPY_INDIRECT_COMMAND_BUFFER => {
            let m: RgfxBlitOpCopyIndirectCommandBuffer = body(tag, payload)?;
            let source = objs.require_indirect_command_buffer(m.source_obj_id)?;
            let destination = objs.require_indirect_command_buffer(m.destination_obj_id)?;
            enc.encode(
                on,
                BlitCmd::CopyIndirectCommandBuffer {
                    source,
                    source_range: NsRange::new(m.source_range_location, m.source_range_length),
                    destination,
                    destination_index: m.destination_index,
                },
            )?;
            objs.icb_commands_copied(
                (m.source_obj_id, m.source_range_location),
                (m.destination_obj_id, m.destination_index),
                m.source_range_length,
            );
        }
        RGFX_OP_BLIT_FILL_BUFFER => {
            let m: RgfxBlitOpFillBuffer = body(tag, payload)?;
            let buffer = objs.require_buffer(m.buffer_obj_id)?;
            enc.encode(
                on,
                BlitCmd::FillBuffer {
                    buffer,
                    range: NsRange::new(m.range_location, m.range_length),
                    value: m.value,
                },
            )?;
        }
        RGFX_OP_BLIT_FILL_BUFFER_PATTERN4 => {
            let m: RgfxBlitOpFillBufferPattern4 = body(tag, payload)?;
            let buffer = objs.require_buffer(m.buffer_obj_id)?;
            enc.encode(
                on,
                BlitCmd::FillBufferPattern4 {
                    buffer,
                    range: NsRange::new(m.range_location, m.range_length),
                    pattern: m.pattern,
                },
            )?;
        }
        RGFX_OP_BLIT_FILL_TEXTURE => {
            let m: RgfxBlitOpFillTexture = body(tag, payload)?;
            let texture = objs.require_texture(m.texture_obj_id)?;
            // The `length` fill bytes follow the fixed head in the op body; a body
            // too short for them faults the exec.
            let start = RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET as usize;
            let bytes = (m.length as usize)
                .checked_add(start)
                .and_then(|end| payload.get(start..end))
                .ok_or_else(|| op_fault("FILL_TEXTURE: body too short for the fill bytes"))?;
            enc.encode(
                on,
                BlitCmd::FillTexture {
                    texture,
                    level: m.level,
                    slice: m.slice,
                    region: MtlRegion::new(
                        MtlOrigin::new(m.region_origin_x, m.region_origin_y, m.region_origin_z),
                        MtlSize::new(
                            m.region_size_width,
                            m.region_size_height,
                            m.region_size_depth,
                        ),
                    ),
                    bytes: HostRange::of_slice(bytes),
                },
            )?;
        }
        RGFX_OP_BLIT_FILL_TEXTURE_COLOR => {
            let m: RgfxBlitOpFillTextureColor = body(tag, payload)?;
            let texture = objs.require_texture(m.texture_obj_id)?;
            enc.encode(
                on,
                BlitCmd::FillTextureColor {
                    texture,
                    level: m.level,
                    slice: m.slice,
                    region: MtlRegion::new(
                        MtlOrigin::new(m.region_origin_x, m.region_origin_y, m.region_origin_z),
                        MtlSize::new(
                            m.region_size_width,
                            m.region_size_height,
                            m.region_size_depth,
                        ),
                    ),
                    color: clear_color(m.color_bits),
                    pixel_format: m.pixel_format,
                },
            )?;
        }
        RGFX_OP_BLIT_GENERATE_MIPMAPS => {
            let m: RgfxBlitOpGenerateMipmaps = body(tag, payload)?;
            let texture = objs.get_texture(m.texture_obj_id)?;
            enc.encode(on, BlitCmd::GenerateMipmaps { texture })?;
        }
        RGFX_OP_BLIT_INVALIDATE_COMPRESSED_TEXTURE => {
            let m: RgfxBlitOpInvalidateCompressedTexture = body(tag, payload)?;
            let texture = objs.require_texture(m.texture_obj_id)?;
            enc.encode(on, BlitCmd::InvalidateCompressedTexture { texture })?;
        }
        RGFX_OP_BLIT_INVALIDATE_COMPRESSED_TEXTURE_SLICE => {
            let m: RgfxBlitOpInvalidateCompressedTextureSlice = body(tag, payload)?;
            let texture = objs.require_texture(m.texture_obj_id)?;
            enc.encode(
                on,
                BlitCmd::InvalidateCompressedTextureSlice {
                    texture,
                    slice: m.slice,
                    level: m.level,
                },
            )?;
        }
        RGFX_OP_BLIT_OPTIMIZE_FOR_CPU => {
            let m: RgfxBlitOpOptimizeForCpu = body(tag, payload)?;
            let texture = objs.require_texture(m.texture_obj_id)?;
            enc.encode(on, BlitCmd::OptimizeForCpu { texture })?;
        }
        RGFX_OP_BLIT_OPTIMIZE_FOR_CPU_SLICE => {
            let m: RgfxBlitOpOptimizeForCpuSlice = body(tag, payload)?;
            let texture = objs.require_texture(m.texture_obj_id)?;
            enc.encode(
                on,
                BlitCmd::OptimizeForCpuSlice {
                    texture,
                    slice: m.slice,
                    level: m.level,
                },
            )?;
        }
        RGFX_OP_BLIT_OPTIMIZE_FOR_GPU => {
            let m: RgfxBlitOpOptimizeForGpu = body(tag, payload)?;
            let texture = objs.require_texture(m.texture_obj_id)?;
            enc.encode(on, BlitCmd::OptimizeForGpu { texture })?;
        }
        RGFX_OP_BLIT_OPTIMIZE_FOR_GPU_SLICE => {
            let m: RgfxBlitOpOptimizeForGpuSlice = body(tag, payload)?;
            let texture = objs.require_texture(m.texture_obj_id)?;
            enc.encode(
                on,
                BlitCmd::OptimizeForGpuSlice {
                    texture,
                    slice: m.slice,
                    level: m.level,
                },
            )?;
        }
        RGFX_OP_BLIT_OPTIMIZE_INDIRECT_COMMAND_BUFFER => {
            let m: RgfxBlitOpOptimizeIndirectCommandBuffer = body(tag, payload)?;
            let icb = objs.require_indirect_command_buffer(m.icb_obj_id)?;
            enc.encode(
                on,
                BlitCmd::OptimizeIndirectCommandBuffer {
                    indirect_command_buffer: icb,
                    range: NsRange::new(m.range_location, m.range_length),
                },
            )?;
        }
        RGFX_OP_BLIT_RESET_COMMANDS_IN_BUFFER => {
            let m: RgfxBlitOpResetCommandsInBuffer = body(tag, payload)?;
            let icb = objs.require_indirect_command_buffer(m.icb_obj_id)?;
            enc.encode(
                on,
                BlitCmd::ResetCommandsInBuffer {
                    buffer: icb,
                    range: NsRange::new(m.range_location, m.range_length),
                },
            )?;
            objs.icb_commands_reset(m.icb_obj_id, m.range_location, m.range_length);
        }
        RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE => {
            let m: RgfxBlitOpSynchronizeResource = body(tag, payload)?;
            // A `Managed` buffer or texture of two copies: its GPU copy to its CPU copy, here.
            if matches!(
                m.resource_kind,
                RGFX_BLIT_RESOURCE_BUFFER | RGFX_BLIT_RESOURCE_TEXTURE
            ) && objs.synchronize_managed(m.resource_obj_id, None, enc, on)?
            {
                return Ok(true);
            }
            // The resource is any `id<MTLResource>`: `resource_kind` picks the
            // resolver. A nil kind (or a 0 id) is a nil resource; an unknown kind
            // faults. The resolved handle is held for the encode.
            match m.resource_kind {
                RGFX_BLIT_RESOURCE_NIL => {
                    enc.encode(on, BlitCmd::SynchronizeResource { resource: None })?;
                }
                RGFX_BLIT_RESOURCE_BUFFER => {
                    let buffer = objs.require_buffer(m.resource_obj_id)?;
                    enc.encode(
                        on,
                        BlitCmd::SynchronizeResource {
                            resource: Some(ResourceRef::Buffer(buffer)),
                        },
                    )?;
                }
                RGFX_BLIT_RESOURCE_TEXTURE => {
                    let texture = objs.require_texture(m.resource_obj_id)?;
                    enc.encode(
                        on,
                        BlitCmd::SynchronizeResource {
                            resource: Some(ResourceRef::Texture(texture)),
                        },
                    )?;
                }
                RGFX_BLIT_RESOURCE_ICB => {
                    let icb = objs.require_indirect_command_buffer(m.resource_obj_id)?;
                    enc.encode(
                        on,
                        BlitCmd::SynchronizeResource {
                            resource: Some(ResourceRef::Icb(icb)),
                        },
                    )?;
                }
                other => {
                    return Err(op_fault(format_args!(
                        "SYNCHRONIZE_RESOURCE: unknown resource kind {other}"
                    )));
                }
            }
        }
        RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE => {
            let m: RgfxBlitOpSynchronizeTexture = body(tag, payload)?;
            let which = Some((m.slice, m.level));
            if objs.synchronize_managed(m.texture_obj_id, which, enc, on)? {
                return Ok(true);
            }
            let texture = objs.require_texture(m.texture_obj_id)?;
            enc.encode(
                on,
                BlitCmd::SynchronizeTexture {
                    texture,
                    slice: m.slice,
                    level: m.level,
                },
            )?;
        }
        RGFX_OP_BLIT_UPDATE_FENCE => {
            let m: RgfxBlitOpUpdateFence = body(tag, payload)?;
            let fence = objs.require_fence(m.fence_obj_id)?;
            enc.encode(on, BlitCmd::UpdateFence { fence })?;
        }
        RGFX_OP_BLIT_WAIT_FOR_FENCE => {
            let m: RgfxBlitOpWaitForFence = body(tag, payload)?;
            let fence = objs.require_fence(m.fence_obj_id)?;
            enc.encode(on, BlitCmd::WaitForFence { fence })?;
        }
        _ => return Ok(false),
    }
    Ok(true)
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

    use reims_graphics::ops::{RGFX_OP_BODY_OFFSET, RgfxOpHeader};
    use reims_graphics::wire::{
        RGFX_CMD_BUFFER_CREATE, RGFX_CMD_FENCE_CREATE, RGFX_CMD_TEXTURE_CREATE, RgfxBufferCreate,
        RgfxMsgHeader, RgfxObjectCreate, RgfxTextureCreate, RgfxTextureDescriptor,
    };
    use reims_vgpu_core::platform::Handle;
    use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::values::IcbDesc;
    use reims_vgpu_core::testkit::FakePlatform;
    use reims_vgpu_core::testkit::calllog::Arg;
    use zerocopy::IntoBytes;

    use super::super::ObjHandle;
    use super::*;

    /// A fresh decoder with two buffers (ids 2, 3), two textures (ids 4, 5), a fence
    /// (id 6) and two indirect command buffers (ids 7, 8), plus a fake platform whose
    /// call log the tests read and an open blit encoder to drive.
    fn setup() -> (
        Arc<FakePlatform>,
        GraphicsDecoder<FakePlatform>,
        <FakePlatform as Platform>::BlitEncoder,
    ) {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );

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
        let fence = RgfxObjectCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_FENCE_CREATE,
                size: size_of::<RgfxObjectCreate>() as u32,
                seqno: 0,
            },
            obj_id: 6,
        };
        assert_eq!(dec.decode(&device, fence.as_bytes()).unwrap(), Ok(()));

        // The foundation has no ICB-create wire command yet; make the two ICBs
        // through the device and place them in the object table directly, so the
        // ICB blit ops have objects to name.
        for id in [7, 8] {
            let icb = device
                .new_indirect_command_buffer(&IcbDesc::default(), 16, 0)
                .unwrap()
                .unwrap();
            assert_eq!(
                dec.insert(id, ObjHandle::Icb(icb)).unwrap(),
                Ok::<(), super::super::DecodeError>(())
            );
        }

        let queue = device.new_command_queue().unwrap();
        let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let enc = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        (p, dec, enc)
    }

    /// One op body, as the op walk hands it to the decoder (the bytes after the
    /// header).
    fn body_of<T: IntoBytes + zerocopy::Immutable>(m: &T) -> Vec<u8> {
        m.as_bytes().to_vec()
    }

    /// The fake id of the buffer, texture, fence or ICB the decoder holds under a
    /// wire id, to compare against a logged object argument.
    fn buffer_id(dec: &GraphicsDecoder<FakePlatform>, id: u64) -> Arg {
        Arg::Obj(Some(dec.require_buffer(id).unwrap().id()))
    }
    fn texture_id(dec: &GraphicsDecoder<FakePlatform>, id: u64) -> Arg {
        Arg::Obj(Some(dec.require_texture(id).unwrap().id()))
    }
    fn fence_id(dec: &GraphicsDecoder<FakePlatform>, id: u64) -> Arg {
        Arg::Obj(Some(dec.require_fence(id).unwrap().id()))
    }
    fn icb_id(dec: &GraphicsDecoder<FakePlatform>, id: u64) -> Arg {
        Arg::Obj(Some(dec.require_indirect_command_buffer(id).unwrap().id()))
    }

    /// Every blit command decodes to its selector, resolving its references and
    /// carrying its arguments in order.
    #[test]
    fn every_blit_command_drives_its_selector() {
        let (p, dec, enc) = setup();

        // copyFromBuffer:...:toBuffer:...
        let mark = p.call_log().mark();
        let m = RgfxBlitOpCopyBufferToBuffer {
            source_buffer_obj_id: 2,
            destination_buffer_obj_id: 3,
            source_offset: 16,
            destination_offset: 32,
            size: 4096,
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c.len(), 1);
        assert_eq!(
            c[0].sel,
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"
        );
        assert_eq!(
            c[0].args,
            vec![
                buffer_id(&dec, 2),
                Arg::U64(16),
                buffer_id(&dec, 3),
                Arg::U64(32),
                Arg::U64(4096),
            ]
        );

        // copyFromBuffer:...:toTexture:... (both required)
        let mark = p.call_log().mark();
        let m = RgfxBlitOpCopyBufferToTexture {
            source_buffer_obj_id: 2,
            destination_texture_obj_id: 4,
            source_offset: 0,
            source_bytes_per_row: 256,
            source_bytes_per_image: 256 * 64,
            source_size_width: 64,
            source_size_height: 64,
            source_size_depth: 1,
            destination_slice: 0,
            destination_level: 1,
            destination_origin_x: 8,
            destination_origin_y: 8,
            destination_origin_z: 0,
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(
            c[0].sel,
            "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"
        );
        assert_eq!(c[0].args[0], buffer_id(&dec, 2));
        assert_eq!(c[0].args[4], Arg::val(&MtlSize::new(64, 64, 1)));
        assert_eq!(c[0].args[5], texture_id(&dec, 4));
        assert_eq!(c[0].args[8], Arg::val(&MtlOrigin::new(8, 8, 0)));

        // copyFromBuffer:...:toTexture:...:options: (nil-allowed, with a nil source)
        let mark = p.call_log().mark();
        let m = RgfxBlitOpCopyBufferToTextureOptions {
            source_buffer_obj_id: 0,
            destination_texture_obj_id: 4,
            options: 1,
            source_size_width: 1,
            ..RgfxBlitOpCopyBufferToTextureOptions::default()
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE_OPTIONS,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].args[0], Arg::Obj(None));
        assert_eq!(c[0].args[5], texture_id(&dec, 4));
        assert_eq!(c[0].args[9], Arg::U64(1));

        // copyFromTexture:...:toBuffer:... (both required)
        let mark = p.call_log().mark();
        let m = RgfxBlitOpCopyTextureToBuffer {
            source_texture_obj_id: 4,
            destination_buffer_obj_id: 2,
            source_slice: 1,
            source_level: 0,
            source_origin_x: 1,
            source_origin_y: 2,
            source_origin_z: 3,
            source_size_width: 16,
            source_size_height: 16,
            source_size_depth: 1,
            destination_offset: 64,
            destination_bytes_per_row: 128,
            destination_bytes_per_image: 128 * 16,
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].args[0], texture_id(&dec, 4));
        assert_eq!(c[0].args[3], Arg::val(&MtlOrigin::new(1, 2, 3)));
        assert_eq!(c[0].args[5], buffer_id(&dec, 2));

        // copyFromTexture:...:toBuffer:...:options: (nil-allowed)
        let mark = p.call_log().mark();
        let m = RgfxBlitOpCopyTextureToBufferOptions {
            source_texture_obj_id: 4,
            destination_buffer_obj_id: 0,
            options: 2,
            source_size_width: 1,
            ..RgfxBlitOpCopyTextureToBufferOptions::default()
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER_OPTIONS,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].args[0], texture_id(&dec, 4));
        assert_eq!(c[0].args[5], Arg::Obj(None));
        assert_eq!(c[0].args[9], Arg::U64(2));

        // copyFromTexture:...:toTexture:... (region form)
        let mark = p.call_log().mark();
        let m = RgfxBlitOpCopyTextureRegion {
            source_texture_obj_id: 4,
            destination_texture_obj_id: 5,
            source_size_width: 16,
            source_size_height: 16,
            source_size_depth: 1,
            destination_origin_x: 4,
            ..RgfxBlitOpCopyTextureRegion::default()
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_COPY_TEXTURE_REGION,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].args[0], texture_id(&dec, 4));
        assert_eq!(c[0].args[5], texture_id(&dec, 5));

        // copyFromTexture:...:toTexture:...:options:
        let mark = p.call_log().mark();
        let m = RgfxBlitOpCopyTextureRegionOptions {
            source_texture_obj_id: 4,
            destination_texture_obj_id: 5,
            options: 4,
            source_size_width: 16,
            ..RgfxBlitOpCopyTextureRegionOptions::default()
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_COPY_TEXTURE_REGION_OPTIONS,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].args[9], Arg::U64(4));

        // copyFromTexture:...:toTexture:...:sliceCount:levelCount:
        let mark = p.call_log().mark();
        let m = RgfxBlitOpCopyTextureSlices {
            source_texture_obj_id: 4,
            destination_texture_obj_id: 5,
            source_slice: 0,
            source_level: 0,
            destination_slice: 1,
            destination_level: 1,
            slice_count: 2,
            level_count: 3,
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_COPY_TEXTURE_SLICES,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].args[6], Arg::U64(2));
        assert_eq!(c[0].args[7], Arg::U64(3));

        // copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:
        let mark = p.call_log().mark();
        let m = RgfxBlitOpCopyIndirectCommandBuffer {
            source_obj_id: 7,
            destination_obj_id: 8,
            source_range_location: 0,
            source_range_length: 4,
            destination_index: 2,
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_COPY_INDIRECT_COMMAND_BUFFER,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].args[0], icb_id(&dec, 7));
        assert_eq!(c[0].args[1], Arg::val(&NsRange::new(0, 4)));
        assert_eq!(c[0].args[2], icb_id(&dec, 8));
        assert_eq!(c[0].args[3], Arg::U64(2));

        // fillBuffer:range:value:
        let mark = p.call_log().mark();
        let m = RgfxBlitOpFillBuffer {
            buffer_obj_id: 2,
            range_location: 0,
            range_length: 1024,
            value: 0xab,
            reserved_19: [0; 7],
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_FILL_BUFFER,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].sel, "fillBuffer:range:value:");
        assert_eq!(
            c[0].args,
            vec![
                buffer_id(&dec, 2),
                Arg::val(&NsRange::new(0, 1024)),
                Arg::U8(0xab)
            ]
        );

        // fillBuffer:range:pattern4:
        let mark = p.call_log().mark();
        let m = RgfxBlitOpFillBufferPattern4 {
            buffer_obj_id: 2,
            range_location: 8,
            range_length: 16,
            pattern: 0xdead_beef,
            reserved_1c: 0,
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_FILL_BUFFER_PATTERN4,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].args[2], Arg::U32(0xdead_beef));

        // generateMipmapsForTexture:
        let mark = p.call_log().mark();
        let m = RgfxBlitOpGenerateMipmaps { texture_obj_id: 4 };
        assert!(
            decode_op(
                RGFX_OP_BLIT_GENERATE_MIPMAPS,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].sel, "generateMipmapsForTexture:");
        assert_eq!(c[0].args[0], texture_id(&dec, 4));

        // The texture-only slice/level ops, each on texture 4.
        for (tag, sel) in [
            (
                RGFX_OP_BLIT_INVALIDATE_COMPRESSED_TEXTURE_SLICE,
                "invalidateCompressedTexture:slice:level:",
            ),
            (
                RGFX_OP_BLIT_OPTIMIZE_FOR_CPU_SLICE,
                "optimizeContentsForCPUAccess:slice:level:",
            ),
            (
                RGFX_OP_BLIT_OPTIMIZE_FOR_GPU_SLICE,
                "optimizeContentsForGPUAccess:slice:level:",
            ),
            (
                RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE,
                "synchronizeTexture:slice:level:",
            ),
        ] {
            let mark = p.call_log().mark();
            let m = RgfxBlitOpSynchronizeTexture {
                texture_obj_id: 4,
                slice: 1,
                level: 2,
            };
            assert!(decode_op(tag, &body_of(&m), &dec, &enc, OnFault::Return).unwrap());
            let c = p.call_log().since(mark);
            assert_eq!(c[0].sel, sel);
            assert_eq!(
                c[0].args,
                vec![texture_id(&dec, 4), Arg::U64(1), Arg::U64(2)]
            );
        }

        // The texture-only whole-resource ops, each on texture 4.
        for (tag, sel) in [
            (
                RGFX_OP_BLIT_INVALIDATE_COMPRESSED_TEXTURE,
                "invalidateCompressedTexture:",
            ),
            (
                RGFX_OP_BLIT_OPTIMIZE_FOR_CPU,
                "optimizeContentsForCPUAccess:",
            ),
            (
                RGFX_OP_BLIT_OPTIMIZE_FOR_GPU,
                "optimizeContentsForGPUAccess:",
            ),
        ] {
            let mark = p.call_log().mark();
            let m = RgfxBlitOpOptimizeForCpu { texture_obj_id: 4 };
            assert!(decode_op(tag, &body_of(&m), &dec, &enc, OnFault::Return).unwrap());
            let c = p.call_log().since(mark);
            assert_eq!(c[0].sel, sel);
            assert_eq!(c[0].args, vec![texture_id(&dec, 4)]);
        }

        // optimizeIndirectCommandBuffer:withRange: and resetCommandsInBuffer:withRange:
        for (tag, sel) in [
            (
                RGFX_OP_BLIT_OPTIMIZE_INDIRECT_COMMAND_BUFFER,
                "optimizeIndirectCommandBuffer:withRange:",
            ),
            (
                RGFX_OP_BLIT_RESET_COMMANDS_IN_BUFFER,
                "resetCommandsInBuffer:withRange:",
            ),
        ] {
            let mark = p.call_log().mark();
            let m = RgfxBlitOpResetCommandsInBuffer {
                icb_obj_id: 7,
                range_location: 0,
                range_length: 8,
            };
            assert!(decode_op(tag, &body_of(&m), &dec, &enc, OnFault::Return).unwrap());
            let c = p.call_log().since(mark);
            assert_eq!(c[0].sel, sel);
            assert_eq!(
                c[0].args,
                vec![icb_id(&dec, 7), Arg::val(&NsRange::new(0, 8))]
            );
        }

        // updateFence: and waitForFence:
        for (tag, sel) in [
            (RGFX_OP_BLIT_UPDATE_FENCE, "updateFence:"),
            (RGFX_OP_BLIT_WAIT_FOR_FENCE, "waitForFence:"),
        ] {
            let mark = p.call_log().mark();
            let m = RgfxBlitOpUpdateFence { fence_obj_id: 6 };
            assert!(decode_op(tag, &body_of(&m), &dec, &enc, OnFault::Return).unwrap());
            let c = p.call_log().since(mark);
            assert_eq!(c[0].sel, sel);
            assert_eq!(c[0].args, vec![fence_id(&dec, 6)]);
        }
    }

    /// `fillTexture:...:color:pixelFormat:` carries the clear colour (as `f64`) and
    /// the pixel format.
    #[test]
    fn fill_texture_color_carries_the_colour() {
        let (p, dec, enc) = setup();
        let mark = p.call_log().mark();
        let m = RgfxBlitOpFillTextureColor {
            texture_obj_id: 4,
            level: 0,
            slice: 0,
            region_origin_x: 0,
            region_origin_y: 0,
            region_origin_z: 0,
            region_size_width: 32,
            region_size_height: 32,
            region_size_depth: 1,
            color_bits: [
                0.0f64.to_bits(),
                0.25f64.to_bits(),
                0.5f64.to_bits(),
                1.0f64.to_bits(),
            ],
            pixel_format: 80,
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_FILL_TEXTURE_COLOR,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(
            c[0].sel,
            "fillTexture:level:slice:region:color:pixelFormat:"
        );
        assert_eq!(c[0].args[0], texture_id(&dec, 4));
        assert_eq!(
            c[0].args[3],
            Arg::val(&MtlRegion::new(
                MtlOrigin::new(0, 0, 0),
                MtlSize::new(32, 32, 1)
            ))
        );
        assert_eq!(
            c[0].args[4],
            Arg::val(&ClearColor {
                red: 0.0,
                green: 0.25,
                blue: 0.5,
                alpha: 1.0,
            })
        );
        assert_eq!(c[0].args[5], Arg::U64(80));
    }

    /// `fillTexture:...:bytes:length:` passes the fill bytes that trail the head, with
    /// their length.
    #[test]
    fn fill_texture_passes_the_trailing_bytes() {
        let (p, dec, enc) = setup();
        let mark = p.call_log().mark();
        let fill = [0x11u8, 0x22, 0x33, 0x44];
        let head = RgfxBlitOpFillTexture {
            texture_obj_id: 4,
            level: 0,
            slice: 0,
            region_origin_x: 0,
            region_origin_y: 0,
            region_origin_z: 0,
            region_size_width: 1,
            region_size_height: 1,
            region_size_depth: 1,
            length: fill.len() as u64,
        };
        let mut payload = head.as_bytes().to_vec();
        payload.extend_from_slice(&fill);
        assert!(
            decode_op(
                RGFX_OP_BLIT_FILL_TEXTURE,
                &payload,
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].sel, "fillTexture:level:slice:region:bytes:length:");
        assert_eq!(c[0].args[0], texture_id(&dec, 4));
        // The last argument is the length; the one before it the host pointer.
        assert_eq!(c[0].args[5], Arg::U64(fill.len() as u64));
        assert!(matches!(c[0].args[4], Arg::Ptr(p) if p != 0));
    }

    /// `synchronizeResource:` resolves its resource by kind, and nil for the nil kind.
    #[test]
    fn synchronize_resource_resolves_by_kind() {
        let (p, dec, enc) = setup();

        for (kind, obj_id) in [
            (RGFX_BLIT_RESOURCE_BUFFER, 2u64),
            (RGFX_BLIT_RESOURCE_TEXTURE, 4),
            (RGFX_BLIT_RESOURCE_ICB, 7),
        ] {
            let mark = p.call_log().mark();
            let m = RgfxBlitOpSynchronizeResource {
                resource_obj_id: obj_id,
                resource_kind: kind,
                reserved_0c: 0,
            };
            assert!(
                decode_op(
                    RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE,
                    &body_of(&m),
                    &dec,
                    &enc,
                    OnFault::Return,
                )
                .unwrap()
            );
            let c = p.call_log().since(mark);
            assert_eq!(c[0].sel, "synchronizeResource:");
            assert!(matches!(c[0].args[0], Arg::Obj(Some(_))));
        }

        // The nil kind sends a nil resource.
        let mark = p.call_log().mark();
        let m = RgfxBlitOpSynchronizeResource {
            resource_obj_id: 0,
            resource_kind: RGFX_BLIT_RESOURCE_NIL,
            reserved_0c: 0,
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].args[0], Arg::Obj(None));

        // An unknown kind faults.
        let m = RgfxBlitOpSynchronizeResource {
            resource_obj_id: 2,
            resource_kind: 99,
            reserved_0c: 0,
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .is_err()
        );
    }

    /// An unknown blit tag is not handled (the walk then faults).
    #[test]
    fn an_unknown_tag_is_unhandled() {
        let (_p, dec, enc) = setup();
        assert!(!decode_op(0x3fff, &[0u8; 8], &dec, &enc, OnFault::Return).unwrap());
    }

    /// A body too short for its op faults the exec.
    #[test]
    fn a_short_body_faults() {
        let (_p, dec, enc) = setup();
        assert!(
            decode_op(
                RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER,
                &[0u8; 8],
                &dec,
                &enc,
                OnFault::Return,
            )
            .is_err()
        );
    }

    /// A required reference the host does not hold raises; a nil-allowed reference of
    /// 0 is fine.
    #[test]
    fn a_bad_reference_raises() {
        let (p, dec, enc) = setup();

        // A required buffer the host does not hold raises.
        let m = RgfxBlitOpFillBuffer {
            buffer_obj_id: 999,
            range_location: 0,
            range_length: 8,
            value: 0,
            reserved_19: [0; 7],
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_FILL_BUFFER,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .is_err()
        );

        // A nil-allowed texture of 0 sends a nil mipmap target, no fault.
        let mark = p.call_log().mark();
        let m = RgfxBlitOpGenerateMipmaps { texture_obj_id: 0 };
        assert!(
            decode_op(
                RGFX_OP_BLIT_GENERATE_MIPMAPS,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .unwrap()
        );
        let c = p.call_log().since(mark);
        assert_eq!(c[0].args[0], Arg::Obj(None));

        // A nil-allowed texture that is non-zero but missing still raises.
        let m = RgfxBlitOpGenerateMipmaps {
            texture_obj_id: 999,
        };
        assert!(
            decode_op(
                RGFX_OP_BLIT_GENERATE_MIPMAPS,
                &body_of(&m),
                &dec,
                &enc,
                OnFault::Return,
            )
            .is_err()
        );
    }

    /// A header-tagged op steps the walk by `len`, so the body offset this module
    /// reads from is the header's end (a guard against drift in the shared offset).
    #[test]
    fn body_offset_is_the_header_end() {
        assert_eq!(RGFX_OP_BODY_OFFSET as usize, size_of::<RgfxOpHeader>());
    }
}
