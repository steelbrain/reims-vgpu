//! The stage-3 Metal-call decoder: it reads the guest bundle's serialised Metal
//! calls off a command queue's ring and drives the library's `Platform`
//! (docs/graphics-kext/design.md, "Stage 3 wire protocol"). It is generic over
//! the platform, so it drives Metal on macOS and the Vulkan platform on Linux,
//! the same trait the Apple lane's deserializer drives.
//!
//! This first slice carries out the descriptor-free **state objects** — events,
//! fences, shared events and command queues — and object destroy. Each maps
//! straight to a [`GpuDevice`] call; the decoder keeps an `obj_id` → handle
//! table, and a destroy drops the handle, releasing the object (handles are
//! refcounted). These objects need none of the resource manager's paging or
//! validity machinery, so a handle table is the whole of their lifetime; the
//! memory resources (buffers, heaps, textures), which do need that machinery,
//! and the command buffer's encoder-op stream follow in later increments and
//! reuse `reims-vgpu-core`'s `resources::manager` (see the effort's notes).
//!
//! The decoder runs under a live raise handler, as the Apple lane's command
//! processing does: a platform raise propagates out of [`GraphicsDecoder::decode`]
//! for the caller to catch.
//!
//! Not yet wired into [`super::GraphicsDevice`] — that lands with the ring loop
//! in a later increment; until then the decoder has no caller outside the tests.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use reims_graphics::wire::{
    RGFX_CMD_BUFFER_CREATE, RGFX_CMD_DEPTH_STENCIL_CREATE, RGFX_CMD_EVENT_CREATE,
    RGFX_CMD_FENCE_CREATE, RGFX_CMD_HEAP_CREATE, RGFX_CMD_LIBRARY_CREATE, RGFX_CMD_OBJECT_DESTROY,
    RGFX_CMD_QUEUE_CREATE, RGFX_CMD_SAMPLER_CREATE, RGFX_CMD_SHARED_EVENT_CREATE,
    RGFX_CMD_TEXTURE_CREATE, RGFX_CMD_TEXTURE_FROM_BUFFER, RGFX_LIBRARY_BYTES_OFFSET,
    RgfxBufferCreate, RgfxDepthStencilCreate, RgfxHeapCreate, RgfxLibraryCreate, RgfxMsgHeader,
    RgfxObjectCreate, RgfxObjectDestroy, RgfxQueueCreate, RgfxSamplerCreate, RgfxSharedEventCreate,
    RgfxStencilDescriptor, RgfxTextureCreate, RgfxTextureDescriptor, RgfxTextureFromBuffer,
};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::resource::Buffer;
use reims_vgpu_core::platform::types::Swizzle;
use reims_vgpu_core::platform::values::{
    BufferDesc, DepthStencilDesc, HeapDesc, OptionSet, SamplerDesc, StencilDesc, TextureDesc,
};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use zerocopy::FromBytes;

/// A host handle for one guest object, by kind. Later increments read these to
/// bind resources into command buffers; this slice only inserts them on a create
/// and drops them on a destroy.
enum ObjHandle<P: Platform> {
    Event(P::Event),
    Fence(P::Fence),
    SharedEvent(P::SharedEvent),
    CommandQueue(P::CommandQueue),
    Library(P::Library),
    Sampler(P::SamplerState),
    DepthStencil(P::DepthStencilState),
    Buffer(P::Buffer),
    Texture(P::Texture),
    Heap(P::Heap),
}

/// Why a message was not carried out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The message was shorter than its command, or unreadable.
    Malformed,
    /// The device has no command of that type.
    UnknownCommand,
    /// The platform made no object.
    Refused,
    /// The message named an object the host does not hold, or of the wrong kind.
    BadReference,
}

/// The stage-3 decoder for one device: the `obj_id` → handle table and the
/// dispatch from a wire message to a `Platform` call.
pub struct GraphicsDecoder<P: Platform> {
    /// The platform whose device the decoder drives, held for the descriptor
    /// builders ([`Platform::texture_descriptor`]) some per-device creates need.
    platform: Arc<P>,
    objects: HashMap<u64, ObjHandle<P>>,
}

impl<P: Platform> GraphicsDecoder<P> {
    /// A new decoder for `platform`, with no objects.
    pub fn new(platform: Arc<P>) -> Self {
        GraphicsDecoder {
            platform,
            objects: HashMap::new(),
        }
    }

    /// The objects the table holds.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Carries out one wire message against `device`: a create inserts the new
    /// handle (keyed by the guest-picked `obj_id`), a destroy drops it. The
    /// inner `Result` says whether the message was carried out; a platform raise
    /// propagates through the outer `RaiseResult`.
    pub fn decode(
        &mut self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(header) = read::<RgfxMsgHeader>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        match header.r#type {
            RGFX_CMD_OBJECT_DESTROY => {
                let Some(m) = read::<RgfxObjectDestroy>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A missing id is not an error: the guest may destroy an object
                // the host never made (a create that the platform refused).
                self.objects.remove(&m.obj_id);
                Ok(Ok(()))
            }
            RGFX_CMD_EVENT_CREATE => {
                let Some(m) = read::<RgfxObjectCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                match device.new_event(OnFault::Return)? {
                    Some(event) => self.insert(m.obj_id, ObjHandle::Event(event)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_FENCE_CREATE => {
                let Some(m) = read::<RgfxObjectCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                match device.new_fence()? {
                    Some(fence) => self.insert(m.obj_id, ObjHandle::Fence(fence)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_SHARED_EVENT_CREATE => {
                let Some(m) = read::<RgfxSharedEventCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // `initial_value` is honoured once a shared event's value can be
                // set from the host; a new shared event starts at 0 today.
                match device.new_shared_event() {
                    Some(event) => self.insert(m.obj_id, ObjHandle::SharedEvent(event)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_QUEUE_CREATE => {
                let Some(m) = read::<RgfxQueueCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // The ring pair this message places (`submit_offset`,
                // `complete_offset`, the data sizes and the vector) is set up
                // with the ring loop in a later increment; here the queue handle
                // is made so later command buffers can name it.
                match device.new_command_queue() {
                    Some(queue) => self.insert(m.obj_id, ObjHandle::CommandQueue(queue)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_LIBRARY_CREATE => {
                let Some(m) = read::<RgfxLibraryCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // The `.metallib`/AIR bytes follow the header; forward them to
                // the platform unchanged.
                let start = RGFX_LIBRARY_BYTES_OFFSET as usize;
                let Some(air) = (m.byte_length as usize)
                    .checked_add(start)
                    .and_then(|end| message.get(start..end))
                else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A nil library and a build failure both mean no usable library;
                // the compile error's text reaches the guest with the completion
                // in a later increment.
                match device.new_library(air)? {
                    Ok(Some(library)) => self.insert(m.obj_id, ObjHandle::Library(library)),
                    Ok(None) | Err(_) => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_SAMPLER_CREATE => {
                let Some(m) = read::<RgfxSamplerCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // Our wire sends the whole descriptor, so every field is set;
                // the LOD clamps travel as `f32` bits, and `flags` packs the
                // booleans. `reduction_mode` is not on the wire yet (the Metal
                // default stands).
                let desc = SamplerDesc {
                    min_filter: Some(m.min_filter.into()),
                    mag_filter: Some(m.mag_filter.into()),
                    mip_filter: Some(m.mip_filter.into()),
                    s_address_mode: Some(m.s_address_mode.into()),
                    t_address_mode: Some(m.t_address_mode.into()),
                    r_address_mode: Some(m.r_address_mode.into()),
                    border_color: Some(m.border_color.into()),
                    compare_function: Some(m.compare_function.into()),
                    max_anisotropy: Some(m.max_anisotropy.into()),
                    lod_min_clamp: Some(f32::from_bits(m.lod_min_clamp_bits)),
                    lod_max_clamp: Some(f32::from_bits(m.lod_max_clamp_bits)),
                    normalized_coordinates: Some((m.flags & 1) != 0),
                    lod_average: Some((m.flags & 2) != 0),
                    support_argument_buffers: Some((m.flags & 4) != 0),
                    reduction_mode: None,
                    resource_index: None,
                };
                match device.new_sampler_state(&desc)? {
                    Some(sampler) => self.insert(m.obj_id, ObjHandle::Sampler(sampler)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_DEPTH_STENCIL_CREATE => {
                let Some(m) = read::<RgfxDepthStencilCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                let desc = DepthStencilDesc {
                    depth_compare_function: Some(m.depth_compare.into()),
                    depth_write_enabled: Some(m.depth_write_enabled != 0),
                    front: Some(stencil(&m.front)),
                    back: Some(stencil(&m.back)),
                };
                match device.new_depth_stencil_state(&desc)? {
                    Some(state) => self.insert(m.obj_id, ObjHandle::DepthStencil(state)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_BUFFER_CREATE => {
                let Some(m) = read::<RgfxBufferCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A device-placed (Private/Shared-on-UMA) buffer. The guest picks
                // the GPU address and the host pins it; a 0 address lets the host
                // choose. `options` carries the whole MTLResourceOptions.
                let desc = BufferDesc {
                    length: m.length,
                    options: vec![OptionSet::ResourceOptions(m.options.into())],
                    pinned_gpu_address: (m.gpu_address != 0).then_some(m.gpu_address),
                    contents: None,
                    no_copy: false,
                    deallocator: None,
                };
                match device.new_buffer(desc)? {
                    Some(buffer) => self.insert(m.obj_id, ObjHandle::Buffer(buffer)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_HEAP_CREATE => {
                let Some(m) = read::<RgfxHeapCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A device-placed heap at a guest-picked GPU address (0 lets the
                // host choose). `heap_type` widens to the `MTLHeapType`. Heaps
                // over host memory land with that region's decode.
                let desc = HeapDesc {
                    size: m.size_bytes,
                    heap_type: m.heap_type as i64,
                    options: vec![OptionSet::ResourceOptions(m.options.into())],
                    address_ranges: None,
                    pinned_gpu_address: (m.gpu_address != 0).then_some(m.gpu_address),
                };
                match device.new_heap(&desc)?.heap {
                    Some(heap) => self.insert(m.obj_id, ObjHandle::Heap(heap)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_TEXTURE_CREATE => {
                let Some(m) = read::<RgfxTextureCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // Heap-placed textures land with the heap decode; this slice
                // makes device-placed textures (`heap_obj_id` 0). The guest
                // picks the resource id, pinned as the texture's resource index
                // so its argument buffers name the same texture the host made.
                if m.heap_obj_id != 0 {
                    return Ok(Err(DecodeError::Refused));
                }
                let desc = texture_desc(&m.desc, m.resource_id);
                // `texture_descriptor` forwards the raw pixel format, so a format
                // Metal refuses aborts here; the guest bundle only ever sends one
                // it read from a real descriptor.
                let descriptor = self.platform.texture_descriptor(&desc);
                match device.new_texture(OnFault::Return, &descriptor)? {
                    Some(texture) => self.insert(m.obj_id, ObjHandle::Texture(texture)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_TEXTURE_FROM_BUFFER => {
                let Some(m) = read::<RgfxTextureFromBuffer>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                let desc = texture_desc(&m.desc, m.resource_id);
                let descriptor = self.platform.texture_descriptor(&desc);
                // The texture reads the guest's named buffer; a buffer the host
                // does not hold is a guest protocol error, not a refusal.
                let made = {
                    let Some(ObjHandle::Buffer(buffer)) = self.objects.get(&m.buffer_obj_id) else {
                        return Ok(Err(DecodeError::BadReference));
                    };
                    buffer.new_texture(OnFault::Return, &descriptor, m.offset, m.bytes_per_row)?
                };
                match made {
                    Some(texture) => self.insert(m.obj_id, ObjHandle::Texture(texture)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            _ => Ok(Err(DecodeError::UnknownCommand)),
        }
    }

    /// Inserts a new object's handle under its guest-picked id.
    fn insert(
        &mut self,
        obj_id: u64,
        handle: ObjHandle<P>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        self.objects.insert(obj_id, handle);
        Ok(Ok(()))
    }
}

/// Reads a message struct from the start of `message`, or `None` if it is too
/// short.
fn read<T: FromBytes>(message: &[u8]) -> Option<T> {
    T::read_from_prefix(message).ok().map(|(t, _)| t)
}

/// The core texture descriptor values from a wire texture descriptor, with the
/// guest-picked resource id pinned as the texture's resource index (the SPI the
/// guest's argument buffers name it by). A 0 `resource_id` leaves the id to the
/// platform, and a 0 swizzle leaves Metal's identity swizzle.
fn texture_desc(d: &RgfxTextureDescriptor, resource_id: u64) -> TextureDesc {
    TextureDesc {
        texture_type: d.texture_type.into(),
        pixel_format: d.pixel_format.into(),
        width: d.width.into(),
        height: d.height.into(),
        depth: d.depth.into(),
        mipmap_level_count: d.mipmap_level_count.into(),
        sample_count: d.sample_count.into(),
        array_length: d.array_length.into(),
        usage: d.usage.into(),
        resource_options: Some(d.options.into()),
        // Four `MTLTextureSwizzle` channels packed one byte each, red lowest.
        swizzle: (d.swizzle != 0).then(|| {
            let [red, green, blue, alpha] = d.swizzle.to_le_bytes();
            Swizzle {
                red,
                green,
                blue,
                alpha,
            }
        }),
        resource_index: (resource_id != 0).then_some((resource_id, true)),
        ..TextureDesc::default()
    }
}

/// The core stencil descriptor for one face from its wire record.
fn stencil(s: &RgfxStencilDescriptor) -> StencilDesc {
    StencilDesc {
        stencil_compare_function: s.stencil_compare.into(),
        stencil_failure_operation: s.stencil_fail_op.into(),
        depth_failure_operation: s.depth_fail_op.into(),
        depth_stencil_pass_operation: s.pass_op.into(),
        read_mask: s.read_mask,
        write_mask: s.write_mask,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_core::testkit::FakePlatform;
    use zerocopy::IntoBytes;

    use super::*;

    /// A create of each state object calls the platform and holds its handle;
    /// an object destroy drops it; an unknown command and a short message are
    /// reported, not carried out.
    #[test]
    fn creates_and_destroys_the_state_objects() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p));

        let obj = |ty: u32, obj_id: u64| RgfxObjectCreate {
            header: RgfxMsgHeader {
                r#type: ty,
                size: size_of::<RgfxObjectCreate>() as u32,
                seqno: 0,
            },
            obj_id,
        };

        // An event, a fence and a shared event, each by its own command.
        assert_eq!(
            dec.decode(&device, obj(RGFX_CMD_EVENT_CREATE, 10).as_bytes())
                .unwrap(),
            Ok(())
        );
        assert_eq!(
            dec.decode(&device, obj(RGFX_CMD_FENCE_CREATE, 11).as_bytes())
                .unwrap(),
            Ok(())
        );
        let shared = RgfxSharedEventCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SHARED_EVENT_CREATE,
                size: size_of::<RgfxSharedEventCreate>() as u32,
                seqno: 0,
            },
            obj_id: 12,
            initial_value: 0,
        };
        assert_eq!(dec.decode(&device, shared.as_bytes()).unwrap(), Ok(()));

        // A command queue and its ring placement.
        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 13,
            submit_offset: 0x2_0000,
            complete_offset: 0x3_0000,
            submit_data_size: 0x1_0000,
            complete_data_size: 0x1_0000,
            priority: 1,
            vector: 1,
        };
        assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));

        // A library from its AIR bytes, which follow the header.
        let air = [1u8, 2, 3, 4, 5, 6];
        let mut lib_msg = RgfxLibraryCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_LIBRARY_CREATE,
                size: (RGFX_LIBRARY_BYTES_OFFSET as usize + air.len()).next_multiple_of(8) as u32,
                seqno: 0,
            },
            obj_id: 14,
            byte_length: air.len() as u64,
        }
        .as_bytes()
        .to_vec();
        lib_msg.extend_from_slice(&air);
        assert_eq!(dec.decode(&device, &lib_msg).unwrap(), Ok(()));

        // A sampler and a depth/stencil state, each from a full descriptor.
        let sampler = RgfxSamplerCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SAMPLER_CREATE,
                size: size_of::<RgfxSamplerCreate>() as u32,
                seqno: 0,
            },
            obj_id: 15,
            max_anisotropy: 1,
            flags: 0b101,
            lod_max_clamp_bits: f32::MAX.to_bits(),
            ..RgfxSamplerCreate::default()
        };
        assert_eq!(dec.decode(&device, sampler.as_bytes()).unwrap(), Ok(()));
        let ds = RgfxDepthStencilCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_DEPTH_STENCIL_CREATE,
                size: size_of::<RgfxDepthStencilCreate>() as u32,
                seqno: 0,
            },
            obj_id: 16,
            depth_compare: 1,
            depth_write_enabled: 1,
            front: RgfxStencilDescriptor {
                read_mask: 0xff,
                write_mask: 0xff,
                ..RgfxStencilDescriptor::default()
            },
            back: RgfxStencilDescriptor::default(),
        };
        assert_eq!(dec.decode(&device, ds.as_bytes()).unwrap(), Ok(()));

        // A device-placed buffer at a guest-picked GPU address.
        let buffer = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: 17,
            gpu_address: 0x4_0000,
            length: 65536,
            options: 0,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));

        // A device-placed render-target texture of a non-default shape, with its
        // resource id pinned and the identity swizzle.
        let texture = RgfxTextureCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_CREATE,
                size: size_of::<RgfxTextureCreate>() as u32,
                seqno: 0,
            },
            obj_id: 18,
            gpu_address: 0x5_0000,
            resource_id: 0x42,
            heap_obj_id: 0,
            heap_offset: 0,
            desc: RgfxTextureDescriptor {
                pixel_format: 80, // BGRA8Unorm
                texture_type: 2,  // 2D
                width: 1920,
                height: 1200,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 4, // RenderTarget
                options: 0,
                swizzle: 0x0403_0201, // identity (red lowest)
                reserved_2c: 0,
            },
        };
        assert_eq!(dec.decode(&device, texture.as_bytes()).unwrap(), Ok(()));

        // A texture over the buffer just made, read from an offset.
        let tex_from_buf = RgfxTextureFromBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_BUFFER,
                size: size_of::<RgfxTextureFromBuffer>() as u32,
                seqno: 0,
            },
            obj_id: 19,
            resource_id: 0x43,
            buffer_obj_id: 17,
            offset: 0,
            bytes_per_row: 1920 * 4,
            desc: RgfxTextureDescriptor {
                pixel_format: 80,
                texture_type: 2,
                width: 1920,
                height: 1200,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 1, // ShaderRead
                options: 0,
                swizzle: 0,
                reserved_2c: 0,
            },
        };
        assert_eq!(
            dec.decode(&device, tex_from_buf.as_bytes()).unwrap(),
            Ok(())
        );

        // A texture over a buffer the host never made is a bad reference, not a
        // refusal, and makes no object.
        let bad = RgfxTextureFromBuffer {
            obj_id: 20,
            buffer_obj_id: 999,
            ..tex_from_buf
        };
        assert_eq!(
            dec.decode(&device, bad.as_bytes()).unwrap(),
            Err(DecodeError::BadReference)
        );

        // A device-placed placement heap at a guest-picked address.
        let heap = RgfxHeapCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_HEAP_CREATE,
                size: size_of::<RgfxHeapCreate>() as u32,
                seqno: 0,
            },
            obj_id: 21,
            gpu_address: 0x6_0000,
            size_bytes: 1 << 20,
            options: 0,
            heap_type: 1, // Placement
        };
        assert_eq!(dec.decode(&device, heap.as_bytes()).unwrap(), Ok(()));

        assert_eq!(dec.object_count(), 11);
        let log = p.call_log();
        assert_eq!(log.calls_to("newEvent").len(), 1);
        assert_eq!(log.calls_to("newFence").len(), 1);
        assert_eq!(log.calls_to("newSharedEvent").len(), 1);
        assert_eq!(log.calls_to("newCommandQueue").len(), 1);
        assert_eq!(log.calls_to("newLibraryWithData:error:").len(), 1);
        assert_eq!(log.calls_to("newSamplerStateWithDescriptor:").len(), 1);
        assert_eq!(log.calls_to("newDepthStencilStateWithDescriptor:").len(), 1);
        assert_eq!(log.calls_to("newBufferWithDescriptor:").len(), 1);
        assert_eq!(log.calls_to("newTextureWithDescriptor:").len(), 1);
        assert_eq!(
            log.calls_to("newTextureWithDescriptor:offset:bytesPerRow:")
                .len(),
            1
        );
        assert_eq!(log.calls_to("newHeapWithDescriptor:").len(), 1);

        // Destroying the event drops its handle; the others stay.
        let destroy = RgfxObjectDestroy {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_OBJECT_DESTROY,
                size: size_of::<RgfxObjectDestroy>() as u32,
                seqno: 0,
            },
            obj_id: 10,
        };
        assert_eq!(dec.decode(&device, destroy.as_bytes()).unwrap(), Ok(()));
        assert_eq!(dec.object_count(), 10);
        // Destroying an id the host never made is not an error.
        assert_eq!(dec.decode(&device, destroy.as_bytes()).unwrap(), Ok(()));

        // An unknown command and a short message are reported.
        let unknown = RgfxMsgHeader {
            r#type: 0xffff,
            size: 16,
            seqno: 0,
        };
        assert_eq!(
            dec.decode(&device, unknown.as_bytes()).unwrap(),
            Err(DecodeError::UnknownCommand)
        );
        assert_eq!(
            dec.decode(&device, &[0u8; 4]).unwrap(),
            Err(DecodeError::Malformed)
        );
    }
}
