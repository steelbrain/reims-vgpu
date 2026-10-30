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

use reims_graphics::wire::{
    RGFX_CMD_EVENT_CREATE, RGFX_CMD_FENCE_CREATE, RGFX_CMD_LIBRARY_CREATE, RGFX_CMD_OBJECT_DESTROY,
    RGFX_CMD_QUEUE_CREATE, RGFX_CMD_SHARED_EVENT_CREATE, RGFX_LIBRARY_BYTES_OFFSET,
    RgfxLibraryCreate, RgfxMsgHeader, RgfxObjectCreate, RgfxObjectDestroy, RgfxQueueCreate,
    RgfxSharedEventCreate,
};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::device::GpuDevice;
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
}

/// The stage-3 decoder for one device: the `obj_id` → handle table and the
/// dispatch from a wire message to a `Platform` call.
pub struct GraphicsDecoder<P: Platform> {
    objects: HashMap<u64, ObjHandle<P>>,
}

impl<P: Platform> Default for GraphicsDecoder<P> {
    fn default() -> Self {
        GraphicsDecoder {
            objects: HashMap::new(),
        }
    }
}

impl<P: Platform> GraphicsDecoder<P> {
    /// A new decoder with no objects.
    pub fn new() -> Self {
        Self::default()
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
        let mut dec = GraphicsDecoder::<FakePlatform>::new();

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

        assert_eq!(dec.object_count(), 5);
        let log = p.call_log();
        assert_eq!(log.calls_to("newEvent").len(), 1);
        assert_eq!(log.calls_to("newFence").len(), 1);
        assert_eq!(log.calls_to("newSharedEvent").len(), 1);
        assert_eq!(log.calls_to("newCommandQueue").len(), 1);
        assert_eq!(log.calls_to("newLibraryWithData:error:").len(), 1);

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
        assert_eq!(dec.object_count(), 4);
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
