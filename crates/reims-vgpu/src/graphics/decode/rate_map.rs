//! Rasterization rate maps: `RATE_MAP_CREATE`, `RATE_MAP_RESET` and `RATE_MAP_QUERY`, the
//! platform's maps under the guest's ids.
//!
//! A map is the platform's (`newRasterizationRateMapWithDescriptor:`), made from the descriptor
//! the message carries: its screen size, its mutability where the guest set it (SPI), and its
//! layers, each a layer descriptor of its sample counts whose samples are written in place through
//! the storage the platform's layer gives. Every answer of the map is the platform's: a create's
//! and a reset's state ([`RgfxRateMapState`]: the answers that take no argument), and each query
//! that takes one ([`RgfxRateMapAnswer`]: a layer's physical size, a coordinate mapped either
//! way). The parameter data never crosses the wire: `copyParameterDataToBuffer:offset:` is the
//! platform's copy into the guest's own buffer, which writes where the buffer's contents lie, the
//! guest's memory, before the answer.
//!
//! What Metal ends the process at, the platform refuses or dies at, and the op's refusal scope
//! answers it [`RGFX_STATUS_REFUSED`] (`graphics.rs`), for the guest's bundle to end its process
//! as Metal's does: a coordinate of a map of no layers or of an empty screen, physical to screen
//! of a mutable map not yet reset. A copy into a `Private` buffer or past a buffer's end, which
//! Metal's copy ends the process at too, is refused here before the platform is called.

use reims_vgpu_graphics::wire::{
    RGFX_EVT_RATE_MAP_ANSWER, RGFX_EVT_RATE_MAP_STATE, RGFX_RATE_MAP_MUTABILITY_SET,
    RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA, RGFX_RATE_MAP_QUERY_PHYSICAL_SIZE,
    RGFX_RATE_MAP_QUERY_PHYSICAL_TO_SCREEN, RGFX_RATE_MAP_QUERY_SCREEN_TO_PHYSICAL,
    RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_NOT_FOUND, RGFX_STATUS_OK, RGFX_STATUS_REFUSED,
    RgfxMsgHeader, RgfxRateMapAnswer, RgfxRateMapCreate, RgfxRateMapLayer, RgfxRateMapQuery,
    RgfxRateMapState,
};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::rate_map::{
    RateMap, RateMapDescriptor, RateMapLayerDescriptor,
};
use reims_vgpu_vulkan::core::platform::resource::{Buffer, GpuResource};
use reims_vgpu_vulkan::core::platform::types::{MtlCoordinate2D, MtlSize};
use reims_vgpu_vulkan::core::platform::values::RateMapDesc;
use reims_vgpu_vulkan::core::raise::{OnFault, RaiseResult};

use super::{GraphicsDecoder, ObjHandle, read, read_at};

/// The raises of the calls here are the op's, which its handler answers.
const ON: OnFault = OnFault::Return;

/// `MTLStorageModePrivate`: a buffer the CPU cannot write, which Metal's copy of parameter data
/// ends the process at.
const PRIVATE: u64 = 2;

/// The `f32` bits at `count` words from byte `offset` of `message`, or `None` where the message
/// does not hold them.
fn samples(message: &[u8], offset: usize, count: usize) -> Option<Vec<u32>> {
    let end = offset.checked_add(count.checked_mul(4)?)?;
    Some(
        message
            .get(offset..end)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect(),
    )
}

/// Writes `bits` at `at`, the sample storage of a layer the platform made of their count.
fn write_samples(at: *mut u32, bits: &[u32]) {
    for (k, &b) in bits.iter().enumerate() {
        // SAFETY: the platform's layer of this many samples holds them at its storage pointer,
        // which stays valid while the layer lives (`RateMapLayerDescriptor`'s doc); the caller
        // holds the layer.
        unsafe { at.add(k).write(b) };
    }
}

impl<P: Platform> GraphicsDecoder<P> {
    /// The platform's descriptor of the map `m` describes, with its layers set; `None` where the
    /// message does not hold them, or the platform makes no layer of a layer's sample counts.
    fn rate_map_descriptor(
        &self,
        message: &[u8],
        m: &RgfxRateMapCreate,
    ) -> RaiseResult<Option<P::RateMapDescriptor>> {
        let desc = self.platform.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(m.screen_width, m.screen_height, m.screen_depth),
            mutability: (m.flags & RGFX_RATE_MAP_MUTABILITY_SET != 0)
                .then_some(u64::from(m.mutability)),
            min_factor: None,
        });
        let stride = size_of::<RgfxRateMapLayer>() as u64;
        for index in 0..u64::from(m.layer_count) {
            let Some(layer) = u32::try_from(u64::from(m.layers_offset) + index * stride)
                .ok()
                .and_then(|at| read_at::<RgfxRateMapLayer>(message, at))
            else {
                return Ok(None);
            };
            let (h, v) = (
                layer.horizontal_count as usize,
                layer.vertical_count as usize,
            );
            let Some(bits) = h
                .checked_add(v)
                .and_then(|n| samples(message, layer.samples_offset as usize, n))
            else {
                return Ok(None);
            };
            let Some(made) = self.platform.new_rate_map_layer(
                ON,
                MtlSize::new(
                    layer.horizontal_count.into(),
                    layer.vertical_count.into(),
                    0,
                ),
            )?
            else {
                return Ok(None);
            };
            let counts = made.sample_count();
            if counts.width != h as u64 || counts.height != v as u64 {
                return Ok(None);
            }
            write_samples(made.horizontal_sample_storage().as_ptr(), &bits[..h]);
            write_samples(made.vertical_sample_storage().as_ptr(), &bits[h..]);
            desc.set_layer(ON, &made, index)?;
        }
        Ok(Some(desc))
    }

    /// `RATE_MAP_CREATE` or `RATE_MAP_RESET` (`reset`): the map made under the message's id, or
    /// the map it names compiled again, and its state, the answer; `None` for a message too short
    /// to carry its seqno. A create's id must be the sender's; a malformed message is answered
    /// invalid, a reset of a map the sender does not hold not found, and a map the platform does
    /// not make, or a layer it makes none of, refused, with nothing made.
    pub(crate) fn rate_map_state(
        &self,
        device: &P::Device,
        message: &[u8],
        reset: bool,
    ) -> RaiseResult<Option<RgfxRateMapState>> {
        let Some(header) = read::<RgfxMsgHeader>(message) else {
            return Ok(None);
        };
        let answer = |status| RgfxRateMapState {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_RATE_MAP_STATE,
                size: size_of::<RgfxRateMapState>() as u32,
                seqno: header.seqno,
            },
            status,
            ..RgfxRateMapState::default()
        };
        let Some(m) = read::<RgfxRateMapCreate>(message) else {
            return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT)));
        };
        if !reset && !self.owned(m.obj_id) {
            return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT)));
        }
        let held = if reset {
            match self.named(m.obj_id) {
                Some(ObjHandle::RateMap(map)) => Some(map),
                _ => return Ok(Some(answer(RGFX_STATUS_NOT_FOUND))),
            }
        } else {
            None
        };
        let Some(desc) = self.rate_map_descriptor(message, &m)? else {
            return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT)));
        };
        let state = |map: &P::RateMap| {
            let screen = map.screen_size();
            let granularity = map.physical_granularity();
            let parameters = map.parameter_buffer_size_and_align();
            RgfxRateMapState {
                screen_width: screen.width,
                screen_height: screen.height,
                screen_depth: screen.depth,
                granularity_width: granularity.width,
                granularity_height: granularity.height,
                granularity_depth: granularity.depth,
                layer_count: map.layer_count(),
                mutability: map.mutability(),
                parameter_size: parameters.size,
                parameter_align: parameters.align,
                ..answer(RGFX_STATUS_OK)
            }
        };
        if let Some(map) = held {
            map.reset(&desc);
            return Ok(Some(state(map)));
        }
        let Some(map) = device.new_rate_map(&desc)? else {
            return Ok(Some(answer(RGFX_STATUS_REFUSED)));
        };
        let made = state(&map);
        self.store(m.obj_id, super::Entry::of(ObjHandle::RateMap(map)));
        Ok(Some(made))
    }

    /// `RATE_MAP_QUERY`: the platform's answer of the map the message names, for the query it
    /// asks; `None` for a message too short to carry its seqno. A map or buffer the sender does
    /// not hold is not found, a query the wire does not define invalid, and a copy Metal ends the
    /// process at (into a `Private` buffer, or past the buffer's end) refused before the platform
    /// is called.
    pub(crate) fn rate_map_answer(&self, message: &[u8]) -> RaiseResult<Option<RgfxRateMapAnswer>> {
        let Some(header) = read::<RgfxMsgHeader>(message) else {
            return Ok(None);
        };
        let answer = |status, values| RgfxRateMapAnswer {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_RATE_MAP_ANSWER,
                size: size_of::<RgfxRateMapAnswer>() as u32,
                seqno: header.seqno,
            },
            status,
            reserved_14: 0,
            values,
        };
        let Some(m) = read::<RgfxRateMapQuery>(message) else {
            return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT, [0; 3])));
        };
        let Some(ObjHandle::RateMap(map)) = self.named(m.obj_id) else {
            return Ok(Some(answer(RGFX_STATUS_NOT_FOUND, [0; 3])));
        };
        let coordinate = MtlCoordinate2D {
            x: f32::from_bits(m.x_bits),
            y: f32::from_bits(m.y_bits),
        };
        let bits = |c: MtlCoordinate2D| [c.x.to_bits().into(), c.y.to_bits().into(), 0];
        let values = match m.query {
            RGFX_RATE_MAP_QUERY_PHYSICAL_SIZE => {
                let size = map.physical_size(m.at)?;
                [size.width, size.height, size.depth]
            }
            RGFX_RATE_MAP_QUERY_SCREEN_TO_PHYSICAL => {
                bits(map.screen_to_physical(coordinate, m.at)?)
            }
            RGFX_RATE_MAP_QUERY_PHYSICAL_TO_SCREEN => {
                bits(map.physical_to_screen(coordinate, m.at)?)
            }
            RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA => {
                let Some(ObjHandle::Buffer(buffer)) = self.named(m.buffer_obj_id) else {
                    return Ok(Some(answer(RGFX_STATUS_NOT_FOUND, [0; 3])));
                };
                let size = map.parameter_buffer_size_and_align().size;
                let fits =
                    m.at.checked_add(size)
                        .is_some_and(|end| end <= buffer.length());
                if !fits || buffer.storage_mode() == PRIVATE {
                    return Ok(Some(answer(RGFX_STATUS_REFUSED, [0; 3])));
                }
                map.copy_parameter_data(buffer, m.at)?;
                [0; 3]
            }
            _ => return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT, [0; 3]))),
        };
        Ok(Some(answer(RGFX_STATUS_OK, values)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_graphics::region::Region;
    use reims_vgpu_graphics::wire::{
        RGFX_CMD_RATE_MAP_CREATE, RGFX_CMD_RATE_MAP_QUERY, RGFX_CMD_RATE_MAP_RESET,
        RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN,
    };
    use reims_vgpu_vulkan::core::platform::rate_map::RateMap;
    use reims_vgpu_vulkan::core::testkit::FakePlatform;
    use zerocopy::{FromBytes, IntoBytes};

    use super::super::{GraphicsDecoder, ObjHandle};
    use super::*;

    /// The id of the map the tests make.
    const MAP: u64 = 0x31;

    fn decoder() -> (Arc<FakePlatform>, GraphicsDecoder<FakePlatform>) {
        let p = Arc::new(FakePlatform::new());
        let size = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
        let dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), region);
        (p, dec)
    }

    /// A create (or a reset) of a map over a `width` by `height` screen, of `layers`, each its
    /// horizontal then its vertical samples.
    fn create(
        ty: u32,
        width: u64,
        height: u64,
        mutability: Option<u32>,
        layers: &[(&[f32], &[f32])],
    ) -> Vec<u8> {
        let head = size_of::<RgfxRateMapCreate>();
        let records = layers.len() * size_of::<RgfxRateMapLayer>();
        let mut records_bytes = Vec::new();
        let mut samples_bytes = Vec::new();
        for (h, v) in layers {
            records_bytes.extend_from_slice(
                RgfxRateMapLayer {
                    horizontal_count: h.len() as u32,
                    vertical_count: v.len() as u32,
                    samples_offset: (head + records + samples_bytes.len()) as u32,
                    reserved_0c: 0,
                }
                .as_bytes(),
            );
            for s in h.iter().chain(v.iter()) {
                samples_bytes.extend_from_slice(&s.to_bits().to_le_bytes());
            }
        }
        while samples_bytes.len() % 8 != 0 {
            samples_bytes.push(0);
        }
        let m = RgfxRateMapCreate {
            header: RgfxMsgHeader {
                r#type: ty,
                size: (head + records + samples_bytes.len()) as u32,
                seqno: 7,
            },
            obj_id: MAP,
            screen_width: width,
            screen_height: height,
            screen_depth: 0,
            flags: if mutability.is_some() {
                RGFX_RATE_MAP_MUTABILITY_SET
            } else {
                0
            },
            mutability: mutability.unwrap_or(0),
            layer_count: layers.len() as u32,
            layers_offset: head as u32,
        };
        let mut bytes = m.as_bytes().to_vec();
        bytes.extend_from_slice(&records_bytes);
        bytes.extend_from_slice(&samples_bytes);
        bytes
    }

    fn query(query: u32, at: u64, buffer: u64, x: f32, y: f32) -> Vec<u8> {
        RgfxRateMapQuery {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_RATE_MAP_QUERY,
                size: size_of::<RgfxRateMapQuery>() as u32,
                seqno: 8,
            },
            obj_id: MAP,
            at,
            buffer_obj_id: buffer,
            query,
            x_bits: x.to_bits(),
            y_bits: y.to_bits(),
            reserved_34: 0,
        }
        .as_bytes()
        .to_vec()
    }

    /// A create makes the platform's map of the descriptor the message carries, its layers'
    /// samples written into the platform's layers, stores it under the guest's id and answers
    /// the platform's state of it.
    #[test]
    fn a_rate_map_create_makes_the_platform_s_map_and_answers_its_state() {
        let (p, dec) = decoder();
        let message = create(
            RGFX_CMD_RATE_MAP_CREATE,
            64,
            32,
            Some(2),
            &[(&[1.0, 0.5], &[0.25, 1.0, 0.75])],
        );
        let m = p.call_log().mark();
        let state = dec
            .rate_map_state(&p.device(), &message, false)
            .unwrap()
            .unwrap();
        assert_eq!(state.header.r#type, RGFX_EVT_RATE_MAP_STATE);
        assert_eq!(state.header.seqno, 7);
        assert_eq!(state.status, RGFX_STATUS_OK);
        assert_eq!(
            (state.screen_width, state.screen_height, state.screen_depth),
            (64, 32, 0)
        );
        assert_eq!(state.layer_count, 1);
        assert_eq!(state.mutability, 2);
        assert_eq!((state.parameter_size, state.parameter_align), (0x100, 0x10));
        let sels = p.call_log().sels_since(m);
        assert_eq!(
            sels,
            [
                "initWithSampleCount:",
                "setLayer:atIndex:",
                "newRasterizationRateMapWithDescriptor:"
            ]
        );
        let Some(ObjHandle::RateMap(map)) = dec.named(MAP) else {
            panic!("no map under the guest's id");
        };
        assert_eq!(map.layer_count(), 1);
    }

    /// The samples the message carries are what the platform's layer holds.
    #[test]
    fn a_rate_map_create_writes_the_layers_samples_in_place() {
        let (p, dec) = decoder();
        let message = create(
            RGFX_CMD_RATE_MAP_CREATE,
            8,
            8,
            None,
            &[(&[1.0, 0.5], &[0.25])],
        );
        let desc = RgfxRateMapCreate::read_from_prefix(&message).unwrap().0;
        let made = dec.rate_map_descriptor(&message, &desc).unwrap().unwrap();
        let layer = made.layer(ON, 0).unwrap().unwrap();
        let read = |at: *mut u32, n: usize| {
            // SAFETY: the layer holds `n` samples at its storage.
            (0..n)
                .map(|k| f32::from_bits(unsafe { at.add(k).read() }))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            read(layer.horizontal_sample_storage().as_ptr(), 2),
            [1.0, 0.5]
        );
        assert_eq!(read(layer.vertical_sample_storage().as_ptr(), 1), [0.25]);
        let _ = p;
    }

    /// A create whose layer records or samples run past the message is answered invalid, and
    /// makes nothing.
    #[test]
    fn a_rate_map_create_past_its_message_is_invalid_and_makes_nothing() {
        let (p, dec) = decoder();
        let mut message = create(RGFX_CMD_RATE_MAP_CREATE, 8, 8, None, &[(&[1.0], &[1.0])]);
        message.truncate(message.len() - 8);
        let state = dec
            .rate_map_state(&p.device(), &message, false)
            .unwrap()
            .unwrap();
        assert_eq!(state.status, RGFX_STATUS_INVALID_ARGUMENT);
        assert!(dec.named(MAP).is_none());
    }

    /// A reset of a map the sender does not hold is not found; of one it holds, it is the
    /// platform's reset, answered with the map's state.
    #[test]
    fn a_rate_map_reset_is_the_platform_s_of_a_map_the_sender_holds() {
        let (p, dec) = decoder();
        let reset = create(RGFX_CMD_RATE_MAP_RESET, 8, 8, None, &[(&[1.0], &[1.0])]);
        let state = dec
            .rate_map_state(&p.device(), &reset, true)
            .unwrap()
            .unwrap();
        assert_eq!(state.status, RGFX_STATUS_NOT_FOUND);
        let made = create(RGFX_CMD_RATE_MAP_CREATE, 8, 8, Some(1), &[]);
        dec.rate_map_state(&p.device(), &made, false)
            .unwrap()
            .unwrap();
        let m = p.call_log().mark();
        let state = dec
            .rate_map_state(&p.device(), &reset, true)
            .unwrap()
            .unwrap();
        assert_eq!(state.status, RGFX_STATUS_OK);
        assert_eq!(state.mutability, 1);
        let sels = p.call_log().sels_since(m);
        assert_eq!(
            sels,
            [
                "initWithSampleCount:",
                "setLayer:atIndex:",
                "resetUsingDescriptor:"
            ]
        );
    }

    /// Each query is the platform's answer of the map, a coordinate's as the bits of its `f32`.
    #[test]
    fn a_rate_map_query_is_the_platform_s_answer() {
        let (p, dec) = decoder();
        let made = create(RGFX_CMD_RATE_MAP_CREATE, 64, 32, None, &[(&[1.0], &[1.0])]);
        dec.rate_map_state(&p.device(), &made, false)
            .unwrap()
            .unwrap();
        let size = dec
            .rate_map_answer(&query(RGFX_RATE_MAP_QUERY_PHYSICAL_SIZE, 0, 0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        assert_eq!(size.header.r#type, RGFX_EVT_RATE_MAP_ANSWER);
        assert_eq!(size.status, RGFX_STATUS_OK);
        assert_eq!(size.values, [64, 32, 0]);
        let mapped = dec
            .rate_map_answer(&query(
                RGFX_RATE_MAP_QUERY_SCREEN_TO_PHYSICAL,
                0,
                0,
                3.5,
                1.25,
            ))
            .unwrap()
            .unwrap();
        assert_eq!(
            mapped.values,
            [3.5f32.to_bits().into(), 1.25f32.to_bits().into(), 0]
        );
        let unknown = dec
            .rate_map_answer(&query(9, 0, 0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        assert_eq!(unknown.status, RGFX_STATUS_INVALID_ARGUMENT);
    }

    /// A copy of the parameter data is the platform's into the sender's buffer; one past the
    /// buffer's end, or into a buffer it does not hold, is not carried out.
    #[test]
    fn a_rate_map_copies_its_parameter_data_into_the_sender_s_buffer() {
        use reims_vgpu_vulkan::core::platform::values::{BufferDesc, OptionSet};
        let (p, dec) = decoder();
        let made = create(RGFX_CMD_RATE_MAP_CREATE, 8, 8, None, &[(&[1.0], &[1.0])]);
        dec.rate_map_state(&p.device(), &made, false)
            .unwrap()
            .unwrap();
        const BUFFER: u64 = 0x32;
        let buffer = p
            .device()
            .new_buffer(BufferDesc {
                length: 0x200,
                options: vec![OptionSet::ResourceOptions(0)],
                pinned_gpu_address: None,
                contents: None,
                no_copy: false,
                deallocator: None,
                zero_from: None,
            })
            .unwrap()
            .unwrap();
        dec.insert(BUFFER, ObjHandle::Buffer(buffer))
            .unwrap()
            .unwrap();
        let copy = |at, buffer| {
            dec.rate_map_answer(&query(
                RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA,
                at,
                buffer,
                0.0,
                0.0,
            ))
            .unwrap()
            .unwrap()
            .status
        };
        let m = p.call_log().mark();
        assert_eq!(copy(0x100, BUFFER), RGFX_STATUS_OK);
        assert_eq!(copy(0x101, BUFFER), RGFX_STATUS_REFUSED);
        assert_eq!(copy(0, 0x33), RGFX_STATUS_NOT_FOUND);
        let sels = p.call_log().sels_since(m);
        assert_eq!(sels, ["copyParameterDataToBuffer:offset:"]);
    }

    /// What a map of one layer whose rates vary answers through the decoder on platform `P`: its
    /// state, its layer 0's physical size, a coordinate mapped either way, and the bytes its copy
    /// wrote into a Shared buffer of the sender's at offset 4, past 4 bytes of 0xa5.
    fn round_trip<P: crate::platform::ApiPlatform>()
    -> (RgfxRateMapState, [u64; 3], [u64; 3], [u64; 3], Vec<u8>) {
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::core::platform::values::{BufferDesc, OptionSet};
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
        );
        let made = create(
            RGFX_CMD_RATE_MAP_CREATE,
            256,
            128,
            None,
            &[(&[1.0, 0.25, 1.0], &[1.0, 0.5])],
        );
        let state = dec.rate_map_state(&device, &made, false).unwrap().unwrap();
        const BUFFER: u64 = 0x32;
        let length = state.parameter_size + 0x40;
        let buffer = device
            .new_buffer(BufferDesc {
                length,
                options: vec![OptionSet::ResourceOptions(0)],
                pinned_gpu_address: None,
                contents: None,
                no_copy: false,
                deallocator: None,
                zero_from: None,
            })
            .unwrap()
            .unwrap();
        let contents = buffer.contents().as_ptr::<u8>();
        // SAFETY: the buffer's `length` bytes, which no command uses.
        unsafe { std::ptr::write_bytes(contents, 0xa5, length as usize) };
        dec.insert(BUFFER, ObjHandle::Buffer(buffer))
            .unwrap()
            .unwrap();
        let ask = |q, at, buffer, x, y| {
            let answer = dec
                .rate_map_answer(&query(q, at, buffer, x, y))
                .unwrap()
                .unwrap();
            assert_eq!(answer.status, RGFX_STATUS_OK, "query {q}");
            answer.values
        };
        let physical = ask(RGFX_RATE_MAP_QUERY_PHYSICAL_SIZE, 0, 0, 0.0, 0.0);
        let to_physical = ask(RGFX_RATE_MAP_QUERY_SCREEN_TO_PHYSICAL, 0, 0, 200.0, 100.0);
        let to_screen = ask(RGFX_RATE_MAP_QUERY_PHYSICAL_TO_SCREEN, 0, 0, 10.0, 10.0);
        ask(RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA, 4, BUFFER, 0.0, 0.0);
        // SAFETY: as above; the copy is done.
        let bytes = unsafe { std::slice::from_raw_parts(contents, length as usize) }.to_vec();
        (state, physical, to_physical, to_screen, bytes)
    }

    /// On the Vulkan platform a map whose rates vary is compiled by the platform: its state is
    /// the runtime's (24,624 bytes of parameter data aligned to 4, a granularity of 32 by 32), its
    /// physical size is below its screen's, its coordinate maps answer the platform's, and its
    /// copy writes the parameter data where the sender's buffer's contents lie, at the offset
    /// asked and nowhere else.
    #[test]
    #[ignore = "rt builds no rasterization rate map (rt-unbuilt-rate-map), so query 0 answers 4, not 0"]
    fn on_vulkan_a_rate_map_is_the_platform_s_through_the_decoder() {
        let (state, physical, to_physical, to_screen, bytes) =
            round_trip::<reims_vgpu_vulkan::RtPlatform>();
        assert_eq!(state.status, RGFX_STATUS_OK);
        assert_eq!((state.screen_width, state.screen_height), (256, 128));
        assert_eq!(
            (
                state.granularity_width,
                state.granularity_height,
                state.granularity_depth
            ),
            (32, 32, 1)
        );
        assert_eq!(state.layer_count, 1);
        assert_eq!((state.parameter_size, state.parameter_align), (24624, 4));
        assert!(physical[0] < 256 && physical[0] > 0, "{physical:?}");
        assert!(physical[1] < 128 && physical[1] > 0, "{physical:?}");
        assert_eq!(physical[2], 1);
        let x = f32::from_bits(to_physical[0] as u32);
        assert!(x > 0.0 && x < 200.0, "{x}");
        let back = f32::from_bits(to_screen[0] as u32);
        assert!(back > 0.0, "{back}");
        assert_eq!(bytes[..4], [0xa5; 4]);
        let end = 4 + state.parameter_size as usize;
        assert!(bytes[4..end].iter().any(|&b| b != 0xa5));
        assert!(bytes[end..].iter().all(|&b| b == 0xa5));
    }

    /// A query of a map the sender does not hold is not found.
    #[test]
    fn a_rate_map_query_of_no_map_is_not_found() {
        let (_, dec) = decoder();
        let answer = dec
            .rate_map_answer(&query(RGFX_RATE_MAP_QUERY_PHYSICAL_SIZE, 0, 0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        assert_eq!(answer.status, RGFX_STATUS_NOT_FOUND);
    }
}
