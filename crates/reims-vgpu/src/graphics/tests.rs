use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use reims_graphics::ops::{RGFX_COMMAND_BUFFER_OPS_OFFSET, RgfxCommandBuffer};
use reims_graphics::wire::*;
use reims_vgpu_core::platform::ObjId;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::types::CommandBufferStatus;
use reims_vgpu_core::testkit::faults::Fault;
use reims_vgpu_core::testkit::objects::CompleteOn;
use reims_vgpu_core::testkit::{Arg, FakePlatform};
use zerocopy::{FromBytes, IntoBytes};

use super::*;

/// A graphics device on the testkit's platform, with no window, raising through `raised`.
fn headless(raised: &Arc<AtomicU32>) -> GraphicsDevice<FakePlatform> {
    let raised = Arc::clone(raised);
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(Arc::new(move |vector| {
                assert_eq!(vector, 0);
                raised.fetch_add(1, Ordering::SeqCst);
            })),
            scanout_written: None,
        },
        window: None,
        memory_map: None,
    };
    GraphicsDevice::with_platform(Arc::new(FakePlatform::new()), config).unwrap()
}

/// A headless device's region is the device's, its registers read as the wire says, and the
/// guest's armed completion reaches the host's interrupt callback; its display offers 1920 ×
/// 1080 at 60 Hz
#[test]
fn a_headless_device_answers_its_guest() {
    let raised = Arc::new(AtomicU32::new(0));
    let device = headless(&raised);
    assert_eq!(device.region_size(), reims_graphics::REGION_SIZE);
    assert!(!device.region_base().is_null());
    assert_eq!(device.mmio_read(RGFX_REG_MAGIC.into()), RGFX_MAGIC);
    assert_eq!(
        device.mmio_read(RGFX_REG_REGION_SIZE_LO.into()),
        (reims_graphics::REGION_SIZE & 0xffff_ffff) as u32
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    // SAFETY: the region is the device's, mapped while it lives, and the rings' fields are at
    // their offsets; nothing else writes the guest's side.
    unsafe {
        let base = device.region_base();
        let ring = base.add(RGFX_CONTROL_SUBMIT_OFFSET as usize + 256);
        let header = RgfxMsgHeader {
            r#type: RGFX_CMD_GET_DISPLAYS,
            size: 16,
            seqno: 1,
        };
        std::ptr::copy_nonoverlapping(
            std::ptr::from_ref(&header).cast::<u8>(),
            ring,
            size_of::<RgfxMsgHeader>(),
        );
        let armed = base.add(RGFX_CONTROL_COMPLETE_OFFSET as usize + 0xc0);
        std::sync::atomic::AtomicU32::from_ptr(armed.cast()).store(1, Ordering::SeqCst);
        let producer = base.add(RGFX_CONTROL_SUBMIT_OFFSET as usize);
        std::sync::atomic::AtomicU64::from_ptr(producer.cast()).store(16, Ordering::SeqCst);
    }
    device.mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    for _ in 0..2000 {
        if raised.load(Ordering::SeqCst) > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(raised.load(Ordering::SeqCst), 1);
    // The answer: DISPLAYS, its one mode 1920 × 1080 at 60 Hz.
    // SAFETY: as above; the device published the answer before it raised.
    let mode = unsafe {
        let answer = device
            .region_base()
            .add(RGFX_CONTROL_COMPLETE_OFFSET as usize + 256 + 36);
        std::slice::from_raw_parts(answer, 12).to_vec()
    };
    let words: Vec<u32> = mode
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(words, [1920, 1080, 60_000]);
}

/// The host screen's rate in thousandths of a hertz, rounded; 60 Hz for a rate that is none
#[test]
fn a_screen_s_rate_is_kept_in_millihertz() {
    assert_eq!(hz_to_mhz(120.0), 120_000);
    assert_eq!(hz_to_mhz(59.94), 59_940);
    assert_eq!(hz_to_mhz(0.0), 60_000);
    assert_eq!(hz_to_mhz(f64::NAN), 60_000);
}

/// A headless graphics device and the platform its decoder drives, raising through `raised`.
fn headless_on(raised: &Arc<AtomicU32>) -> (GraphicsDevice<FakePlatform>, Arc<FakePlatform>) {
    let platform = Arc::new(FakePlatform::new());
    let raised = Arc::clone(raised);
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(Arc::new(move |vector| {
                assert_eq!(vector, 0);
                raised.fetch_add(1, Ordering::SeqCst);
            })),
            scanout_written: None,
        },
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(Arc::clone(&platform), config).unwrap();
    (device, platform)
}

/// The device's first call reserves the guest's texture resource ids, every id of the kext's
/// chunks, before any queue or texture on it; the plain textures that check it take ids past them
#[test]
fn the_device_reserves_the_guest_s_texture_ids_first() {
    let raised = Arc::new(AtomicU32::new(0));
    let (_device, platform) = headless_on(&raised);
    let calls = platform.call_log().calls();
    let reserve = "reserveResourceIndicesForResourceType:indices:indexCount:";
    let first = calls
        .iter()
        .position(|c| c.sel == reserve)
        .expect("a reservation");
    let ids: Vec<u64> = (1..=u64::from(RGFX_RESOURCE_ID_CHUNKS) * RGFX_RESOURCE_ID_CHUNK).collect();
    assert_eq!(ids.len(), 16320);
    assert_eq!(calls[first].args, vec![Arg::U64(0), Arg::U64s(ids)]);
    assert!(
        calls[..first]
            .iter()
            .all(|c| !c.sel.starts_with("newCommandQueue") && !c.sel.starts_with("newTexture")),
        "{:?}",
        calls[..first].iter().map(|c| c.sel).collect::<Vec<_>>()
    );
    let checks = calls
        .iter()
        .filter(|c| c.sel == "newTextureWithDescriptor:")
        .count();
    assert_eq!(checks, RESERVATION_CHECKS);
}

/// On Metal, a texture the guest places at one of its ids and the plain textures the host makes
/// never share an id, and one placed at the guest's last id, past the table a fresh device has,
/// is made and lives (Metal's heap is not corrupted). Each test runs in a process of its own, so
/// the device's ids start fresh
#[cfg(target_os = "macos")]
#[test]
fn on_metal_guest_texture_ids_are_kept_from_the_host_s() {
    use reims_vgpu_core::platform::resource::Texture as _;
    use reims_vgpu_core::platform::values::TextureDesc;
    let platform = reims_vgpu_metal::MetalPlatform::process();
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: None,
            scanout_written: None,
        },
        window: None,
        memory_map: None,
    };
    let _device = GraphicsDevice::with_platform(Arc::clone(&platform), config).unwrap();
    let gpu = platform.metal_device(None).unwrap();
    let desc = |resource_index| TextureDesc {
        pixel_format: 80,
        width: 4,
        height: 4,
        usage: 1,
        resource_options: Some(0x20),
        resource_index,
        ..TextureDesc::default()
    };
    let last = u64::from(RGFX_RESOURCE_ID_CHUNKS) * RGFX_RESOURCE_ID_CHUNK;
    let mut placed = Vec::new();
    for id in [1, 2, 65, last / 2, last - 1, last] {
        let d = platform.texture_descriptor(&desc(Some((id, true))));
        let t = gpu.new_texture(OnFault::Return, &d).unwrap().unwrap();
        assert_eq!(t.gpu_resource_id(), id);
        placed.push(t);
    }
    let plain_desc = platform.texture_descriptor(&desc(None));
    let plain: Vec<_> = (0..256)
        .map(|_| {
            gpu.new_texture(OnFault::Return, &plain_desc)
                .unwrap()
                .unwrap()
        })
        .collect();
    let inside: Vec<u64> = plain
        .iter()
        .map(|t| t.gpu_resource_id())
        .filter(|id| (1..=last).contains(id))
        .collect();
    assert!(
        inside.is_empty(),
        "plain textures took the guest's ids {inside:?}"
    );
    for (t, id) in placed.iter().zip([1, 2, 65, last / 2, last - 1, last]) {
        assert_eq!(t.gpu_resource_id(), id);
    }
}

/// The control ring's data-ring size.
const RING_DATA: u64 = RGFX_CONTROL_RING_DATA_SIZE as u64;

/// A stand-in guest driving the control rings of a headless device through its region.
struct Guest {
    base: *mut u8,
    produced: u64,
    consumed: u64,
    seqno: u64,
}

impl Guest {
    fn new(device: &GraphicsDevice<FakePlatform>) -> Guest {
        Guest {
            base: device.region_base(),
            produced: 0,
            consumed: 0,
            seqno: 0,
        }
    }

    // SAFETY (all the pointer helpers): the region is the device's, mapped at this address while
    // the device lives, and each access lies in the ring fields the wire names; nothing else
    // writes the guest's side.
    unsafe fn write(&self, off: usize, bytes: &[u8]) {
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(off), bytes.len()) };
    }
    unsafe fn read(&self, off: usize, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        unsafe { std::ptr::copy_nonoverlapping(self.base.add(off), v.as_mut_ptr(), len) };
        v
    }
    unsafe fn store_u64(&self, off: usize, val: u64) {
        unsafe { AtomicU64::from_ptr(self.base.add(off).cast()).store(val, Ordering::SeqCst) };
    }
    unsafe fn load_u64(&self, off: usize) -> u64 {
        unsafe { AtomicU64::from_ptr(self.base.add(off).cast()).load(Ordering::SeqCst) }
    }

    /// Publishes `command` on the submission ring (its `seqno` the next, its `size` its length)
    /// and rings the doorbell.
    fn send(&mut self, device: &GraphicsDevice<FakePlatform>, mut command: Vec<u8>) {
        self.seqno += 1;
        command[8..16].copy_from_slice(&self.seqno.to_le_bytes());
        let len = command.len() as u32;
        command[4..8].copy_from_slice(&len.to_le_bytes());
        let at =
            |count: u64| RGFX_CONTROL_SUBMIT_OFFSET as usize + 256 + (count % RING_DATA) as usize;
        let left = RING_DATA - self.produced % RING_DATA;
        // SAFETY: as the helpers above.
        unsafe {
            if left < command.len() as u64 {
                let mut pad = [0u8; 8];
                pad[4..8].copy_from_slice(&(left as u32).to_le_bytes());
                self.write(at(self.produced), &pad);
                self.produced += left;
            }
            self.write(at(self.produced), &command);
            self.produced += command.len() as u64;
            self.store_u64(RGFX_CONTROL_SUBMIT_OFFSET as usize, self.produced);
        }
        device.mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    }

    /// The next completion-ring event, waiting about 2 s; `None` where none came.
    fn event(&mut self) -> Option<Vec<u8>> {
        let at =
            |count: u64| RGFX_CONTROL_COMPLETE_OFFSET as usize + 256 + (count % RING_DATA) as usize;
        for _ in 0..2000 {
            // SAFETY: as the helpers above.
            let produced = unsafe { self.load_u64(RGFX_CONTROL_COMPLETE_OFFSET as usize) };
            if produced != self.consumed {
                // SAFETY: as the helpers above; the host published the message before producer.
                let header = unsafe { self.read(at(self.consumed), 8) };
                let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
                let r#type = u32::from_le_bytes(header[0..4].try_into().unwrap());
                let bytes = unsafe { self.read(at(self.consumed), size) };
                self.consumed += size as u64;
                // SAFETY: as the helpers above.
                unsafe {
                    self.store_u64(RGFX_CONTROL_COMPLETE_OFFSET as usize + 0x40, self.consumed)
                };
                if r#type == RGFX_MSG_PAD {
                    continue;
                }
                return Some(bytes);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        None
    }
}

/// An object-create message ([`RGFX_CMD_EVENT_CREATE`]) for `obj_id`.
fn event_create(obj_id: u64) -> Vec<u8> {
    RgfxObjectCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_EVENT_CREATE,
            size: size_of::<RgfxObjectCreate>() as u32,
            seqno: 0,
        },
        obj_id,
    }
    .as_bytes()
    .to_vec()
}

/// A [`RGFX_CMD_COMMAND_BUFFER`] for command buffer `cb_obj_id` on queue `queue_obj_id`, carrying
/// no ops (an empty command buffer, which still opens, commits and completes).
fn command_buffer(queue_obj_id: u64, cb_obj_id: u64) -> Vec<u8> {
    RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: RGFX_COMMAND_BUFFER_OPS_OFFSET,
            seqno: 0,
        },
        queue_obj_id,
        cb_obj_id,
        op_bytes: 0,
    }
    .as_bytes()
    .to_vec()
}

/// The device-info caps blob is written into the region at [`RGFX_DEVICE_INFO_OFFSET`] at init,
/// mirroring the GPU device's caps and features the guest bundle reads before it makes any queue;
/// on the fake (an M3 Max's answers) it answers no Apple family and no iOS feature set, as an
/// Intel or AMD Mac's GPU does, and the platform's Mac2 and macOS_GPUFamily2_v1
#[test]
fn device_info_is_written_into_the_region_at_init() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    // SAFETY: the region is the device's, mapped while it lives, and the blob lies in the
    // completion ring span's unused tail the wire names.
    let bytes = unsafe {
        let p = device.region_base().add(RGFX_DEVICE_INFO_OFFSET as usize);
        std::slice::from_raw_parts(p, size_of::<RgfxDeviceInfo>()).to_vec()
    };
    let info = RgfxDeviceInfo::read_from_bytes(&bytes).unwrap();
    let caps = platform.device().caps();
    assert_eq!(info.version, RGFX_DEVICE_INFO_VERSION);
    assert_eq!(
        u32::from(info.features_lo),
        device.mmio_read(RGFX_REG_FEATURES_LO.into()) & 0xff,
        "the features the processes read"
    );
    assert_ne!(info.features_lo, 0);
    assert_eq!(info.registry_id, caps.registry_id);
    assert!(info.name_len > 0);
    assert_eq!(
        &info.name[..info.name_len as usize],
        caps.name.as_deref().unwrap().as_bytes()
    );
    let family = |value: i64| {
        let n = info.family_count as usize;
        let k = info.family_values[..n].iter().position(|&f| f == value);
        k.map(|k| info.family_answers[k])
    };
    let feature_set = |value: u64| {
        let n = info.feature_set_count as usize;
        let k = info.feature_set_values[..n]
            .iter()
            .position(|&f| f == value);
        k.map(|k| info.feature_set_answers[k])
    };
    for apple in 1001..=1011 {
        assert_eq!(family(apple), Some(0), "family {apple}");
    }
    assert_eq!(family(2002), Some(1));
    // iOS_GPUFamily1_v1 (0) to iOS_GPUFamily5_v1 (16); macOS_GPUFamily2_v1 (10005).
    for set in 0..=16 {
        assert_eq!(feature_set(set), Some(0), "feature set {set}");
    }
    assert_eq!(feature_set(10005), Some(1));
}

/// An object-create message on ring 0 reaches the decoder: its handle is held
#[test]
fn an_object_create_reaches_the_decoder() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    assert_eq!(device.decoder_object_count(), 0);
    let mut guest = Guest::new(&device);
    guest.send(&device, event_create(obj(1, 10)));
    let mut grew = false;
    for _ in 0..2000 {
        if device.decoder_object_count() == 1 {
            grew = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(grew, "the create never reached the decoder");
}

/// A `RESOURCE_ID_CHUNK` giving `context` chunk `chunk` of the texture resource ids.
fn resource_id_chunk(context: u32, chunk: u32) -> Vec<u8> {
    RgfxResourceIdChunk {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_RESOURCE_ID_CHUNK,
            size: 0,
            seqno: 0,
        },
        context,
        chunk,
    }
    .as_bytes()
    .to_vec()
}

/// A `TEXTURE_CREATE` of a 64 x 64 BGRA8 texture `obj_id` at resource id `resource_id`.
fn texture_create(obj_id: u64, resource_id: u64) -> Vec<u8> {
    RgfxTextureCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_TEXTURE_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        resource_id,
        desc: RgfxTextureDescriptor {
            pixel_format: 80,
            texture_type: 2,
            width: 64,
            height: 64,
            depth: 1,
            mipmap_level_count: 1,
            array_length: 1,
            sample_count: 1,
            usage: 5,
            ..RgfxTextureDescriptor::default()
        },
        ..RgfxTextureCreate::default()
    }
    .as_bytes()
    .to_vec()
}

/// The kext's `RESOURCE_ID_CHUNK` on ring 0 lets its context make textures at the chunk's ids;
/// a texture at an id the context holds no chunk of, the limit's among them, is refused and the
/// device goes on; at the context's close the chunk goes with it, so the next context's texture
/// at its ids is refused until the kext gives it the chunk.
#[test]
fn a_texture_create_needs_its_context_s_chunk_of_resource_ids() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    // Ring 0 is carried out in order, so once the close of a context that holds nothing is
    // answered, everything sent before it is.
    let carried_out = |guest: &mut Guest| {
        assert_eq!(answered(&device, guest, context_close(99)), RGFX_STATUS_OK);
        device.decoder_object_count()
    };
    let first = 1 + 7 * RGFX_RESOURCE_ID_CHUNK;
    guest.send(&device, resource_id_chunk(3, 7));
    guest.send(&device, texture_create(obj(3, 1), first));
    guest.send(
        &device,
        texture_create(obj(3, 2), first + RGFX_RESOURCE_ID_CHUNK),
    );
    guest.send(&device, texture_create(obj(3, 3), RGFX_RESOURCE_ID_LIMIT));
    guest.send(&device, texture_create(obj(3, 4), 0));
    guest.send(&device, event_create(obj(3, 5)));
    assert_eq!(
        carried_out(&mut guest),
        2,
        "the texture in the chunk, and the event"
    );
    assert_eq!(
        answered(&device, &mut guest, context_close(3)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 0);
    guest.send(&device, texture_create(obj(4, 1), first));
    assert_eq!(carried_out(&mut guest), 0, "the closed context's chunk");
    guest.send(&device, resource_id_chunk(4, 7));
    guest.send(&device, texture_create(obj(4, 1), first));
    assert_eq!(carried_out(&mut guest), 1, "the chunk given again");
}

/// A command buffer on a queue is decoded and committed once ring 0 has carried out what was
/// published on it before; completing it on the GPU posts a `CB_COMPLETE` onto the queue's
/// completion ring, raising the queue's vector (0 here) when the guest armed it; the context's
/// close drops its objects and is answered
#[test]
fn a_command_buffer_submission_completes_on_the_ring() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    let context = 2;
    let mut guest = Guest::new(&device);
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(context, 1, SUBMIT_OFF, COMPLETE_OFF, RING_DATA, 0)
        ),
        RGFX_STATUS_OK
    );
    // A create on ring 0, then the buffer on the queue, which runs after it.
    guest.send(&device, event_create(obj(context, 10)));
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    ring.arm();
    ring.send(&device, command_buffer(obj(context, 1), 100));
    // The host decodes and commits asynchronously; complete the committed buffer on the GPU once
    // it is there (a no-op before it is), until the handler posts onto the completion ring.
    for _ in 0..3000 {
        platform.world().complete_all(CompleteOn::Caller);
        if ring.has_event() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let bytes = ring.event().expect("a CB_COMPLETE");
    let done = RgfxCbComplete::read_from_bytes(&bytes).unwrap();
    assert_eq!(done.header.r#type, RGFX_EVT_CB_COMPLETE);
    assert_eq!(done.cb_obj_id, 100);
    assert_eq!(done.status, RGFX_CB_OK);
    assert_eq!(raised.load(Ordering::SeqCst), 1);
    // The context's close drops its objects, the queue and the event, before it is answered.
    assert_eq!(device.decoder_object_count(), 2);
    assert_eq!(
        answered(&device, &mut guest, context_close(context)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.graphics_queue_count(), 0);
    assert_eq!(device.decoder_object_count(), 0);
}

/// Object id `n` of context `context`.
fn obj(context: u32, n: u64) -> u64 {
    (u64::from(context) << RGFX_CONTEXT_OBJ_ID_SHIFT) | n
}

/// A `QUEUE_CREATE` of queue `n` of `context` with both rings `data` bytes, MSI-X vector `vector`.
fn queue_create(
    context: u32,
    n: u64,
    submit_offset: u64,
    complete_offset: u64,
    data: u32,
    vector: u32,
) -> Vec<u8> {
    RgfxQueueCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_QUEUE_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id: obj(context, n),
        submit_offset,
        complete_offset,
        submit_data_size: data,
        complete_data_size: data,
        priority: 2,
        vector,
    }
    .as_bytes()
    .to_vec()
}

/// A `CONTEXT_CLOSE` of `context`.
fn context_close(context: u32) -> Vec<u8> {
    RgfxContextClose {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_CONTEXT_CLOSE,
            size: 0,
            seqno: 0,
        },
        context,
        reserved_14: 0,
    }
    .as_bytes()
    .to_vec()
}

/// An `OBJECT_DESTROY` of `obj_id`.
fn object_destroy(obj_id: u64) -> Vec<u8> {
    RgfxObjectDestroy {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_OBJECT_DESTROY,
            size: 0,
            seqno: 0,
        },
        obj_id,
    }
    .as_bytes()
    .to_vec()
}

/// Sends `command` (one answered by `COMPLETE`: a `QUEUE_CREATE`, a `CONTEXT_CLOSE`, a library
/// create or lookup) on ring 0, as the guest's kext does, and waits for its `COMPLETE`: its
/// status.
fn answered(device: &GraphicsDevice<FakePlatform>, guest: &mut Guest, command: Vec<u8>) -> u32 {
    guest.send(device, command);
    let bytes = guest.event().expect("a COMPLETE");
    let complete = RgfxComplete::read_from_bytes(&bytes).unwrap();
    assert_eq!(complete.header.r#type, RGFX_EVT_COMPLETE);
    assert_eq!(complete.header.seqno, guest.seqno);
    complete.status
}

/// A stand-in guest driving one command queue's own submission and completion rings, at the region
/// offsets the queue was created with.
struct QueueRing {
    base: *mut u8,
    submit_off: u64,
    complete_off: u64,
    data: u64,
    produced: u64,
    consumed: u64,
    /// The `(seqno, cb_obj_id)` of each `CB_SCHEDULED` taken off the completion ring, in order.
    scheduled: Vec<(u64, u64)>,
}

impl QueueRing {
    fn new(
        device: &GraphicsDevice<FakePlatform>,
        submit_off: u64,
        complete_off: u64,
        data: u64,
    ) -> QueueRing {
        QueueRing {
            base: device.region_base(),
            submit_off,
            complete_off,
            data,
            produced: 0,
            consumed: 0,
            scheduled: Vec::new(),
        }
    }

    // SAFETY (the pointer helpers): the region is the device's, mapped at this address while the
    // device lives, and each access lies in the ring fields at the queue's offsets; nothing else
    // writes the guest's side of this queue's rings.
    unsafe fn write(&self, off: usize, bytes: &[u8]) {
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(off), bytes.len()) };
    }
    unsafe fn read(&self, off: usize, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        unsafe { std::ptr::copy_nonoverlapping(self.base.add(off), v.as_mut_ptr(), len) };
        v
    }
    unsafe fn store_u64(&self, off: usize, val: u64) {
        unsafe { AtomicU64::from_ptr(self.base.add(off).cast()).store(val, Ordering::SeqCst) };
    }
    unsafe fn load_u64(&self, off: usize) -> u64 {
        unsafe { AtomicU64::from_ptr(self.base.add(off).cast()).load(Ordering::SeqCst) }
    }

    /// Arms the completion ring so the next completion raises the queue's vector.
    fn arm(&self) {
        // SAFETY: as the helpers above; `armed` is at 0xc0 of the completion ring header.
        unsafe {
            AtomicU32::from_ptr(self.base.add(self.complete_off as usize + 0xc0).cast())
                .store(1, Ordering::SeqCst);
        }
    }

    /// Publishes `command` (its `size` its length) on the submission ring and rings the queue's
    /// doorbell (its ring id is the submission ring's offset).
    fn send(&mut self, device: &GraphicsDevice<FakePlatform>, command: Vec<u8>) {
        self.publish(command);
        device.mmio_write(RGFX_REG_DOORBELL.into(), self.submit_off as u32);
    }

    /// Publishes `command` (its `size` its length) on the submission ring, ringing no doorbell.
    fn publish(&mut self, mut command: Vec<u8>) {
        let len = command.len() as u32;
        command[4..8].copy_from_slice(&len.to_le_bytes());
        let (submit_off, data) = (self.submit_off, self.data);
        let at = |count: u64| submit_off as usize + 256 + (count % data) as usize;
        let left = self.data - self.produced % self.data;
        // SAFETY: as the helpers above.
        unsafe {
            if left < command.len() as u64 {
                let mut pad = [0u8; 8];
                pad[4..8].copy_from_slice(&(left as u32).to_le_bytes());
                self.write(at(self.produced), &pad);
                self.produced += left;
            }
            self.write(at(self.produced), &command);
            self.produced += command.len() as u64;
            self.store_u64(self.submit_off as usize, self.produced);
        }
    }

    /// Whether the completion ring has an event the guest has not taken, other than a
    /// `CB_SCHEDULED`: those at its head are taken into `scheduled`.
    fn has_event(&mut self) -> bool {
        let (complete_off, data) = (self.complete_off, self.data);
        let at = |count: u64| complete_off as usize + 256 + (count % data) as usize;
        // SAFETY (this block): as the helpers above; `producer` is at the ring's start, and the
        // host published each message before producer.
        unsafe {
            while self.load_u64(self.complete_off as usize) != self.consumed {
                let header = self.read(at(self.consumed), 8);
                let r#type = u32::from_le_bytes(header[0..4].try_into().unwrap());
                let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
                match r#type {
                    RGFX_MSG_PAD => {}
                    RGFX_EVT_CB_SCHEDULED => {
                        let bytes = self.read(at(self.consumed), size);
                        let scheduled = RgfxCbScheduled::read_from_bytes(&bytes).unwrap();
                        self.scheduled
                            .push((scheduled.header.seqno, scheduled.cb_obj_id));
                    }
                    _ => return true,
                }
                self.consumed += size as u64;
                self.store_u64(self.complete_off as usize + 0x40, self.consumed);
            }
        }
        false
    }

    /// The next completion-ring event other than a `CB_SCHEDULED` (see `has_event`), waiting
    /// about 2 s; `None` where none came.
    fn event(&mut self) -> Option<Vec<u8>> {
        let (complete_off, data) = (self.complete_off, self.data);
        let at = |count: u64| complete_off as usize + 256 + (count % data) as usize;
        for _ in 0..2000 {
            if self.has_event() {
                // SAFETY: as the helpers above; `has_event` stopped at a published message.
                let header = unsafe { self.read(at(self.consumed), 8) };
                let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
                let bytes = unsafe { self.read(at(self.consumed), size) };
                self.consumed += size as u64;
                // SAFETY: as the helpers above.
                unsafe { self.store_u64(self.complete_off as usize + 0x40, self.consumed) };
                return Some(bytes);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        None
    }
}

/// A command queue created with valid ring parameters gets its own submission and completion rings
/// and a service thread; a command buffer submitted on its submission ring is decoded onto the
/// platform and committed, and completing it on the GPU posts a `CB_COMPLETE` onto the queue's own
/// completion ring, raising that queue's MSI-X vector
#[test]
fn a_per_queue_ring_decodes_and_completes_a_command_buffer() {
    let vectors = Arc::new(super::Mutex::new(Vec::<u32>::new()));
    let platform = Arc::new(FakePlatform::new());
    let vrec = Arc::clone(&vectors);
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(Arc::new(move |vector| vrec.lock().push(vector))),
            scanout_written: None,
        },
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(Arc::clone(&platform), config).unwrap();
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);

    // The queue's two rings live in the guest's allocations, well clear of each other.
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const RING_DATA: u32 = 0x1_0000;

    // Create the queue (queue 1 of context 3) with its own rings and MSI-X vector 1, on ring 0: it
    // is there when the create is answered.
    let context = 3;
    let queue = obj(context, 1);
    let mut guest = Guest::new(&device);
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(context, 1, SUBMIT_OFF, COMPLETE_OFF, RING_DATA, 1)
        ),
        RGFX_STATUS_OK
    );
    assert_eq!(device.graphics_queue_count(), 1);

    // Submit a command buffer (obj_id 100) on the queue's own submission ring, with its completion
    // ring armed so the completion raises the queue's vector.
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    ring.arm();
    ring.send(&device, command_buffer(queue, 100));

    // The host decodes and commits asynchronously; complete the committed buffer on the GPU once it
    // is there until the handler posts onto the queue's completion ring.
    for _ in 0..3000 {
        platform.world().complete_all(CompleteOn::Caller);
        if ring.has_event() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let bytes = ring.event().expect("a CB_COMPLETE on the queue ring");
    let done = RgfxCbComplete::read_from_bytes(&bytes).unwrap();
    assert_eq!(done.header.r#type, RGFX_EVT_CB_COMPLETE);
    assert_eq!(done.cb_obj_id, 100);
    assert_eq!(done.status, RGFX_CB_OK);
    // Its scheduled record came ahead of it on the same ring (`event` takes only those at the
    // ring's head aside).
    assert_eq!(ring.scheduled, [(done.header.seqno, 100)]);
    // The queue's own vector was raised, not the control vector.
    assert_eq!(*vectors.lock(), vec![1]);

    // Destroying the queue ends its per-queue service, and its handle goes.
    guest.send(&device, object_destroy(queue));
    let mut gone = false;
    for _ in 0..2000 {
        if device.graphics_queue_count() == 0 && device.decoder_object_count() == 0 {
            gone = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(gone, "the queue's destroy never ended it");
}

/// A command buffer that asks for its `CB_SUBMITTED` and need wait for nothing, rung through
/// `mmio_write_taking`, is taken on the ringing thread: its `CB_SUBMITTED` is on its queue's
/// completion ring, the queue's vector raised on that thread, before the write returns; its queue's
/// thread walks and commits it, and it completes as one that thread took. One that does not ask is
/// left to the queue's thread, and completes as before
#[test]
fn a_doorbell_write_takes_a_command_buffer_on_its_thread() {
    use reims_graphics::ops::{RGFX_OP_SUBMITTED, RgfxOpHeader};

    let raised = Arc::new(super::Mutex::new(Vec::<(u32, Option<String>)>::new()));
    let platform = Arc::new(FakePlatform::new());
    let record = Arc::clone(&raised);
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(Arc::new(move |vector| {
                let name = std::thread::current().name().map(str::to_owned);
                record.lock().push((vector, name));
            })),
            scanout_written: None,
        },
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(Arc::clone(&platform), config).unwrap();
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const RING_DATA: u32 = 0x1_0000;
    let (context, mut guest) = (3, Guest::new(&device));
    let queue = obj(context, 1);
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(context, 1, SUBMIT_OFF, COMPLETE_OFF, RING_DATA, 1)
        ),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    // The queue's thread sleeps with the ring idle, as the guest rings for a buffer.
    for _ in 0..2000 {
        // SAFETY: as `QueueRing`'s helpers; `consumer_idle` is at 0x80 of the submission ring.
        let idle = unsafe {
            AtomicU32::from_ptr(device.region_base().add(SUBMIT_OFF as usize + 0x80).cast())
                .load(Ordering::SeqCst)
        };
        if idle == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    ring.arm();

    let mut asking = command_buffer(queue, 100);
    let submitted = RgfxOpHeader {
        tag: RGFX_OP_SUBMITTED,
        flags: 0,
        len: 8,
    };
    asking.extend_from_slice(submitted.as_bytes());
    asking[RGFX_COMMAND_BUFFER_OPS_OFFSET as usize - 8..RGFX_COMMAND_BUFFER_OPS_OFFSET as usize]
        .copy_from_slice(&8u64.to_le_bytes());
    ring.publish(asking);
    device.mmio_write_taking(RGFX_REG_DOORBELL.into(), SUBMIT_OFF as u32);
    assert!(
        ring.has_event(),
        "the CB_SUBMITTED is posted before the write returns"
    );
    let bytes = ring.event().unwrap();
    let posted = RgfxCbSubmitted::read_from_bytes(&bytes).unwrap();
    assert_eq!(
        (posted.header.r#type, posted.cb_obj_id),
        (RGFX_EVT_CB_SUBMITTED, 100)
    );
    let here = std::thread::current().name().map(str::to_owned);
    assert_eq!(raised.lock().first(), Some(&(1, here)));

    let complete = |ring: &mut QueueRing| {
        for _ in 0..3000 {
            platform.world().complete_all(CompleteOn::Caller);
            if ring.has_event() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        RgfxCbComplete::read_from_bytes(&ring.event().expect("a CB_COMPLETE")).unwrap()
    };
    let done = complete(&mut ring);
    assert_eq!((done.cb_obj_id, done.status), (100, RGFX_CB_OK));

    ring.publish(command_buffer(queue, 101));
    device.mmio_write_taking(RGFX_REG_DOORBELL.into(), SUBMIT_OFF as u32);
    let done = complete(&mut ring);
    assert_eq!((done.cb_obj_id, done.status), (101, RGFX_CB_OK));
    assert_eq!(ring.scheduled, [(0, 100), (0, 101)]);
}

/// The window shows the guest's cursor once it has an image and while the guest shows it,
/// telling the window only of a change
#[test]
fn the_cursor_shows_with_an_image_while_the_guest_shows_it() {
    let window = Window::new(WindowConfig::default(), 0, "test".to_owned());
    let mut cursor = CursorState {
        image: false,
        visible: true,
        shown: None,
    };
    cursor.update(&window);
    assert_eq!(cursor.shown, Some(false));
    cursor.image = true;
    cursor.update(&window);
    assert_eq!(cursor.shown, Some(true));
    cursor.visible = false;
    cursor.update(&window);
    assert_eq!(cursor.shown, Some(false));
    window.close();
}

/// The window's sink imports nothing at its making, then only the host pages under the scanout
/// it draws (to `stride × height` bytes on), again only when the scanout moves or changes size,
/// and kept to the region
#[test]
fn the_sink_imports_only_the_scanout_s_pages() {
    use reims_vgpu_core::platform::resource::Buffer as _;

    const IMPORT: &str = "newBufferWithBytesNoCopy:length:options:deallocator:";
    let platform = Arc::new(FakePlatform::new());
    let region = Arc::new(Region::new(reims_graphics::REGION_SIZE as usize).unwrap());
    let shown = Arc::new(Shown::default());
    let presented = Arc::new(Presented::<FakePlatform>::default());
    let draw = sink::ScanoutDraw::new(Arc::clone(&platform), &region, &shown, &presented).unwrap();
    assert_eq!(platform.call_log().count(IMPORT), 0);

    // SAFETY: sysconf reads a constant of the host's.
    let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap();
    let base = region.base() as u64;
    let addr = |b: &<FakePlatform as Platform>::Buffer| b.contents().addr() as u64;
    let scanout = |width: u32, height: u32, offset: u64| Scanout {
        width,
        height,
        stride: (width * 4).next_multiple_of(256),
        offset,
    };
    let at = RGFX_ALLOCATIONS_OFFSET;
    let full_hd = scanout(1920, 1080, at);
    let (buffer, source) = draw.scanout_buffer(&full_hd).unwrap();
    assert_eq!(source, 0);
    assert_eq!(addr(&buffer), base + at);
    assert_eq!(buffer.length(), (7680 * 1080u64).next_multiple_of(page));
    assert_eq!(platform.call_log().count(IMPORT), 1);
    // The same scanout again: the same import.
    draw.scanout_buffer(&full_hd).unwrap();
    assert_eq!(platform.call_log().count(IMPORT), 1);

    // Moved off a page boundary: the import starts at its page, the copy past it.
    let moved = scanout(640, 480, at + page + 0x100);
    let (buffer, source) = draw.scanout_buffer(&moved).unwrap();
    assert_eq!(source, 0x100);
    assert_eq!(addr(&buffer), base + at + page);
    assert_eq!(
        buffer.length(),
        (0x100 + 2560 * 480u64).next_multiple_of(page)
    );
    assert_eq!(platform.call_log().count(IMPORT), 2);

    // At the region's end, the rows' padding past it is not imported.
    let size = region.size();
    let last = scanout(600, 4, size - 2560 * 3 - 2400);
    let (buffer, source) = draw.scanout_buffer(&last).unwrap();
    assert_eq!(addr(&buffer) + source, base + last.offset);
    assert_eq!(addr(&buffer) + buffer.length(), base + size);
    assert_eq!(platform.call_log().count(IMPORT), 3);
}

/// The display pipe's present: a surface the decoder holds no texture over is answered invalid
/// with nothing shown, and 0 (the scanout) is answered at once on a device with no window
#[test]
fn a_present_needs_the_surface_s_texture() {
    let raised = Arc::new(AtomicU32::new(0));
    let device = headless(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let present = |id: u32| {
        RgfxPresentSurface {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_PRESENT_SURFACE,
                size: 32,
                seqno: 0,
            },
            display: 0,
            iosurface_id: id,
            transaction_id: 1,
            reserved_1c: 0,
        }
        .as_bytes()
        .to_vec()
    };
    for (id, status) in [(0x77, RGFX_STATUS_INVALID_ARGUMENT), (0, RGFX_STATUS_OK)] {
        guest.send(&device, present(id));
        let bytes = guest.event().expect("a COMPLETE");
        let done = RgfxComplete::read_from_bytes(&bytes).unwrap();
        assert_eq!(done.header.r#type, RGFX_EVT_COMPLETE);
        assert_eq!(done.status, status, "IOSurface {id}");
        assert!(device.control.presented.texture.lock().is_none());
    }
}

/// A `LIBRARY_CREATE` of library `obj_id`, whose bytes the fake platform takes as a library.
fn library_create(obj_id: u64) -> Vec<u8> {
    keyed_library_create(obj_id, &[1, 2, 3, 4], None)
}

/// A `LIBRARY_CREATE` of library `obj_id` from `bytes`, with `key` where there is one.
fn keyed_library_create(obj_id: u64, bytes: &[u8], key: Option<[u8; 32]>) -> Vec<u8> {
    let mut message = RgfxLibraryCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_LIBRARY_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        byte_length: bytes.len() as u64,
        flags: if key.is_some() {
            RGFX_LIBRARY_FLAG_KEYED
        } else {
            0
        },
        reserved_24: 0,
        key: key.unwrap_or_default(),
    }
    .as_bytes()
    .to_vec();
    message.extend_from_slice(bytes);
    message.resize(message.len().next_multiple_of(8), 0);
    message
}

/// A `LIBRARY_LOOKUP` of library `obj_id` by `key`, for `size` bytes.
fn library_lookup(obj_id: u64, size: usize, key: [u8; 32]) -> Vec<u8> {
    RgfxLibraryLookup {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_LIBRARY_LOOKUP,
            size: 0,
            seqno: 0,
        },
        obj_id,
        slice_size: size as u64,
        key,
    }
    .as_bytes()
    .to_vec()
}

/// A `.metallib` of two functions whose `HASH` tags are their bitcode's, and its key.
fn keyed_metallib() -> (Vec<u8>, [u8; 32]) {
    let bitcode = b"vertex-bitcode fragment-bitcode";
    let bytes = decode::test_metallib(
        bitcode,
        &[
            (0, 15, decode::test_sha(&bitcode[..15])),
            (15, 16, decode::test_sha(&bitcode[15..])),
        ],
    );
    let key = decode::test_key(&bytes).expect("the test library's key");
    (bytes, key)
}

/// Makes library `obj_id` as the guest's bundle does: its create, then the wait for its
/// `COMPLETE`, which must be OK.
fn make_library(device: &GraphicsDevice<FakePlatform>, guest: &mut Guest, obj_id: u64) {
    assert_eq!(
        answered(device, guest, library_create(obj_id)),
        RGFX_STATUS_OK
    );
}

/// A `RENDER_PIPELINE_CREATE` of pipeline `obj_id` from functions `vmain` and `fmain` of library
/// `library`, colour attachment 0 BGRA8Unorm.
fn render_pipeline_create(obj_id: u64, library: u64) -> Vec<u8> {
    let head = size_of::<RgfxRenderPipelineCreate>() as u32;
    let (vertex, fragment) = (b"vmain".as_slice(), b"fmain".as_slice());
    let mut colors = [RgfxColorAttachmentDesc::default(); 8];
    colors[0].pixel_format = 80;
    colors[0].write_mask = 0xf;
    let m = RgfxRenderPipelineCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_RENDER_PIPELINE_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        vertex_library_obj_id: library,
        fragment_library_obj_id: library,
        vertex_name_offset: head,
        vertex_name_len: vertex.len() as u32,
        fragment_name_offset: head + vertex.len() as u32,
        fragment_name_len: fragment.len() as u32,
        raster_sample_count: 1,
        color_attachment_count: 1,
        colors,
        ..RgfxRenderPipelineCreate::default()
    };
    let mut bytes = m.as_bytes().to_vec();
    bytes.extend_from_slice(vertex);
    bytes.extend_from_slice(fragment);
    bytes.resize(bytes.len().next_multiple_of(8), 0);
    bytes
}

/// A `COMPUTE_PIPELINE_CREATE` of pipeline `obj_id` from function `cmain` of library `library`.
fn compute_pipeline_create(obj_id: u64, library: u64) -> Vec<u8> {
    let head = size_of::<RgfxComputePipelineCreate>() as u32;
    let name = b"cmain".as_slice();
    let m = RgfxComputePipelineCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMPUTE_PIPELINE_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        function_library_obj_id: library,
        name_offset: head,
        name_len: name.len() as u32,
        ..RgfxComputePipelineCreate::default()
    };
    let mut bytes = m.as_bytes().to_vec();
    bytes.extend_from_slice(name);
    bytes.resize(bytes.len().next_multiple_of(8), 0);
    bytes
}

/// A `COMMAND_BUFFER` `cb_obj_id` on queue `queue_obj_id` whose one compute encoder sets compute
/// pipeline `pipeline`.
fn compute_command_buffer(queue_obj_id: u64, cb_obj_id: u64, pipeline: u64) -> Vec<u8> {
    use reims_graphics::ops::compute::{
        RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
    };
    use reims_graphics::ops::{
        RGFX_OP_ENCODER_BEGIN_COMPUTE, RGFX_OP_ENCODER_END, RgfxComputePassDesc, RgfxOpHeader,
    };
    let op = |tag: u16, body: &[u8]| {
        let mut bytes = RgfxOpHeader {
            tag,
            flags: 0,
            len: (8 + body.len()) as u32,
        }
        .as_bytes()
        .to_vec();
        bytes.extend_from_slice(body);
        bytes
    };
    let mut ops = op(
        RGFX_OP_ENCODER_BEGIN_COMPUTE,
        RgfxComputePassDesc::default().as_bytes(),
    );
    ops.extend(op(
        RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
        RgfxComputeOpSetComputePipelineState {
            pipeline_obj_id: pipeline,
        }
        .as_bytes(),
    ));
    ops.extend(op(RGFX_OP_ENCODER_END, &[]));
    let mut bytes = RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: 0,
            seqno: 0,
        },
        queue_obj_id,
        cb_obj_id,
        op_bytes: ops.len() as u64,
    }
    .as_bytes()
    .to_vec();
    bytes.extend(ops);
    bytes
}

/// The next event of ring 0's completion ring, which must be of `type`.
fn next_event(guest: &mut Guest, r#type: u32) -> Vec<u8> {
    let bytes = guest.event().expect("an event on ring 0");
    let (header, _) = RgfxMsgHeader::read_from_prefix(&bytes).unwrap();
    assert_eq!(header.r#type, r#type);
    bytes
}

/// Completes the platform's committed command buffers until `ring` has an event, about 3 s.
fn complete_until_event(platform: &FakePlatform, ring: &mut QueueRing) -> bool {
    for _ in 0..3000 {
        platform.world().complete_all(CompleteOn::Caller);
        if ring.has_event() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    false
}

/// One process's pipeline compile holds nothing of another's: while context 1's render pipeline
/// compiles, context 2's queue is created and its command buffer runs and completes; context 1's
/// reflection comes once the compile ends
#[test]
fn another_context_s_command_buffer_is_not_held_by_a_slow_compile() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    make_library(&device, &mut guest, obj(1, 10));
    device.control.pool.hold();
    guest.send(&device, render_pipeline_create(obj(1, 11), obj(1, 10)));
    let create = guest.seqno;

    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(2, 1, SUBMIT_OFF, COMPLETE_OFF, RING_DATA, 0)
        ),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    ring.send(&device, command_buffer(obj(2, 1), 100));
    assert!(
        complete_until_event(&platform, &mut ring),
        "context 2's command buffer was held by context 1's compile"
    );
    let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!(done.cb_obj_id, 100);
    assert_eq!(done.status, RGFX_CB_OK);
    // The library and the queue; the pipeline is still compiling.
    assert_eq!(device.decoder_object_count(), 2);

    device.control.pool.release();
    let bytes = next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    let reflection = RgfxPipelineReflection::read_from_bytes(&bytes).unwrap();
    assert_eq!(reflection.header.seqno, create);
    assert_eq!(reflection.obj_id, obj(1, 11));
    assert_eq!(reflection.status, RGFX_PIPELINE_OK);
    assert_eq!(device.decoder_object_count(), 3);
}

/// A display command behind a pipeline create on ring 0 is answered while the pipeline compiles
#[test]
fn a_display_command_behind_a_slow_compile_is_answered() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    make_library(&device, &mut guest, obj(1, 10));
    device.control.pool.hold();
    guest.send(&device, compute_pipeline_create(obj(1, 11), obj(1, 10)));
    let create = guest.seqno;
    guest.send(
        &device,
        RgfxMsgHeader {
            r#type: RGFX_CMD_GET_DISPLAYS,
            size: 0,
            seqno: 0,
        }
        .as_bytes()
        .to_vec(),
    );
    let displays = next_event(&mut guest, RGFX_EVT_DISPLAYS);
    let (header, _) = RgfxMsgHeader::read_from_prefix(&displays).unwrap();
    assert_eq!(header.seqno, guest.seqno);

    device.control.pool.release();
    let bytes = next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    let reflection = RgfxPipelineReflection::read_from_bytes(&bytes).unwrap();
    assert_eq!(reflection.header.seqno, create);
    assert_eq!(reflection.status, RGFX_PIPELINE_OK);
}

/// A command buffer that names a pipeline still compiling waits for it, then runs with it
#[test]
fn a_command_buffer_naming_a_pending_pipeline_waits_for_it() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(3, 1, SUBMIT_OFF, COMPLETE_OFF, RING_DATA, 0)
        ),
        RGFX_STATUS_OK
    );
    make_library(&device, &mut guest, obj(3, 10));
    device.control.pool.hold();
    guest.send(&device, compute_pipeline_create(obj(3, 11), obj(3, 10)));
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    ring.send(&device, compute_command_buffer(obj(3, 1), 100, obj(3, 11)));
    for _ in 0..100 {
        platform.world().complete_all(CompleteOn::Caller);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        !ring.has_event(),
        "the command buffer ran before the pipeline it names was made"
    );

    device.control.pool.release();
    assert!(
        complete_until_event(&platform, &mut ring),
        "the command buffer never ran"
    );
    let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!(done.cb_obj_id, 100);
    assert_eq!(done.status, RGFX_CB_OK);
    next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
}

/// An `OBJECT_DESTROY` of a pipeline still compiling keeps its place in ring order: the pipeline
/// is not stored when its compile ends, and its create is still answered
#[test]
fn a_destroy_of_a_pending_pipeline_cancels_it() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    make_library(&device, &mut guest, obj(1, 10));
    device.control.pool.hold();
    guest.send(&device, compute_pipeline_create(obj(1, 11), obj(1, 10)));
    guest.send(&device, object_destroy(obj(1, 11)));
    // The destroy is carried out before the GET_DISPLAYS after it is answered.
    guest.send(
        &device,
        RgfxMsgHeader {
            r#type: RGFX_CMD_GET_DISPLAYS,
            size: 0,
            seqno: 0,
        }
        .as_bytes()
        .to_vec(),
    );
    next_event(&mut guest, RGFX_EVT_DISPLAYS);
    device.control.pool.release();
    next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    assert_eq!(device.decoder_object_count(), 1);
}

/// A heap texture size query on ring 0 is answered on the control completion ring, at its seqno,
/// with the platform's size and alignment for the shape, and makes no object
#[test]
fn a_heap_texture_size_query_is_answered_on_ring_0() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let query = RgfxHeapTextureSizeAndAlign {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN,
            size: 0,
            seqno: 0,
        },
        obj_id: obj(1, 3),
        desc: RgfxTextureDescriptor {
            pixel_format: 80,
            texture_type: 2,
            width: 256,
            height: 128,
            depth: 1,
            mipmap_level_count: 1,
            array_length: 1,
            sample_count: 1,
            ..RgfxTextureDescriptor::default()
        },
    };
    guest.send(&device, query.as_bytes().to_vec());
    let bytes = guest.event().expect("a SIZE_AND_ALIGN");
    let answer = RgfxSizeAndAlign::read_from_bytes(&bytes).unwrap();
    assert_eq!(answer.header.r#type, RGFX_EVT_SIZE_AND_ALIGN);
    assert_eq!(answer.header.seqno, guest.seqno);
    // The fake's size: 4 bytes a texel, at its 16 KiB alignment.
    assert_eq!(
        (answer.status, answer.size, answer.align),
        (RGFX_STATUS_OK, 0x2_0000, 0x4000)
    );
    assert_eq!(
        platform
            .call_log()
            .calls_to("heapTextureSizeAndAlignWithDescriptor:")
            .len(),
        1
    );
    assert_eq!(device.decoder_object_count(), 0);
}

/// A sampler create on ring 0 is answered on the control completion ring, at its seqno, with
/// the sampler's `gpuResourceID`: the platform's, which a command buffer naming the sampler finds
#[test]
fn a_sampler_create_is_answered_on_ring_0() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let create = RgfxSamplerCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SAMPLER_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id: obj(1, 4),
        max_anisotropy: 1,
        flags: 0b101,
        lod_max_clamp_bits: f32::MAX.to_bits(),
        ..RgfxSamplerCreate::default()
    };
    guest.send(&device, create.as_bytes().to_vec());
    let bytes = guest.event().expect("a SAMPLER_STATE");
    let answer = RgfxSamplerState::read_from_bytes(&bytes).unwrap();
    assert_eq!(answer.header.r#type, RGFX_EVT_SAMPLER_STATE);
    assert_eq!(answer.header.seqno, guest.seqno);
    assert_eq!(answer.status, RGFX_STATUS_OK);
    assert_ne!(answer.gpu_resource_id, 0);
    assert_eq!(
        platform
            .call_log()
            .calls_to("newSamplerStateWithDescriptor:")
            .len(),
        1
    );
    assert_eq!(device.decoder_object_count(), 1);
}

/// An argument layout query on ring 0 writes the platform's layout into the sender's block and is
/// answered by a `COMPLETE` at its seqno, making no object; one whose block is another process's
/// is refused, with nothing written
#[test]
fn an_argument_layout_query_is_answered_on_ring_0() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SIZE: u64 = 0x1000;
    guest.send(&device, block_grant(1, BLOCK, SIZE));
    let descriptors = [
        RgfxArgumentDescriptor {
            data_type: 58, // MTLDataTypeTexture
            index: 0,
            texture_type: 2,
            ..RgfxArgumentDescriptor::default()
        },
        RgfxArgumentDescriptor {
            data_type: 60, // MTLDataTypePointer
            index: 1,
            array_length: 4,
            ..RgfxArgumentDescriptor::default()
        },
    ];
    let query = |obj_id: u64| {
        let head = RgfxArgumentLayoutQuery {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ARGUMENT_LAYOUT,
                size: 0,
                seqno: 0,
            },
            obj_id,
            reply_offset: BLOCK,
            reply_length: SIZE,
            source: RGFX_ARGUMENT_SOURCE_DESCRIPTORS,
            count: 2,
            descriptors_offset: size_of::<RgfxArgumentLayoutQuery>() as u32,
            ..RgfxArgumentLayoutQuery::default()
        };
        let mut bytes = head.as_bytes().to_vec();
        bytes.extend_from_slice(descriptors.as_bytes());
        bytes
    };
    // Process 2 names process 1's block: refused, and the block keeps its bytes.
    guest.send(&device, query(obj(2, 1)));
    let seqno = guest.seqno;
    assert_ne!(answer_of(&mut guest, seqno), RGFX_STATUS_OK);
    // SAFETY: as `Guest`'s helpers; the block lies in the region.
    assert_eq!(unsafe { guest.load_u64(BLOCK as usize) }, 0);
    guest.send(&device, query(obj(1, 1)));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_OK);
    // SAFETY: as above.
    let head = unsafe { guest.read(BLOCK as usize, size_of::<RgfxArgumentLayoutReply>()) };
    let head = RgfxArgumentLayoutReply::read_from_bytes(&head).unwrap();
    assert_eq!((head.layout_count, head.member_count), (1, 2));
    // SAFETY: as above.
    let layout = unsafe {
        guest.read(
            BLOCK as usize + size_of::<RgfxArgumentLayoutReply>(),
            size_of::<RgfxArgumentLayoutRecord>(),
        )
    };
    let layout = RgfxArgumentLayoutRecord::read_from_bytes(&layout).unwrap();
    // The fake's layout: 8 bytes an element, in order.
    assert_eq!((layout.encoded_length, layout.member_count), (40, 2));
    assert_eq!(device.decoder_object_count(), 0);
}

/// The calls the platform had to make a library from bytes.
fn library_makes(platform: &FakePlatform) -> usize {
    platform
        .call_log()
        .calls_to("newLibraryWithData:error:")
        .len()
}

/// A library one process made from keyed bytes is found by another's lookup: the first lookup
/// misses, the create makes the library and keeps it, and the second process's lookup binds its
/// own id to that library, with no bytes sent and nothing made; a pipeline from it compiles
#[test]
fn a_library_lookup_misses_then_hits_across_two_contexts() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let (bytes, key) = keyed_metallib();
    assert_eq!(
        answered(
            &device,
            &mut guest,
            library_lookup(obj(1, 10), bytes.len(), key)
        ),
        RGFX_STATUS_NOT_FOUND
    );
    assert_eq!(
        answered(
            &device,
            &mut guest,
            keyed_library_create(obj(1, 10), &bytes, Some(key))
        ),
        RGFX_STATUS_OK
    );
    assert_eq!(library_makes(&platform), 1);

    let lookup = library_lookup(obj(2, 10), bytes.len(), key);
    assert_eq!(lookup.len(), size_of::<RgfxLibraryLookup>());
    assert_eq!(answered(&device, &mut guest, lookup), RGFX_STATUS_OK);
    assert_eq!(library_makes(&platform), 1);
    assert_eq!(device.decoder_object_count(), 2);
    // The same key at another size is not that library.
    assert_eq!(
        answered(
            &device,
            &mut guest,
            library_lookup(obj(2, 11), bytes.len() + 8, key)
        ),
        RGFX_STATUS_NOT_FOUND
    );

    // Each id is a handle of its own: the first process's destroy leaves the second's.
    guest.send(&device, object_destroy(obj(1, 10)));
    guest.send(&device, compute_pipeline_create(obj(2, 12), obj(2, 10)));
    let bytes = next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    let reflection = RgfxPipelineReflection::read_from_bytes(&bytes).unwrap();
    assert_eq!(reflection.obj_id, obj(2, 12));
    assert_eq!(reflection.status, RGFX_PIPELINE_OK);
}

/// A create whose key is not the one the host computes from its bytes makes the library for its
/// request, and keeps it under no key: neither key finds it
#[test]
fn a_library_with_a_wrong_key_is_not_kept() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let (bytes, key) = keyed_metallib();
    let mut wrong = key;
    wrong[0] ^= 1;
    assert_eq!(
        answered(
            &device,
            &mut guest,
            keyed_library_create(obj(1, 10), &bytes, Some(wrong))
        ),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 1);
    for key in [wrong, key] {
        assert_eq!(
            answered(
                &device,
                &mut guest,
                library_lookup(obj(2, 10), bytes.len(), key)
            ),
            RGFX_STATUS_NOT_FOUND
        );
    }
    // Bytes with no key are made, and kept under none either.
    assert_eq!(
        answered(
            &device,
            &mut guest,
            keyed_library_create(obj(1, 11), &bytes, None)
        ),
        RGFX_STATUS_OK
    );
    assert_eq!(
        answered(
            &device,
            &mut guest,
            library_lookup(obj(2, 10), bytes.len(), key)
        ),
        RGFX_STATUS_NOT_FOUND
    );
    assert_eq!(library_makes(&platform), 2);
}

/// A library no id names any more stays in the cache: after its process closes, the next
/// process's lookup still finds it
#[test]
fn a_released_library_stays_in_the_cache() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let (bytes, key) = keyed_metallib();
    assert_eq!(
        answered(
            &device,
            &mut guest,
            keyed_library_create(obj(1, 10), &bytes, Some(key))
        ),
        RGFX_STATUS_OK
    );
    assert_eq!(
        answered(&device, &mut guest, context_close(1)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 0);
    assert_eq!(
        answered(
            &device,
            &mut guest,
            library_lookup(obj(2, 10), bytes.len(), key)
        ),
        RGFX_STATUS_OK
    );
    assert_eq!(library_makes(&platform), 1);
    assert_eq!(device.decoder_object_count(), 1);
}

/// A display command behind a library create on ring 0 is answered while the library is made
#[test]
fn a_display_command_behind_a_slow_library_create_is_answered() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    device.control.pool.hold();
    guest.send(&device, library_create(obj(1, 10)));
    let create = guest.seqno;
    guest.send(
        &device,
        RgfxMsgHeader {
            r#type: RGFX_CMD_GET_DISPLAYS,
            size: 0,
            seqno: 0,
        }
        .as_bytes()
        .to_vec(),
    );
    let displays = next_event(&mut guest, RGFX_EVT_DISPLAYS);
    let (header, _) = RgfxMsgHeader::read_from_prefix(&displays).unwrap();
    assert_eq!(header.seqno, guest.seqno);
    assert_eq!(device.decoder_object_count(), 0);

    device.control.pool.release();
    let done = RgfxComplete::read_from_bytes(&next_event(&mut guest, RGFX_EVT_COMPLETE)).unwrap();
    assert_eq!(done.header.seqno, create);
    assert_eq!(done.status, RGFX_STATUS_OK);
    assert_eq!(device.decoder_object_count(), 1);
}

/// The events of ring 0's completion ring until `n` have come, by type.
fn events(guest: &mut Guest, n: usize) -> Vec<(u32, Vec<u8>)> {
    (0..n)
        .map(|_| {
            let bytes = guest.event().expect("an event on ring 0");
            let (header, _) = RgfxMsgHeader::read_from_prefix(&bytes).unwrap();
            (header.r#type, bytes)
        })
        .collect()
}

/// A pipeline create that names a library still being made is prepared at once, ring 0 going on,
/// and compiles once the library is made
#[test]
fn a_pipeline_naming_a_pending_library_waits_for_it() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    device.control.pool.hold();
    guest.send(&device, library_create(obj(1, 10)));
    guest.send(&device, compute_pipeline_create(obj(1, 11), obj(1, 10)));
    let pipeline = guest.seqno;
    guest.send(
        &device,
        RgfxMsgHeader {
            r#type: RGFX_CMD_GET_DISPLAYS,
            size: 0,
            seqno: 0,
        }
        .as_bytes()
        .to_vec(),
    );
    next_event(&mut guest, RGFX_EVT_DISPLAYS);
    assert_eq!(library_makes(&platform), 0);

    device.control.pool.release();
    let mut got = events(&mut guest, 2);
    got.sort_by_key(|(r#type, _)| *r#type);
    assert_eq!(got[0].0, RGFX_EVT_COMPLETE);
    assert_eq!(
        RgfxComplete::read_from_bytes(&got[0].1).unwrap().status,
        RGFX_STATUS_OK
    );
    assert_eq!(got[1].0, RGFX_EVT_PIPELINE_REFLECTION);
    let reflection = RgfxPipelineReflection::read_from_bytes(&got[1].1).unwrap();
    assert_eq!(reflection.header.seqno, pipeline);
    assert_eq!(reflection.status, RGFX_PIPELINE_OK);
    assert_eq!(device.decoder_object_count(), 2);
}

/// An `OBJECT_DESTROY` of a library still being made keeps ring order: the pipeline created from
/// it before the destroy compiles with it, the one after is refused and answered at once, and the
/// library is not stored
#[test]
fn a_destroy_of_a_pending_library_keeps_the_pipeline_before_it() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    device.control.pool.hold();
    guest.send(&device, library_create(obj(1, 10)));
    guest.send(&device, compute_pipeline_create(obj(1, 11), obj(1, 10)));
    guest.send(&device, object_destroy(obj(1, 10)));
    // A pipeline create after the destroy names no library.
    guest.send(&device, compute_pipeline_create(obj(1, 12), obj(1, 10)));
    guest.send(
        &device,
        RgfxMsgHeader {
            r#type: RGFX_CMD_GET_DISPLAYS,
            size: 0,
            seqno: 0,
        }
        .as_bytes()
        .to_vec(),
    );
    let refused = next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    let refused = RgfxPipelineReflection::read_from_bytes(&refused).unwrap();
    assert_eq!(refused.obj_id, obj(1, 12));
    assert_eq!(refused.status, RGFX_PIPELINE_REFUSED);
    next_event(&mut guest, RGFX_EVT_DISPLAYS);
    device.control.pool.release();
    let got = events(&mut guest, 2);
    let reflection = got
        .iter()
        .find(|(r#type, _)| *r#type == RGFX_EVT_PIPELINE_REFLECTION)
        .map(|(_, bytes)| RgfxPipelineReflection::read_from_bytes(bytes).unwrap())
        .expect("the pipeline's reflection");
    assert_eq!(reflection.obj_id, obj(1, 11));
    assert_eq!(reflection.status, RGFX_PIPELINE_OK);
    assert!(guest.event().is_none());
    assert_eq!(device.decoder_object_count(), 1);
}

/// A pipeline create refused at its op is answered at once, refused, as its process waits for the
/// reply: one naming another process's library, and one whose function name lies outside it
#[test]
fn a_pipeline_create_refused_at_its_op_is_answered_at_once() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    make_library(&device, &mut guest, obj(1, 10));
    // A function name past the message's end.
    let mut malformed = compute_pipeline_create(obj(1, 12), obj(1, 10));
    let (mut head, _) = RgfxComputePipelineCreate::read_from_prefix(&malformed).unwrap();
    head.name_len = 0x1_0000;
    malformed[..size_of::<RgfxComputePipelineCreate>()].copy_from_slice(head.as_bytes());
    for (create, obj_id) in [
        (compute_pipeline_create(obj(2, 11), obj(1, 10)), obj(2, 11)),
        (render_pipeline_create(obj(2, 13), obj(1, 10)), obj(2, 13)),
        (malformed, obj(1, 12)),
    ] {
        guest.send(&device, create);
        let bytes = next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
        let reflection = RgfxPipelineReflection::read_from_bytes(&bytes).unwrap();
        assert_eq!(reflection.header.seqno, guest.seqno);
        assert_eq!(reflection.obj_id, obj_id);
        assert_eq!(reflection.status, RGFX_PIPELINE_REFUSED);
    }
    // Only the library.
    assert_eq!(device.decoder_object_count(), 1);
}

/// A process's command buffer names only its own objects, and runs only on the queue whose ring
/// carried it: one naming another process's pipeline, and one naming another process's queue,
/// each fail; a queue's ring carries nothing but command buffers; and the other process's objects
/// are untouched
#[test]
fn a_command_buffer_names_only_its_own_process_s_objects() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT_1: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_1: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const SUBMIT_2: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4_0000;
    const COMPLETE_2: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x6_0000;
    const RING_DATA: u32 = 0x1_0000;
    for (context, submit, complete) in [(1, SUBMIT_1, COMPLETE_1), (2, SUBMIT_2, COMPLETE_2)] {
        assert_eq!(
            answered(
                &device,
                &mut guest,
                queue_create(context, 1, submit, complete, RING_DATA, 0)
            ),
            RGFX_STATUS_OK
        );
    }
    make_library(&device, &mut guest, obj(1, 10));
    guest.send(&device, compute_pipeline_create(obj(1, 11), obj(1, 10)));
    next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);

    let mut ring = QueueRing::new(&device, SUBMIT_2, COMPLETE_2, u64::from(RING_DATA));
    for (cb, message) in [
        (100, compute_command_buffer(obj(2, 1), 100, obj(1, 11))),
        (101, command_buffer(obj(1, 1), 101)),
    ] {
        ring.send(&device, message);
        assert!(
            complete_until_event(&platform, &mut ring),
            "command buffer {cb} was not answered"
        );
        let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
        assert_eq!(done.cb_obj_id, cb);
        assert_ne!(done.status, RGFX_CB_OK, "command buffer {cb} ran");
    }
    // A queue's ring carries only command buffers: a destroy or a create on it, which would pass
    // the kext by, is refused there. Process 2's own empty buffer behind them shows they were
    // taken.
    ring.send(&device, object_destroy(obj(1, 11)));
    ring.send(&device, library_create(obj(2, 30)));
    ring.send(&device, command_buffer(obj(2, 1), 103));
    assert!(complete_until_event(&platform, &mut ring));
    let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (103, RGFX_CB_OK));
    assert_eq!(library_makes(&platform), 1);
    // Process 1's own buffer on its own queue still runs with its pipeline.
    let mut ring1 = QueueRing::new(&device, SUBMIT_1, COMPLETE_1, u64::from(RING_DATA));
    ring1.send(&device, compute_command_buffer(obj(1, 1), 102, obj(1, 11)));
    assert!(complete_until_event(&platform, &mut ring1));
    let done = RgfxCbComplete::read_from_bytes(&ring1.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (102, RGFX_CB_OK));
}

/// A create the kext posted but whose answer it gave up on is destroyed right behind it, and the
/// process makes it again under a new id: the device then holds the retry alone, though the
/// first was still being made when its destroy came
#[test]
fn a_timed_out_create_s_destroy_leaves_only_the_retry() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    device.control.pool.hold();
    guest.send(&device, library_create(obj(1, 10)));
    guest.send(&device, object_destroy(obj(1, 10)));
    guest.send(&device, library_create(obj(1, 11)));
    let retry = guest.seqno;
    device.control.pool.release();
    let got = events(&mut guest, 2);
    assert!(got.iter().all(|(r#type, _)| *r#type == RGFX_EVT_COMPLETE));
    let retried = got
        .iter()
        .map(|(_, bytes)| RgfxComplete::read_from_bytes(bytes).unwrap())
        .find(|c| c.header.seqno == retry)
        .expect("the retry's answer");
    assert_eq!(retried.status, RGFX_STATUS_OK);
    assert_eq!(library_makes(&platform), 2);
    assert_eq!(device.decoder_object_count(), 1);
    // The retry's id names the library; the destroyed one's names nothing.
    guest.send(&device, compute_pipeline_create(obj(1, 12), obj(1, 11)));
    let ok = RgfxPipelineReflection::read_from_bytes(&next_event(
        &mut guest,
        RGFX_EVT_PIPELINE_REFLECTION,
    ))
    .unwrap();
    assert_eq!(ok.status, RGFX_PIPELINE_OK);
    guest.send(&device, compute_pipeline_create(obj(1, 13), obj(1, 10)));
    let refused = RgfxPipelineReflection::read_from_bytes(&next_event(
        &mut guest,
        RGFX_EVT_PIPELINE_REFLECTION,
    ))
    .unwrap();
    assert_eq!(refused.status, RGFX_PIPELINE_REFUSED);
}

/// A `BLOCK_RELEASE` of context `context`'s block of `size` bytes at `offset`, as the kext posts
/// it.
fn block_release(context: u32, offset: u64, size: u64) -> Vec<u8> {
    RgfxBlockRelease {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_BLOCK_RELEASE,
            size: 0,
            seqno: 0,
        },
        context,
        flags: 0,
        offset,
        size,
    }
    .as_bytes()
    .to_vec()
}

/// A `BLOCK_GRANT` of the region block of `size` bytes at `offset` to context `context`, as the
/// kext posts it.
fn block_grant(context: u32, offset: u64, size: u64) -> Vec<u8> {
    let mut grant = block_release(context, offset, size);
    grant[0..4].copy_from_slice(&RGFX_CMD_BLOCK_GRANT.to_le_bytes());
    grant
}

/// A `BUFFER_OVER_HOST_MEMORY` of `length` bytes at region offset `offset`.
fn buffer_over_host_memory(obj_id: u64, offset: u64, length: u64) -> Vec<u8> {
    RgfxBufferOverHostMemory {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_BUFFER_OVER_HOST_MEMORY,
            size: 0,
            seqno: 0,
        },
        obj_id,
        region_offset: offset,
        length,
        ..RgfxBufferOverHostMemory::default()
    }
    .as_bytes()
    .to_vec()
}

/// A `TEXTURE_FROM_BUFFER`: a 64x64 BGRA8 texture at resource id `resource_id` over the start of
/// `buffer`.
fn texture_from_buffer(obj_id: u64, resource_id: u64, buffer: u64) -> Vec<u8> {
    let (create, _) = RgfxTextureCreate::read_from_prefix(&texture_create(0, 0)).unwrap();
    RgfxTextureFromBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_TEXTURE_FROM_BUFFER,
            size: 0,
            seqno: 0,
        },
        obj_id,
        resource_id,
        buffer_obj_id: buffer,
        offset: 0,
        bytes_per_row: 256,
        desc: create.desc,
    }
    .as_bytes()
    .to_vec()
}

/// A `LINEAR_TEXTURE_FROM_BUFFER` of a 2D array is made over its buffer by the platform's linear
/// form, with the offset, the bytes per row and the bytes per image the guest asked, and lies in
/// the buffer's block as a texture over a buffer does
#[test]
fn a_linear_texture_is_made_by_the_platform_s_linear_form() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SIZE: u64 = 0x1_0000;
    guest.send(&device, block_grant(1, BLOCK, SIZE));
    guest.send(&device, resource_id_chunk(1, 0));
    guest.send(&device, buffer_over_host_memory(obj(1, 1), BLOCK, 0x8000));
    let (create, _) = RgfxTextureCreate::read_from_prefix(&texture_create(0, 0)).unwrap();
    let linear = RgfxLinearTextureFromBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER,
            size: 0,
            seqno: 0,
        },
        obj_id: obj(1, 2),
        resource_id: 1,
        buffer_obj_id: obj(1, 1),
        offset: 0x1000,
        bytes_per_row: 256,
        bytes_per_image: 0x4000,
        desc: RgfxTextureDescriptor {
            texture_type: 3,
            width: 16,
            height: 16,
            array_length: 3,
            ..create.desc
        },
    };
    guest.send(&device, linear.as_bytes().to_vec());
    // The block holds the texture until it goes.
    let status = answered(&device, &mut guest, block_release(1, BLOCK, SIZE));
    assert_eq!(
        status, RGFX_STATUS_REFUSED,
        "the buffer and its texture lie in it"
    );
    assert_eq!(device.decoder_object_count(), 2);
    let calls = platform
        .call_log()
        .calls_to("newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:");
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].args[1..],
        [Arg::U64(0x1000), Arg::U64(256), Arg::U64(0x4000)]
    );
    assert!(
        platform
            .call_log()
            .calls_to("newTextureWithDescriptor:offset:bytesPerRow:")
            .is_empty()
    );
}

/// Whether ring 0's completion ring stays without a new event for about 100 ms.
fn quiet(guest: &Guest) -> bool {
    std::thread::sleep(Duration::from_millis(100));
    // SAFETY: as `Guest`'s helpers.
    unsafe { guest.load_u64(RGFX_CONTROL_COMPLETE_OFFSET as usize) == guest.consumed }
}

/// The `COMPLETE` of ring-0 command `seqno`, the next event: its status.
fn answer_of(guest: &mut Guest, seqno: u64) -> u32 {
    let complete = RgfxComplete::read_from_bytes(&next_event(guest, RGFX_EVT_COMPLETE)).unwrap();
    assert_eq!(complete.header.seqno, seqno);
    complete.status
}

/// Waits about 2 s for the platform to hold `n` committed command buffers not yet completed;
/// their ids, in commit order.
fn committed(platform: &FakePlatform, n: usize) -> Vec<ObjId> {
    for _ in 0..2000 {
        let ids = platform.world().committed();
        if ids.len() == n {
            return ids;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("{n} command buffers were not committed");
}

/// A block is given out again only once nothing the host holds lies in it: its release is refused
/// while a buffer over it lives, and still while a texture made over that buffer lives after the
/// buffer's destroy; once both are destroyed it is answered OK. A block nothing lies in is
/// released at once, whatever lies in the blocks beside it. Only the process the kext granted a
/// block places objects over it
#[test]
fn a_block_is_released_only_once_nothing_lies_in_it() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SIZE: u64 = 0x1_0000;
    guest.send(&device, block_grant(1, BLOCK, SIZE));
    guest.send(&device, block_grant(1, BLOCK + SIZE, SIZE));
    guest.send(&device, resource_id_chunk(1, 0));
    // Process 2's buffer over process 1's block is refused.
    guest.send(&device, buffer_over_host_memory(obj(2, 1), BLOCK, 0x4000));
    guest.send(&device, buffer_over_host_memory(obj(1, 1), BLOCK, 0x4000));
    guest.send(&device, texture_from_buffer(obj(1, 2), 1, obj(1, 1)));
    let status = answered(&device, &mut guest, block_release(1, BLOCK + SIZE, SIZE));
    assert_eq!(status, RGFX_STATUS_OK, "the block beside them");
    assert_eq!(
        device.decoder_object_count(),
        2,
        "process 2's buffer was made"
    );
    let status = answered(&device, &mut guest, block_release(1, BLOCK, SIZE));
    assert_eq!(status, RGFX_STATUS_REFUSED, "the buffer lies in it");
    guest.send(&device, object_destroy(obj(1, 1)));
    let status = answered(&device, &mut guest, block_release(1, BLOCK, SIZE));
    assert_eq!(
        status, RGFX_STATUS_REFUSED,
        "the texture over the buffer lies in it"
    );
    guest.send(&device, object_destroy(obj(1, 2)));
    let status = answered(&device, &mut guest, block_release(1, BLOCK, SIZE));
    assert_eq!(status, RGFX_STATUS_OK);
}

/// A `Managed` buffer's range on ring 0 reaches the platform before a command buffer its process
/// commits after it on a queue's own ring: the range's `didModifyRange:` comes before the
/// buffer's commit
#[test]
fn a_managed_range_is_carried_out_before_a_later_command_buffer() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const RING_DATA: u32 = 0x1_0000;
    let mut guest = Guest::new(&device);
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(1, 1, SUBMIT, COMPLETE, RING_DATA, 0)
        ),
        RGFX_STATUS_OK
    );
    guest.send(&device, block_grant(1, BLOCK, 0x1_0000));
    let mut managed = buffer_over_host_memory(obj(1, 2), BLOCK, 0x1000);
    let options = std::mem::offset_of!(RgfxBufferOverHostMemory, options);
    managed[options..options + 4].copy_from_slice(&(1u32 << 4).to_le_bytes());
    guest.send(&device, managed);
    let range = RgfxBufferDidModifyRange {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_BUFFER_DID_MODIFY_RANGE,
            size: 0,
            seqno: 0,
        },
        obj_id: obj(1, 2),
        location: 0x40,
        length: 0x10,
    };
    guest.send(&device, range.as_bytes().to_vec());
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    ring.send(&device, command_buffer(obj(1, 1), 100));
    committed(&platform, 1);
    let calls = platform.call_log().calls();
    let at = |sel: &str, arg: Option<Arg>| {
        calls
            .iter()
            .position(|c| c.sel == sel && arg.as_ref().is_none_or(|a| c.args.first() == Some(a)))
            .unwrap_or_else(|| panic!("no {sel}"))
    };
    let modified = at(
        "didModifyRange:",
        Some(Arg::val(&reims_vgpu_core::platform::types::NsRange::new(
            0x40, 0x10,
        ))),
    );
    assert!(modified < at("commit", None));
}

/// A library from region chunks is checked against its own process's blocks, whichever process
/// sent the message before it: process 1's chunk over its block, sent after process 2's create,
/// is made; process 2's chunk over process 1's block is refused
#[test]
fn a_library_chunk_is_checked_against_its_sender_s_blocks() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SIZE: u64 = 0x1_0000;
    let bytes = [1u8, 2, 3, 4];
    // SAFETY: as `Guest`'s helpers; the span lies in the region.
    unsafe { guest.write((BLOCK + 0x1000) as usize, &bytes) };
    guest.send(&device, block_grant(1, BLOCK, SIZE));
    let chunk = |obj_id: u64| {
        RgfxLibraryFromRegion {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_LIBRARY_FROM_REGION,
                size: 0,
                seqno: 0,
            },
            obj_id,
            region_offset: BLOCK + 0x1000,
            byte_length: bytes.len() as u64,
            total_length: bytes.len() as u64,
            chunk_offset: 0,
            ack_offset: BLOCK,
            ack_value: 7,
            ..RgfxLibraryFromRegion::default()
        }
        .as_bytes()
        .to_vec()
    };
    // The control thread's last message is process 2's (refused: its block is not process 2's).
    guest.send(&device, buffer_over_host_memory(obj(2, 1), BLOCK, 0x1000));
    assert_eq!(
        answered(&device, &mut guest, chunk(obj(1, 10))),
        RGFX_STATUS_OK
    );
    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.load_u64(BLOCK as usize) },
        7,
        "the chunk is acked"
    );
    // Now the last message is process 1's.
    assert_eq!(
        answered(&device, &mut guest, chunk(obj(2, 10))),
        RGFX_STATUS_INVALID_ARGUMENT
    );
}

/// A block's release, and a context's close, wait for the command buffers the context committed
/// before them, and only those: another context's in-flight buffer, and the context's own buffer
/// committed after the release, hold neither
#[test]
fn a_release_and_a_close_wait_for_their_context_s_command_buffers() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT_1: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_1: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const SUBMIT_2: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4_0000;
    const COMPLETE_2: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x6_0000;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const RING_DATA: u32 = 0x1_0000;
    for (context, submit, complete) in [(1, SUBMIT_1, COMPLETE_1), (2, SUBMIT_2, COMPLETE_2)] {
        assert_eq!(
            answered(
                &device,
                &mut guest,
                queue_create(context, 1, submit, complete, RING_DATA, 0)
            ),
            RGFX_STATUS_OK
        );
    }
    let mut ring1 = QueueRing::new(&device, SUBMIT_1, COMPLETE_1, u64::from(RING_DATA));
    let mut ring2 = QueueRing::new(&device, SUBMIT_2, COMPLETE_2, u64::from(RING_DATA));
    ring1.send(&device, command_buffer(obj(1, 1), 100));
    let first = committed(&platform, 1)[0];
    ring2.send(&device, command_buffer(obj(2, 1), 200));
    let other = committed(&platform, 2)[1];

    guest.send(&device, block_release(1, BLOCK, 0x1_0000));
    let release = guest.seqno;
    assert!(quiet(&guest), "answered with context 1's buffer in flight");
    ring1.send(&device, command_buffer(obj(1, 1), 101));
    let later = committed(&platform, 3)[2];
    let world = platform.world();
    let done = |id| world.complete(id, CommandBufferStatus::COMPLETED, None, CompleteOn::Caller);
    assert!(done(other));
    assert!(
        quiet(&guest),
        "answered when another context's buffer finished"
    );
    assert!(done(first));
    assert_eq!(answer_of(&mut guest, release), RGFX_STATUS_OK);

    // The close waits for the buffer committed after the release.
    guest.send(&device, context_close(1));
    let close = guest.seqno;
    assert!(
        quiet(&guest),
        "the close was answered with its buffer in flight"
    );
    assert!(done(later));
    assert_eq!(answer_of(&mut guest, close), RGFX_STATUS_OK);
}

/// A block a live queue's ring lies in is not released; once the queue is destroyed, it is
#[test]
fn a_queue_s_rings_keep_their_block() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(1, 1, SUBMIT, COMPLETE, RING_DATA, 0)
        ),
        RGFX_STATUS_OK
    );
    for at in [SUBMIT, COMPLETE] {
        let status = answered(&device, &mut guest, block_release(1, at, 0x2_0000));
        assert_eq!(status, RGFX_STATUS_REFUSED, "a ring at {at:#x}");
    }
    guest.send(&device, object_destroy(obj(1, 1)));
    let status = answered(&device, &mut guest, block_release(1, SUBMIT, 0x4_0000));
    assert_eq!(status, RGFX_STATUS_OK);
}

/// A `BUFFER_CREATE` of a 4 KiB Private buffer `obj_id` the host places.
fn buffer_create(obj_id: u64) -> Vec<u8> {
    RgfxBufferCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_BUFFER_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        length: 0x1000,
        options: 0x20,
        ..RgfxBufferCreate::default()
    }
    .as_bytes()
    .to_vec()
}

/// The reflection the next event of ring 0 is: of pipeline `obj_id`, `status`.
fn expect_reflection(guest: &mut Guest, obj_id: u64, status: u32) {
    let bytes = next_event(guest, RGFX_EVT_PIPELINE_REFLECTION);
    let reflection = RgfxPipelineReflection::read_from_bytes(&bytes).unwrap();
    assert_eq!(reflection.header.seqno, guest.seqno);
    assert_eq!(reflection.obj_id, obj_id);
    assert_eq!(reflection.status, status);
}

/// The platform ending the process at a pipeline's compile or at its function's specialisation
/// (the Vulkan runtime refusing the compile, which ends the process on the paravirt device)
/// refuses the create in this lane: the reflection is refused, nothing is stored, and the device
/// goes on, the next creates made
#[test]
fn a_compile_the_platform_refuses_is_a_refused_reflection() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    make_library(&device, &mut guest, obj(1, 10));
    let faults = platform.faults();
    faults.at(
        "MTLDevice",
        "newRenderPipelineStateWithDescriptor:error:",
        0,
        Fault::Crash,
    );
    guest.send(&device, render_pipeline_create(obj(1, 11), obj(1, 10)));
    expect_reflection(&mut guest, obj(1, 11), RGFX_PIPELINE_REFUSED);
    faults.at(
        "MTLDevice",
        "newComputePipelineStateWithDescriptor:options:reflection:error:",
        0,
        Fault::Crash,
    );
    guest.send(&device, compute_pipeline_create(obj(1, 12), obj(1, 10)));
    expect_reflection(&mut guest, obj(1, 12), RGFX_PIPELINE_REFUSED);
    faults.at("MTLLibrary", "newFunctionWithName:", 0, Fault::Crash);
    guest.send(&device, compute_pipeline_create(obj(1, 13), obj(1, 10)));
    expect_reflection(&mut guest, obj(1, 13), RGFX_PIPELINE_REFUSED);
    assert_eq!(device.decoder_object_count(), 1, "only the library");
    // The device goes on: the same creates are made.
    guest.send(&device, render_pipeline_create(obj(1, 21), obj(1, 10)));
    expect_reflection(&mut guest, obj(1, 21), RGFX_PIPELINE_OK);
    guest.send(&device, compute_pipeline_create(obj(1, 22), obj(1, 10)));
    expect_reflection(&mut guest, obj(1, 22), RGFX_PIPELINE_OK);
    assert_eq!(device.decoder_object_count(), 3);
}

/// With no compile thread, a create's compile runs on the control thread, where the platform
/// ending the process refuses it as on the pool: the reflection or the `COMPLETE` is refused, and
/// the next creates are made
#[test]
fn a_create_compiled_on_the_control_thread_the_platform_refuses_is_refused() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.control.pool.stop();
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    make_library(&device, &mut guest, obj(1, 10));
    let faults = platform.faults();
    faults.at(
        "MTLDevice",
        "newRenderPipelineStateWithDescriptor:error:",
        0,
        Fault::Crash,
    );
    guest.send(&device, render_pipeline_create(obj(1, 11), obj(1, 10)));
    expect_reflection(&mut guest, obj(1, 11), RGFX_PIPELINE_REFUSED);
    faults.at(
        "MTLDevice",
        "newComputePipelineStateWithDescriptor:options:reflection:error:",
        0,
        Fault::Crash,
    );
    guest.send(&device, compute_pipeline_create(obj(1, 12), obj(1, 10)));
    expect_reflection(&mut guest, obj(1, 12), RGFX_PIPELINE_REFUSED);
    faults.at("MTLDevice", "newLibraryWithData:error:", 0, Fault::Crash);
    assert_eq!(
        answered(&device, &mut guest, library_create(obj(1, 13))),
        RGFX_STATUS_REFUSED
    );
    assert_eq!(device.decoder_object_count(), 1, "only the first library");
    guest.send(&device, render_pipeline_create(obj(1, 21), obj(1, 10)));
    expect_reflection(&mut guest, obj(1, 21), RGFX_PIPELINE_OK);
    guest.send(&device, compute_pipeline_create(obj(1, 22), obj(1, 10)));
    expect_reflection(&mut guest, obj(1, 22), RGFX_PIPELINE_OK);
    make_library(&device, &mut guest, obj(1, 23));
    assert_eq!(device.decoder_object_count(), 4);
}

/// The platform ending the process at a library's make (the Vulkan runtime refusing a library
/// kind it makes none of yet, a Core Image kernel library) refuses the create in this lane: its
/// `COMPLETE` is refused, a pipeline of it is refused, and the next library is made
#[test]
fn a_library_the_platform_refuses_is_a_refused_create() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    platform
        .faults()
        .at("MTLDevice", "newLibraryWithData:error:", 0, Fault::Crash);
    assert_eq!(
        answered(&device, &mut guest, library_create(obj(1, 10))),
        RGFX_STATUS_REFUSED
    );
    guest.send(&device, compute_pipeline_create(obj(1, 11), obj(1, 10)));
    expect_reflection(&mut guest, obj(1, 11), RGFX_PIPELINE_REFUSED);
    assert_eq!(device.decoder_object_count(), 0, "nothing made");
    // The device goes on: the same create is made.
    make_library(&device, &mut guest, obj(1, 20));
    guest.send(&device, compute_pipeline_create(obj(1, 21), obj(1, 20)));
    expect_reflection(&mut guest, obj(1, 21), RGFX_PIPELINE_OK);
    assert_eq!(device.decoder_object_count(), 2);
}

/// A `HEAP_CREATE` of an Automatic heap `obj_id` of 1 MiB.
fn heap_create(obj_id: u64) -> Vec<u8> {
    RgfxHeapCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_HEAP_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        gpu_address: 0x6_0000,
        size_bytes: 1 << 20,
        options: 0,
        heap_type: 0,
    }
    .as_bytes()
    .to_vec()
}

/// A `TEXTURE_CREATE` of texture `obj_id` at `offset` in heap `heap`, at `resource_id`.
fn heap_texture_create(obj_id: u64, resource_id: u64, heap: u64, offset: u64) -> Vec<u8> {
    let mut m = RgfxTextureCreate::read_from_bytes(&texture_create(obj_id, resource_id)).unwrap();
    m.heap_obj_id = heap;
    m.heap_offset = offset;
    m.as_bytes().to_vec()
}

/// The platform ending the process at a texture's or a buffer's create, or at a texture's
/// placement in a heap (the Vulkan runtime refusing a placement at an offset in an Automatic
/// heap, which Metal's validation fails), refuses the create: nothing is made, the device goes
/// on, and the next creates are made
#[test]
fn a_create_the_platform_refuses_makes_nothing() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let first = 1 + 7 * RGFX_RESOURCE_ID_CHUNK;
    guest.send(&device, resource_id_chunk(3, 7));
    guest.send(&device, heap_create(obj(3, 9)));
    let faults = platform.faults();
    faults.at("MTLDevice", "newTextureWithDescriptor:", 0, Fault::Crash);
    faults.at("MTLDevice", "newBufferWithDescriptor:", 0, Fault::Crash);
    faults.at(
        "MTLHeap",
        "newTextureWithDescriptor:offset:",
        0,
        Fault::Crash,
    );
    guest.send(&device, texture_create(obj(3, 1), first));
    guest.send(&device, buffer_create(obj(3, 2)));
    guest.send(
        &device,
        heap_texture_create(obj(3, 5), first + 2, obj(3, 9), 0x1000),
    );
    assert_eq!(
        answered(&device, &mut guest, context_close(99)),
        RGFX_STATUS_OK
    );
    assert_eq!(
        device.decoder_object_count(),
        1,
        "the three creates refused"
    );
    guest.send(&device, texture_create(obj(3, 3), first + 1));
    guest.send(&device, buffer_create(obj(3, 4)));
    guest.send(
        &device,
        heap_texture_create(obj(3, 6), first + 3, obj(3, 9), 0x1000),
    );
    assert_eq!(
        answered(&device, &mut guest, context_close(99)),
        RGFX_STATUS_OK
    );
    assert_eq!(
        device.decoder_object_count(),
        4,
        "the heap, and the next texture, buffer and placed texture"
    );
}

/// A `TEXTURE_VIEW` in `pixel_format` of levels `levels` and slices `slices` of texture `parent`,
/// as view `obj_id` at resource id `resource_id`.
fn texture_view(
    obj_id: u64,
    parent: u64,
    resource_id: u64,
    pixel_format: u32,
    levels: (u32, u32),
    slices: (u32, u32),
) -> Vec<u8> {
    RgfxTextureView {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_TEXTURE_VIEW,
            size: 0,
            seqno: 0,
        },
        obj_id,
        parent_obj_id: parent,
        resource_id,
        pixel_format,
        flags: RGFX_TEXTURE_VIEW_SUBRANGE,
        texture_type: 2,
        level_start: levels.0,
        level_count: levels.1,
        slice_start: slices.0,
        slice_count: slices.1,
        swizzle: 0,
    }
    .as_bytes()
    .to_vec()
}

/// The platform ending the process at a view, as it does at one Metal makes not (levels outside
/// the texture, a format of another class), refuses the view: nothing is made, the device goes
/// on, and the next view of the texture is made
#[test]
fn a_view_the_platform_refuses_makes_nothing() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let first = 1 + 7 * RGFX_RESOURCE_ID_CHUNK;
    guest.send(&device, resource_id_chunk(3, 7));
    guest.send(&device, texture_create(obj(3, 1), first));
    platform.faults().at(
        "MTLTexture",
        "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:",
        0,
        Fault::Crash,
    );
    guest.send(
        &device,
        texture_view(obj(3, 2), obj(3, 1), first + 1, 80, (3, 1), (0, 1)),
    );
    assert_eq!(
        answered(&device, &mut guest, context_close(99)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 1, "the view refused");
    guest.send(
        &device,
        texture_view(obj(3, 3), obj(3, 1), first + 2, 81, (0, 1), (0, 1)),
    );
    assert_eq!(
        answered(&device, &mut guest, context_close(99)),
        RGFX_STATUS_OK
    );
    assert_eq!(
        device.decoder_object_count(),
        2,
        "the texture and its next view"
    );
}

/// The platform ending the process at an encode in a command buffer fails that command buffer:
/// its completion carries an error, and the queue's next command buffer runs and completes
#[test]
fn an_encode_the_platform_refuses_fails_its_command_buffer() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(3, 1, SUBMIT_OFF, COMPLETE_OFF, RING_DATA, 0)
        ),
        RGFX_STATUS_OK
    );
    make_library(&device, &mut guest, obj(3, 10));
    guest.send(&device, compute_pipeline_create(obj(3, 11), obj(3, 10)));
    expect_reflection(&mut guest, obj(3, 11), RGFX_PIPELINE_OK);
    platform.faults().at(
        "MTLComputeCommandEncoder",
        "setComputePipelineState:",
        0,
        Fault::Crash,
    );
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    ring.send(&device, compute_command_buffer(obj(3, 1), 100, obj(3, 11)));
    assert!(complete_until_event(&platform, &mut ring), "no completion");
    let failed = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!(failed.cb_obj_id, 100);
    assert_ne!(failed.status, RGFX_CB_OK);
    ring.send(&device, compute_command_buffer(obj(3, 1), 101, obj(3, 11)));
    assert!(complete_until_event(&platform, &mut ring), "no completion");
    let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!(done.cb_obj_id, 101);
    assert_eq!(done.status, RGFX_CB_OK);
}

/// A `SHARED_EVENT_CREATE` of shared event `obj_id`, at `initial_value`.
fn shared_event_create(obj_id: u64, initial_value: u64) -> Vec<u8> {
    RgfxSharedEventCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SHARED_EVENT_CREATE,
            size: size_of::<RgfxSharedEventCreate>() as u32,
            seqno: 0,
        },
        obj_id,
        initial_value,
    }
    .as_bytes()
    .to_vec()
}

/// A `COMMAND_BUFFER` `cb_obj_id` on queue `queue_obj_id` that signals shared event `event` to 4,
/// draws with render pipeline `pipeline` in a pass of no attachments, and signals `event` to 5.
fn render_command_buffer(queue_obj_id: u64, cb_obj_id: u64, pipeline: u64, event: u64) -> Vec<u8> {
    use reims_graphics::ops::render::{
        RGFX_OP_RENDER_DRAW_PRIMITIVES, RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
        RgfxOpDrawPrimitives, RgfxOpSetRenderPipelineState,
    };
    use reims_graphics::ops::{
        RGFX_OP_ENCODER_BEGIN_RENDER, RGFX_OP_ENCODER_END, RGFX_OP_SIGNAL_EVENT,
        RGFX_PASS_FLAG_RT_SIZE, RgfxOpEvent, RgfxOpHeader, RgfxRenderPassDesc,
    };
    let op = |tag: u16, body: &[u8]| {
        let mut bytes = RgfxOpHeader {
            tag,
            flags: 0,
            len: (8 + body.len()) as u32,
        }
        .as_bytes()
        .to_vec();
        bytes.extend_from_slice(body);
        bytes
    };
    let signal = |value: u64| {
        op(
            RGFX_OP_SIGNAL_EVENT,
            RgfxOpEvent {
                event_obj_id: event,
                value,
            }
            .as_bytes(),
        )
    };
    let mut ops = signal(4);
    ops.extend(op(
        RGFX_OP_ENCODER_BEGIN_RENDER,
        RgfxRenderPassDesc {
            flags: RGFX_PASS_FLAG_RT_SIZE,
            render_target_width: 64,
            render_target_height: 64,
            ..RgfxRenderPassDesc::default()
        }
        .as_bytes(),
    ));
    ops.extend(op(
        RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
        RgfxOpSetRenderPipelineState {
            pipeline_obj_id: pipeline,
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
    ops.extend(op(RGFX_OP_ENCODER_END, &[]));
    ops.extend(signal(5));
    let mut bytes = RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: 0,
            seqno: 0,
        },
        queue_obj_id,
        cb_obj_id,
        op_bytes: ops.len() as u64,
    }
    .as_bytes()
    .to_vec();
    bytes.extend(ops);
    bytes
}

/// A command buffer the host cannot carry out, one whose draw sets a render pipeline the host
/// refused, fails as a buffer Metal could not run does: in its queue's order, once the buffers
/// committed before it have finished, and with the shared event values it signals posted ahead
/// of its failure, so nothing waiting for one of them waits for good. The buffer after it runs
#[test]
fn a_command_buffer_naming_a_refused_pipeline_fails_in_order_and_signals_its_events() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(3, 1, SUBMIT_OFF, COMPLETE_OFF, RING_DATA, 0)
        ),
        RGFX_STATUS_OK
    );
    guest.send(&device, shared_event_create(obj(3, 20), 0));
    make_library(&device, &mut guest, obj(3, 10));
    platform.faults().at(
        "MTLDevice",
        "newRenderPipelineStateWithDescriptor:error:",
        0,
        Fault::Crash,
    );
    guest.send(&device, render_pipeline_create(obj(3, 11), obj(3, 10)));
    expect_reflection(&mut guest, obj(3, 11), RGFX_PIPELINE_REFUSED);

    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    ring.send(&device, command_buffer(obj(3, 1), 100));
    ring.send(
        &device,
        render_command_buffer(obj(3, 1), 101, obj(3, 11), obj(3, 20)),
    );
    ring.send(&device, command_buffer(obj(3, 1), 102));
    // Three buffers wait on the GPU, the refused one's stand-in among them, and nothing is
    // answered before the first finishes.
    committed(&platform, 3);
    assert!(
        !ring.has_event(),
        "a buffer was answered before the one before it finished"
    );

    platform.world().complete_all(CompleteOn::Caller);
    let mut got = Vec::new();
    for _ in 0..5 {
        let bytes = ring
            .event()
            .expect("an event on the queue's completion ring");
        let (header, _) = RgfxMsgHeader::read_from_prefix(&bytes).unwrap();
        got.push(match header.r#type {
            RGFX_EVT_CB_COMPLETE => {
                let done = RgfxCbComplete::read_from_bytes(&bytes).unwrap();
                (header.r#type, done.cb_obj_id, u64::from(done.status))
            }
            RGFX_EVT_EVENT_SIGNALLED => {
                let signalled = RgfxEventSignalled::read_from_bytes(&bytes).unwrap();
                (header.r#type, signalled.obj_id, signalled.value)
            }
            other => panic!("event {other:#x} on the queue's completion ring"),
        });
    }
    assert_eq!(
        got,
        [
            (RGFX_EVT_CB_COMPLETE, 100, u64::from(RGFX_CB_OK)),
            (RGFX_EVT_EVENT_SIGNALLED, obj(3, 20), 4),
            (RGFX_EVT_EVENT_SIGNALLED, obj(3, 20), 5),
            (RGFX_EVT_CB_COMPLETE, 101, u64::from(RGFX_CB_INTERNAL)),
            (RGFX_EVT_CB_COMPLETE, 102, u64::from(RGFX_CB_OK)),
        ]
    );
    // The two buffers the GPU ran were scheduled; the refused one's stand-in posts only its
    // completion.
    let scheduled: Vec<u64> = ring.scheduled.iter().map(|&(_, cb)| cb).collect();
    assert_eq!(scheduled, [100, 102]);
    assert!(!ring.has_event());
}

/// A `TEXTURE_FROM_IOSURFACE`: a BGRA8 texture `width` pixels square at resource id
/// `resource_id`, over the start of `buffer`, naming plane 0 of IOSurface `iosurface`, its copy's
/// writers `writers_offset` bytes into the buffer.
fn texture_from_iosurface(
    obj_id: u64,
    resource_id: u64,
    buffer: u64,
    iosurface: u32,
    width: u32,
    writers_offset: u64,
) -> Vec<u8> {
    let (create, _) = RgfxTextureCreate::read_from_prefix(&texture_create(0, 0)).unwrap();
    RgfxTextureFromIOSurface {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_TEXTURE_FROM_IOSURFACE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        resource_id,
        buffer_obj_id: buffer,
        offset: 0,
        bytes_per_row: u64::from(width) * 4,
        iosurface_id: iosurface,
        plane: 0,
        desc: RgfxTextureDescriptor {
            width,
            height: width,
            ..create.desc
        },
        writers_offset,
    }
    .as_bytes()
    .to_vec()
}

/// A display pipe's `PRESENT_SURFACE` of IOSurface `id` (0: the scanout), as the kext posts it.
fn present_surface(id: u32) -> Vec<u8> {
    RgfxPresentSurface {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_PRESENT_SURFACE,
            size: 0,
            seqno: 0,
        },
        display: 0,
        iosurface_id: id,
        transaction_id: 1,
        reserved_1c: 0,
    }
    .as_bytes()
    .to_vec()
}

/// [`command_buffer`] at `seqno` on its queue's ring, as the guest's ring writer numbers it.
fn command_buffer_at(queue_obj_id: u64, cb_obj_id: u64, seqno: u64) -> Vec<u8> {
    let mut message = command_buffer(queue_obj_id, cb_obj_id);
    message[8..16].copy_from_slice(&seqno.to_le_bytes());
    message
}

/// A surface copy's writers, as the guest's processes keep them: `writers` in the first slots.
fn write_writers(device: &GraphicsDevice<FakePlatform>, at: u64, writers: &[(u64, u64)]) {
    let mut record = RgfxSurfaceWriters::default();
    for (slot, &(queue_obj_id, seqno)) in record.writers.iter_mut().zip(writers) {
        *slot = RgfxSurfaceWriter {
            queue_obj_id,
            seqno,
        };
    }
    // SAFETY: the record lies in the device's region, mapped while the device lives, in a block
    // the test granted; the host only reads it.
    unsafe {
        std::ptr::copy_nonoverlapping(
            record.as_bytes().as_ptr(),
            device.region_base().add(at as usize),
            size_of::<RgfxSurfaceWriters>(),
        );
    }
}

/// Waits about 2 s for `n` presents to wait for their writers.
fn presents_waiting(device: &GraphicsDevice<FakePlatform>, n: usize) {
    for _ in 0..2000 {
        if device.control.presents.waiting() == n {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("{n} presents were not waiting");
}

/// The width of the texture the display shows; `None` for the scanout.
fn shown_width(device: &GraphicsDevice<FakePlatform>) -> Option<u64> {
    use reims_vgpu_core::platform::resource::Texture as _;
    device
        .control
        .presented
        .texture
        .lock()
        .as_ref()
        .map(|t| t.width())
}

/// Context 1 with a queue (its rings at `SUBMIT` and `COMPLETE`), resource-id chunk 0, and a
/// buffer over the block at `BLOCK` that a texture over IOSurface 0x77 lies over, its copy's
/// writers at `WRITERS` in the block: the guest's ring 0 and the queue's rings.
const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
const WRITERS: u64 = 0x8000;

fn surface_and_queue(device: &GraphicsDevice<FakePlatform>) -> (Guest, QueueRing) {
    const RING_DATA: u32 = 0x1_0000;
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(device);
    assert_eq!(
        answered(
            device,
            &mut guest,
            queue_create(1, 1, SUBMIT, COMPLETE, RING_DATA, 0)
        ),
        RGFX_STATUS_OK
    );
    guest.send(device, block_grant(1, BLOCK, 0x1_0000));
    guest.send(device, resource_id_chunk(1, 0));
    guest.send(device, buffer_over_host_memory(obj(1, 2), BLOCK, 0x1_0000));
    guest.send(
        device,
        texture_from_iosurface(obj(1, 3), 1, obj(1, 2), 0x77, 64, WRITERS),
    );
    let ring = QueueRing::new(device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    (guest, ring)
}

/// A present of a surface is shown only once the command buffer committed before it that writes
/// the surface's copy has finished, however late the host takes that buffer: a buffer before it
/// on the queue finishing shows nothing, and the present is answered once it is shown. The process
/// that made the surface's texture owns the scanout from the present on: its compiles go first,
/// until its close
#[test]
fn a_present_waits_for_the_command_buffer_that_wrote_its_surface() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    let (mut guest, mut ring) = surface_and_queue(&device);
    assert_eq!(device.control.pool.favoured(), None);
    // The guest committed buffer 5 on the queue, which draws the surface, then presented it; the
    // host has taken neither buffer 4 nor 5 yet.
    write_writers(&device, BLOCK + WRITERS, &[(obj(1, 1), 5)]);
    guest.send(&device, present_surface(0x77));
    presents_waiting(&device, 1);
    assert_eq!(device.control.pool.favoured(), Some(1));
    assert!(quiet(&guest), "answered before it was shown");
    assert_eq!(shown_width(&device), None);

    // Buffer 4 runs and finishes: the present still waits.
    ring.send(&device, command_buffer_at(obj(1, 1), 100, 4));
    assert!(complete_until_event(&platform, &mut ring));
    ring.event().expect("buffer 4's CB_COMPLETE");
    presents_waiting(&device, 1);
    assert_eq!(shown_width(&device), None);

    // Buffer 5 finishes: the surface shows, and the present is answered.
    ring.send(&device, command_buffer_at(obj(1, 1), 101, 5));
    assert!(complete_until_event(&platform, &mut ring));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_OK);
    assert_eq!(shown_width(&device), Some(64));
    presents_waiting(&device, 0);
    assert_eq!(
        answered(&device, &mut guest, context_close(1)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.control.pool.favoured(), None);
}

/// Presents are shown in the order made: one with nothing to wait for, made after one that waits,
/// waits behind it; a present waiting for a queue that is destroyed is shown, as nothing of that
/// queue will finish; and a writer naming no queue the host holds holds nothing
#[test]
fn presents_keep_their_order_and_a_gone_queue_holds_none() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    let (mut guest, _ring) = surface_and_queue(&device);
    write_writers(&device, BLOCK + WRITERS, &[(obj(1, 1), 9), (obj(1, 7), 3)]);
    guest.send(&device, present_surface(0x77));
    presents_waiting(&device, 1);
    // The scanout, presented next, waits behind it.
    guest.send(&device, present_surface(0));
    presents_waiting(&device, 2);
    assert!(quiet(&guest));

    guest.send(&device, object_destroy(obj(1, 1)));
    let first = guest.seqno - 2;
    assert_eq!(answer_of(&mut guest, first), RGFX_STATUS_OK);
    assert_eq!(answer_of(&mut guest, first + 1), RGFX_STATUS_OK);
    presents_waiting(&device, 0);
    assert_eq!(shown_width(&device), None, "the scanout, presented last");
}

/// A present shows the latest texture a process made over its surface, and an earlier one once
/// the latest is destroyed; a surface with no texture is answered invalid at once
#[test]
fn a_present_shows_the_latest_texture_over_its_surface() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    let (mut guest, _ring) = surface_and_queue(&device);
    guest.send(
        &device,
        texture_from_iosurface(obj(1, 4), 2, obj(1, 2), 0x77, 32, RGFX_SURFACE_NO_WRITERS),
    );
    guest.send(&device, present_surface(0x77));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_OK);
    assert_eq!(shown_width(&device), Some(32));

    guest.send(&device, object_destroy(obj(1, 4)));
    guest.send(&device, present_surface(0x77));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_OK);
    assert_eq!(shown_width(&device), Some(64));

    guest.send(&device, present_surface(0x78));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_INVALID_ARGUMENT);
    assert_eq!(shown_width(&device), Some(64));
}

/// The platform dying at its read of a finished command buffer on its own thread (a failure it has
/// no answer of Metal's for), which ends the process on the paravirt device, fails that command
/// buffer and its context: the buffer's completion is an internal error, the context's later
/// command buffers and creates (its sampler creates among them) are refused at once, its
/// destroys are carried out, another context's work goes on, and once the context closes its
/// number is fresh for the next process
#[test]
fn a_completion_the_platform_refuses_fails_its_context() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT_3: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_3: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const SUBMIT_4: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4_0000;
    const COMPLETE_4: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x6_0000;
    const SUBMIT_NEXT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x8_0000;
    const COMPLETE_NEXT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0xa_0000;
    const RING_DATA: u32 = 0x1_0000;
    for (context, submit, complete) in [(3, SUBMIT_3, COMPLETE_3), (4, SUBMIT_4, COMPLETE_4)] {
        assert_eq!(
            answered(
                &device,
                &mut guest,
                queue_create(context, 1, submit, complete, RING_DATA, 0)
            ),
            RGFX_STATUS_OK
        );
    }
    let mut ring3 = QueueRing::new(&device, SUBMIT_3, COMPLETE_3, u64::from(RING_DATA));
    let mut ring4 = QueueRing::new(&device, SUBMIT_4, COMPLETE_4, u64::from(RING_DATA));
    guest.send(&device, event_create(obj(3, 5)));
    ring3.send(&device, command_buffer(obj(3, 1), 100));
    let cb = committed(&platform, 1)[0];
    assert!(platform.world().complete_dying_read(
        cb,
        "a failure none of Metal's",
        CompleteOn::Thread
    ));
    let failed = ring3.event().expect("the failed buffer's completion");
    let failed = RgfxCbComplete::read_from_bytes(&failed).unwrap();
    assert_eq!((failed.cb_obj_id, failed.status), (100, RGFX_CB_INTERNAL));

    // The context's later command buffer is refused at once, unrun, and so is its create.
    ring3.send(&device, command_buffer(obj(3, 1), 101));
    let refused = ring3.event().expect("the refused buffer's completion");
    let refused = RgfxCbComplete::read_from_bytes(&refused).unwrap();
    assert_eq!((refused.cb_obj_id, refused.status), (101, RGFX_CB_INTERNAL));
    assert!(platform.world().committed().is_empty(), "the buffer ran");
    guest.send(&device, event_create(obj(3, 6)));
    let sampler = RgfxSamplerCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SAMPLER_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id: obj(3, 7),
        max_anisotropy: 1,
        lod_max_clamp_bits: f32::MAX.to_bits(),
        ..RgfxSamplerCreate::default()
    };
    guest.send(&device, sampler.as_bytes().to_vec());
    let answer = guest.event().expect("the sampler create's SAMPLER_STATE");
    let answer = RgfxSamplerState::read_from_bytes(&answer).unwrap();
    assert_eq!(
        (answer.header.seqno, answer.status, answer.gpu_resource_id),
        (guest.seqno, RGFX_STATUS_REFUSED, 0)
    );
    assert_eq!(
        platform
            .call_log()
            .calls_to("newSamplerStateWithDescriptor:")
            .len(),
        0
    );
    assert_eq!(
        answered(&device, &mut guest, context_close(99)),
        RGFX_STATUS_OK
    );
    assert_eq!(
        device.decoder_object_count(),
        3,
        "the two queues and context 3's first event"
    );

    // Another context's work goes on.
    ring4.send(&device, command_buffer(obj(4, 1), 200));
    assert!(complete_until_event(&platform, &mut ring4), "no completion");
    let done = RgfxCbComplete::read_from_bytes(&ring4.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (200, RGFX_CB_OK));

    // The failed context's destroys are carried out.
    guest.send(&device, object_destroy(obj(3, 5)));
    assert_eq!(
        answered(&device, &mut guest, context_close(99)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 2, "the event destroyed");

    // Closed, the context's number is fresh: the next process given it runs its work.
    assert_eq!(
        answered(&device, &mut guest, context_close(3)),
        RGFX_STATUS_OK
    );
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(3, 2, SUBMIT_NEXT, COMPLETE_NEXT, RING_DATA, 0)
        ),
        RGFX_STATUS_OK
    );
    let mut next = QueueRing::new(&device, SUBMIT_NEXT, COMPLETE_NEXT, u64::from(RING_DATA));
    next.send(&device, command_buffer(obj(3, 2), 300));
    assert!(complete_until_event(&platform, &mut next), "no completion");
    let done = RgfxCbComplete::read_from_bytes(&next.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (300, RGFX_CB_OK));
}

/// The platform dying as it lets go of an object, at the object's destroy or at its context's
/// close, refuses that release alone: the destroy and the close are carried out whole, the close
/// is answered, and the device goes on
#[test]
fn a_release_the_platform_dies_at_is_refused() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let world = platform.world();
    let before = world.live("MTLEvent");
    for n in 1..=3 {
        guest.send(&device, event_create(obj(3, n)));
    }
    assert_eq!(
        answered(&device, &mut guest, context_close(99)),
        RGFX_STATUS_OK
    );
    let events: Vec<ObjId> = (world.live("MTLEvent").into_iter())
        .filter(|id| !before.contains(id))
        .collect();
    assert_eq!(events.len(), 3);
    assert_eq!(device.decoder_object_count(), 3);

    // At a destroy.
    world.die_at_release(events[0]);
    guest.send(&device, object_destroy(obj(3, 1)));
    assert_eq!(
        answered(&device, &mut guest, context_close(99)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 2, "the destroy carried out");

    // At the close: the context's other objects go all the same.
    world.die_at_release(events[1]);
    assert_eq!(
        answered(&device, &mut guest, context_close(3)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 0, "the close carried out");

    // The device goes on.
    guest.send(&device, event_create(obj(3, 4)));
    assert_eq!(
        answered(&device, &mut guest, context_close(99)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 1);
}

/// Fails context `context` as the platform does off its op's thread: makes its queue 1 over
/// `submit` and `complete`, runs command buffer 100 on it, and completes the buffer with a read
/// that dies ([`World::complete_dying_read`]), taking its failed completion.
///
/// [`World::complete_dying_read`]: reims_vgpu_core::testkit::objects::World::complete_dying_read
fn fail_context(
    device: &GraphicsDevice<FakePlatform>,
    platform: &FakePlatform,
    guest: &mut Guest,
    context: u32,
    (submit, complete): (u64, u64),
) {
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        answered(
            device,
            guest,
            queue_create(context, 1, submit, complete, RING_DATA, 0)
        ),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(device, submit, complete, u64::from(RING_DATA));
    ring.send(device, command_buffer(obj(context, 1), 100));
    let cb = committed(platform, 1)[0];
    assert!(platform.world().complete_dying_read(
        cb,
        "a failure none of Metal's",
        CompleteOn::Thread
    ));
    let failed = ring.event().expect("the failed buffer's completion");
    let failed = RgfxCbComplete::read_from_bytes(&failed).unwrap();
    assert_eq!((failed.cb_obj_id, failed.status), (100, RGFX_CB_INTERNAL));
}

/// A failed context's queue creates and queries are refused as its other work is: a
/// `QUEUE_CREATE` makes no rings, and a heap texture size query and an argument layout query are
/// answered refused, the platform not asked and nothing written; another context's are answered
#[test]
fn a_failed_context_s_queues_and_queries_are_refused() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const NEXT_SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4_0000;
    const NEXT_COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x6_0000;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const OTHER_BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    const SIZE: u64 = 0x1000;
    fail_context(&device, &platform, &mut guest, 3, (SUBMIT, COMPLETE));

    // The queue create.
    assert_eq!(
        answered(
            &device,
            &mut guest,
            queue_create(3, 2, NEXT_SUBMIT, NEXT_COMPLETE, 0x1_0000, 0)
        ),
        RGFX_STATUS_REFUSED
    );
    assert_eq!(
        device.graphics_queue_count(),
        1,
        "the refused queue's rings"
    );

    // The heap texture size query.
    let size = |obj_id: u64| {
        RgfxHeapTextureSizeAndAlign {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN,
                size: 0,
                seqno: 0,
            },
            obj_id,
            desc: RgfxTextureDescriptor {
                pixel_format: 80,
                texture_type: 2,
                width: 256,
                height: 128,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                ..RgfxTextureDescriptor::default()
            },
        }
        .as_bytes()
        .to_vec()
    };
    let sizes = || {
        platform
            .call_log()
            .calls_to("heapTextureSizeAndAlignWithDescriptor:")
            .len()
    };
    guest.send(&device, size(obj(3, 8)));
    let answer = RgfxSizeAndAlign::read_from_bytes(&guest.event().unwrap()).unwrap();
    assert_eq!(
        (answer.status, answer.size, answer.align),
        (RGFX_STATUS_REFUSED, 0, 0)
    );
    assert_eq!(sizes(), 0);
    guest.send(&device, size(obj(4, 8)));
    let answer = RgfxSizeAndAlign::read_from_bytes(&guest.event().unwrap()).unwrap();
    assert_eq!(answer.status, RGFX_STATUS_OK);
    assert_eq!(sizes(), 1);

    // The argument layout query.
    guest.send(&device, block_grant(3, BLOCK, SIZE));
    guest.send(&device, block_grant(4, OTHER_BLOCK, SIZE));
    let descriptor = RgfxArgumentDescriptor {
        data_type: 60, // MTLDataTypePointer
        index: 0,
        ..RgfxArgumentDescriptor::default()
    };
    let layout = |obj_id: u64, reply_offset: u64| {
        let mut bytes = RgfxArgumentLayoutQuery {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ARGUMENT_LAYOUT,
                size: 0,
                seqno: 0,
            },
            obj_id,
            reply_offset,
            reply_length: SIZE,
            source: RGFX_ARGUMENT_SOURCE_DESCRIPTORS,
            count: 1,
            descriptors_offset: size_of::<RgfxArgumentLayoutQuery>() as u32,
            ..RgfxArgumentLayoutQuery::default()
        }
        .as_bytes()
        .to_vec();
        bytes.extend_from_slice(descriptor.as_bytes());
        bytes
    };
    guest.send(&device, layout(obj(3, 9), BLOCK));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_REFUSED);
    // SAFETY: as `Guest`'s helpers; the block lies in the region.
    assert_eq!(unsafe { guest.load_u64(BLOCK as usize) }, 0, "written");
    guest.send(&device, layout(obj(4, 9), OTHER_BLOCK));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_OK);
}

/// With a memory map the device carries out `SURFACE_PAGES` where the host's page is the
/// guest's 4 KiB, and says so in `FEATURES_LO`; without one it never does. Every device answers
/// a command buffer's ask for its submission (`CB_SUBMITTED`)
#[test]
fn a_memory_map_gives_the_surface_pages_feature_on_a_guest_sized_page() {
    use reims_vgpu_core::testkit::GuestRam;

    let platform = Arc::new(FakePlatform::new());
    let ram = GuestRam::with_range(0x10_0000, 0x10_0000).unwrap();
    platform.fake_vm().adopt(&ram);
    let mut map = crate::MemoryMap::new();
    for r in ram.ranges() {
        // SAFETY: the fake RAM stays mapped while the test, which holds it, runs.
        unsafe {
            map.push(
                r.physical_address,
                r.physical().physical_length,
                r.host().as_ptr(),
            );
        }
    }
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks::default(),
        window: None,
        memory_map: Some(map),
    };
    let device = GraphicsDevice::with_platform(Arc::clone(&platform), config).unwrap();
    let want = if platform.vm().page_size() == 4096 {
        RGFX_FEATURE_SURFACE_PAGES
    } else {
        0
    };
    assert_eq!(
        device.mmio_read(RGFX_REG_FEATURES_LO.into()),
        want | RGFX_FEATURE_CB_SUBMITTED
    );
    assert_eq!(device.mmio_read(RGFX_REG_FEATURES_HI.into()), 0);

    let (device, _) = headless_on(&Arc::new(AtomicU32::new(0)));
    assert_eq!(
        device.mmio_read(RGFX_REG_FEATURES_LO.into()),
        RGFX_FEATURE_CB_SUBMITTED
    );
}
