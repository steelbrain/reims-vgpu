use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use reims_vgpu_graphics::ops::{RGFX_COMMAND_BUFFER_OPS_OFFSET, RgfxCommandBuffer};
use reims_vgpu_graphics::wire::*;
use reims_vgpu_vulkan::core::platform::ObjId;
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;
use reims_vgpu_vulkan::core::testkit::faults::Fault;
use reims_vgpu_vulkan::core::testkit::impls::device::{FAKE_LINEAR_ROW_ALIGNMENT, FAKE_OWN_MEMORY};
use reims_vgpu_vulkan::core::testkit::objects::CompleteOn;
use reims_vgpu_vulkan::core::testkit::{Arg, FakePlatform, HeldCall};
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
            ..GraphicsCallbacks::default()
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
    assert_eq!(device.region_size(), reims_vgpu_graphics::REGION_SIZE);
    assert!(!device.region_base().is_null());
    assert_eq!(device.mmio_read(RGFX_REG_MAGIC.into()), RGFX_MAGIC);
    assert_eq!(
        device.mmio_read(RGFX_REG_REGION_SIZE_LO.into()),
        (reims_vgpu_graphics::REGION_SIZE & 0xffff_ffff) as u32
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
            ..GraphicsCallbacks::default()
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

/// On the Vulkan platform, a texture the guest places at one of its ids and the plain textures the
/// host makes never share an id, and one placed at the guest's last id, past the table a fresh
/// device has, is made and lives. Each test runs in a process of its own, so the device's ids
/// start fresh
#[test]
fn on_vulkan_guest_texture_ids_are_kept_from_the_host_s() {
    use reims_vgpu_vulkan::core::platform::resource::Texture as _;
    use reims_vgpu_vulkan::core::platform::values::TextureDesc;
    let platform = crate::singleton::platform();
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: None,
            ..GraphicsCallbacks::default()
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
    fn new<P: ApiPlatform>(device: &GraphicsDevice<P>) -> Guest {
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
    fn send<P: ApiPlatform>(&mut self, device: &GraphicsDevice<P>, command: Vec<u8>) {
        self.publish(command);
        device.mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    }

    /// [`Self::send`], with no doorbell rung.
    fn publish(&mut self, mut command: Vec<u8>) {
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
    }

    /// Whether the control thread has `consumer_idle` set: it sleeps, or is about to.
    fn control_idle(&self) -> bool {
        // SAFETY: as the helpers above; `consumer_idle` is at 0x80 of the submission ring.
        unsafe {
            AtomicU32::from_ptr(
                self.base
                    .add(RGFX_CONTROL_SUBMIT_OFFSET as usize + 0x80)
                    .cast(),
            )
            .load(Ordering::SeqCst)
                == 1
        }
    }

    /// Arms the control completion ring: the next record raises the control vector.
    fn arm(&self) {
        // SAFETY: as the helpers above; `armed` is at 0xc0 of the completion ring.
        unsafe {
            AtomicU32::from_ptr(
                self.base
                    .add(RGFX_CONTROL_COMPLETE_OFFSET as usize + 0xc0)
                    .cast(),
            )
            .store(1, Ordering::SeqCst);
        }
    }

    /// Whether the completion ring holds an event not taken yet.
    fn event_waiting(&self) -> bool {
        // SAFETY: as the helpers above.
        let produced = unsafe { self.load_u64(RGFX_CONTROL_COMPLETE_OFFSET as usize) };
        produced != self.consumed
    }

    /// The next completion-ring event where one is there now; else `None`, waiting for none.
    fn event_now(&mut self) -> Option<Vec<u8>> {
        // SAFETY: as the helpers above.
        let produced = unsafe { self.load_u64(RGFX_CONTROL_COMPLETE_OFFSET as usize) };
        if produced == self.consumed {
            return None;
        }
        self.event()
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

/// [`command_buffer`] with one op, `SUBMITTED`: a buffer that asks for its `CB_SUBMITTED`, as
/// `commitAndWaitUntilSubmitted` commits it.
fn asking(queue_obj_id: u64, cb_obj_id: u64) -> Vec<u8> {
    use reims_vgpu_graphics::ops::{RGFX_OP_SUBMITTED, RgfxOpHeader};

    let mut message = command_buffer(queue_obj_id, cb_obj_id);
    let submitted = RgfxOpHeader {
        tag: RGFX_OP_SUBMITTED,
        flags: 0,
        len: 8,
    };
    message.extend_from_slice(submitted.as_bytes());
    message[RGFX_COMMAND_BUFFER_OPS_OFFSET as usize - 8..RGFX_COMMAND_BUFFER_OPS_OFFSET as usize]
        .copy_from_slice(&8u64.to_le_bytes());
    message
}

/// The device-info caps blob `device` wrote into its region at [`RGFX_DEVICE_INFO_OFFSET`].
fn device_info_of(device: &GraphicsDevice<FakePlatform>) -> RgfxDeviceInfo {
    // SAFETY: the region is the device's, mapped while it lives, and the blob lies in the
    // completion ring span's unused tail the wire names.
    let bytes = unsafe {
        let p = device.region_base().add(RGFX_DEVICE_INFO_OFFSET as usize);
        std::slice::from_raw_parts(p, size_of::<RgfxDeviceInfo>()).to_vec()
    };
    RgfxDeviceInfo::read_from_bytes(&bytes).unwrap()
}

/// The device-info caps blob is written into the region at [`RGFX_DEVICE_INFO_OFFSET`] at init,
/// mirroring the GPU device's caps and features the guest bundle reads before it makes any queue;
/// on the fake (an M3 Max's answers) it answers no Apple family and no iOS feature set, as an
/// Intel or AMD Mac's GPU does, and the platform's Mac2 and macOS_GPUFamily2_v1
#[test]
fn device_info_is_written_into_the_region_at_init() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    let info = device_info_of(&device);
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
    id_grant(RGFX_CMD_RESOURCE_ID_CHUNK, context, chunk)
}

/// A grant of type `type`, `RESOURCE_ID_CHUNK` or `RESOURCE_ID_SMALL_CHUNK`, giving `context`
/// chunk `chunk` of the texture resource ids.
fn id_grant(r#type: u32, context: u32, chunk: u32) -> Vec<u8> {
    RgfxResourceIdChunk {
        header: RgfxMsgHeader {
            r#type,
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
    // Once a lookup of the context's is answered, everything of its sent before it is.
    let carried_out = |guest: &mut Guest, context: u32| {
        carried_out_of(&device, guest, context);
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
        carried_out(&mut guest, 3),
        2,
        "the texture in the chunk, and the event"
    );
    assert_eq!(
        answered(&device, &mut guest, context_close(3)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 0);
    guest.send(&device, texture_create(obj(4, 1), first));
    assert_eq!(carried_out(&mut guest, 4), 0, "the closed context's chunk");
    guest.send(&device, resource_id_chunk(4, 7));
    guest.send(&device, texture_create(obj(4, 1), first));
    assert_eq!(carried_out(&mut guest, 4), 1, "the chunk given again");
}

/// The kext's `RESOURCE_ID_SMALL_CHUNK` on ring 0 lets its context make textures at its small
/// chunk's 16 ids, as a `RESOURCE_ID_CHUNK` does at its 64, and at no other; at the context's
/// close the small chunk goes with it, and the next context's grant of it, posted after the
/// close, lets that context make textures there.
#[test]
fn a_small_chunk_of_resource_ids_reaches_the_decoder_on_ring_0() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let carried_out = |guest: &mut Guest, context: u32| {
        carried_out_of(&device, guest, context);
        device.decoder_object_count()
    };
    // Small chunk 29: ids 465 to 480, inside chunk 7 of 64.
    let first = 1 + 29 * RGFX_RESOURCE_ID_SMALL_CHUNK;
    let last = first + RGFX_RESOURCE_ID_SMALL_CHUNK - 1;
    guest.send(&device, id_grant(RGFX_CMD_RESOURCE_ID_SMALL_CHUNK, 3, 29));
    guest.send(&device, texture_create(obj(3, 1), first));
    guest.send(&device, texture_create(obj(3, 2), last));
    guest.send(&device, texture_create(obj(3, 3), first - 1));
    guest.send(&device, texture_create(obj(3, 4), last + 1));
    assert_eq!(
        carried_out(&mut guest, 3),
        2,
        "the textures in the small chunk"
    );
    assert_eq!(
        answered(&device, &mut guest, context_close(3)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 0);
    guest.send(&device, texture_create(obj(4, 1), first));
    assert_eq!(
        carried_out(&mut guest, 4),
        0,
        "the closed context's small chunk"
    );
    guest.send(&device, id_grant(RGFX_CMD_RESOURCE_ID_SMALL_CHUNK, 4, 29));
    guest.send(&device, texture_create(obj(4, 1), first));
    assert_eq!(carried_out(&mut guest, 4), 1, "the small chunk given again");
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
        create_queue(
            &device,
            &mut guest,
            context,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            0
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

/// A create posted after its context's close (a submit in flight in the kext as its process
/// closes the device or dies) makes nothing: the close dropped the context's objects once, and no
/// later message drops one made after it.
#[test]
#[ignore = "aud-tests-06: a create that follows its context's close makes an object nothing drops"]
fn a_create_after_its_context_s_close_makes_nothing() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let context = 2;
    let mut guest = Guest::new(&device);
    guest.send(&device, event_create(obj(context, 10)));
    assert_eq!(
        answered(&device, &mut guest, context_close(context)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 0);
    guest.send(&device, event_create(obj(context, 11)));
    // Another context's close after it, answered once ring 0 has carried out the create.
    assert_eq!(
        answered(&device, &mut guest, context_close(context + 1)),
        RGFX_STATUS_OK
    );
    assert_eq!(
        device.decoder_object_count(),
        0,
        "the create after context {context}'s close made an object"
    );
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

/// Creates queue `n` of `context` with its rings at `rings` (submission, completion), each of
/// `data` bytes, as the guest's kext does: each ring's block granted to the process, then the
/// `QUEUE_CREATE`, waiting for its `COMPLETE`. Its status.
fn create_queue(
    device: &GraphicsDevice<FakePlatform>,
    guest: &mut Guest,
    context: u32,
    n: u64,
    rings: (u64, u64),
    data: u32,
    vector: u32,
) -> u32 {
    let span = u64::from(RGFX_RING_HEADER_SIZE + data);
    guest.send(device, block_grant(context, rings.0, span));
    guest.send(device, block_grant(context, rings.1, span));
    let create = queue_create(context, n, rings.0, rings.1, data, vector);
    answered(device, guest, create)
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
fn answered<P: ApiPlatform>(
    device: &GraphicsDevice<P>,
    guest: &mut Guest,
    command: Vec<u8>,
) -> u32 {
    guest.send(device, command);
    let bytes = guest.event().expect("a COMPLETE");
    let complete = RgfxComplete::read_from_bytes(&bytes).unwrap();
    assert_eq!(complete.header.r#type, RGFX_EVT_COMPLETE);
    assert_eq!(complete.header.seqno, guest.seqno);
    complete.status
}

/// Waits for every ring-0 message of `context`'s sent before it to be carried out, as the guest's
/// process knows its messages done: by an answer of its own context's, which the context's lane
/// gives after them. A lookup of a library key no library has, which makes nothing.
fn carried_out_of(device: &GraphicsDevice<FakePlatform>, guest: &mut Guest, context: u32) {
    let lookup = library_lookup(obj(context, 0xfff), 0, [0xee; 32]);
    assert_ne!(
        answered(device, guest, lookup),
        RGFX_STATUS_OK,
        "a lookup's miss"
    );
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

    /// Publishes `command` (its `size` its length) on the submission ring and rings the queue's
    /// doorbell through `mmio_write_taking`, as QEMU rings, only where the host marked the ring
    /// idle, as the guest's writer does (`RgfxRingWriter`): whether it rang.
    fn send_as_guest(&mut self, device: &GraphicsDevice<FakePlatform>, command: Vec<u8>) -> bool {
        self.publish(command);
        let idle = self.marked_idle();
        if idle {
            device.mmio_write_taking(RGFX_REG_DOORBELL.into(), self.submit_off as u32);
        }
        idle
    }

    /// Whether the host marked the submission ring idle (`consumer_idle` 1), so the guest's
    /// writer rings the doorbell for what it has just published.
    fn marked_idle(&self) -> bool {
        // SAFETY: as the helpers above; `consumer_idle` is at 0x80 of the submission ring header,
        // read after the publish's store, as the writer reads it after a full fence.
        let idle = unsafe {
            AtomicU32::from_ptr(self.base.add(self.submit_off as usize + 0x80).cast())
                .load(Ordering::SeqCst)
        };
        idle == 1
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
            ..GraphicsCallbacks::default()
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
        create_queue(
            &device,
            &mut guest,
            context,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            1
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

/// A command buffer too large for its queue's ring, whose whole message lies in a block of its
/// process's (`COMMAND_BUFFER_IN_REGION`), is copied out of the block once, by the queue's service
/// thread, and walked and committed from that copy on its lane: no thread makes a second copy of
/// its message's length on the way, and it completes as one from the ring does.
#[test]
fn an_out_of_line_command_buffer_is_carried_out_from_one_copy() {
    use reims_vgpu_graphics::ops::{RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_END, RgfxOpHeader};

    let platform = Arc::new(FakePlatform::new());
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: None,
            ..GraphicsCallbacks::default()
        },
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(Arc::clone(&platform), config).unwrap();
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    const RING_DATA: u32 = 0x1_0000;
    let (context, mut guest) = (3, Guest::new(&device));
    let queue = obj(context, 1);
    assert_eq!(
        create_queue(
            &device,
            &mut guest,
            context,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            1
        ),
        RGFX_STATUS_OK
    );

    // The buffer's message: 4099 empty blit encoders, larger than the ring, of a length nothing
    // else allocates. The kext granted its block to the queue's process before the buffer.
    let mut message = command_buffer(queue, 100);
    for tag in [RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_END].repeat(4099) {
        let op = RgfxOpHeader {
            tag,
            flags: 0,
            len: 8,
        };
        message.extend_from_slice(op.as_bytes());
    }
    let len = message.len();
    assert!(len > RING_DATA as usize);
    let op_bytes = (len - RGFX_COMMAND_BUFFER_OPS_OFFSET as usize) as u64;
    message[RGFX_COMMAND_BUFFER_OPS_OFFSET as usize - 8..RGFX_COMMAND_BUFFER_OPS_OFFSET as usize]
        .copy_from_slice(&op_bytes.to_le_bytes());
    message[4..8].copy_from_slice(&(len as u32).to_le_bytes());
    guest.send(&device, block_grant(context, BLOCK, 0x2_0000));
    // SAFETY: the block lies inside the device's region, mapped at its base while the device
    // lives, and nothing else writes it.
    unsafe {
        std::ptr::copy_nonoverlapping(
            message.as_ptr(),
            device.region_base().add(BLOCK as usize),
            len,
        );
    }
    let reference = RgfxCommandBufferInRegion {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER_IN_REGION,
            size: size_of::<RgfxCommandBufferInRegion>() as u32,
            seqno: 0,
        },
        region_offset: BLOCK,
        byte_length: len as u64,
    };

    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    let (completed, copies) = crate::counted::allocations_of(len, || {
        ring.send(&device, reference.as_bytes().to_vec());
        complete_until_event(&platform, &mut ring)
    });
    assert!(completed, "no completion");
    let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (100, RGFX_CB_OK));
    assert_eq!(copies, 1, "copies of the {len}-byte message");
}

/// A command buffer that asks for its `CB_SUBMITTED` and need wait for nothing, rung through
/// `mmio_write_taking`, is answered on the ringing thread: its `CB_SUBMITTED` is on its queue's
/// completion ring, the queue's vector raised on that thread, before the write returns; its
/// queue's thread takes, walks and commits it, and it completes as one that thread answered. One
/// that does not ask is left to the queue's thread, and completes as before
#[test]
fn a_doorbell_write_answers_a_command_buffer_on_its_thread() {
    let raised = Arc::new(super::Mutex::new(Vec::<(u32, Option<String>)>::new()));
    let platform = Arc::new(FakePlatform::new());
    let record = Arc::clone(&raised);
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(Arc::new(move |vector| {
                let name = std::thread::current().name().map(str::to_owned);
                record.lock().push((vector, name));
            })),
            ..GraphicsCallbacks::default()
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
        create_queue(
            &device,
            &mut guest,
            context,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            1
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

    ring.publish(asking(queue, 100));
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

/// A doorbell write's answer neither allocates nor frees on the ringing thread, the guest's vCPU,
/// where the library's allocator may purge pages in either call: each of four buffers that ask is
/// answered inside the write, which the allocator counts, its `CB_SUBMITTED` posted and its vector
/// raised there, then completed by its queue's thread; and a write of a buffer that does not ask,
/// which only wakes the queue's thread, calls the allocator no more.
///
/// The write answers a buffer only where the queue's thread holds neither its take nor its ring
/// (`answer_on_doorbell`), which `consumer_idle` does not show alone: a wake that came while the
/// thread was awake (the ring of the buffer before) has it wake once more and look at the ring,
/// though `consumer_idle` read 1 before it woke. A buffer the write left is answered by the
/// thread, as any is, and the next buffer is rung, up to 64 (measured on the M3 Max: in 1 of 30
/// runs of the package's graphics tests, the write left buffer 101).
#[test]
fn a_doorbell_s_answer_allocates_nothing() {
    use std::sync::atomic::AtomicUsize;

    let raised = Arc::new(AtomicUsize::new(0));
    let platform = Arc::new(FakePlatform::new());
    let count = Arc::clone(&raised);
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(Arc::new(move |_| {
                count.fetch_add(1, Ordering::SeqCst);
            })),
            ..GraphicsCallbacks::default()
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
        create_queue(
            &device,
            &mut guest,
            context,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            1
        ),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    let mut answered = 0;
    for cb in 100..164 {
        if answered == 4 {
            break;
        }
        // The queue's thread sleeps with the ring idle, as the guest rings for a buffer.
        wait_idle(&device, SUBMIT_OFF);
        ring.arm();
        ring.publish(asking(queue, cb));
        let raises = raised.load(Ordering::SeqCst);
        let ((), calls) = crate::counted::allocator_calls_in(|| {
            device.mmio_write_taking(RGFX_REG_DOORBELL.into(), SUBMIT_OFF as u32);
        });
        if ring.has_event() {
            assert!(
                raised.load(Ordering::SeqCst) > raises,
                "no raise in the write"
            );
            assert_eq!(
                calls,
                (0, 0),
                "the answer of buffer {cb} called the allocator (allocations, frees)"
            );
            answered += 1;
        } else {
            // The queue's thread held the ring: it answers the buffer itself.
            assert!(
                eventually(|| ring.has_event()),
                "buffer {cb} was never answered"
            );
        }
        let posted = RgfxCbSubmitted::read_from_bytes(&ring.event().unwrap()).unwrap();
        assert_eq!(
            (posted.header.r#type, posted.cb_obj_id),
            (RGFX_EVT_CB_SUBMITTED, cb)
        );
        for _ in 0..3000 {
            platform.world().complete_all(CompleteOn::Caller);
            if ring.has_event() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let done = RgfxCbComplete::read_from_bytes(&ring.event().expect("a CB_COMPLETE")).unwrap();
        assert_eq!((done.cb_obj_id, done.status), (cb, RGFX_CB_OK));
    }
    assert_eq!(answered, 4, "the writes answered {answered} buffers of 64");
    // One that does not ask, with an id the loop's buffers do not use: the write leaves it to
    // the queue's thread.
    wait_idle(&device, SUBMIT_OFF);
    ring.publish(command_buffer(queue, 164));
    let ((), calls) = crate::counted::allocator_calls_in(|| {
        device.mmio_write_taking(RGFX_REG_DOORBELL.into(), SUBMIT_OFF as u32);
    });
    assert_eq!(
        calls,
        (0, 0),
        "a write that only woke the queue's thread called the allocator (allocations, frees)"
    );
    for _ in 0..3000 {
        platform.world().complete_all(CompleteOn::Caller);
        if ring.has_event() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let done = RgfxCbComplete::read_from_bytes(&ring.event().expect("a CB_COMPLETE")).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (164, RGFX_CB_OK));
}

/// A landed download's answer at its ring-0 doorbell (`mmio_write_taking`) neither allocates nor
/// frees on the ringing thread, the guest's vCPU: each of four downloads of a block with no device
/// cache, whose pages hold it, is answered inside the write, its `COMPLETE` posted and the control
/// vector raised on that thread. The write answers only where the control thread sleeps with
/// nothing in flight, which `consumer_idle` does not show alone (the thread sets it before it
/// sleeps): a download the write left is answered by the control thread, and the next is rung,
/// up to 64.
#[test]
fn a_landed_download_s_answer_at_its_doorbell_allocates_nothing() {
    use std::cell::Cell;
    use std::sync::atomic::AtomicUsize;

    thread_local! {
        /// Set while this thread rings the doorbell.
        static RINGING: Cell<bool> = const { Cell::new(false) };
    }
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let here = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&here);
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(Arc::new(move |_| {
                if RINGING.with(Cell::get) {
                    count.fetch_add(1, Ordering::SeqCst);
                }
            })),
            ..GraphicsCallbacks::default()
        },
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(Arc::new(FakePlatform::new()), config).unwrap();
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let download = RgfxSurfaceDownload {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SURFACE_DOWNLOAD,
            size: 0,
            seqno: 0,
        },
        block_offset: BLOCK,
    };
    let mut answered = 0;
    for _ in 0..64 {
        if answered == 4 {
            break;
        }
        assert!(
            eventually(|| guest.control_idle()),
            "the control thread never slept"
        );
        guest.arm();
        guest.publish(download.as_bytes().to_vec());
        let raised = here.load(Ordering::SeqCst);
        let ((), calls) = crate::counted::allocator_calls_in(|| {
            RINGING.with(|r| r.set(true));
            device.mmio_write_taking(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
            RINGING.with(|r| r.set(false));
        });
        if here.load(Ordering::SeqCst) > raised {
            assert!(guest.event_waiting(), "raised with no answer");
            assert_eq!(
                calls,
                (0, 0),
                "the answer of download {} called the allocator (allocations, frees)",
                guest.seqno
            );
            answered += 1;
        }
        let bytes = guest.event().expect("an answer");
        let complete = RgfxComplete::read_from_bytes(&bytes).unwrap();
        assert_eq!(
            (
                complete.header.r#type,
                complete.header.seqno,
                complete.status
            ),
            (RGFX_EVT_COMPLETE, guest.seqno, RGFX_STATUS_OK)
        );
    }
    assert_eq!(
        answered, 4,
        "the writes answered {answered} downloads of 64"
    );
}

/// What a lone command buffer that asks for its `CB_SUBMITTED` costs from its doorbell write to
/// the record on its queue's completion ring, the queue's thread asleep before each, as for a
/// guest's `commitAndWaitUntilSubmitted` on an idle queue: answered inside the write
/// (`mmio_write_taking`, the vCPU's time), and taken by the queue's thread that the write wakes
/// (`mmio_write`). Prints the mean, median, 90th and 99th percentiles and the most, in
/// microseconds: a measurement, run by hand (`--run-ignored only --no-capture`, in a release
/// build).
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_lone_buffer_s_doorbell_to_its_cb_submitted() {
    use std::time::Instant;

    const ROUNDS: u64 = 2000;
    let platform = Arc::new(FakePlatform::new());
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(Arc::new(|_| {})),
            ..GraphicsCallbacks::default()
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
        create_queue(
            &device,
            &mut guest,
            context,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            1
        ),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    let stats = |name: &str, mut us: Vec<f64>| {
        us.sort_by(f64::total_cmp);
        let n = us.len();
        let mean = us.iter().sum::<f64>() / n as f64;
        println!(
            "{name}: mean {mean:.2} p50 {:.2} p90 {:.2} p99 {:.2} max {:.2} us (n {n})",
            us[n / 2],
            us[n * 9 / 10],
            us[n * 99 / 100],
            us[n - 1]
        );
    };
    let finish = |ring: &mut QueueRing, cb: u64| {
        for _ in 0..3000 {
            platform.world().complete_all(CompleteOn::Caller);
            if ring.has_event() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let done = RgfxCbComplete::read_from_bytes(&ring.event().expect("a CB_COMPLETE")).unwrap();
        assert_eq!((done.cb_obj_id, done.status), (cb, RGFX_CB_OK));
        ring.scheduled.clear();
    };
    for taking in [true, false] {
        let mut us = Vec::new();
        for round in 0..ROUNDS {
            let cb = 1000 + round + if taking { 0 } else { ROUNDS };
            wait_idle(&device, SUBMIT_OFF);
            // A pause as the guest's, so the queue's thread sleeps as it does between frames.
            std::thread::sleep(Duration::from_micros(200));
            ring.arm();
            ring.publish(asking(queue, cb));
            let start = Instant::now();
            if taking {
                device.mmio_write_taking(RGFX_REG_DOORBELL.into(), SUBMIT_OFF as u32);
                assert!(
                    ring.has_event(),
                    "buffer {cb} was not answered in the write"
                );
            } else {
                device.mmio_write(RGFX_REG_DOORBELL.into(), SUBMIT_OFF as u32);
                while !ring.has_event() {
                    assert!(start.elapsed() < Duration::from_secs(2), "no CB_SUBMITTED");
                    std::hint::spin_loop();
                }
            }
            us.push(start.elapsed().as_secs_f64() * 1e6);
            let posted = RgfxCbSubmitted::read_from_bytes(&ring.event().unwrap()).unwrap();
            assert_eq!(posted.header.r#type, RGFX_EVT_CB_SUBMITTED);
            finish(&mut ring, cb);
        }
        stats(
            if taking {
                "answered in the doorbell write"
            } else {
                "taken by the queue's thread"
            },
            us,
        );
    }
}

/// What a command buffer that asks for its `CB_SUBMITTED` costs from its doorbell write to the
/// record on its queue's completion ring, as for a guest's `commitAndWaitUntilSubmitted` of a
/// tiny buffer, rung for through `mmio_write_taking` as QEMU rings, where the host marked the ring
/// idle, as the guest rings: on an idle queue, and just
/// behind a short buffer (a sampler bind, walked on the queue's service thread where its commit
/// waits for nothing) that does not ask, rung for `GAP` µs before it with the queue's thread
/// asleep, whose walk is made to take `WALK` µs by a spin after its first op, standing for a GPU
/// driver's record and submit of a few fills. The buffer behind is split by where it was
/// answered: inside its write, or after it (by the queue's thread, which had to wake), and the
/// answers that came only once the walk before it had ended are counted. Beside
/// them, round by round so each sees the same load on the host, the idle buffer rung through
/// `mmio_write`, which only wakes the queue's thread: what that wake costs here. Prints the mean,
/// median, 90th and 99th percentiles and the most, in microseconds: a measurement, run by hand
/// (`--run-ignored only --no-capture`, in a release build).
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn an_asking_buffer_s_doorbell_to_its_cb_submitted_behind_a_walk() {
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;

    const ROUNDS: usize = 2000;
    /// The behind buffer's walk, in µs.
    const WALK: u64 = 100;
    /// From the doorbell write of the buffer before to the asking one's, in µs: the guest's make,
    /// encode and commit of its next buffer.
    const GAP: u64 = 20;
    let spin_for = |us: u64| {
        let start = Instant::now();
        while start.elapsed() < Duration::from_micros(us) {
            std::hint::spin_loop();
        }
    };
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, _guest, mut ring) = one_queue(&raised);
    let (walking, walked) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let (slow, ended) = (Arc::clone(&walking), Arc::clone(&walked));
    device
        .control
        .decoder()
        .set_walk_hook(Some(Arc::new(move |index: usize| {
            if index == 0 && slow.swap(false, Ordering::SeqCst) {
                spin_for(WALK);
                ended.store(true, Ordering::SeqCst);
            }
        })));
    let submit_off = ring.submit_off;
    let taking = || device.mmio_write_taking(RGFX_REG_DOORBELL.into(), submit_off as u32);
    // (idle, answered in the write), (idle, the queue's thread woken), (behind, in the write),
    // (behind, after it).
    let mut us: [Vec<f64>; 4] = Default::default();
    let (mut cb, mut after_the_walk) = (1000, 0);
    for round in 0..3 * ROUNDS {
        let (behind, plain) = (round % 3 == 2, round % 3 == 1);
        wait_idle(&device, submit_off);
        // A pause as the guest's, so the queue's thread sleeps as it does between frames.
        std::thread::sleep(Duration::from_micros(200));
        ring.arm();
        let first = cb;
        if behind {
            walked.store(false, Ordering::SeqCst);
            walking.store(true, Ordering::SeqCst);
            ring.publish(sampler_command_buffer(obj(1, 1), cb, obj(1, 4)));
            taking();
            spin_for(GAP);
            cb += 1;
        }
        ring.publish(asking(obj(1, 1), cb));
        let start = Instant::now();
        if plain {
            device.mmio_write(RGFX_REG_DOORBELL.into(), submit_off as u32);
        } else if ring.marked_idle() {
            taking();
        }
        let in_write = !plain && ring.has_event();
        while !ring.has_event() {
            assert!(start.elapsed() < Duration::from_secs(2), "no CB_SUBMITTED");
            std::hint::spin_loop();
        }
        let took = start.elapsed().as_secs_f64() * 1e6;
        if behind && walked.load(Ordering::SeqCst) {
            after_the_walk += 1;
        }
        us[match (behind, plain, in_write) {
            (false, false, _) => 0,
            (false, true, _) => 1,
            (true, _, true) => 2,
            (true, _, false) => 3,
        }]
        .push(took);
        let posted = RgfxCbSubmitted::read_from_bytes(&ring.event().unwrap()).unwrap();
        assert_eq!(
            (posted.header.r#type, posted.cb_obj_id),
            (RGFX_EVT_CB_SUBMITTED, cb)
        );
        for done in first..=cb {
            assert!(complete_until_event(&platform, &mut ring), "no completion");
            let complete = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
            assert_eq!((complete.cb_obj_id, complete.status), (done, RGFX_CB_OK));
        }
        ring.scheduled.clear();
        cb += 1;
    }
    let stats = |name: &str, us: &[f64]| {
        let mut us = us.to_vec();
        us.sort_by(f64::total_cmp);
        let n = us.len();
        if n == 0 {
            println!("{name}: none");
            return;
        }
        let mean = us.iter().sum::<f64>() / n as f64;
        println!(
            "{name}: mean {mean:.2} p50 {:.2} p90 {:.2} p99 {:.2} max {:.2} us (n {n})",
            us[n / 2],
            us[n * 9 / 10],
            us[n * 99 / 100],
            us[n - 1]
        );
    };
    stats("idle, answered in the write", &us[0]);
    stats("idle, rung plainly: the queue's thread's wake", &us[1]);
    let behind = format!("behind a {WALK} us walk rung for {GAP} us before");
    stats(&behind, &[us[2].as_slice(), us[3].as_slice()].concat());
    stats(&format!("{behind}, answered in the write"), &us[2]);
    stats(&format!("{behind}, answered after it"), &us[3]);
    println!("{behind}: {after_the_walk} of {ROUNDS} answered once the walk had ended");
}

/// What a lone command buffer costs from its doorbell write to its `CB_SCHEDULED` on its queue's
/// completion ring, as for a guest's commit on an idle queue whose thread sleeps before each: a
/// sampler bind whose walk is made to take `WALK` us by a spin after its first op, standing for a
/// GPU runtime's record and submit before it takes the buffer's work (the fake takes it as it is
/// committed). One that asks for its `CB_SUBMITTED`, rung for through `mmio_write_taking` as QEMU
/// rings, and one that does not, rung for the same way, so the queue's thread takes it. Prints the
/// mean, median, 90th and 99th percentiles and the most, in microseconds: a measurement, run by
/// hand (`--run-ignored only --no-capture`, in a release build).
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_lone_buffer_s_doorbell_to_its_cb_scheduled() {
    use reims_vgpu_graphics::ops::{RGFX_OP_SUBMITTED, RgfxOpHeader};
    use std::time::Instant;

    const ROUNDS: usize = 2000;
    /// The walk, in us.
    const WALK: u64 = 150;
    let spin_for = |us: u64| {
        let start = Instant::now();
        while start.elapsed() < Duration::from_micros(us) {
            std::hint::spin_loop();
        }
    };
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, _guest, mut ring) = one_queue(&raised);
    device
        .control
        .decoder()
        .set_walk_hook(Some(Arc::new(move |index: usize| {
            if index == 0 {
                spin_for(WALK);
            }
        })));
    let asking_sampler = |cb: u64| {
        let mut message = sampler_command_buffer(obj(1, 1), cb, obj(1, 4));
        let ops_at = RGFX_COMMAND_BUFFER_OPS_OFFSET as usize;
        let op_bytes = u64::from_le_bytes(message[ops_at - 8..ops_at].try_into().unwrap()) + 8;
        message[ops_at - 8..ops_at].copy_from_slice(&op_bytes.to_le_bytes());
        let submitted = RgfxOpHeader {
            tag: RGFX_OP_SUBMITTED,
            flags: 0,
            len: 8,
        };
        message.extend_from_slice(submitted.as_bytes());
        message
    };
    let submit_off = ring.submit_off;
    // (asks, does not ask)
    let mut us: [Vec<f64>; 2] = Default::default();
    for (round, cb) in (0..2 * ROUNDS).zip(3000_u64..) {
        let asks = round % 2 == 0;
        wait_idle(&device, submit_off);
        // A pause as the guest's, so the queue's thread sleeps as it does between frames.
        std::thread::sleep(Duration::from_micros(200));
        ring.arm();
        ring.publish(if asks {
            asking_sampler(cb)
        } else {
            sampler_command_buffer(obj(1, 1), cb, obj(1, 4))
        });
        let start = Instant::now();
        device.mmio_write_taking(RGFX_REG_DOORBELL.into(), submit_off as u32);
        while !ring.scheduled.iter().any(|&(_, id)| id == cb) {
            if ring.has_event() {
                let posted = RgfxCbSubmitted::read_from_bytes(&ring.event().unwrap()).unwrap();
                assert_eq!(
                    (posted.header.r#type, posted.cb_obj_id),
                    (RGFX_EVT_CB_SUBMITTED, cb)
                );
            }
            assert!(start.elapsed() < Duration::from_secs(2), "no CB_SCHEDULED");
            std::hint::spin_loop();
        }
        us[usize::from(!asks)].push(start.elapsed().as_secs_f64() * 1e6);
        assert!(complete_until_event(&platform, &mut ring), "no completion");
        let complete = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
        assert_eq!((complete.cb_obj_id, complete.status), (cb, RGFX_CB_OK));
        ring.scheduled.clear();
    }
    let stats = |name: &str, us: &[f64]| {
        let mut us = us.to_vec();
        us.sort_by(f64::total_cmp);
        let n = us.len();
        let mean = us.iter().sum::<f64>() / n as f64;
        println!(
            "{name}: mean {mean:.2} p50 {:.2} p90 {:.2} p99 {:.2} max {:.2} us (n {n})",
            us[n / 2],
            us[n * 9 / 10],
            us[n * 99 / 100],
            us[n - 1]
        );
    };
    stats(
        &format!("doorbell to CB_SCHEDULED, asking, a {WALK} us walk"),
        &us[0],
    );
    stats(
        &format!("doorbell to CB_SCHEDULED, not asking, a {WALK} us walk"),
        &us[1],
    );
}

/// Waits about 2 s at most for the service thread of the queue whose submission ring is at region
/// offset `submit_off` to sleep with its ring idle (its `consumer_idle` 1), as the guest finds it
/// when it rings for a buffer.
fn wait_idle(device: &GraphicsDevice<FakePlatform>, submit_off: u64) {
    for _ in 0..2000 {
        // SAFETY: the region is the device's, mapped while it lives; `consumer_idle` is at 0x80
        // of the submission ring's header, which the host writes atomically.
        let idle = unsafe {
            AtomicU32::from_ptr(device.region_base().add(submit_off as usize + 0x80).cast())
                .load(Ordering::SeqCst)
        };
        if idle == 1 {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The name of the thread each raise that called the device ran on, once its calls returned.
type Returned = Arc<super::Mutex<Vec<Option<String>>>>;

/// A headless device whose first `raise_interrupt` calls the device at once, as the header lets
/// it: it reads `STATUS`, rings the queue whose submission ring is at `ring` through
/// `mmio_write_taking`, and resets the device, then records the thread it ran on.
fn calling_back(ring: u64) -> (Arc<GraphicsDevice<FakePlatform>>, Returned) {
    use std::sync::atomic::AtomicBool;
    use std::sync::{OnceLock, Weak};

    let handle = Arc::new(OnceLock::<Weak<GraphicsDevice<FakePlatform>>>::new());
    let returned = Returned::default();
    let (from, record, once) = (
        Arc::clone(&handle),
        Arc::clone(&returned),
        AtomicBool::new(false),
    );
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(Arc::new(move |_vector| {
                if once.swap(true, Ordering::SeqCst) {
                    return;
                }
                let Some(device) = from.get().and_then(Weak::upgrade) else {
                    return;
                };
                device.mmio_read(RGFX_REG_STATUS.into());
                device.mmio_write_taking(RGFX_REG_DOORBELL.into(), ring as u32);
                device.reset();
                drop(device);
                let name = std::thread::current().name().map(str::to_owned);
                record.lock().push(name);
            })),
            ..GraphicsCallbacks::default()
        },
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(Arc::new(FakePlatform::new()), config).unwrap();
    let device = Arc::new(device);
    handle.set(Arc::downgrade(&device)).unwrap();
    (device, returned)
}

/// A raise may call the device at once, as the header lets it, wherever it comes from: at a
/// command buffer's `CB_SUBMITTED` posted on its queue's thread and on a doorbell's thread, and at
/// a present the host answers at once with no window, on the presents' lane. Its `STATUS` read, its doorbell through `mmio_write_taking` and its reset each
/// return, and the device is reset
#[test]
fn a_raise_may_call_the_device_at_once() {
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const RING_DATA: u32 = 0x1_0000;
    let (context, queue) = (3, obj(3, 1));
    // A started device with the queue made, its thread asleep with the ring idle, and its
    // completion ring armed.
    let started = || {
        let (device, returned) = calling_back(SUBMIT_OFF);
        device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
        let mut guest = Guest::new(&device);
        let created = create_queue(
            &device,
            &mut guest,
            context,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            1,
        );
        assert_eq!(created, RGFX_STATUS_OK);
        let ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
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
        (device, returned, guest, ring)
    };
    let returned_on = |returned: &Returned| {
        for _ in 0..3000 {
            if let Some(name) = returned.lock().first() {
                return name.clone();
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("the raise's calls never returned");
    };
    let reset = |device: &GraphicsDevice<FakePlatform>| {
        assert_eq!(device.mmio_read(RGFX_REG_STATUS.into()), RGFX_STATUS_RESET);
    };

    // On the queue's thread, which the doorbell wakes to take the buffer.
    let (device, returned, _guest, mut ring) = started();
    ring.publish(asking(queue, 100));
    device.mmio_write(RGFX_REG_DOORBELL.into(), SUBMIT_OFF as u32);
    let on = returned_on(&returned);
    assert!(
        on.as_deref()
            .is_some_and(|name| name.starts_with("reims-vgpu.graphics-queue-")),
        "raised on {on:?}"
    );
    reset(&device);

    // On the doorbell's thread, inside the write that takes the buffer.
    let (device, returned, _guest, mut ring) = started();
    ring.publish(asking(queue, 100));
    device.mmio_write_taking(RGFX_REG_DOORBELL.into(), SUBMIT_OFF as u32);
    let here = std::thread::current().name().map(str::to_owned);
    assert_eq!(returned.lock().first(), Some(&here));
    reset(&device);

    // On the presents' lane (one of ring 0's threads), at a present of the scanout, which the host
    // shows and answers at once.
    let (device, returned, mut guest, _ring) = started();
    // SAFETY: as `Guest`'s helpers; `armed` is at 0xc0 of the control completion ring.
    unsafe {
        AtomicU32::from_ptr(
            device
                .region_base()
                .add(RGFX_CONTROL_COMPLETE_OFFSET as usize + 0xc0)
                .cast(),
        )
        .store(1, Ordering::SeqCst);
    }
    guest.send(&device, present_surface(0));
    let on = returned_on(&returned);
    assert!(
        (on.as_deref()).is_some_and(|name| name.starts_with("reims-vgpu.graphics-ring0-")),
        "{on:?}"
    );
    reset(&device);
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

/// What the host's cursor callbacks were told, in order.
#[derive(Debug, PartialEq)]
enum Told {
    Glyph(u32, u32, (u16, u16), Vec<u8>),
    Show(u8),
    Move,
}

/// The host's cursor callbacks, recording what they are told and the threads they run on.
fn telling(told: &Arc<Mutex<Vec<(Told, std::thread::ThreadId)>>>) -> GraphicsCallbacks {
    let (glyph, show, moved) = (Arc::clone(told), Arc::clone(told), Arc::clone(told));
    let here = || std::thread::current().id();
    GraphicsCallbacks {
        cursor_glyph: Some(Arc::new(move |g: &CursorGlyph<'_>| {
            let told = Told::Glyph(g.width, g.height, g.hot_spot, g.rgba.to_vec());
            glyph.lock().push((told, here()));
        })),
        cursor_show: Some(Arc::new(move |s| show.lock().push((Told::Show(s), here())))),
        cursor_move: Some(Arc::new(move || moved.lock().push((Told::Move, here())))),
        ..GraphicsCallbacks::default()
    }
}

/// The guest's cursor reaches the host's callbacks off the threads that took it, in the order the
/// device took its commands: the image as straight RGBA with its hot spot, that it shows once it
/// has an image while the guest shows it and hides when the guest hides it, and each move, the
/// hot spot's place kept to 0 to 65535; a reset puts it at (0, 0), still hidden with no image
#[test]
fn the_guest_s_cursor_reaches_the_host() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let config = GraphicsConfig {
        callbacks: telling(&told),
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(Arc::new(FakePlatform::new()), config).unwrap();
    let wait = |count: usize| {
        for _ in 0..2000 {
            if told.lock().len() >= count {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let told = told.lock();
        assert_eq!(told.len(), count, "{told:?}");
        told.iter()
            .map(|(t, _)| format!("{t:?}"))
            .collect::<Vec<_>>()
    };
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    // The kext's cursor slot, at the region's end: two premultiplied BGRA pixels.
    let slot = device.region_size() - RGFX_CURSOR_SPAN;
    // SAFETY: the slot lies in the device's region, mapped while the device lives.
    unsafe {
        guest.write(slot as usize, &[10, 20, 30, 255, 64, 32, 0, 128]);
    }
    let set = RgfxSetCursor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_CURSOR,
            size: 0,
            seqno: 0,
        },
        display: 0,
        width: 2,
        height: 1,
        hot_x: 1,
        hot_y: 0,
        reserved_24: 0,
        offset: slot,
    };
    assert_eq!(
        answered(&device, &mut guest, set.as_bytes().to_vec()),
        RGFX_STATUS_OK
    );
    let move_to = |x, y, visible| {
        RgfxMoveCursor {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_MOVE_CURSOR,
                size: 0,
                seqno: 0,
            },
            display: 0,
            x,
            y,
            visible,
        }
        .as_bytes()
        .to_vec()
    };
    guest.send(&device, move_to(100, 50, 1));
    let glyph = Told::Glyph(2, 1, (1, 0), vec![30, 20, 10, 255, 0, 64, 128, 128]);
    assert_eq!(
        wait(3),
        [format!("{glyph:?}"), "Show(1)".into(), "Move".into()]
    );
    assert_eq!(device.cursor_position(), (100, 50));

    guest.send(&device, move_to(-5, 70_000, 0));
    assert_eq!(wait(5)[3..], ["Move", "Show(0)"]);
    assert_eq!(device.cursor_position(), (0, 65535));

    device.reset();
    assert_eq!(wait(6)[5..], ["Move"]);
    assert_eq!(device.cursor_position(), (0, 0));
    let here = std::thread::current().id();
    assert!(told.lock().iter().all(|(_, on)| *on != here));
}

/// The window's sink makes its queue unshared and imports nothing at its making, then only the
/// host pages under the scanout it draws (to `stride × height` bytes on), again only when the
/// scanout moves or changes size, and kept to the region
#[test]
fn the_sink_imports_only_the_scanout_s_pages() {
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;

    const IMPORT: &str = "newBufferWithBytesNoCopy:length:options:deallocator:";
    let platform = Arc::new(FakePlatform::new());
    let region = Arc::new(Region::new(reims_vgpu_graphics::REGION_SIZE as usize).unwrap());
    let shown = Arc::new(Shown::default());
    let presented = Arc::new(Presented::<FakePlatform>::default());
    let draw = sink::ScanoutDraw::new(Arc::clone(&platform), &region, &shown, &presented).unwrap();
    assert_eq!(platform.call_log().count(IMPORT), 0);
    // Its queue is unshared: each draw is on a present's path.
    {
        use reims_vgpu_vulkan::core::testkit::impls::device::UNSHARED_QUEUE;
        assert_eq!(platform.call_log().count(UNSHARED_QUEUE), 1);
        assert_eq!(platform.call_log().count("newCommandQueue"), 0);
    }

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

/// A graphics device on the testkit's platform with a display window. The platform draws no
/// sink, so the window never opens: the test draws what it shows through a sink of its own
/// ([`windowed_draw`]), and counts the frames it is asked for ([`frames_asked`]).
fn windowed() -> (GraphicsDevice<FakePlatform>, Arc<FakePlatform>) {
    let platform = Arc::new(FakePlatform::new());
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks::default(),
        window: Some(WindowConfig::default()),
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(Arc::clone(&platform), config).unwrap();
    (device, platform)
}

/// A sink over what `device`'s window shows, as the window's own draws it, and a drawable of
/// `width` x `height` for it.
fn windowed_draw(
    device: &GraphicsDevice<FakePlatform>,
    platform: &Arc<FakePlatform>,
    width: u64,
    height: u64,
) -> (
    sink::ScanoutDraw<FakePlatform>,
    <FakePlatform as Platform>::Texture,
) {
    let shown = device.shown.as_ref().expect("a window");
    let draw = sink::ScanoutDraw::new(
        Arc::clone(platform),
        &device.control.region,
        shown,
        &device.control.presented,
    )
    .unwrap();
    let drawable = platform.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width,
        height,
        usage: 4,
        ..TextureDesc::default()
    });
    let drawable = platform
        .metal_device(None)
        .unwrap()
        .new_texture(OnFault::Return, &drawable)
        .unwrap()
        .unwrap();
    (draw, drawable)
}

/// `copyFromBuffer:...toTexture:...`, the blit of the scanout into a drawable.
const SCANOUT_BLIT: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:\
                            sourceSize:toTexture:destinationSlice:destinationLevel:\
                            destinationOrigin:";

/// Whether the window's next draw blits the scanout at `stride` bytes a row into `drawable`, at
/// its whole size: what the window shows then.
fn draws_the_scanout(
    platform: &FakePlatform,
    draw: &sink::ScanoutDraw<FakePlatform>,
    drawable: &<FakePlatform as Platform>::Texture,
    stride: u64,
) -> bool {
    use reims_vgpu_vulkan::core::platform::resource::Texture as _;
    let blits = platform.call_log().count(SCANOUT_BLIT);
    let rect = reims_vgpu_window::PixelRect {
        x: 0,
        y: 0,
        width: drawable.width(),
        height: drawable.height(),
    };
    if draw.encode(drawable, rect).is_none() || platform.call_log().count(SCANOUT_BLIT) != blits + 1
    {
        return false;
    }
    let calls = platform.call_log().calls();
    let blit = calls.iter().rev().find(|c| c.sel == SCANOUT_BLIT).unwrap();
    blit.args[1..3] == [Arg::U64(0), Arg::U64(stride)]
}

/// The frames `device`'s window was asked to draw so far.
fn frames_asked(device: &GraphicsDevice<FakePlatform>) -> u64 {
    device
        .window
        .as_ref()
        .expect("a window")
        .window()
        .frames_asked()
}

/// Waits ([`eventually`]) for `device`'s window to be asked for `count` frames past `from`.
fn frames_asked_past(device: &GraphicsDevice<FakePlatform>, from: u64, count: u64) -> bool {
    eventually(|| frames_asked(device) >= from + count)
}

/// The guest's driver fails after it took the device, as the kext does where the device lists no
/// usable display or `IOFramebuffer`'s start fails: the firmware sets its scanout, the kext resets
/// the device to take it, starts it and asks `GET_DISPLAYS`, and resets it on the way out. The
/// window keeps the firmware's scanout, so the frames the OS then draws into the firmware's
/// framebuffer reach it: each refresh asks the window to draw that scanout again, and the
/// window's draw blits it out of the region
#[test]
fn a_failure_after_the_take_keeps_the_firmware_s_scanout_in_the_window() {
    let (device, platform) = windowed();
    let (draw, drawable) = windowed_draw(&device, &platform, 64, 32);
    let scanout = Scanout {
        width: 64,
        height: 32,
        stride: 256,
        offset: RGFX_ALLOCATIONS_OFFSET,
    };
    let firmware = || {
        RgfxSetScanout {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SET_SCANOUT,
                size: 0,
                seqno: 0,
            },
            display: 0,
            width: 64,
            height: 32,
            stride: 256,
            offset: RGFX_ALLOCATIONS_OFFSET,
        }
        .as_bytes()
        .to_vec()
    };
    let get_displays = RgfxGetDisplays {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_GET_DISPLAYS,
            size: 0,
            seqno: 0,
        },
    }
    .as_bytes()
    .to_vec();

    // The firmware's start and its mode.
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    assert_eq!(answered(&device, &mut guest, firmware()), RGFX_STATUS_OK);
    assert!(draws_the_scanout(&platform, &draw, &drawable, 256));

    // The kext's take, its start and its list of the displays; then its failure.
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    guest.send(&device, get_displays);
    let displays = guest.event().expect("DISPLAYS");
    assert_eq!(
        RgfxMsgHeader::read_from_prefix(&displays).unwrap().0.r#type,
        RGFX_EVT_DISPLAYS
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    reset_done(&device);
    // Past a refresh that read the scanout before the reset.
    std::thread::sleep(Duration::from_millis(50));

    // The OS draws its next frame into the firmware's framebuffer.
    let frames = frames_asked(&device);
    // SAFETY: the scanout lies in the device's region, mapped while the device lives.
    unsafe {
        std::ptr::write_bytes(
            device.region_base().add(RGFX_ALLOCATIONS_OFFSET as usize),
            0x5a,
            256 * 32,
        );
    }
    assert!(
        frames_asked_past(&device, frames, 2),
        "no refresh drew the firmware's scanout after the failure"
    );
    let shown = device.shown.as_ref().unwrap();
    assert_eq!(*shown.scanout.lock(), Some(scanout));
    assert!(
        draws_the_scanout(&platform, &draw, &drawable, 256),
        "the window's draw shows nothing of the firmware's scanout"
    );
}

/// A reset of the device while its display pipe shows a surface, as the kext's stop makes: the
/// window lets go of the surface and shows the scanout again, as a GPU's display goes back to
/// its framebuffer across its driver's reset, and the next present of the guest's driver shows
/// its surface
#[test]
fn a_reset_shows_the_scanout_in_place_of_the_surface_shown() {
    let (device, platform) = windowed();
    let (draw, drawable) = windowed_draw(&device, &platform, 64, 32);
    let (mut guest, _ring) = surface_and_queue(&device);
    let scanout = RgfxSetScanout {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_SCANOUT,
            size: 0,
            seqno: 0,
        },
        display: 0,
        width: 64,
        height: 32,
        stride: 256,
        offset: RGFX_ALLOCATIONS_OFFSET,
    }
    .as_bytes()
    .to_vec();
    assert_eq!(answered(&device, &mut guest, scanout), RGFX_STATUS_OK);
    // With no window open, the present is answered as it is shown.
    let status = answered(&device, &mut guest, present_surface(0x77));
    assert_eq!(status, RGFX_STATUS_OK);
    assert_eq!(shown_width(&device), Some(64));

    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    reset_done(&device);
    for _ in 0..2000 {
        if device.control.presented.shown().is_none() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        device.control.presented.shown().is_none(),
        "the surface stayed"
    );
    assert!(
        draws_the_scanout(&platform, &draw, &drawable, 256),
        "the window's draw shows nothing after the reset"
    );

    // The guest's driver starts again and presents its surface.
    zero_ring_headers(&device, [SUBMIT, COMPLETE]);
    let (mut guest, _ring) = surface_and_queue(&device);
    let status = answered(&device, &mut guest, present_surface(0x77));
    assert_eq!(status, RGFX_STATUS_OK);
    assert_eq!(shown_width(&device), Some(64));
}

/// While the window shows the guest's scanout, each refresh asks it to draw the scanout again,
/// as the guest may have written it since; before the guest sets one, a refresh asks for nothing
#[test]
fn a_refresh_draws_the_scanout_again_while_no_surface_shows() {
    let (device, _platform) = windowed();
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    // Six refreshes at the window's 60 Hz.
    let frames = frames_asked(&device);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        frames_asked(&device),
        frames,
        "a refresh with no scanout asked for a frame"
    );

    assert_eq!(
        answered(&device, &mut guest, set_scanout(64, 32, 256)),
        RGFX_STATUS_OK
    );
    let frames = frames_asked(&device);
    assert!(
        frames_asked_past(&device, frames, 3),
        "the refreshes drew the scanout {} times",
        frames_asked(&device) - frames
    );
}

/// While the window shows a presented surface, a refresh asks it for no frame: the guest's
/// presents alone draw it. Once the guest presents the scanout again, each refresh draws it
#[test]
fn a_refresh_draws_nothing_while_a_presented_surface_shows() {
    let (device, _platform) = windowed();
    let (mut guest, _ring) = surface_and_queue(&device);
    assert_eq!(
        answered(&device, &mut guest, set_scanout(64, 32, 256)),
        RGFX_STATUS_OK
    );
    assert!(frames_asked_past(&device, frames_asked(&device), 2));

    // With no window open, the present is answered as it is shown.
    let status = answered(&device, &mut guest, present_surface(0x77));
    assert_eq!(status, RGFX_STATUS_OK);
    assert_eq!(shown_width(&device), Some(64));
    // Past a refresh that found the scanout shown before the present.
    std::thread::sleep(Duration::from_millis(20));
    let frames = frames_asked(&device);
    // Six refreshes at the window's 60 Hz.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        frames_asked(&device),
        frames,
        "a refresh asked for a frame while the surface showed"
    );

    let status = answered(&device, &mut guest, present_surface(0));
    assert_eq!(status, RGFX_STATUS_OK);
    assert_eq!(shown_width(&device), None);
    assert!(
        frames_asked_past(&device, frames, 2),
        "no refresh drew the scanout presented again"
    );
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
            flags: 0,
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

/// A `DISPLAY_COLOR` part of display 0's transaction 1: `stage`'s `count` entries, `part` of them
/// from `first` on, entry `k`'s red `k`.
fn display_color(stage: u32, count: u32, first: u32, part: u32) -> Vec<u8> {
    let mut bytes = RgfxDisplayColor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_DISPLAY_COLOR,
            size: 40 + 16 * part,
            seqno: 0,
        },
        display: 0,
        transaction_id: 1,
        stage,
        entry_count: count,
        first_entry: first,
        part_entries: part,
    }
    .as_bytes()
    .to_vec();
    for k in first..first + part {
        #[expect(clippy::as_conversions, reason = "entry indices far below 2^24")]
        let red = (k as f32).to_bits();
        bytes.extend_from_slice(
            RgfxColorEntry {
                x: red,
                y: 0,
                z: 0,
                w: 0,
            }
            .as_bytes(),
        );
    }
    bytes
}

/// The display's colour stages: a stage's parts take effect with the next present the host
/// accepts, a kept present (`RGFX_PRESENT_KEEP`) changes the colour and nothing else, a refused
/// present leaves what is staged for the next, and the reset turns every stage off; the device
/// offers the feature
#[test]
fn display_colors_take_effect_with_the_next_accepted_present() {
    let raised = Arc::new(AtomicU32::new(0));
    let device = headless(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let features = device.mmio_read(RGFX_REG_FEATURES_LO.into());
    assert_ne!(features & RGFX_FEATURE_DISPLAY_COLOR, 0);
    let mut guest = Guest::new(&device);
    let present = |id: u32, flags: u32| {
        RgfxPresentSurface {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_PRESENT_SURFACE,
                size: 32,
                seqno: 0,
            },
            display: 0,
            iosurface_id: id,
            transaction_id: 1,
            flags,
        }
        .as_bytes()
        .to_vec()
    };
    let answered = |guest: &mut Guest, command: Vec<u8>| {
        guest.send(&device, command);
        let bytes = guest.event().expect("a COMPLETE");
        RgfxComplete::read_from_bytes(&bytes).unwrap().status
    };
    let reds = |stage: u32| -> Option<Vec<f32>> {
        let (_, color) = device.control.presented.color();
        let table = color.stages[usize::try_from(stage).unwrap()].as_ref()?;
        Some(table.iter().map(|e| e[0]).collect())
    };

    // 300 entries in two parts, then a present of a surface the decoder holds no texture over:
    // refused, and the colour keeps its stages.
    guest.send(&device, display_color(RGFX_COLOR_POST_GAMMA, 300, 0, 256));
    guest.send(&device, display_color(RGFX_COLOR_POST_GAMMA, 300, 256, 44));
    assert_eq!(
        answered(&mut guest, present(0x77, 0)),
        RGFX_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(reds(RGFX_COLOR_POST_GAMMA), None);
    let (before, _) = device.control.presented.color();

    // The kept present after it takes the staged table, and shows what it showed.
    assert_eq!(
        answered(&mut guest, present(0, RGFX_PRESENT_KEEP)),
        RGFX_STATUS_OK
    );
    let table = reds(RGFX_COLOR_POST_GAMMA).expect("the post-gamma table");
    assert_eq!(table.len(), 300);
    assert_eq!((table[0], table[255], table[299]), (0.0, 255.0, 299.0));
    assert_ne!(device.control.presented.color().0, before);
    assert!(device.control.presented.texture.lock().is_none());

    // A matrix of three rows, and the pre-gamma table turned off, with a present of the scanout:
    // the post-gamma table stays.
    guest.send(&device, display_color(RGFX_COLOR_MATRIX, 3, 0, 3));
    guest.send(&device, display_color(RGFX_COLOR_PRE_GAMMA, 0, 0, 0));
    assert_eq!(answered(&mut guest, present(0, 0)), RGFX_STATUS_OK);
    assert_eq!(reds(RGFX_COLOR_MATRIX), Some(vec![0.0, 1.0, 2.0]));
    assert_eq!(reds(RGFX_COLOR_PRE_GAMMA), None);
    assert_eq!(reds(RGFX_COLOR_POST_GAMMA).map(|t| t.len()), Some(300));

    // A part out of order is dropped: the stage keeps its table.
    guest.send(&device, display_color(RGFX_COLOR_POST_GAMMA, 2, 1, 1));
    assert_eq!(
        answered(&mut guest, present(0, RGFX_PRESENT_KEEP)),
        RGFX_STATUS_OK
    );
    assert_eq!(reds(RGFX_COLOR_POST_GAMMA).map(|t| t.len()), Some(300));

    // The reset, which the control thread tells the host: every stage off.
    device.reset();
    for _ in 0..3000 {
        if device.control.presented.color().1.is_identity() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(device.control.presented.color().1.is_identity());
}

/// A `DISPLAY_COLOR` of display 0's transaction 1 carrying all of `stage`'s entries, each red,
/// green and blue with a fourth word of 0.
fn display_color_entries(stage: u32, entries: &[[f32; 3]]) -> Vec<u8> {
    let count = u32::try_from(entries.len()).unwrap();
    let mut bytes = RgfxDisplayColor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_DISPLAY_COLOR,
            size: 40 + 16 * count,
            seqno: 0,
        },
        display: 0,
        transaction_id: 1,
        stage,
        entry_count: count,
        first_entry: 0,
        part_entries: count,
    }
    .as_bytes()
    .to_vec();
    for [x, y, z] in entries {
        bytes.extend_from_slice(
            RgfxColorEntry {
                x: x.to_bits(),
                y: y.to_bits(),
                z: z.to_bits(),
                w: 0,
            }
            .as_bytes(),
        );
    }
    bytes
}

/// A fade's last steps, each a transaction that sets only the pipe's gamma table after its matrix
/// (`DISPLAY_COLOR` and a kept present, as the kext sends a colour-only transaction): the window's
/// next draw after each kept present puts the display through that step's table, and the last
/// step's table, with no present after it, is what every later draw shows, as the display holds
/// the black a fade ends on
#[test]
fn a_kept_present_s_table_reaches_the_next_drawn_frame() {
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
    use reims_vgpu_vulkan::core::testkit::objects::kind;
    const FRAGMENT_BUFFER: &str = "setFragmentBuffer:offset:atIndex:";
    /// `displayPipeColorScanoutFragment`'s buffer of the post-gamma table.
    const POST_TABLE: u64 = 3;

    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let shown = Arc::new(Shown::default());
    *shown.scanout.lock() = Some(Scanout {
        width: 64,
        height: 32,
        stride: 256,
        offset: RGFX_ALLOCATIONS_OFFSET,
    });
    let draw = sink::ScanoutDraw::new(
        Arc::clone(&platform),
        &device.control.region,
        &shown,
        &device.control.presented,
    )
    .unwrap();
    let drawable = platform.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: 64,
        height: 32,
        usage: 4,
        ..TextureDesc::default()
    });
    let drawable = platform
        .metal_device(None)
        .unwrap()
        .new_texture(OnFault::Return, &drawable)
        .unwrap()
        .unwrap();
    let made: Arc<Mutex<Vec<<FakePlatform as Platform>::Buffer>>> = Arc::default();
    for _ in 0..64 {
        let made = Arc::clone(&made);
        platform
            .world()
            .on_new::<kind::Buffer>(move |b| made.lock().push(b.clone_ref()));
    }
    let rect = reims_vgpu_window::PixelRect {
        x: 0,
        y: 0,
        width: 64,
        height: 32,
    };
    // The post-gamma table the window's next draw binds, as its floats.
    let drawn_table = || -> Vec<f32> {
        let log = platform.call_log();
        let mark = log.mark();
        let _drawn = draw.encode(&drawable, rect).expect("a draw");
        let bound = log
            .since(mark)
            .into_iter()
            .filter(|c| c.sel == FRAGMENT_BUFFER && c.args[2].as_u64() == Some(POST_TABLE))
            .map(|c| c.args[0].as_obj().unwrap())
            .next_back()
            .expect("a draw through the colour");
        let made = made.lock();
        let buffer = made.iter().find(|b| b.id() == bound).expect("the table");
        let length = usize::try_from(buffer.length()).unwrap();
        let contents: *const u8 = buffer.contents().as_ptr();
        // SAFETY: the fake buffer's contents are `length` bytes it holds while it lives.
        let bytes = unsafe { std::slice::from_raw_parts(contents, length) };
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect()
    };

    for step in [0.75f32, 0.5, 0.25, 0.0] {
        guest.send(
            &device,
            display_color_entries(RGFX_COLOR_POST_GAMMA, &[[0.0; 3], [step; 3]]),
        );
        let keep = RgfxPresentSurface {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_PRESENT_SURFACE,
                size: 32,
                seqno: 0,
            },
            display: 0,
            iosurface_id: 0,
            transaction_id: 1,
            flags: RGFX_PRESENT_KEEP,
        };
        assert_eq!(
            answered(&device, &mut guest, keep.as_bytes().to_vec()),
            RGFX_STATUS_OK
        );
        assert_eq!(
            drawn_table(),
            [0.0, 0.0, 0.0, 0.0, step, step, step, 0.0],
            "the draw after step {step}"
        );
    }
    // Nothing presented after the last step: the window's later draws still show it.
    for _ in 0..2 {
        assert_eq!(drawn_table(), [0.0; 8], "a later draw");
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
    use reims_vgpu_graphics::ops::compute::{
        RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
    };
    use reims_vgpu_graphics::ops::{
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
        create_queue(
            &device,
            &mut guest,
            2,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            0
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
        create_queue(
            &device,
            &mut guest,
            3,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            0
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
    // The destroy is carried out before a lookup of its context's after it is answered.
    carried_out_of(&device, &mut guest, 1);
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

/// A `SAMPLER_CREATE` of `obj_id`, from a whole descriptor.
fn sampler_create(obj_id: u64) -> Vec<u8> {
    RgfxSamplerCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SAMPLER_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        max_anisotropy: 1,
        flags: 0b101,
        lod_max_clamp_bits: f32::MAX.to_bits(),
        ..RgfxSamplerCreate::default()
    }
    .as_bytes()
    .to_vec()
}

/// Sends a sampler create of `obj_id` on ring 0 and takes its `SAMPLER_STATE`: its status.
fn sampler_answered(device: &GraphicsDevice<FakePlatform>, guest: &mut Guest, obj_id: u64) -> u32 {
    guest.send(device, sampler_create(obj_id));
    let bytes = guest.event().expect("a SAMPLER_STATE");
    let answer = RgfxSamplerState::read_from_bytes(&bytes).unwrap();
    assert_eq!(answer.header.r#type, RGFX_EVT_SAMPLER_STATE);
    assert_eq!(answer.header.seqno, guest.seqno);
    answer.status
}

/// A command buffer whose compute pass binds the sampler `sampler` at index 0.
fn sampler_command_buffer(queue_obj_id: u64, cb_obj_id: u64, sampler: u64) -> Vec<u8> {
    use reims_vgpu_graphics::ops::compute::{
        RGFX_OP_COMPUTE_SET_SAMPLER_STATES, RgfxComputeOpSetSamplerStates,
        RgfxComputeSamplerBinding,
    };
    use reims_vgpu_graphics::ops::{
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
    let mut bind = RgfxComputeOpSetSamplerStates {
        location: 0,
        length: 1,
    }
    .as_bytes()
    .to_vec();
    bind.extend_from_slice(
        RgfxComputeSamplerBinding {
            sampler_obj_id: sampler,
        }
        .as_bytes(),
    );
    ops.extend(op(RGFX_OP_COMPUTE_SET_SAMPLER_STATES, &bind));
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

/// A gate the first command buffer to walk stops at, after its first op, until the test opens it:
/// a walk in flight, which holds nothing device-wide.
/// Whether a walk took the gate, stopped at it, and whether it is open; and their moves.
type GateState = Arc<(
    reims_vgpu_vulkan::core::sync::Mutex<(bool, bool, bool)>,
    reims_vgpu_vulkan::core::sync::Condvar,
)>;

struct WalkGate {
    state: GateState,
    /// The ops walked past the gate once a walk took it: other walks' while it is shut.
    passed: Arc<AtomicU32>,
    /// The name of the thread whose walk took the gate, once one did.
    taken_on: Arc<super::Mutex<Option<String>>>,
}

impl WalkGate {
    /// The gate, set on `device`'s decoder: the first walk to reach it stops there. A walk held
    /// at it stands for a long one, so every buffer goes to its queue's lane, as a long one does
    /// (`GraphicsControl::carry_out`).
    fn on(device: &GraphicsDevice<FakePlatform>) -> WalkGate {
        device.control.inline_walk_max.store(0, Ordering::Relaxed);
        WalkGate::in_place(device)
    }

    /// The gate, set on `device`'s decoder with each buffer carried out where it would be: a
    /// short buffer whose commit would wait for nothing stops there on its queue's service thread.
    fn in_place(device: &GraphicsDevice<FakePlatform>) -> WalkGate {
        let state = Arc::new((
            reims_vgpu_vulkan::core::sync::Mutex::new((false, false, false)),
            reims_vgpu_vulkan::core::sync::Condvar::new(),
        ));
        let passed = Arc::new(AtomicU32::new(0));
        let taken_on = Arc::new(super::Mutex::new(None));
        let (hook, past, on) = (
            Arc::clone(&state),
            Arc::clone(&passed),
            Arc::clone(&taken_on),
        );
        device
            .control
            .decoder()
            .set_walk_hook(Some(Arc::new(move |index: usize| {
                let (lock, moved) = &*hook;
                let mut s = lock.lock();
                // (taken by a walk, the walk stopped at it, open)
                if s.0 {
                    past.fetch_add(1, Ordering::SeqCst);
                    return;
                }
                if index != 0 {
                    return;
                }
                *on.lock() = std::thread::current().name().map(str::to_owned);
                s.0 = true;
                s.1 = true;
                moved.notify_all();
                while !s.2 {
                    moved.wait(&mut s);
                }
            })));
        WalkGate {
            state,
            passed,
            taken_on,
        }
    }

    /// The name of the thread whose walk took the gate, once one did.
    fn taken_on(&self) -> Option<String> {
        self.taken_on.lock().clone()
    }

    /// Waits, about 3 s at most, for `n` ops to be walked past the gate once a walk took it;
    /// whether they were.
    fn passed(&self, n: u32) -> bool {
        for _ in 0..3000 {
            if self.passed.load(Ordering::SeqCst) >= n {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        false
    }

    /// Waits, about 3 s at most, for a walk to stop at the gate; whether one did.
    fn reached(&self) -> bool {
        let (lock, moved) = &*self.state;
        let mut s = lock.lock();
        for _ in 0..3000 {
            if s.1 {
                return true;
            }
            moved.wait_for(&mut s, Duration::from_millis(1));
        }
        false
    }

    /// Lets the walk stopped at the gate go on.
    fn open(&self) {
        let (lock, moved) = &*self.state;
        lock.lock().2 = true;
        moved.notify_all();
    }
}

/// Queue `n` of `context` created as [`create_queue`] does, its `QUEUE_CREATE` carrying the flags
/// tail with `flags`, as the kext sends it with a flag set. Its status.
fn create_queue_flagged(
    device: &GraphicsDevice<FakePlatform>,
    guest: &mut Guest,
    (context, n): (u32, u64),
    rings: (u64, u64),
    data: u32,
    flags: u32,
) -> u32 {
    let span = u64::from(RGFX_RING_HEADER_SIZE + data);
    guest.send(device, block_grant(context, rings.0, span));
    guest.send(device, block_grant(context, rings.1, span));
    let mut create = queue_create(context, n, rings.0, rings.1, data, 0);
    let tail = RgfxQueueCreateFlags {
        flags,
        reserved_04: 0,
    };
    create.extend_from_slice(tail.as_bytes());
    answered(device, guest, create)
}

/// The platform's queue creators' calls, newest last, once `count` of them were made (about 2 s):
/// each queue's handle is made on its service thread, after its create is answered.
fn queue_calls(platform: &FakePlatform, count: usize) -> Vec<&'static str> {
    use reims_vgpu_vulkan::core::testkit::impls::device::UNSHARED_QUEUE;
    let made = || -> Vec<&'static str> {
        let calls = platform.call_log().calls();
        let creators = calls.iter().map(|c| c.sel);
        creators
            .filter(|&sel| sel == "newCommandQueue" || sel == UNSHARED_QUEUE)
            .collect()
    };
    for _ in 0..2000 {
        if made().len() >= count {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    made()
}

/// Where the kext took `UNSHARED_QUEUES`, a `QUEUE_CREATE` flagged `RGFX_QUEUE_UNSHARED` is made
/// on the platform's unshared queue, and one with no flag on its plain `newCommandQueue`. Where it
/// did not take the feature, a flagged create is refused and the platform makes no queue for it.
#[test]
fn a_queue_the_kext_asks_unshared_is_the_platforms_unshared_queue() {
    use reims_vgpu_vulkan::core::testkit::impls::device::UNSHARED_QUEUE;
    const RING_DATA: u32 = 0x1_0000;
    let rings = |n: u64| {
        let at = RGFX_CONTEXT_ALLOCATIONS_OFFSET + n * 0x4_0000;
        (at, at + 0x2_0000)
    };
    let raised = Arc::new(AtomicU32::new(0));

    let (device, platform) = headless_on(&raised);
    let unshared = RGFX_FEATURE_UNSHARED_QUEUES;
    device.mmio_write(RGFX_REG_DRIVER_FEATURES_LO.into(), unshared);
    assert_eq!(
        device.mmio_read(RGFX_REG_DRIVER_FEATURES_LO.into()),
        unshared
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let flagged = |guest: &mut Guest, n: u64, flags: u32| {
        create_queue_flagged(&device, guest, (1, n), rings(n), RING_DATA, flags)
    };
    assert_eq!(flagged(&mut guest, 1, RGFX_QUEUE_UNSHARED), RGFX_STATUS_OK);
    assert_eq!(queue_calls(&platform, 1), [UNSHARED_QUEUE]);
    let plain = create_queue(&device, &mut guest, 1, 2, rings(2), RING_DATA, 0);
    assert_eq!(plain, RGFX_STATUS_OK);
    assert_eq!(
        queue_calls(&platform, 2),
        [UNSHARED_QUEUE, "newCommandQueue"]
    );
    assert_eq!(flagged(&mut guest, 3, 0), RGFX_STATUS_OK);
    assert_eq!(
        queue_calls(&platform, 3),
        [UNSHARED_QUEUE, "newCommandQueue", "newCommandQueue"]
    );

    // A kext that did not take the feature.
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let refused = create_queue_flagged(
        &device,
        &mut guest,
        (1, 1),
        rings(1),
        RING_DATA,
        RGFX_QUEUE_UNSHARED,
    );
    assert_eq!(refused, RGFX_STATUS_INVALID_ARGUMENT);
    let plain = create_queue(&device, &mut guest, 1, 2, rings(2), RING_DATA, 0);
    assert_eq!(plain, RGFX_STATUS_OK);
    assert_eq!(queue_calls(&platform, 1), ["newCommandQueue"]);
}

/// Ring 0 is carried out while a command buffer's walk is in flight on a queue: a sampler create
/// is answered, and a destroy of the sampler the walk names is carried out, with the walk stopped
/// at its gate. The walk names the sampler as it was at the buffer's begin, as one carried out
/// whole before the destroy would, and completes; the sampler is gone after it.
#[test]
fn ring_0_is_carried_out_while_a_walk_is_in_flight() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        create_queue(&device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(1, 4)),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    let gate = WalkGate::on(&device);
    ring.send(&device, sampler_command_buffer(obj(1, 1), 100, obj(1, 4)));
    assert!(gate.reached(), "the walk did not start");
    // With the walk stopped: a create answered, and the walked sampler's destroy carried out
    // (a lookup of its context's is answered once it is).
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(1, 5)),
        RGFX_STATUS_OK
    );
    guest.send(&device, object_destroy(obj(1, 4)));
    carried_out_of(&device, &mut guest, 1);
    assert_eq!(
        device.decoder_object_count(),
        2,
        "the queue and the new sampler"
    );
    gate.open();
    assert!(complete_until_event(&platform, &mut ring), "no completion");
    let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (100, RGFX_CB_OK));
    // A buffer begun after the destroy does not find it.
    ring.send(&device, sampler_command_buffer(obj(1, 1), 101, obj(1, 4)));
    assert!(complete_until_event(&platform, &mut ring), "no completion");
    let failed = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!(failed.cb_obj_id, 101);
    assert_ne!(failed.status, RGFX_CB_OK);
}

/// A close of a context whose command buffer's walk is in flight ends its queues, drops its
/// objects, and is answered once the buffer's completion handler has run: the walk, which goes on
/// naming the objects, commits after the close, on a queue that posts nothing more
#[test]
fn a_close_during_a_walk_is_answered_after_the_buffer() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        create_queue(&device, &mut guest, 2, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(2, 4)),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    let gate = WalkGate::on(&device);
    ring.send(&device, sampler_command_buffer(obj(2, 1), 100, obj(2, 4)));
    assert!(gate.reached(), "the walk did not start");
    guest.send(&device, context_close(2));
    let close = guest.seqno;
    let commits = platform.call_log().count("commit");
    gate.open();
    for _ in 0..3000 {
        if platform.call_log().count("commit") > commits {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        platform.call_log().count("commit") > commits,
        "not committed"
    );
    // The buffer's completion is its ended queue's, which posts nothing more; the close is
    // answered once its completion handler has run.
    platform.world().complete_all(CompleteOn::Caller);
    let bytes = guest.event().expect("the close's COMPLETE");
    let complete = RgfxComplete::read_from_bytes(&bytes).unwrap();
    assert_eq!(
        (complete.header.seqno, complete.status),
        (close, RGFX_STATUS_OK)
    );
    assert_eq!(device.decoder_object_count(), 0, "the close dropped them");
}

/// A close of a context whose command buffer's walk is in flight holds up nothing on ring 0:
/// the close is carried out at once (its objects leave the table, the walk keeps naming them),
/// and the ring-0 message after it is answered while the walk is still stopped.
#[test]
fn a_close_during_a_walk_holds_up_nothing_on_ring_0() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        create_queue(&device, &mut guest, 2, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(2, 4)),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    let gate = WalkGate::on(&device);
    ring.send(&device, sampler_command_buffer(obj(2, 1), 100, obj(2, 4)));
    assert!(gate.reached(), "the walk did not start");
    guest.send(&device, context_close(2));
    // Another context's create after the close, answered with the walk still stopped.
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(3, 4)),
        RGFX_STATUS_OK
    );
    assert_eq!(
        device.decoder_object_count(),
        1,
        "the close was not carried out: only context 3's sampler is left"
    );
    let commits = platform.call_log().count("commit");
    gate.open();
    for _ in 0..3000 {
        if platform.call_log().count("commit") > commits {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    platform.world().complete_all(CompleteOn::Caller);
    let bytes = guest.event().expect("the close's COMPLETE");
    let complete = RgfxComplete::read_from_bytes(&bytes).unwrap();
    assert_eq!(complete.status, RGFX_STATUS_OK);
}

/// A reset while a command buffer's walk is in flight comes after the walk, as one under a lock
/// over the decoder would: the device's reset ends the walk's queue at once, but the decoder's
/// reset waits for the walk, which names its objects and commits, and then drops every object,
/// none outliving it. Nothing of the buffer reaches the guest after the reset.
#[test]
fn a_reset_during_a_walk_comes_after_it() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        create_queue(&device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(1, 4)),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    let gate = WalkGate::on(&device);
    ring.send(&device, sampler_command_buffer(obj(1, 1), 100, obj(1, 4)));
    assert!(gate.reached(), "the walk did not start");
    let commits = platform.call_log().count("commit");
    device.reset();
    // The decoder's reset, on the control thread, waits for the walk.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        device.decoder_object_count(),
        2,
        "the reset dropped the walk's objects before the walk ended"
    );
    gate.open();
    for _ in 0..3000 {
        if device.decoder_object_count() == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(device.decoder_object_count(), 0, "the reset left objects");
    assert!(
        platform.call_log().count("commit") > commits,
        "the walk was not committed before the reset"
    );
    platform.world().complete_all(CompleteOn::Caller);
    assert!(!ring.has_event(), "the buffer posted after the reset");
}

/// A command buffer on one queue is carried out while another queue's walk is in flight: it names
/// a sampler made meanwhile, which it finds, and completes first
#[test]
fn a_queue_s_buffer_runs_beside_another_queue_s_walk() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const RING_DATA: u32 = 0x1_0000;
    let rings = [
        (
            RGFX_CONTEXT_ALLOCATIONS_OFFSET,
            RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000,
        ),
        (
            RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4_0000,
            RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x6_0000,
        ),
    ];
    for (n, &(submit, complete)) in rings.iter().enumerate() {
        assert_eq!(
            create_queue(
                &device,
                &mut guest,
                1,
                n as u64 + 1,
                (submit, complete),
                RING_DATA,
                0
            ),
            RGFX_STATUS_OK
        );
    }
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(1, 4)),
        RGFX_STATUS_OK
    );
    let mut first = QueueRing::new(&device, rings[0].0, rings[0].1, u64::from(RING_DATA));
    let mut second = QueueRing::new(&device, rings[1].0, rings[1].1, u64::from(RING_DATA));
    let gate = WalkGate::on(&device);
    first.send(&device, sampler_command_buffer(obj(1, 1), 100, obj(1, 4)));
    assert!(gate.reached(), "the walk did not start");
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(1, 5)),
        RGFX_STATUS_OK
    );
    second.send(&device, sampler_command_buffer(obj(1, 2), 200, obj(1, 5)));
    assert!(
        complete_until_event(&platform, &mut second),
        "no completion beside the walk"
    );
    let done = RgfxCbComplete::read_from_bytes(&second.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (200, RGFX_CB_OK));
    assert!(!first.has_event(), "the stopped walk completed");
    gate.open();
    assert!(complete_until_event(&platform, &mut first), "no completion");
    let done = RgfxCbComplete::read_from_bytes(&first.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (100, RGFX_CB_OK));
}

/// `message`, a command buffer, with a `SUBMITTED` op after its ops: it asks for its
/// `CB_SUBMITTED`, as `commitAndWaitUntilSubmitted` commits it.
fn with_ask(mut message: Vec<u8>) -> Vec<u8> {
    use reims_vgpu_graphics::ops::{RGFX_OP_SUBMITTED, RgfxOpHeader};

    let submitted = RgfxOpHeader {
        tag: RGFX_OP_SUBMITTED,
        flags: 0,
        len: 8,
    };
    message.extend_from_slice(submitted.as_bytes());
    let field =
        RGFX_COMMAND_BUFFER_OPS_OFFSET as usize - 8..RGFX_COMMAND_BUFFER_OPS_OFFSET as usize;
    let ops = u64::from_le_bytes(message[field.clone()].try_into().unwrap()) + 8;
    message[field].copy_from_slice(&ops.to_le_bytes());
    message
}

/// A device with process 1's queues 1 and 2 and process 2's queue 1 made, their rings in that
/// order, each process's shared event 20, library 10 and render pipeline 11, and process 1's
/// sampler 4.
fn two_processes(
    raised: &Arc<AtomicU32>,
) -> (
    GraphicsDevice<FakePlatform>,
    Arc<FakePlatform>,
    Guest,
    [QueueRing; 3],
) {
    const RING_DATA: u32 = 0x1_0000;
    let rings = |n: u64| {
        let at = RGFX_CONTEXT_ALLOCATIONS_OFFSET + n * 0x4_0000;
        (at, at + 0x2_0000)
    };
    let (device, platform) = headless_on(raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    for (at, (context, n)) in [(1, 1), (1, 2), (2, 1)].into_iter().enumerate() {
        let made = create_queue(
            &device,
            &mut guest,
            context,
            n,
            rings(at as u64),
            RING_DATA,
            0,
        );
        assert_eq!(made, RGFX_STATUS_OK);
    }
    for context in [1, 2] {
        guest.send(&device, shared_event_create(obj(context, 20), 0));
        make_library(&device, &mut guest, obj(context, 10));
        guest.send(
            &device,
            render_pipeline_create(obj(context, 11), obj(context, 10)),
        );
        expect_reflection(&mut guest, obj(context, 11), RGFX_PIPELINE_OK);
    }
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(1, 4)),
        RGFX_STATUS_OK
    );
    let ring = |at: u64| {
        let (submit, complete) = rings(at);
        QueueRing::new(&device, submit, complete, u64::from(RING_DATA))
    };
    let queues = [ring(0), ring(1), ring(2)];
    (device, platform, guest, queues)
}

/// Holds the next draw the platform is sent, a walk's, as a driver's cold pipeline make holds
/// a draw: the hold, once a walk reached it.
fn hold_next_draw(platform: &FakePlatform) -> HeldCall {
    let draw = HeldCall::new();
    platform.faults().at(
        "MTLRenderCommandEncoder",
        "drawPrimitives:vertexStart:vertexCount:",
        0,
        Fault::Hold(draw.clone()),
    );
    draw
}

/// The `(type, object, value)` of `ring`'s next `n` events but `CB_SCHEDULED`, completing the
/// platform's committed buffers meanwhile: a `CB_COMPLETE`'s buffer and status, an
/// `EVENT_SIGNALLED`'s event and value, a `CB_SUBMITTED`'s buffer.
fn ring_events(platform: &FakePlatform, ring: &mut QueueRing, n: usize) -> Vec<(u32, u64, u64)> {
    let mut got = Vec::new();
    while got.len() < n {
        assert!(
            complete_until_event(platform, ring),
            "no event, after {got:?}"
        );
        let bytes = ring.event().unwrap();
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
            RGFX_EVT_CB_SUBMITTED => {
                let posted = RgfxCbSubmitted::read_from_bytes(&bytes).unwrap();
                (header.r#type, posted.cb_obj_id, 0)
            }
            other => panic!("event {other:#x} on a queue's completion ring"),
        });
    }
    got
}

/// A `SHARED_EVENT_NAME` or `SHARED_EVENT_IMPORT` as the kext builds it.
fn shared_event_share(r#type: u32, obj_id: u64, name: u64, listen_queue_obj_id: u64) -> Vec<u8> {
    RgfxSharedEventName {
        header: RgfxMsgHeader {
            r#type,
            size: size_of::<RgfxSharedEventName>() as u32,
            seqno: 0,
        },
        obj_id,
        name,
        listen_queue_obj_id,
    }
    .as_bytes()
    .to_vec()
}

/// A shared event one process names and another makes one of its own from is one event across
/// them, as a Mac's is: the import is answered the event's value; a command buffer's signals in
/// the namer are posted to the importer's listen queue, for the importer's own id, as they reach
/// the namer's own ring; and the importer's CPU set is posted to the namer's listen queue. A name
/// of an event the process does not hold, and an import of a name the kext let go of, are refused
#[test]
fn a_named_shared_event_is_one_event_across_two_processes() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, mut guest, [mut namer, mut namer_listen, mut importer]) =
        two_processes(&raised);
    // Process 1 names its event 20, the other holders' values going to its queue 2.
    let name = shared_event_share(RGFX_CMD_SHARED_EVENT_NAME, obj(1, 20), 7, obj(1, 2));
    assert_eq!(answered(&device, &mut guest, name), RGFX_STATUS_OK);
    let none = shared_event_share(RGFX_CMD_SHARED_EVENT_NAME, obj(1, 99), 8, obj(1, 2));
    assert_eq!(answered(&device, &mut guest, none), RGFX_STATUS_NOT_FOUND);
    // Process 2 makes its event 21 of the name, its values going to its queue 1.
    let import = shared_event_share(RGFX_CMD_SHARED_EVENT_IMPORT, obj(2, 21), 7, obj(2, 1));
    guest.send(&device, import);
    let value = RgfxSharedEventValue::read_from_bytes(&guest.event().unwrap()).unwrap();
    assert_eq!(
        (value.header.r#type, value.header.seqno),
        (RGFX_EVT_SHARED_EVENT_VALUE, guest.seqno)
    );
    assert_eq!((value.status, value.value), (RGFX_STATUS_OK, 0));

    // Process 1's buffer signals 4 and 5.
    namer.send(
        &device,
        render_command_buffer(obj(1, 1), 100, obj(1, 11), obj(1, 20)),
    );
    assert_eq!(
        ring_events(&platform, &mut namer, 3),
        [
            (RGFX_EVT_EVENT_SIGNALLED, obj(1, 20), 4),
            (RGFX_EVT_EVENT_SIGNALLED, obj(1, 20), 5),
            (RGFX_EVT_CB_COMPLETE, 100, u64::from(RGFX_CB_OK)),
        ]
    );
    assert_eq!(
        ring_events(&platform, &mut importer, 2),
        [
            (RGFX_EVT_EVENT_SIGNALLED, obj(2, 21), 4),
            (RGFX_EVT_EVENT_SIGNALLED, obj(2, 21), 5),
        ]
    );
    // Process 2's CPU sets it to 9.
    let set = RgfxSharedEventSetValue {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SHARED_EVENT_SET_VALUE,
            size: size_of::<RgfxSharedEventSetValue>() as u32,
            seqno: 0,
        },
        obj_id: obj(2, 21),
        value: 9,
    };
    guest.send(&device, set.as_bytes().to_vec());
    assert_eq!(
        ring_events(&platform, &mut namer_listen, 1),
        [(RGFX_EVT_EVENT_SIGNALLED, obj(1, 20), 9)]
    );
    assert!(!namer.has_event() && !importer.has_event());

    // The kext lets go of the name: a later import finds none, and makes nothing.
    let unname = RgfxSharedEventUnname {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SHARED_EVENT_UNNAME,
            size: size_of::<RgfxSharedEventUnname>() as u32,
            seqno: 0,
        },
        reserved_10: 0,
        name: 7,
    };
    guest.send(&device, unname.as_bytes().to_vec());
    carried_out_of(&device, &mut guest, 0);
    let again = shared_event_share(RGFX_CMD_SHARED_EVENT_IMPORT, obj(2, 22), 7, obj(2, 1));
    guest.send(&device, again);
    let refused = RgfxSharedEventValue::read_from_bytes(&guest.event().unwrap()).unwrap();
    assert_eq!((refused.status, refused.value), (RGFX_STATUS_NOT_FOUND, 0));
}

/// A command buffer whose walk is held inside the platform's draw, as a driver's cold pipeline
/// make holds one, holds up no other queue's: with the draw held, another process's ring-0
/// messages are carried out and its pipeline compiled, and its buffers are taken, begun, answered
/// their `CB_SUBMITTED`, committed and completed, one of them drawing too; so is a buffer on the
/// held process's other queue. The held buffer completes once its draw returns
#[test]
fn a_walk_held_in_a_draw_holds_up_no_other_queue() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, mut guest, [mut held, mut beside, mut other]) = two_processes(&raised);
    let draw = hold_next_draw(&platform);
    held.send(
        &device,
        render_command_buffer(obj(1, 1), 100, obj(1, 11), obj(1, 20)),
    );
    assert!(
        draw.reached(Duration::from_secs(3)),
        "the walk did not reach its draw"
    );

    // Process 2, with the draw held: ring 0 carried out, a pipeline compiled.
    make_library(&device, &mut guest, obj(2, 30));
    guest.send(&device, render_pipeline_create(obj(2, 31), obj(2, 30)));
    expect_reflection(&mut guest, obj(2, 31), RGFX_PIPELINE_OK);
    // Its buffer that asks for its `CB_SUBMITTED`, and one that draws.
    other.send(&device, asking(obj(2, 1), 200));
    other.send(
        &device,
        render_command_buffer(obj(2, 1), 201, obj(2, 31), obj(2, 20)),
    );
    assert_eq!(
        ring_events(&platform, &mut other, 5),
        [
            (RGFX_EVT_CB_SUBMITTED, 200, 0),
            (RGFX_EVT_CB_COMPLETE, 200, u64::from(RGFX_CB_OK)),
            (RGFX_EVT_EVENT_SIGNALLED, obj(2, 20), 4),
            (RGFX_EVT_EVENT_SIGNALLED, obj(2, 20), 5),
            (RGFX_EVT_CB_COMPLETE, 201, u64::from(RGFX_CB_OK)),
        ]
    );
    // The held process's other queue.
    beside.send(&device, sampler_command_buffer(obj(1, 2), 300, obj(1, 4)));
    assert_eq!(
        ring_events(&platform, &mut beside, 1),
        [(RGFX_EVT_CB_COMPLETE, 300, u64::from(RGFX_CB_OK))]
    );
    assert!(!held.has_event(), "the held buffer completed");

    draw.open();
    assert_eq!(
        ring_events(&platform, &mut held, 3),
        [
            (RGFX_EVT_EVENT_SIGNALLED, obj(1, 20), 4),
            (RGFX_EVT_EVENT_SIGNALLED, obj(1, 20), 5),
            (RGFX_EVT_CB_COMPLETE, 100, u64::from(RGFX_CB_OK)),
        ]
    );
}

/// A command buffer answered its `CB_SUBMITTED` whose walk is held inside the platform's draw
/// holds up no other queue's take or begin, and only the commits of its own process's buffers
/// begun after its answer, which come after it ([`decode::sequencer`]): with the draw held,
/// another process's buffer that asks for its `CB_SUBMITTED` is answered, committed and
/// completed, as no tracked resource of one process is another's; a buffer of the held
/// process's other queue is answered, and committed once the held buffer is
#[test]
fn a_held_walk_answered_its_submitted_holds_only_its_process_s_later_commits() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, _guest, [mut held, mut beside, mut other]) = two_processes(&raised);
    let draw = hold_next_draw(&platform);
    held.send(
        &device,
        with_ask(render_command_buffer(
            obj(1, 1),
            100,
            obj(1, 11),
            obj(1, 20),
        )),
    );
    assert!(
        draw.reached(Duration::from_secs(3)),
        "the walk did not reach its draw"
    );
    assert_eq!(
        ring_events(&platform, &mut held, 1),
        [(RGFX_EVT_CB_SUBMITTED, 100, 0)]
    );

    // Another process's buffer, begun after the answer: committed with the draw held.
    other.send(&device, asking(obj(2, 1), 200));
    assert_eq!(
        ring_events(&platform, &mut other, 2),
        [
            (RGFX_EVT_CB_SUBMITTED, 200, 0),
            (RGFX_EVT_CB_COMPLETE, 200, u64::from(RGFX_CB_OK)),
        ],
        "another process's buffer waited for the held walk's commit"
    );

    // The held process's other queue's buffer, begun after the answer: answered, and committed
    // only after the held buffer.
    let commits = platform.call_log().count("commit");
    beside.send(&device, asking(obj(1, 2), 300));
    assert_eq!(
        ring_events(&platform, &mut beside, 1),
        [(RGFX_EVT_CB_SUBMITTED, 300, 0)],
        "the held process's other queue's buffer was not begun beside the held walk"
    );
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        platform.call_log().count("commit"),
        commits,
        "a buffer of the held process begun after the held one's answer was committed before it"
    );

    draw.open();
    assert_eq!(
        ring_events(&platform, &mut held, 3),
        [
            (RGFX_EVT_EVENT_SIGNALLED, obj(1, 20), 4),
            (RGFX_EVT_EVENT_SIGNALLED, obj(1, 20), 5),
            (RGFX_EVT_CB_COMPLETE, 100, u64::from(RGFX_CB_OK)),
        ]
    );
    assert_eq!(
        ring_events(&platform, &mut beside, 1),
        [(RGFX_EVT_CB_COMPLETE, 300, u64::from(RGFX_CB_OK))]
    );
}

/// Two processes that share a surface stay in the order of IOSurface accesses with no wait for
/// each other's commits (`decode`'s `sequencer`, `surface_order`): process 1's buffer that writes
/// a surface, answered its `CB_SUBMITTED` and held in its walk, and process 2's buffer that reads
/// the surface after it, naming it in its `SURFACE_ORDER` as the bundle does. The reader is
/// committed with the writer still walking, and waits on the GPU for the writer's queue's surface
/// timeline to reach the writer's seqno, the signal the writer's commit carries once its walk
/// ends: the GPU runs the write first, as a Mac orders such accesses on the GPU alone
/// (docs/graphics-kext/dgpu-oracle.md, Q16), though the reader reached the platform first. A write
/// after another process's answered read is the same to the host: the bundle names the reader's
/// queue and seqno from the surface's reader slots, and the host waits for a named buffer
/// whichever slot named it
#[test]
fn a_reader_in_another_process_waits_on_the_gpu_for_an_answered_writer() {
    use reims_vgpu_graphics::ops::{
        RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_END, RGFX_OP_SURFACE_ORDER,
        RgfxSurfaceOrderWait,
    };

    const WAIT: &str = "encodeWaitForEvent:value:";
    const SIGNAL: &str = "encodeSignalEvent:value:";
    let at = |mut message: Vec<u8>, seqno: u64| {
        message[8..16].copy_from_slice(&seqno.to_le_bytes());
        message
    };
    let events = |calls: &[reims_vgpu_vulkan::core::testkit::calllog::Call], sel: &str| {
        (calls.iter())
            .filter(|c| c.sel == sel)
            .map(|c| (c.args[0].as_obj(), c.args[1].as_u64()))
            .collect::<Vec<_>>()
    };
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, _guest, [mut writer, _, mut reader]) = two_processes(&raised);
    let gate = WalkGate::on(&device);
    // Process 1's writer, at seqno 1: it holds the order (it writes a shared surface) and asks
    // for its submission; it stops at the gate.
    let write = command_buffer_of(
        obj(1, 1),
        100,
        &[
            (RGFX_OP_SURFACE_ORDER, &[]),
            (RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
            (RGFX_OP_ENCODER_END, &[]),
        ],
    );
    writer.send(&device, at(with_ask(write), 1));
    assert!(gate.reached(), "the writer's walk did not start");
    assert_eq!(
        ring_events(&platform, &mut writer, 1),
        [(RGFX_EVT_CB_SUBMITTED, 100, 0)]
    );

    // Process 2's reader names the writer's queue at its seqno: committed and completed with the
    // writer's walk held, its commit carrying the GPU wait.
    let mark = platform.call_log().mark();
    let wait = RgfxSurfaceOrderWait {
        queue_obj_id: obj(1, 1),
        seqno: 1,
    };
    let read = command_buffer_of(
        obj(2, 1),
        200,
        &[
            (RGFX_OP_SURFACE_ORDER, wait.as_bytes()),
            (RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
            (RGFX_OP_ENCODER_END, &[]),
        ],
    );
    reader.send(&device, at(read, 1));
    assert_eq!(
        ring_events(&platform, &mut reader, 1),
        [(RGFX_EVT_CB_COMPLETE, 200, u64::from(RGFX_CB_OK))],
        "the reader waited for the writer's commit"
    );
    let waits = events(&platform.call_log().since(mark), WAIT);
    assert_eq!(
        waits.len(),
        1,
        "the reader does not wait for the writer: {waits:?}"
    );
    assert_eq!(waits[0].1, Some(1), "the reader waits for another seqno");

    // The writer's commit signals that timeline to that value.
    let mark = platform.call_log().mark();
    gate.open();
    assert_eq!(
        ring_events(&platform, &mut writer, 1),
        [(RGFX_EVT_CB_COMPLETE, 100, u64::from(RGFX_CB_OK))]
    );
    let signals = events(&platform.call_log().since(mark), SIGNAL);
    assert!(
        signals.contains(&waits[0]),
        "the writer does not signal what the reader waits for: {signals:?} against {waits:?}"
    );
}

/// A queue is known to the order of IOSurface accesses from its create's answer, before its
/// service thread hands the create to the host and has the decoder make its handle
/// (`surface_order`): with the service thread of a new queue of process 1 held there, as a
/// no-copy buffer of the process being made holds it (`pages_buffers`), process 2's buffer that
/// names the new queue's first buffer, published on its ring, waits on the GPU for that buffer,
/// which signals what it waits for once the service thread goes on and the buffer is taken
#[test]
fn a_queue_s_buffers_are_waited_for_before_its_handle_is_made() {
    use reims_vgpu_graphics::ops::{
        RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_END, RGFX_OP_SURFACE_ORDER,
        RgfxSurfaceOrderWait,
    };

    const WAIT: &str = "encodeWaitForEvent:value:";
    const SIGNAL: &str = "encodeSignalEvent:value:";
    const RING_DATA: u32 = 0x1_0000;
    let at = |mut message: Vec<u8>, seqno: u64| {
        message[8..16].copy_from_slice(&seqno.to_le_bytes());
        message
    };
    let events = |calls: &[reims_vgpu_vulkan::core::testkit::calllog::Call], sel: &str| {
        (calls.iter())
            .filter(|c| c.sel == sel)
            .map(|c| (c.args[0].as_obj(), c.args[1].as_u64()))
            .collect::<Vec<_>>()
    };
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, mut guest, [_, _, mut reader]) = two_processes(&raised);

    // Process 1's third queue, answered while its service thread waits for the process's no-copy
    // buffer before it hands the create to the host.
    let made = platform.call_log().mark();
    let pages = &device.control.pages_buffers;
    let held = pages.begin(pages_context(1), 0, device.control.pool.now());
    let (submit, complete) = {
        let at = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 3 * 0x4_0000;
        (at, at + 0x2_0000)
    };
    assert_eq!(
        create_queue(&device, &mut guest, 1, 3, (submit, complete), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    // Its first buffer, which writes a shared surface, on its ring.
    let mut fresh = QueueRing::new(&device, submit, complete, u64::from(RING_DATA));
    let write = command_buffer_of(
        obj(1, 3),
        300,
        &[
            (RGFX_OP_SURFACE_ORDER, &[]),
            (RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
            (RGFX_OP_ENCODER_END, &[]),
        ],
    );
    fresh.send(&device, at(write, 1));

    // Process 2's reader names it, and is committed with a GPU wait for it.
    let mark = platform.call_log().mark();
    let wait = RgfxSurfaceOrderWait {
        queue_obj_id: obj(1, 3),
        seqno: 1,
    };
    let read = command_buffer_of(
        obj(2, 1),
        200,
        &[
            (RGFX_OP_SURFACE_ORDER, wait.as_bytes()),
            (RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
            (RGFX_OP_ENCODER_END, &[]),
        ],
    );
    reader.send(&device, at(read, 1));
    assert_eq!(
        ring_events(&platform, &mut reader, 1),
        [(RGFX_EVT_CB_COMPLETE, 200, u64::from(RGFX_CB_OK))]
    );
    let waits = events(&platform.call_log().since(mark), WAIT);
    assert_eq!(
        waits.len(),
        1,
        "the reader does not wait for the new queue's buffer: {waits:?}"
    );
    assert_eq!(waits[0].1, Some(1), "the reader waits for another seqno");

    let sels = platform.call_log().sels_since(made);
    assert!(
        !sels.contains(&"newCommandQueue"),
        "the new queue's handle was made: {sels:?}"
    );

    // The service thread goes on: the new queue's handle is made, its buffer taken, and its
    // commit signals what the reader waits for.
    let mark = platform.call_log().mark();
    drop(held);
    assert_eq!(
        ring_events(&platform, &mut fresh, 1),
        [(RGFX_EVT_CB_COMPLETE, 300, u64::from(RGFX_CB_OK))]
    );
    let signals = events(&platform.call_log().since(mark), SIGNAL);
    assert!(
        signals.contains(&waits[0]),
        "the new queue's buffer does not signal what the reader waits for: {signals:?} against \
         {waits:?}"
    );
}

/// A command buffer with the ops `ops`, each a tag and its body, on queue `queue_obj_id`.
fn command_buffer_of(queue_obj_id: u64, cb_obj_id: u64, ops: &[(u16, &[u8])]) -> Vec<u8> {
    use reims_vgpu_graphics::ops::RgfxOpHeader;

    let mut stream = Vec::new();
    for &(tag, body) in ops {
        let header = RgfxOpHeader {
            tag,
            flags: 0,
            len: (8 + body.len()) as u32,
        };
        stream.extend_from_slice(header.as_bytes());
        stream.extend_from_slice(body);
    }
    let mut bytes = RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: 0,
            seqno: 0,
        },
        queue_obj_id,
        cb_obj_id,
        op_bytes: stream.len() as u64,
    }
    .as_bytes()
    .to_vec();
    bytes.extend(stream);
    bytes
}

/// A headless device with queue 1 of process 1 made, its ring, and the process's sampler 4.
fn one_queue(
    raised: &Arc<AtomicU32>,
) -> (
    GraphicsDevice<FakePlatform>,
    Arc<FakePlatform>,
    Guest,
    QueueRing,
) {
    const RING_DATA: u32 = 0x1_0000;
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    let (device, platform) = headless_on(raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let rings = (SUBMIT_OFF, COMPLETE_OFF);
    assert_eq!(
        create_queue(&device, &mut guest, 1, 1, rings, RING_DATA, 0),
        RGFX_STATUS_OK
    );
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(1, 4)),
        RGFX_STATUS_OK
    );
    let ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    (device, platform, guest, ring)
}

/// Two command buffers of one queue are walked at once, and committed in the order they came
/// (`decode`'s `lane`): the second, sent while the first's walk is stopped at the gate, is walked
/// whole beside it, and committed only once the first is. They complete in that order
#[test]
fn a_queue_s_buffers_are_walked_at_once_and_committed_in_order() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, _guest, mut ring) = one_queue(&raised);
    let gate = WalkGate::on(&device);
    ring.send(&device, sampler_command_buffer(obj(1, 1), 100, obj(1, 4)));
    assert!(gate.reached(), "the first walk did not start");
    ring.send(&device, sampler_command_buffer(obj(1, 1), 101, obj(1, 4)));
    assert!(
        gate.passed(3),
        "the second buffer was not walked beside the first"
    );
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        platform.world().committed().is_empty(),
        "the second was committed before the first"
    );
    gate.open();
    committed(&platform, 2);
    for cb in [100, 101] {
        assert!(complete_until_event(&platform, &mut ring), "no completion");
        let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
        assert_eq!((done.cb_obj_id, done.status), (cb, RGFX_CB_OK));
    }
}

/// What a drain of small command buffers costs the device's own threads, from the first one's
/// doorbell write to the last one's `CB_COMPLETE`, on the fake platform, whose commits and
/// completions cost next to nothing, the queue's thread asleep before each drain: a lone buffer,
/// and 8 committed back to back and waited for, as cbwait's drain8 commits them, each ringing its
/// own doorbell as the bundle's commit does. Prints the median, 90th and 99th percentiles in
/// microseconds, and how many threads the last drain's commits ran on: a measurement, run by hand
/// (`--run-ignored only --no-capture`, in a release build).
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_drain_s_first_doorbell_to_its_last_cb_complete() {
    use std::time::Instant;

    const ROUNDS: usize = 1000;
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, _guest, mut ring) = one_queue(&raised);
    let idle = || {
        for _ in 0..2000 {
            // SAFETY: as `QueueRing`'s helpers; `consumer_idle` is at 0x80 of the submission ring.
            let idle = unsafe {
                AtomicU32::from_ptr(device.region_base().add(SUBMIT_OFF as usize + 0x80).cast())
                    .load(Ordering::SeqCst)
            };
            if idle == 1 {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("the queue's thread did not go idle");
    };
    let mut cb = 1000;
    for buffers in [1, 8] {
        let mut us = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            idle();
            // A pause as the guest's between frames, so the device's threads sleep.
            std::thread::sleep(Duration::from_micros(500));
            ring.arm();
            let start = Instant::now();
            for _ in 0..buffers {
                cb += 1;
                ring.send(&device, sampler_command_buffer(obj(1, 1), cb, obj(1, 4)));
            }
            let mut done = 0;
            while done < buffers {
                platform.world().complete_all(CompleteOn::Caller);
                while ring.has_event() {
                    let bytes = ring.event().unwrap();
                    let record = RgfxCbComplete::read_from_bytes(&bytes).unwrap();
                    assert_eq!(record.header.r#type, RGFX_EVT_CB_COMPLETE);
                    assert_eq!(record.status, RGFX_CB_OK);
                    done += 1;
                }
                assert!(
                    start.elapsed() < Duration::from_secs(2),
                    "the drain stalled"
                );
                std::hint::spin_loop();
            }
            us.push(start.elapsed().as_secs_f64() * 1e6);
            ring.scheduled.clear();
        }
        us.sort_by(f64::total_cmp);
        let n = us.len();
        let calls = platform.call_log().calls_to("commit");
        let threads: std::collections::BTreeSet<_> = (calls.iter().rev().take(buffers))
            .filter_map(|c| c.thread.clone())
            .collect();
        println!(
            "drain of {buffers}: p50 {:.1} p90 {:.1} p99 {:.1} us, {:.1} us a buffer at p50; \
             the last drain committed on {} thread(s) {threads:?}",
            us[n / 2],
            us[n * 9 / 10],
            us[n * 99 / 100],
            us[n / 2] / buffers as f64,
            threads.len(),
        );
    }
}

/// What a CPU lock's `SURFACE_DOWNLOAD` costs the device from its doorbell to its `COMPLETE`
/// where the surface's pages already hold its cache (a decoder's output whose early copy has
/// landed, or a cache nothing wrote), with the device's threads asleep before each, as between a
/// client's frames: rung by `mmio_write`, the control thread's wake and take, a block lane's wake
/// where the download goes to the lane, and the answer; rung by `mmio_write_taking`, the answer
/// on the ringing thread inside the write where it is answered there, else as by `mmio_write`
/// (how many were, it prints). The fake platform's GPU shares the CPU's memory, so the
/// block has no cache and the host answers at once, as it does a landed copy. The two writes
/// take turns, [`SIDE_ROUNDS`] rounds of 1000 downloads each after 100 to warm up, so a load
/// on the host falls on both: each round's p50, p90 and p99 in microseconds are printed, then
/// each write's median of its rounds' p50s and their spread. A measurement, run by hand
/// (`--run-ignored only --no-capture`, in a release build).
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_landed_download_s_doorbell_to_its_answer() {
    use std::time::Instant;

    const ROUNDS: usize = 1000;
    const WARM: usize = 100;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let download = RgfxSurfaceDownload {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SURFACE_DOWNLOAD,
            size: 0,
            seqno: 0,
        },
        block_offset: BLOCK,
    };
    let writes = [("mmio_write", false), ("mmio_write_taking", true)];
    let mut p50s: [Vec<f64>; 2] = Default::default();
    for side_round in 0..SIDE_ROUNDS {
        // Each round in turn order and its reverse, so neither write always goes first.
        let order = if side_round % 2 == 0 { [0, 1] } else { [1, 0] };
        for side in order {
            let (write, taking) = writes[side];
            let mut us = Vec::with_capacity(ROUNDS);
            let mut inside = 0;
            for round in 0..WARM + ROUNDS {
                // A pause as between a client's frames, so the device's threads sleep.
                std::thread::sleep(Duration::from_millis(2));
                let start = Instant::now();
                guest.publish(download.as_bytes().to_vec());
                if taking {
                    device.mmio_write_taking(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
                    if round >= WARM && guest.event_waiting() {
                        inside += 1;
                    }
                } else {
                    device.mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
                }
                let bytes = loop {
                    if let Some(bytes) = guest.event_now() {
                        break bytes;
                    }
                    assert!(
                        start.elapsed() < Duration::from_secs(2),
                        "the download went unanswered"
                    );
                    std::hint::spin_loop();
                };
                let elapsed = start.elapsed();
                let complete = RgfxComplete::read_from_bytes(&bytes).unwrap();
                assert_eq!(
                    (complete.header.r#type, complete.status),
                    (RGFX_EVT_COMPLETE, RGFX_STATUS_OK)
                );
                if round >= WARM {
                    us.push(elapsed.as_secs_f64() * 1e6);
                }
            }
            us.sort_by(f64::total_cmp);
            let n = us.len();
            p50s[side].push(us[n / 2]);
            println!(
                "landed download, doorbell ({write}) to COMPLETE, round {side_round}: p50 {:.1} \
                 p90 {:.1} p99 {:.1} max {:.1} us; answered inside the write {inside} of {ROUNDS}",
                us[n / 2],
                us[n * 9 / 10],
                us[n * 99 / 100],
                us[n - 1],
            );
        }
    }
    for (side, (write, _)) in writes.iter().enumerate() {
        let mut rounds = p50s[side].clone();
        rounds.sort_by(f64::total_cmp);
        println!(
            "landed download, doorbell ({write}) to COMPLETE: median of {SIDE_ROUNDS} rounds' p50 \
             {:.1} us, spread {:.1} to {:.1} us",
            rounds[rounds.len() / 2],
            rounds[0],
            rounds[rounds.len() - 1],
        );
    }
}

/// How many rounds each side of a before-and-after measurement takes, in turns.
const SIDE_ROUNDS: usize = 5;

/// A command buffer that asks for its `CB_SUBMITTED`, sent behind one of its queue that waits for
/// an event and is still being walked, is answered at its scheduling, after that one's, not at its
/// take: the waiting one holds its queue from its begin, as it would from its commit had it been
/// carried out whole first (the fake platform schedules each buffer at its commit)
#[test]
fn a_buffer_behind_one_that_waits_and_is_still_walked_is_answered_at_its_scheduling() {
    use reims_vgpu_graphics::ops::{RGFX_OP_WAIT_EVENT, RgfxOpEvent};

    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, mut guest, mut ring) = one_queue(&raised);
    guest.send(&device, shared_event_create(obj(1, 7), 0));
    let wait = RgfxOpEvent {
        event_obj_id: obj(1, 7),
        value: 1,
    };
    let gate = WalkGate::on(&device);
    let waits = command_buffer_of(obj(1, 1), 100, &[(RGFX_OP_WAIT_EVENT, wait.as_bytes())]);
    ring.send(&device, waits);
    assert!(gate.reached(), "the first walk did not start");
    ring.send(&device, asking(obj(1, 1), 101));
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        !ring.has_event(),
        "answered while the buffer before it is walked"
    );
    gate.open();
    committed(&platform, 2);
    let bytes = ring.event().expect("a CB_SUBMITTED");
    let posted = RgfxCbSubmitted::read_from_bytes(&bytes).unwrap();
    assert_eq!(
        (posted.header.r#type, posted.cb_obj_id),
        (RGFX_EVT_CB_SUBMITTED, 101)
    );
    assert_eq!(
        ring.scheduled.first().map(|&(_, cb)| cb),
        Some(100),
        "the one before it was scheduled first"
    );
}

/// A command buffer that holds a video encoder is walked in its turn, after the buffers of its
/// queue before it, not beside them: its session's work starts as it is walked. Here its walk
/// waits for the stopped first buffer, then faults at the encoder (its process holds no session),
/// and its failure comes after the first's completion
#[test]
fn a_video_buffer_is_walked_in_its_turn() {
    use reims_vgpu_graphics::ops::{
        RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_BEGIN_VIDEO, RGFX_OP_ENCODER_END,
        RgfxOpVideoPass,
    };

    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, _guest, mut ring) = one_queue(&raised);
    let gate = WalkGate::on(&device);
    ring.send(&device, sampler_command_buffer(obj(1, 1), 100, obj(1, 4)));
    assert!(gate.reached(), "the first walk did not start");
    let pass = RgfxOpVideoPass {
        session_obj_id: obj(1, 9),
    };
    let video = command_buffer_of(
        obj(1, 1),
        101,
        &[
            (RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
            (RGFX_OP_ENCODER_END, &[]),
            (RGFX_OP_ENCODER_BEGIN_VIDEO, pass.as_bytes()),
        ],
    );
    ring.send(&device, video);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        gate.passed.load(Ordering::SeqCst),
        0,
        "the video buffer was walked beside the first"
    );
    gate.open();
    assert!(gate.passed(2), "the video buffer was not walked");
    for (cb, ok) in [(100, true), (101, false)] {
        assert!(complete_until_event(&platform, &mut ring), "no completion");
        let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
        assert_eq!((done.cb_obj_id, done.status == RGFX_CB_OK), (cb, ok));
    }
}

/// A command buffer that holds a video encoder, sent to a queue with no buffer before it still to
/// be carried out, is walked on the queue's service thread, not handed to a lane thread: its turn
/// is next, so it would wait there for nothing. The one sent behind a buffer still walked goes to
/// the lane ([`a_video_buffer_is_walked_in_its_turn`])
#[test]
fn a_lone_video_buffer_is_walked_on_its_queue_s_service_thread() {
    use reims_vgpu_graphics::ops::{
        RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_BEGIN_VIDEO, RGFX_OP_ENCODER_END,
        RgfxOpVideoPass,
    };

    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, _guest, mut ring) = one_queue(&raised);
    let threads = Arc::new(reims_vgpu_vulkan::core::sync::Mutex::new(Vec::new()));
    let seen = Arc::clone(&threads);
    device
        .control
        .decoder()
        .set_walk_hook(Some(Arc::new(move |index: usize| {
            if index == 0 {
                let name = std::thread::current().name().map(str::to_owned);
                seen.lock().push(name.unwrap_or_default());
            }
        })));
    let pass = RgfxOpVideoPass {
        session_obj_id: obj(1, 9),
    };
    let video = |cb| {
        command_buffer_of(
            obj(1, 1),
            cb,
            &[
                (RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
                (RGFX_OP_ENCODER_END, &[]),
                (RGFX_OP_ENCODER_BEGIN_VIDEO, pass.as_bytes()),
            ],
        )
    };
    // Two, one after the other: the second is sent once the first has completed.
    for cb in [100, 101] {
        ring.send(&device, video(cb));
        assert!(complete_until_event(&platform, &mut ring), "no completion");
        let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
        assert_eq!((done.cb_obj_id, done.status == RGFX_CB_OK), (cb, false));
    }
    let threads = threads.lock().clone();
    assert_eq!(threads.len(), 2, "walks seen: {threads:?}");
    for name in &threads {
        assert!(
            name.starts_with("reims-vgpu.graphics-queue-"),
            "a lone video buffer walked on {name:?}"
        );
    }
}

/// The names of the threads that walk command buffers on `device`, one a walk, in the order
/// their walks began.
fn walk_threads(
    device: &GraphicsDevice<FakePlatform>,
) -> Arc<reims_vgpu_vulkan::core::sync::Mutex<Vec<String>>> {
    let threads = Arc::new(reims_vgpu_vulkan::core::sync::Mutex::new(Vec::new()));
    let seen = Arc::clone(&threads);
    device
        .control
        .decoder()
        .set_walk_hook(Some(Arc::new(move |index: usize| {
            if index == 0 {
                let name = std::thread::current().name().map(str::to_owned);
                seen.lock().push(name.unwrap_or_default());
            }
        })));
    threads
}

/// A short command buffer whose commit would wait for nothing is walked and committed on its
/// queue's service thread, not handed to a lane thread that would first be woken: here 8 sent
/// back to back, as a guest's drain. A long one goes to its queue's lane, so the queue's next
/// buffers are taken beside its walk, and so does a short one whose commit would wait: here
/// behind its process's buffer on another queue answered its `CB_SUBMITTED` and still walked
#[test]
fn a_short_buffer_that_waits_for_nothing_is_walked_on_its_queue_s_service_thread() {
    use reims_vgpu_graphics::ops::{RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_END};

    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, _guest, mut ring) = one_queue(&raised);
    let threads = walk_threads(&device);
    for cb in 100..108 {
        ring.send(&device, sampler_command_buffer(obj(1, 1), cb, obj(1, 4)));
    }
    for cb in 100..108 {
        assert!(complete_until_event(&platform, &mut ring), "no completion");
        let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
        assert_eq!((done.cb_obj_id, done.status), (cb, RGFX_CB_OK));
    }
    // A long one: past the bound of ops, empty blit passes.
    let passes: Vec<(u16, &[u8])> = (0..200)
        .flat_map(|_| {
            [
                (RGFX_OP_ENCODER_BEGIN_BLIT, &[][..]),
                (RGFX_OP_ENCODER_END, &[][..]),
            ]
        })
        .collect();
    let long = command_buffer_of(obj(1, 1), 108, &passes);
    assert!(long.len() > super::INLINE_WALK_MAX);
    ring.send(&device, long);
    assert!(complete_until_event(&platform, &mut ring), "no completion");
    let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (108, RGFX_CB_OK));
    let walked = threads.lock().clone();
    assert_eq!(walked.len(), 9, "walks seen: {walked:?}");
    for name in &walked[..8] {
        assert!(
            name.starts_with("reims-vgpu.graphics-queue-"),
            "a short buffer walked on {name:?}"
        );
    }
    assert!(
        walked[8].starts_with("reims-vgpu.graphics-lane-"),
        "a long buffer walked on {:?}",
        walked[8]
    );

    // A short one behind its process's answered buffer still walked on another queue.
    let (device, platform, _guest, [mut held, mut beside, _]) = two_processes(&raised);
    let draw = hold_next_draw(&platform);
    held.send(
        &device,
        with_ask(render_command_buffer(
            obj(1, 1),
            100,
            obj(1, 11),
            obj(1, 20),
        )),
    );
    assert!(
        draw.reached(Duration::from_secs(3)),
        "the walk did not reach its draw"
    );
    assert_eq!(
        ring_events(&platform, &mut held, 1),
        [(RGFX_EVT_CB_SUBMITTED, 100, 0)]
    );
    let threads = walk_threads(&device);
    beside.send(&device, sampler_command_buffer(obj(1, 2), 300, obj(1, 4)));
    for _ in 0..3000 {
        if !threads.lock().is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    draw.open();
    assert_eq!(
        ring_events(&platform, &mut beside, 1),
        [(RGFX_EVT_CB_COMPLETE, 300, u64::from(RGFX_CB_OK))]
    );
    let walked = threads.lock().clone();
    assert_eq!(walked.len(), 1, "walks seen: {walked:?}");
    assert!(
        walked[0].starts_with("reims-vgpu.graphics-lane-"),
        "a buffer whose commit waits walked on {:?}",
        walked[0]
    );
}

/// A command buffer that asks for its `CB_SUBMITTED` behind a short buffer its queue's service
/// thread walks is answered before that walk ends, as behind one a lane's thread walks:
/// published during the walk, rung for, as the guest rings only for a ring the host marked idle,
/// which the service thread marks it through the walk, and answered inside the write; and
/// published before the service thread took the buffer before it, by that thread, before the
/// walk. Each completes after the buffer before it. The service thread held its claim on the
/// queue's messages through the walk, so the write answered nothing and the buffer was answered
/// only at its take, after the walk and the commit before it; and once it let the claim go, it
/// still kept the ring marked busy, so the guest rang no doorbell for a buffer it published
/// during the walk, which was answered after it all the same: a guest's
/// `commitAndWaitUntilSubmitted` behind a buffer of fills waited for them
#[test]
fn a_buffer_that_asks_behind_a_walk_on_its_service_thread_is_answered_before_the_walk_ends() {
    let raised = Arc::new(AtomicU32::new(0));
    let ok = u64::from(RGFX_CB_OK);
    // Published, and rung for, during the walk.
    let (device, platform, _guest, mut ring) = one_queue(&raised);
    let gate = WalkGate::in_place(&device);
    ring.send(&device, sampler_command_buffer(obj(1, 1), 100, obj(1, 4)));
    assert!(gate.reached(), "the short buffer's walk did not start");
    let walked_on = gate.taken_on().unwrap_or_default();
    assert!(
        walked_on.starts_with("reims-vgpu.graphics-queue-"),
        "the short buffer walked on {walked_on:?}"
    );
    assert!(
        ring.send_as_guest(&device, asking(obj(1, 1), 101)),
        "the ring was not marked idle during the walk: the guest rang no doorbell"
    );
    assert!(
        ring.has_event(),
        "the write did not answer the buffer behind the walk"
    );
    let posted = RgfxCbSubmitted::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!(
        (posted.header.r#type, posted.cb_obj_id),
        (RGFX_EVT_CB_SUBMITTED, 101)
    );
    gate.open();
    assert_eq!(
        ring_events(&platform, &mut ring, 2),
        [
            (RGFX_EVT_CB_COMPLETE, 100, ok),
            (RGFX_EVT_CB_COMPLETE, 101, ok)
        ]
    );
    assert_eq!(ring.scheduled, [(0, 100), (0, 101)]);

    // Published, and rung for, before the service thread took the buffer before it.
    let (device, platform, _guest, mut ring) = one_queue(&raised);
    let gate = WalkGate::in_place(&device);
    wait_idle(&device, ring.submit_off);
    ring.publish(sampler_command_buffer(obj(1, 1), 200, obj(1, 4)));
    ring.publish(asking(obj(1, 1), 201));
    device.mmio_write(RGFX_REG_DOORBELL.into(), ring.submit_off as u32);
    assert!(gate.reached(), "the short buffer's walk did not start");
    let walked_on = gate.taken_on().unwrap_or_default();
    assert!(
        walked_on.starts_with("reims-vgpu.graphics-queue-"),
        "the short buffer walked on {walked_on:?}"
    );
    assert!(
        ring.has_event(),
        "the buffer behind the walk was not answered before it"
    );
    let posted = RgfxCbSubmitted::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!(
        (posted.header.r#type, posted.cb_obj_id),
        (RGFX_EVT_CB_SUBMITTED, 201)
    );
    gate.open();
    assert_eq!(
        ring_events(&platform, &mut ring, 2),
        [
            (RGFX_EVT_CB_COMPLETE, 200, ok),
            (RGFX_EVT_CB_COMPLETE, 201, ok)
        ]
    );
    assert_eq!(ring.scheduled, [(0, 200), (0, 201)]);
}

/// A queue's buffers decide their signals of its surface timeline in their turns: a buffer of
/// another queue that names the first of two buffers, sent while the first is still walked and
/// the second, which holds no `SURFACE_ORDER`, has been walked whole beside it, waits on the GPU
/// for the first's signal, which comes once the first is walked. Had the second noted itself
/// taken at its walk's end, the order would read the first as taken with no signal and the
/// buffer would wait for nothing
#[test]
fn a_buffer_named_while_its_queue_s_later_buffer_is_walked_first_is_waited_for() {
    use reims_vgpu_graphics::ops::{
        RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_END, RGFX_OP_SURFACE_ORDER,
        RgfxSurfaceOrderWait,
    };

    const WAIT: &str = "encodeWaitForEvent:value:";
    const SIGNAL: &str = "encodeSignalEvent:value:";
    const RING_DATA: u32 = 0x1_0000;
    let at = |mut message: Vec<u8>, seqno: u64| {
        message[8..16].copy_from_slice(&seqno.to_le_bytes());
        message
    };
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, mut guest, mut first) = one_queue(&raised);
    let rings = (
        RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4_0000,
        RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x6_0000,
    );
    assert_eq!(
        create_queue(&device, &mut guest, 1, 2, rings, RING_DATA, 0),
        RGFX_STATUS_OK
    );
    let mut second = QueueRing::new(&device, rings.0, rings.1, u64::from(RING_DATA));
    let gate = WalkGate::on(&device);
    // Queue 1's buffer at seqno 1 names the op (it uses a shared surface); it stops at the gate.
    let named = command_buffer_of(
        obj(1, 1),
        100,
        &[
            (RGFX_OP_SURFACE_ORDER, &[]),
            (RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
            (RGFX_OP_ENCODER_END, &[]),
        ],
    );
    first.send(&device, at(named, 1));
    assert!(gate.reached(), "the first walk did not start");
    // Its buffer at seqno 2 holds no op, and is walked whole beside it.
    first.send(
        &device,
        at(sampler_command_buffer(obj(1, 1), 101, obj(1, 4)), 2),
    );
    assert!(gate.passed(3), "the second buffer was not walked");
    std::thread::sleep(Duration::from_millis(50));
    // Queue 2's buffer names queue 1's at seqno 1.
    let mark = platform.call_log().mark();
    let wait = RgfxSurfaceOrderWait {
        queue_obj_id: obj(1, 1),
        seqno: 1,
    };
    let waiting = command_buffer_of(obj(1, 2), 200, &[(RGFX_OP_SURFACE_ORDER, wait.as_bytes())]);
    second.send(&device, at(waiting, 1));
    committed(&platform, 1);
    assert_eq!(
        platform
            .call_log()
            .sels_since(mark)
            .iter()
            .filter(|&&s| s == WAIT)
            .count(),
        1,
        "the buffer named does not wait for queue 1's buffer"
    );
    let mark = platform.call_log().mark();
    gate.open();
    committed(&platform, 3);
    assert!(
        platform.call_log().sels_since(mark).contains(&SIGNAL),
        "queue 1's named buffer did not signal its timeline"
    );
    for cb in [100, 101] {
        assert!(complete_until_event(&platform, &mut first), "no completion");
        let done = RgfxCbComplete::read_from_bytes(&first.event().unwrap()).unwrap();
        assert_eq!((done.cb_obj_id, done.status), (cb, RGFX_CB_OK));
    }
    assert!(
        complete_until_event(&platform, &mut second),
        "no completion"
    );
    let done = RgfxCbComplete::read_from_bytes(&second.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (200, RGFX_CB_OK));
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
    carried_out_of(&device, &mut guest, 1);
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
    guest.send(&device, library_lookup(obj(1, 0xfff), 0, [0xee; 32]));
    let refused = next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    let refused = RgfxPipelineReflection::read_from_bytes(&refused).unwrap();
    assert_eq!(refused.obj_id, obj(1, 12));
    assert_eq!(refused.status, RGFX_PIPELINE_REFUSED);
    next_event(&mut guest, RGFX_EVT_COMPLETE);
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
            create_queue(
                &device,
                &mut guest,
                context,
                1,
                (submit, complete),
                RING_DATA,
                0
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

/// A `BLOCK_SHARE` by context `context` of the shared block of `size` bytes at `offset`, or, where
/// not `holds`, its `BLOCK_UNSHARE`, as the kext posts them.
fn block_share(context: u32, offset: u64, size: u64, holds: bool) -> Vec<u8> {
    let r#type = if holds {
        RGFX_CMD_BLOCK_SHARE
    } else {
        RGFX_CMD_BLOCK_UNSHARE
    };
    let mut share = block_release(context, offset, size);
    share[0..4].copy_from_slice(&r#type.to_le_bytes());
    share
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

/// A `TEXTURE_FROM_BUFFER` of more than one level, WindowServer's 54 x 54 BGRA8 of 6 levels at
/// 256 bytes a row, is made by the platform's tiled form with the offset and the bytes per row
/// the guest asked, and one of one level by the public form
#[test]
fn a_texture_of_more_levels_over_a_buffer_is_made_by_the_platform_s_tiled_form() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SIZE: u64 = 0x1_0000;
    guest.send(&device, block_grant(1, BLOCK, SIZE));
    guest.send(&device, resource_id_chunk(1, 0));
    guest.send(&device, buffer_over_host_memory(obj(1, 1), BLOCK, 0x8000));
    let (mut tiled, _) =
        RgfxTextureFromBuffer::read_from_prefix(&texture_from_buffer(obj(1, 2), 1, obj(1, 1)))
            .unwrap();
    tiled.offset = 0x4000;
    tiled.desc = RgfxTextureDescriptor {
        width: 54,
        height: 54,
        mipmap_level_count: 6,
        usage: 1,
        ..tiled.desc
    };
    guest.send(&device, tiled.as_bytes().to_vec());
    guest.send(&device, texture_from_buffer(obj(1, 3), 2, obj(1, 1)));
    // The block holds the buffer and both textures until they go.
    let status = answered(&device, &mut guest, block_release(1, BLOCK, SIZE));
    assert_eq!(status, RGFX_STATUS_REFUSED, "the buffer and its textures");
    assert_eq!(device.decoder_object_count(), 3);
    let log = platform.call_log();
    let tiled = log.calls_to("newTiledTextureWithDescriptor:offset:bytesPerRow:");
    assert_eq!(tiled.len(), 1);
    assert_eq!(tiled[0].args[1..], [Arg::U64(0x4000), Arg::U64(256)]);
    let public = log.calls_to("newTextureWithDescriptor:offset:bytesPerRow:");
    assert_eq!(public.len(), 1, "the one of one level");
    assert_eq!(public[0].args[1..], [Arg::U64(0), Arg::U64(256)]);
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

/// How many times [`eventually`] polls, a millisecond's sleep apart, for a state the device's
/// own threads reach: at least a minute, since a loaded host starves those threads for seconds
/// (a present's wait was not seen within 2,000 polls, 2.2 s and 3.3 s, on msi beside other
/// lanes' builds, where the test passes in under 0.4 s alone). A test that passes polls only
/// until the state holds.
const PATIENCE: u32 = 60_000;

/// Polls `done` until it holds, [`PATIENCE`] times at most: whether it held.
fn eventually(mut done: impl FnMut() -> bool) -> bool {
    for _ in 0..PATIENCE {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    done()
}

/// Waits ([`eventually`]) for the platform to hold `n` committed command buffers not yet
/// completed; their ids, in commit order.
fn committed(platform: &FakePlatform, n: usize) -> Vec<ObjId> {
    let mut ids = Vec::new();
    let held = eventually(|| {
        ids = platform.world().committed();
        ids.len() == n
    });
    assert!(held, "{n} command buffers were not committed");
    ids
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
        create_queue(&device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
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
        Some(Arg::val(
            &reims_vgpu_vulkan::core::platform::types::NsRange::new(0x40, 0x10),
        )),
    );
    assert!(modified < at("commit", None));
}

/// A `Managed` buffer's range sent behind a command buffer of its process answered at its take
/// and still walked holds nothing up on ring 0, which answers the next message meanwhile: its
/// `didModifyRange:` is made once that buffer is committed, and before the commit of a buffer the
/// process sends after it
#[test]
fn a_managed_range_behind_an_answered_walk_holds_up_nothing_on_ring_0() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const RING_DATA: u32 = 0x1_0000;
    let mut guest = Guest::new(&device);
    assert_eq!(
        create_queue(&device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    guest.send(&device, block_grant(1, BLOCK, 0x1_0000));
    let mut managed = buffer_over_host_memory(obj(1, 2), BLOCK, 0x1000);
    let options = std::mem::offset_of!(RgfxBufferOverHostMemory, options);
    managed[options..options + 4].copy_from_slice(&(1u32 << 4).to_le_bytes());
    guest.send(&device, managed);
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    let gate = WalkGate::on(&device);
    ring.send(&device, asking(obj(1, 1), 100));
    assert!(gate.reached(), "the walk did not start");
    let bytes = ring.event().expect("a CB_SUBMITTED");
    assert_eq!(
        RgfxCbSubmitted::read_from_bytes(&bytes).unwrap().cb_obj_id,
        100
    );
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
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(1, 4)),
        RGFX_STATUS_OK,
        "ring 0 waited for the walk"
    );
    let modified = |calls: &[reims_vgpu_vulkan::core::testkit::calllog::Call]| {
        let range = Arg::val(&reims_vgpu_vulkan::core::platform::types::NsRange::new(
            0x40, 0x10,
        ));
        (calls.iter()).position(|c| c.sel == "didModifyRange:" && c.args.first() == Some(&range))
    };
    assert_eq!(
        modified(&platform.call_log().calls()),
        None,
        "carried out before the answered buffer's commit"
    );
    ring.send(&device, command_buffer(obj(1, 1), 101));
    gate.open();
    committed(&platform, 2);
    let calls = platform.call_log().calls();
    let commits: Vec<usize> = (calls.iter().enumerate())
        .filter(|(_, c)| c.sel == "commit")
        .map(|(i, _)| i)
        .collect();
    let modified = modified(&calls).expect("no didModifyRange:");
    assert!(
        commits[commits.len() - 2] < modified && modified < commits[commits.len() - 1],
        "between the two buffers' commits"
    );
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
    // Ring 0's message before it is process 2's (refused: its block is not process 2's).
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
            create_queue(
                &device,
                &mut guest,
                context,
                1,
                (submit, complete),
                RING_DATA,
                0
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

/// A context's close, and a later release of one of its blocks, wait for the command buffer the
/// context committed before an earlier release, still in flight: each release begins a drain for
/// the buffers after it, and a later drain ends only after the earlier ones
#[test]
fn a_close_waits_for_a_buffer_committed_before_an_earlier_release() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        create_queue(&device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    ring.send(&device, command_buffer(obj(1, 1), 100));
    let held = committed(&platform, 1)[0];

    guest.send(&device, block_release(1, BLOCK, 0x1_0000));
    let first = guest.seqno;
    guest.send(&device, block_release(1, BLOCK + 0x1_0000, 0x1_0000));
    let second = guest.seqno;
    guest.send(&device, context_close(1));
    let close = guest.seqno;
    assert!(
        quiet(&guest),
        "answered with a buffer committed before the first release in flight"
    );
    let world = platform.world();
    assert!(world.complete(
        held,
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
    assert_eq!(answer_of(&mut guest, first), RGFX_STATUS_OK);
    assert_eq!(answer_of(&mut guest, second), RGFX_STATUS_OK);
    assert_eq!(answer_of(&mut guest, close), RGFX_STATUS_OK);
}

/// A release the kext posted behind its context's close waits for the context's command buffer
/// still in flight, as the close does: the kext gives the block out again at its answer. It makes
/// the closed context no state again
#[test]
fn a_release_behind_a_close_waits_for_the_close_s_buffers() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const RING_DATA: u32 = 0x1_0000;
    guest.send(&device, block_grant(1, BLOCK, 0x1_0000));
    assert_eq!(
        create_queue(&device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    ring.send(&device, command_buffer(obj(1, 1), 100));
    let held = committed(&platform, 1)[0];

    guest.send(&device, context_close(1));
    let close = guest.seqno;
    guest.send(&device, block_release(1, BLOCK, 0x1_0000));
    let release = guest.seqno;
    assert!(
        quiet(&guest),
        "answered with a buffer of the closed context in flight"
    );
    let world = platform.world();
    assert!(world.complete(
        held,
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
    assert_eq!(answer_of(&mut guest, close), RGFX_STATUS_OK);
    assert_eq!(answer_of(&mut guest, release), RGFX_STATUS_OK);
    // The release made the closed context no state again.
    assert_eq!(device.control.decoder().context_states(), 0);
}

/// A pipeline create whose compile its context's close cancelled (a process that quit with a
/// create it posted without waiting still compiling) ends after the close and stores nothing: it
/// makes the closed context no state again, as a late buffer or release does not.
#[test]
fn a_compile_ending_after_its_context_s_close_makes_it_no_state() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    make_library(&device, &mut guest, obj(1, 10));
    device.control.pool.hold();
    guest.send(&device, compute_pipeline_create(obj(1, 11), obj(1, 10)));
    assert_eq!(
        answered(&device, &mut guest, context_close(1)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.control.decoder().context_states(), 0);
    device.control.pool.release();
    next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    assert_eq!(device.decoder_object_count(), 0);
    assert_eq!(
        device.control.decoder().context_states(),
        0,
        "the compile's end made the closed context a state again"
    );
}

/// An `ICB_CREATE` made on the ICB worker after its context's close (its process died waiting
/// for the answer, and the kext destroys behind it) stores nothing, and makes the closed context
/// no state again.
#[test]
fn an_icb_made_after_its_context_s_close_makes_it_no_state() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = icb_device(&raised);
    let mut guest = Guest::new(&device);
    let before = device.decoder_object_count();
    device.control.icb_worker.hold();
    guest.send(&device, icb_create(obj(1, 50)));
    assert_eq!(
        answered(&device, &mut guest, context_close(1)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.control.decoder().context_states(), 0);
    guest.send(&device, object_destroy(obj(1, 50)));
    guest.send(&device, library_lookup(obj(1, 0xfff), 0, [0xee; 32]));
    let control = Arc::clone(&device.control);
    let opener = std::thread::Builder::new()
        .spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            control.icb_worker.release();
        })
        .unwrap();
    next_event(&mut guest, RGFX_EVT_ICB_STATE);
    next_event(&mut guest, RGFX_EVT_COMPLETE);
    opener.join().unwrap();
    assert_eq!(device.decoder_object_count(), before);
    assert_eq!(
        device.control.decoder().context_states(),
        0,
        "the ICB's make made the closed context a state again"
    );
}

/// A shared block's release waits for the command buffer an earlier holder committed before its
/// `BLOCK_UNSHARE`, still in flight, as for the releasing context's own: the kext unwired the
/// earlier holder's pages at its unshare and gives the block's memory back at the release's
/// answer, so an answer before the buffer finished would let its GPU work land in memory the
/// guest's system gave out again.
#[test]
fn a_shared_block_s_release_waits_for_an_earlier_holder_s_buffers() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SIZE: u64 = 0x1_0000;
    const RING_DATA: u32 = 0x1_0000;
    guest.send(&device, block_share(1, BLOCK, SIZE, true));
    guest.send(&device, block_share(2, BLOCK, SIZE, true));
    assert_eq!(
        create_queue(&device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    ring.send(&device, command_buffer(obj(1, 1), 100));
    let held = committed(&platform, 1)[0];

    guest.send(&device, block_share(1, BLOCK, SIZE, false));
    guest.send(&device, block_release(2, BLOCK, SIZE));
    let release = guest.seqno;
    assert!(
        quiet(&guest),
        "the release was answered with a buffer of an earlier holder's in flight"
    );
    let world = platform.world();
    assert!(world.complete(
        held,
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
    assert_eq!(answer_of(&mut guest, release), RGFX_STATUS_OK);

    // A later share of the block's span, released by its one holder, waits for nothing.
    guest.send(&device, block_share(2, BLOCK, SIZE, true));
    let status = answered(&device, &mut guest, block_release(2, BLOCK, SIZE));
    assert_eq!(status, RGFX_STATUS_OK);
}

/// A shared block's release waits for the command buffer of a holder whose `BLOCK_UNSHARE` has not
/// come, still in flight: the kext posts an unshare out of its lock, so another holder's last
/// let-go may post the block's release first, and gives the block's memory back at the release's
/// answer. The unshare that comes after the release holds back no later release at the offset.
#[test]
fn a_shared_block_s_release_waits_for_a_holder_whose_unshare_has_not_come() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SIZE: u64 = 0x1_0000;
    const RING_DATA: u32 = 0x1_0000;
    guest.send(&device, block_share(1, BLOCK, SIZE, true));
    guest.send(&device, block_share(2, BLOCK, SIZE, true));
    assert_eq!(
        create_queue(&device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    ring.send(&device, command_buffer(obj(1, 1), 100));
    let held = committed(&platform, 1)[0];

    guest.send(&device, block_release(2, BLOCK, SIZE));
    let release = guest.seqno;
    assert!(
        quiet(&guest),
        "the release was answered with a buffer of a holder in flight whose unshare had not come"
    );
    let world = platform.world();
    assert!(world.complete(
        held,
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
    assert_eq!(answer_of(&mut guest, release), RGFX_STATUS_OK);

    // The holder's unshare comes now, with another buffer of its in flight: a later block at the
    // offset, one context's alone, is released without waiting for it.
    ring.send(&device, command_buffer(obj(1, 1), 101));
    let later = committed(&platform, 1)[0];
    guest.send(&device, block_share(1, BLOCK, SIZE, false));
    guest.send(&device, block_share(2, BLOCK, SIZE, true));
    let status = answered(&device, &mut guest, block_release(2, BLOCK, SIZE));
    assert_eq!(
        status, RGFX_STATUS_OK,
        "the late unshare held back a later release at its offset"
    );
    assert!(world.complete(
        later,
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
}

/// A shared block's release by its other holder waits for the command buffer of a holder whose
/// close is not yet answered, still in flight: the kext may let go of the closing context's wire
/// of the block's pages before the close is answered (a close it took for refused), and gives the
/// block's memory back at the release's answer.
#[test]
fn a_shared_block_s_release_waits_for_a_closing_holder_s_buffers() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SIZE: u64 = 0x1_0000;
    const RING_DATA: u32 = 0x1_0000;
    guest.send(&device, block_share(1, BLOCK, SIZE, true));
    guest.send(&device, block_share(2, BLOCK, SIZE, true));
    assert_eq!(
        create_queue(&device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT, COMPLETE, u64::from(RING_DATA));
    ring.send(&device, command_buffer(obj(1, 1), 100));
    let held = committed(&platform, 1)[0];

    guest.send(&device, context_close(1));
    let close = guest.seqno;
    guest.send(&device, block_release(2, BLOCK, SIZE));
    let release = guest.seqno;
    assert!(
        quiet(&guest),
        "the close or the release was answered with a buffer of the closing holder in flight"
    );
    let world = platform.world();
    assert!(world.complete(
        held,
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
    let mut answers: Vec<(u64, u32)> = (0..2)
        .map(|_| {
            let bytes = next_event(&mut guest, RGFX_EVT_COMPLETE);
            let complete = RgfxComplete::read_from_bytes(&bytes).unwrap();
            (complete.header.seqno, complete.status)
        })
        .collect();
    answers.sort_unstable();
    assert_eq!(
        answers,
        [(close, RGFX_STATUS_OK), (release, RGFX_STATUS_OK)]
    );
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
        create_queue(&device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
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
/// (the Vulkan runtime refusing the compile, which ends the process outside a refusal scope)
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
    faults.at(
        "MTLLibrary",
        "newFunctionWithName:constantValues:error:",
        0,
        Fault::Crash,
    );
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

/// With no compile thread, a create's compile runs on its context's lane, where the platform
/// ending the process refuses it as on the pool: the reflection or the `COMPLETE` is refused, and
/// the next creates are made
#[test]
fn a_create_compiled_on_its_lane_the_platform_refuses_is_refused() {
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
    carried_out_of(&device, &mut guest, 3);
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
    carried_out_of(&device, &mut guest, 3);
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
    carried_out_of(&device, &mut guest, 3);
    assert_eq!(device.decoder_object_count(), 1, "the view refused");
    guest.send(
        &device,
        texture_view(obj(3, 3), obj(3, 1), first + 2, 81, (0, 1), (0, 1)),
    );
    carried_out_of(&device, &mut guest, 3);
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
        create_queue(
            &device,
            &mut guest,
            3,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            0
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
    // It fails as an internal error, `MTLCommandBufferErrorInternal` the code the guest's buffer
    // reports, never `MTLCommandBufferErrorNone` (0).
    assert_eq!((failed.status, failed.error_code), (RGFX_CB_INTERNAL, 1));
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
    use reims_vgpu_graphics::ops::render::{
        RGFX_OP_RENDER_DRAW_PRIMITIVES, RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
        RgfxOpDrawPrimitives, RgfxOpSetRenderPipelineState,
    };
    use reims_vgpu_graphics::ops::{
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

/// A command buffer whose draw sets a render pipeline the host refused (a create the bundle
/// posts unanswered, as one equal to an accepted create, "Known pipeline creates") is carried out
/// as Metal would carry it out with the pipeline made, but for the draw, which is dropped: it
/// completes in its queue's order, unfailed, with the shared event values it signals posted ahead
/// of its completion, and the buffer after it runs (`decode::refused`).
#[test]
fn a_command_buffer_naming_a_refused_pipeline_drops_its_draw_and_completes() {
    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    assert_eq!(
        create_queue(
            &device,
            &mut guest,
            3,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            0
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
    assert_eq!(device.decoder_refused_ids(), [obj(3, 11)]);

    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    ring.send(&device, command_buffer(obj(3, 1), 100));
    ring.send(
        &device,
        render_command_buffer(obj(3, 1), 101, obj(3, 11), obj(3, 20)),
    );
    ring.send(&device, command_buffer(obj(3, 1), 102));
    // Three buffers wait on the GPU, and nothing is answered before the first finishes.
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
            (RGFX_EVT_CB_COMPLETE, 101, u64::from(RGFX_CB_OK)),
            (RGFX_EVT_CB_COMPLETE, 102, u64::from(RGFX_CB_OK)),
        ]
    );
    // All three ran on the GPU; the draw with the refused pipeline did not.
    let scheduled: Vec<u64> = ring.scheduled.iter().map(|&(_, cb)| cb).collect();
    assert_eq!(scheduled, [100, 101, 102]);
    assert_eq!(platform.call_log().count(DRAW), 0);
    assert!(!ring.has_event());
}

/// The op `tag` with `body`, as a command buffer's stream holds it.
fn op_of(tag: u16, body: &[u8]) -> (u16, Vec<u8>) {
    (tag, body.to_vec())
}

/// An object whose posted create the host refused fails no command buffer that names it, as
/// Metal fails none for an object it made (`decode::refused`): a pass with it as an attachment is
/// skipped, a binding of it binds nil and the draws after it run, as with nil bound, a copy from
/// it is dropped and the fill beside it is not, a wait for an event of it is
/// dropped, and a signal of a shared event of it still reaches the guest. A create over it stands
/// refused in turn, and its destroy lets the id go.
#[test]
fn what_names_a_refused_create_is_dropped_and_its_command_buffer_completes() {
    use reims_vgpu_graphics::ops::blit::{
        RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER, RGFX_OP_BLIT_FILL_BUFFER,
        RgfxBlitOpCopyTextureToBuffer, RgfxBlitOpFillBuffer,
    };
    use reims_vgpu_graphics::ops::render::{
        RGFX_OP_RENDER_DRAW_PRIMITIVES, RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
        RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, RgfxOpDrawPrimitives, RgfxOpSetFragmentTexture,
        RgfxOpSetRenderPipelineState,
    };
    use reims_vgpu_graphics::ops::{
        RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_BEGIN_RENDER, RGFX_OP_ENCODER_END,
        RGFX_OP_SIGNAL_EVENT, RGFX_OP_WAIT_EVENT, RGFX_PASS_FLAG_RT_SIZE,
        RGFX_RENDER_PASS_COLOR_OFFSET, RgfxOpEvent, RgfxRenderPassColorAttachment,
        RgfxRenderPassDesc,
    };
    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
    const BIND: &str = "setFragmentTexture:atIndex:";
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, mut guest, mut ring) = one_queue(&raised);
    let first = 1 + 7 * RGFX_RESOURCE_ID_CHUNK;
    guest.send(&device, resource_id_chunk(1, 7));
    let (texture, refused_texture, buffer) = (obj(1, 10), obj(1, 11), obj(1, 12));
    let (refused_event, view, pipeline) = (obj(1, 13), obj(1, 14), obj(1, 21));
    guest.send(&device, texture_create(texture, first));
    guest.send(&device, buffer_create(buffer));
    make_library(&device, &mut guest, obj(1, 20));
    guest.send(&device, render_pipeline_create(pipeline, obj(1, 20)));
    expect_reflection(&mut guest, pipeline, RGFX_PIPELINE_OK);
    let faults = platform.faults();
    faults.at("MTLDevice", "newTextureWithDescriptor:", 0, Fault::Nil);
    faults.at("MTLDevice", "newSharedEvent", 0, Fault::Nil);
    guest.send(&device, texture_create(refused_texture, first + 1));
    guest.send(&device, shared_event_create(refused_event, 0));
    guest.send(
        &device,
        texture_view(view, refused_texture, first + 2, 80, (0, 1), (0, 1)),
    );
    carried_out_of(&device, &mut guest, 1);
    assert_eq!(
        device.decoder_refused_ids(),
        [refused_texture, refused_event, view]
    );

    let mut attached = RgfxRenderPassDesc {
        color_count: 1,
        ..RgfxRenderPassDesc::default()
    }
    .as_bytes()
    .to_vec();
    attached.resize(RGFX_RENDER_PASS_COLOR_OFFSET as usize, 0);
    attached.extend_from_slice(
        RgfxRenderPassColorAttachment {
            texture_obj_id: refused_texture,
            ..RgfxRenderPassColorAttachment::default()
        }
        .as_bytes(),
    );
    let sized = RgfxRenderPassDesc {
        flags: RGFX_PASS_FLAG_RT_SIZE,
        render_target_width: 64,
        render_target_height: 64,
        ..RgfxRenderPassDesc::default()
    };
    let set_pipeline = RgfxOpSetRenderPipelineState {
        pipeline_obj_id: pipeline,
    };
    let draw = RgfxOpDrawPrimitives {
        primitive_type: 3,
        vertex_start: 0,
        vertex_count: 3,
    };
    let bind = |texture_obj_id| RgfxOpSetFragmentTexture {
        texture_obj_id,
        index: 0,
    };
    let event = |value| RgfxOpEvent {
        event_obj_id: refused_event,
        value,
    };
    let ops = [
        // A pass with the refused texture as its attachment: skipped whole.
        op_of(RGFX_OP_ENCODER_BEGIN_RENDER, &attached),
        op_of(
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            set_pipeline.as_bytes(),
        ),
        op_of(RGFX_OP_RENDER_DRAW_PRIMITIVES, draw.as_bytes()),
        op_of(RGFX_OP_ENCODER_END, &[]),
        // A pass that binds it: nil is bound there, and its draws run, as Metal's do with nil
        // bound (oracle_nilbind).
        op_of(RGFX_OP_ENCODER_BEGIN_RENDER, sized.as_bytes()),
        op_of(
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            set_pipeline.as_bytes(),
        ),
        op_of(
            RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
            bind(refused_texture).as_bytes(),
        ),
        op_of(RGFX_OP_RENDER_DRAW_PRIMITIVES, draw.as_bytes()),
        op_of(
            RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
            bind(texture).as_bytes(),
        ),
        op_of(RGFX_OP_RENDER_DRAW_PRIMITIVES, draw.as_bytes()),
        op_of(RGFX_OP_ENCODER_END, &[]),
        // A copy from it is dropped; the fill beside it is not.
        op_of(RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
        op_of(
            RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER,
            RgfxBlitOpCopyTextureToBuffer {
                source_texture_obj_id: refused_texture,
                destination_buffer_obj_id: buffer,
                source_size_width: 64,
                source_size_height: 1,
                source_size_depth: 1,
                destination_bytes_per_row: 256,
                destination_bytes_per_image: 256,
                ..RgfxBlitOpCopyTextureToBuffer::default()
            }
            .as_bytes(),
        ),
        op_of(
            RGFX_OP_BLIT_FILL_BUFFER,
            RgfxBlitOpFillBuffer {
                buffer_obj_id: buffer,
                range_location: 0,
                range_length: 0x100,
                value: 7,
                ..RgfxBlitOpFillBuffer::default()
            }
            .as_bytes(),
        ),
        op_of(RGFX_OP_ENCODER_END, &[]),
        // The refused shared event: its wait is dropped, its signal reaches the guest.
        op_of(RGFX_OP_WAIT_EVENT, event(1).as_bytes()),
        op_of(RGFX_OP_SIGNAL_EVENT, event(2).as_bytes()),
    ];
    let ops: Vec<(u16, &[u8])> = ops.iter().map(|(tag, body)| (*tag, &body[..])).collect();
    ring.send(&device, command_buffer_of(obj(1, 1), 100, &ops));
    committed(&platform, 1);
    platform.world().complete_all(CompleteOn::Caller);
    let signalled = ring.event().expect("the shared event's value");
    let signalled = RgfxEventSignalled::read_from_bytes(&signalled).unwrap();
    assert_eq!((signalled.obj_id, signalled.value), (refused_event, 2));
    let done = ring.event().expect("the buffer's completion");
    let done = RgfxCbComplete::read_from_bytes(&done).unwrap();
    assert_eq!(
        (done.cb_obj_id, done.status, done.error_code),
        (100, RGFX_CB_OK, 0)
    );

    let log = platform.call_log();
    assert_eq!(
        log.count("renderCommandEncoderWithDescriptor:"),
        1,
        "the pass skipped"
    );
    assert_eq!(
        log.count(DRAW),
        2,
        "the draw while it was bound runs, with nil there"
    );
    let binds: Vec<Vec<Arg>> = (log.calls().into_iter())
        .filter(|c| c.sel == BIND)
        .map(|c| c.args)
        .collect();
    assert_eq!(binds.len(), 2);
    assert_eq!(
        binds[0][0],
        Arg::Obj(None),
        "the refused texture bound as nil"
    );
    assert_ne!(binds[1][0], Arg::Obj(None));
    assert_eq!(log.count("fillBuffer:range:value:"), 1);
    assert_eq!(
        log.count(
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:\
             destinationOffset:destinationBytesPerRow:destinationBytesPerImage:"
        ),
        0
    );
    assert_eq!(log.count("encodeWaitForEvent:value:"), 0);
    assert_eq!(log.count("encodeSignalEvent:value:"), 0);

    // Its destroy lets the id go, as any object's.
    guest.send(&device, object_destroy(refused_texture));
    carried_out_of(&device, &mut guest, 1);
    assert_eq!(device.decoder_refused_ids(), [refused_event, view]);
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
        flags: 0,
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

/// Waits ([`eventually`]) for `n` presents to wait for their writers.
fn presents_waiting(device: &GraphicsDevice<FakePlatform>, n: usize) {
    assert!(
        eventually(|| device.control.presents.waiting() == n),
        "{n} presents were not waiting, but {}",
        device.control.presents.waiting()
    );
}

/// The width of the texture the display shows; `None` for the scanout.
fn shown_width(device: &GraphicsDevice<FakePlatform>) -> Option<u64> {
    use reims_vgpu_vulkan::core::platform::resource::Texture as _;
    device
        .control
        .presented
        .texture
        .lock()
        .as_ref()
        .map(|t| t.texture.width())
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
        create_queue(device, &mut guest, 1, 1, (SUBMIT, COMPLETE), RING_DATA, 0),
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
    // The guest committed buffers 4 and 5 on the queue, 5 drawing the surface, then presented it;
    // the host has taken neither yet (no doorbell was rung).
    ring.publish(command_buffer_at(obj(1, 1), 100, 4));
    ring.publish(command_buffer_at(obj(1, 1), 101, 5));
    write_writers(&device, BLOCK + WRITERS, &[(obj(1, 1), 5)]);
    guest.send(&device, present_surface(0x77));
    presents_waiting(&device, 1);
    assert_eq!(device.control.pool.favoured(), Some(1));
    assert!(quiet(&guest), "answered before it was shown");
    assert_eq!(shown_width(&device), None);

    // The host takes both, and buffer 4 finishes: the present still waits.
    device.mmio_write(RGFX_REG_DOORBELL.into(), SUBMIT as u32);
    let ids = committed(&platform, 2);
    let world = platform.world();
    let done = |id| world.complete(id, CommandBufferStatus::COMPLETED, None, CompleteOn::Caller);
    assert!(done(ids[0]));
    ring.event().expect("buffer 4's CB_COMPLETE");
    presents_waiting(&device, 1);
    assert_eq!(shown_width(&device), None);

    // Buffer 5 finishes: the surface shows, and the present is answered.
    assert!(done(ids[1]));
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

/// A present waits for a writer of a queue created just before it whose handle the host has yet to
/// make: the device tells the host of the queue before it answers the create, and the guest names
/// the queue's buffers only after the answer, however late the queue's service thread has the
/// decoder make the handle. With the service thread of process 1's new queue held there, as a
/// no-copy buffer of the process being made holds it (`pages_buffers`), a present whose writer is
/// the queue's first buffer, published on its ring, waits for it; once the thread goes on, the
/// buffer is taken and committed, and the present is shown, and answered, when it finishes
#[test]
fn a_present_waits_for_a_writer_of_a_queue_whose_handle_is_not_yet_made() {
    const RING_DATA: u32 = 0x1_0000;
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    let (mut guest, _ring) = surface_and_queue(&device);
    // Process 1's messages so far carried out (the texture over the surface made), and its first
    // queue's handle made.
    carried_out_of(&device, &mut guest, 1);
    let log = platform.call_log();
    assert!(eventually(|| log.count("newCommandQueue") == 1));

    // Process 1's second queue, answered while its service thread waits for the process's no-copy
    // buffer before it hands the create to the host.
    let pages = &device.control.pages_buffers;
    let held = pages.begin(pages_context(1), 0, device.control.pool.now());
    let (submit, complete) = (
        RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4_0000,
        RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x6_0000,
    );
    assert_eq!(
        create_queue(&device, &mut guest, 1, 4, (submit, complete), RING_DATA, 0),
        RGFX_STATUS_OK
    );
    // Its first buffer draws the surface, and the surface is presented.
    let mut fresh = QueueRing::new(&device, submit, complete, u64::from(RING_DATA));
    fresh.publish(command_buffer_at(obj(1, 4), 400, 1));
    write_writers(&device, BLOCK + WRITERS, &[(obj(1, 4), 1)]);
    guest.send(&device, present_surface(0x77));
    let world = platform.world();
    assert!(
        eventually(|| device.control.presents.waiting() == 1 || shown_width(&device).is_some()),
        "the present was not carried out"
    );
    assert_eq!(shown_width(&device), None, "shown before its writer ran");
    assert_eq!(device.control.presents.waiting(), 1);
    assert!(quiet(&guest), "answered before it was shown");
    assert_eq!(
        log.count("newCommandQueue"),
        1,
        "the new queue's handle was made"
    );

    // The service thread goes on: the new queue's handle is made and its buffer committed, and
    // once the buffer finishes the surface shows and the present is answered.
    drop(held);
    let ids = committed(&platform, 1);
    assert_eq!(log.count("newCommandQueue"), 2);
    presents_waiting(&device, 1);
    assert!(world.complete(
        ids[0],
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_OK);
    assert_eq!(shown_width(&device), Some(64));
    presents_waiting(&device, 0);
}

/// A failure the host posts at once, ahead of the buffers before it on its queue, says nothing of
/// them: a present waiting for the earlier buffer that wrote its surface still waits for that
/// buffer to finish, and is shown only then
#[test]
fn a_later_buffer_failed_at_once_shows_no_present_of_an_earlier_writer() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    let (mut guest, mut ring) = surface_and_queue(&device);
    // Buffer 4 draws the surface; buffer 5, on the same ring, names a queue that is not the
    // ring's, so the host neither carries it out nor stands in for it, and fails it at once.
    ring.publish(command_buffer_at(obj(1, 1), 100, 4));
    ring.publish(command_buffer_at(obj(1, 9), 101, 5));
    write_writers(&device, BLOCK + WRITERS, &[(obj(1, 1), 4)]);
    guest.send(&device, present_surface(0x77));
    presents_waiting(&device, 1);
    device.mmio_write(RGFX_REG_DOORBELL.into(), SUBMIT as u32);
    let ids = committed(&platform, 1);
    let failed =
        RgfxCbComplete::read_from_bytes(&ring.event().expect("buffer 5's failure")).unwrap();
    assert_eq!((failed.cb_obj_id, failed.header.seqno), (101, 5));
    assert_ne!(failed.status, RGFX_CB_OK);
    presents_waiting(&device, 1);
    assert!(quiet(&guest), "answered before buffer 4 finished");
    assert_eq!(shown_width(&device), None, "shown before buffer 4 finished");

    // Buffer 4 finishes: the surface shows, and the present is answered.
    let world = platform.world();
    assert!(world.complete(
        ids[0],
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_OK);
    assert_eq!(shown_width(&device), Some(64));
    presents_waiting(&device, 0);
}

/// A presented surface's block is given back only once the display is done with it, as a
/// compositor holds the IOSurface it shows: its release, after the texture's and its buffer's
/// destroys, waits while the display shows the texture, and then while a window draw of it (a
/// new layout, the cursor) is on the GPU; once the release is answered, no draw reads the block
#[test]
fn a_presented_surface_s_block_waits_for_the_display() {
    use reims_vgpu_vulkan::core::platform::command::CommandBuffer as _;
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    let (mut guest, _ring) = surface_and_queue(&device);
    let draw = sink::ScanoutDraw::new(
        Arc::clone(&platform),
        &device.control.region,
        &Arc::new(Shown::default()),
        &device.control.presented,
    )
    .unwrap();
    // With no window, the present is answered as it is shown.
    let status = answered(&device, &mut guest, present_surface(0x77));
    assert_eq!(status, RGFX_STATUS_OK);
    let (drawable, _) = device.control.presented.shown().expect("the surface shown");
    let rect = reims_vgpu_window::PixelRect {
        x: 0,
        y: 0,
        width: 64,
        height: 64,
    };
    let (command_buffer, presents) = draw.encode(&drawable, rect).expect("a draw of it");
    presents.after(&command_buffer);
    no_catch(command_buffer.commit(OnFault::Die));
    let drawing = committed(&platform, 1)[0];

    guest.send(&device, object_destroy(obj(1, 3)));
    guest.send(&device, object_destroy(obj(1, 2)));
    guest.send(&device, block_release(1, BLOCK, 0x1_0000));
    let release = guest.seqno;
    assert!(
        quiet(&guest),
        "the release was answered while the display shows the surface"
    );

    // The scanout is presented: the display lets go of the surface, its draw still on the GPU.
    guest.send(&device, present_surface(0));
    let scanout = guest.seqno;
    assert_eq!(answer_of(&mut guest, scanout), RGFX_STATUS_OK);
    assert!(
        quiet(&guest),
        "the release was answered while a window draw reads the surface"
    );
    // A draw from here reads nothing of the block: the guest set no scanout either.
    assert!(device.control.presented.shown().is_none());
    assert!(draw.encode(&drawable, rect).is_none());

    let world = platform.world();
    assert!(world.complete(
        drawing,
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
    assert_eq!(answer_of(&mut guest, release), RGFX_STATUS_OK);
}

/// A present waiting for the command buffer that writes its surface holds the surface's block
/// from the present on: a release of the block that comes before the present is shown waits for
/// it to be shown, then for the display to show something else
#[test]
fn a_waiting_present_holds_its_surface_s_block() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    let (mut guest, mut ring) = surface_and_queue(&device);
    ring.publish(command_buffer_at(obj(1, 1), 100, 4));
    write_writers(&device, BLOCK + WRITERS, &[(obj(1, 1), 4)]);
    guest.send(&device, present_surface(0x77));
    let present = guest.seqno;
    presents_waiting(&device, 1);

    guest.send(&device, object_destroy(obj(1, 3)));
    guest.send(&device, object_destroy(obj(1, 2)));
    guest.send(&device, block_release(1, BLOCK, 0x1_0000));
    let release = guest.seqno;
    assert!(
        quiet(&guest),
        "the release was answered while a present of the surface waits"
    );

    // The writer finishes: the surface shows, and the present is answered; the release waits.
    device.mmio_write(RGFX_REG_DOORBELL.into(), SUBMIT as u32);
    let ids = committed(&platform, 1);
    let world = platform.world();
    assert!(world.complete(
        ids[0],
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
    ring.event().expect("the writer's CB_COMPLETE");
    assert_eq!(answer_of(&mut guest, present), RGFX_STATUS_OK);
    assert_eq!(shown_width(&device), Some(64));
    assert!(
        quiet(&guest),
        "the release was answered while the display shows the surface"
    );

    guest.send(&device, present_surface(0));
    let scanout = guest.seqno;
    assert_eq!(answer_of(&mut guest, scanout), RGFX_STATUS_OK);
    assert_eq!(answer_of(&mut guest, release), RGFX_STATUS_OK);
}

/// Presents are shown in the order made: one with nothing to wait for, made after one that waits,
/// waits behind it; a present waiting for a queue that is destroyed is shown, as nothing of that
/// queue will finish; and a writer naming no queue the host holds holds nothing
#[test]
fn presents_keep_their_order_and_a_gone_queue_holds_none() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    let (mut guest, mut ring) = surface_and_queue(&device);
    ring.publish(command_buffer_at(obj(1, 1), 100, 9));
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

/// A writer whose seqno is past every seqno its queue's ring has published, taken or not, names
/// no command buffer: the present is shown at once, not held for ever, and so is the one after
/// it; a writer the ring has published holds its present as before
#[test]
fn a_writer_past_what_its_queue_published_holds_nothing() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    let (mut guest, mut ring) = surface_and_queue(&device);
    ring.send(&device, command_buffer_at(obj(1, 1), 100, 1));
    let taken = committed(&platform, 1);
    ring.publish(command_buffer_at(obj(1, 1), 101, 2));
    for bogus in [3, u64::MAX] {
        write_writers(&device, BLOCK + WRITERS, &[(obj(1, 1), bogus)]);
        guest.send(&device, present_surface(0x77));
        let seqno = guest.seqno;
        assert_eq!(
            answer_of(&mut guest, seqno),
            RGFX_STATUS_OK,
            "seqno {bogus}"
        );
        presents_waiting(&device, 0);
    }
    guest.send(&device, present_surface(0));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_OK);

    // The published buffer 2, not yet taken, holds its present.
    write_writers(&device, BLOCK + WRITERS, &[(obj(1, 1), 2)]);
    guest.send(&device, present_surface(0x77));
    presents_waiting(&device, 1);
    let world = platform.world();
    assert!(world.complete(
        taken[0],
        CommandBufferStatus::COMPLETED,
        None,
        CompleteOn::Caller
    ));
    ring.event().expect("buffer 1's CB_COMPLETE");
    presents_waiting(&device, 1);
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
/// no answer of Metal's for), which ends the process outside a refusal scope, fails that command
/// buffer alone, as a GPU error on Metal fails the buffer it struck: its completion is an internal
/// error, and its context's later work is carried out as any context's, its command buffers,
/// its creates (its sampler creates and its queue creates among them) and its destroys, as is
/// another context's
#[test]
fn a_completion_the_platform_refuses_fails_that_command_buffer_alone() {
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
            create_queue(
                &device,
                &mut guest,
                context,
                1,
                (submit, complete),
                RING_DATA,
                0
            ),
            RGFX_STATUS_OK
        );
    }
    // Each queue's handle is made on its own service thread, which may start after its create is
    // answered: both are made before the counts below.
    assert!(
        holds_objects(&device, 2),
        "the two queues' handles were not made"
    );
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
    // Its error is `MTLCommandBufferErrorInternal`, which the guest's buffer reports, never 0.
    assert_eq!(failed.error_code, 1);

    // The context's later command buffer runs.
    ring3.send(&device, command_buffer(obj(3, 1), 101));
    assert!(complete_until_event(&platform, &mut ring3), "no completion");
    let done = RgfxCbComplete::read_from_bytes(&ring3.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (101, RGFX_CB_OK));

    // Its creates are carried out: a sampler's, answered with its id, and a queue's.
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
        (answer.header.seqno, answer.status),
        (guest.seqno, RGFX_STATUS_OK)
    );
    assert_ne!(answer.gpu_resource_id, 0);
    assert_eq!(
        platform
            .call_log()
            .calls_to("newSamplerStateWithDescriptor:")
            .len(),
        1
    );
    assert_eq!(
        create_queue(
            &device,
            &mut guest,
            3,
            2,
            (SUBMIT_NEXT, COMPLETE_NEXT),
            RING_DATA,
            0
        ),
        RGFX_STATUS_OK
    );
    let mut next = QueueRing::new(&device, SUBMIT_NEXT, COMPLETE_NEXT, u64::from(RING_DATA));
    next.send(&device, command_buffer(obj(3, 2), 300));
    assert!(complete_until_event(&platform, &mut next), "no completion");
    let done = RgfxCbComplete::read_from_bytes(&next.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (300, RGFX_CB_OK));

    // Another context's work goes on.
    ring4.send(&device, command_buffer(obj(4, 1), 200));
    assert!(complete_until_event(&platform, &mut ring4), "no completion");
    let done = RgfxCbComplete::read_from_bytes(&ring4.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (200, RGFX_CB_OK));

    // Its destroys are carried out.
    let before = device.decoder_object_count();
    guest.send(&device, object_destroy(obj(3, 5)));
    carried_out_of(&device, &mut guest, 3);
    assert_eq!(
        device.decoder_object_count(),
        before - 1,
        "the event destroyed"
    );
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
    carried_out_of(&device, &mut guest, 3);
    let events: Vec<ObjId> = (world.live("MTLEvent").into_iter())
        .filter(|id| !before.contains(id))
        .collect();
    assert_eq!(events.len(), 3);
    assert_eq!(device.decoder_object_count(), 3);

    // At a destroy.
    world.die_at_release(events[0]);
    guest.send(&device, object_destroy(obj(3, 1)));
    carried_out_of(&device, &mut guest, 3);
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
    carried_out_of(&device, &mut guest, 3);
    assert_eq!(device.decoder_object_count(), 1);
}

/// With a memory map the device carries out `SURFACE_PAGES`, and `PAGES_BUFFERS` and
/// `HEAP_OVER_PAGES` over it, where
/// the host's page is the guest's 4 KiB, and says so in `FEATURES_LO`; without one it never does.
/// Every device answers a command buffer's ask for its submission (`CB_SUBMITTED`), orders command
/// buffers over the IOSurfaces they share (`SURFACE_ORDER`), makes indirect command buffers
/// (`INDIRECT_COMMAND_BUFFERS`), makes the queues the kext asks unshared on the platform's
/// unshared queue (`UNSHARED_QUEUES`), tells a producer waiting for room on a full submission
/// ring of the device's takes (`SUBMIT_ROOM`), reads indirect command buffers' CPU encodes from
/// slots in the region (`ICB_SLOTS`), lets the guest read the clock its VBLANKs are stamped on
/// (`HOST_TIME`), and shows the display pipe's colour (`DISPLAY_COLOR`). The fake's GPU shares
/// the CPU's memory, so a device with `SURFACE_PAGES` answers which surfaces need a device cache
/// (`SURFACE_CACHE_REFUSED`), keeps none of the rest, defers a fresh one of those that would need
/// one whatever its pages (`SURFACE_DEFERRED_ASK`), within a quarter of its GPU's own memory's
/// budget, and has the guest align the surfaces' rows as its GPU's images take them in place (the
/// device info's `linear_row_alignment`), which a device of no surface pages leaves as they were
#[test]
fn a_memory_map_gives_the_surface_pages_feature_on_a_guest_sized_page() {
    use reims_vgpu_vulkan::core::testkit::GuestRam;

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
            | RGFX_FEATURE_PAGES_BUFFERS
            | RGFX_FEATURE_HEAP_OVER_PAGES
            | RGFX_FEATURE_SURFACE_CACHE_REFUSED
            | RGFX_FEATURE_SURFACE_DEFERRED_ASK
    } else {
        0
    };
    assert_eq!(
        device.mmio_read(RGFX_REG_FEATURES_LO.into()),
        want | RGFX_FEATURE_CB_SUBMITTED
            | RGFX_FEATURE_SURFACE_ORDER
            | RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS
            | RGFX_FEATURE_UNSHARED_QUEUES
            | RGFX_FEATURE_SUBMIT_ROOM
            | RGFX_FEATURE_ICB_SLOTS
            | RGFX_FEATURE_COMMIT_ORDER
            | RGFX_FEATURE_HOST_TIME
            | RGFX_FEATURE_MANAGED_TEXTURE_SLABS
            | RGFX_FEATURE_MANAGED_BUFFER_SLABS
            | RGFX_FEATURE_SMALL_ID_CHUNKS
            | RGFX_FEATURE_DISPLAY_COLOR
            | RGFX_FEATURE_MESH_PIPELINES
            | RGFX_FEATURE_SHARED_EVENT_NAMES
    );
    assert_eq!(device.mmio_read(RGFX_REG_FEATURES_HI.into()), 0);
    // The fake's GPU shares the CPU's memory: deferred blocks only where they ask.
    let budget = if want & RGFX_FEATURE_SURFACE_DEFERRED_ASK != 0 {
        FAKE_OWN_MEMORY / 4
    } else {
        0
    };
    assert_eq!(
        u64::from(device.mmio_read(RGFX_REG_DEFERRED_BUDGET_LO.into())),
        budget & 0xffff_ffff
    );
    assert_eq!(
        u64::from(device.mmio_read(RGFX_REG_DEFERRED_BUDGET_HI.into())),
        budget >> 32
    );
    let aligned = want & RGFX_FEATURE_SURFACE_PAGES != 0;
    assert_eq!(
        u64::from(device_info_of(&device).linear_row_alignment),
        if aligned {
            FAKE_LINEAR_ROW_ALIGNMENT
        } else {
            0
        },
        "the rows of the surfaces that lie in place in their pages"
    );

    let (device, _) = headless_on(&Arc::new(AtomicU32::new(0)));
    assert_eq!(device_info_of(&device).linear_row_alignment, 0);
    assert_eq!(
        device.mmio_read(RGFX_REG_FEATURES_LO.into()),
        RGFX_FEATURE_CB_SUBMITTED
            | RGFX_FEATURE_SURFACE_ORDER
            | RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS
            | RGFX_FEATURE_UNSHARED_QUEUES
            | RGFX_FEATURE_SUBMIT_ROOM
            | RGFX_FEATURE_ICB_SLOTS
            | RGFX_FEATURE_COMMIT_ORDER
            | RGFX_FEATURE_HOST_TIME
            | RGFX_FEATURE_MANAGED_TEXTURE_SLABS
            | RGFX_FEATURE_MANAGED_BUFFER_SLABS
            | RGFX_FEATURE_SMALL_ID_CHUNKS
            | RGFX_FEATURE_DISPLAY_COLOR
            | RGFX_FEATURE_MESH_PIPELINES
            | RGFX_FEATURE_SHARED_EVENT_NAMES
    );
    assert_eq!(device.mmio_read(RGFX_REG_DEFERRED_BUDGET_LO.into()), 0);
    assert_eq!(device.mmio_read(RGFX_REG_DEFERRED_BUDGET_HI.into()), 0);
}

/// The deferred blocks' budget is a quarter of the GPU's own memory's where the device offers
/// `SURFACE_DEFERRED`, or `SURFACE_DEFERRED_ASK` on a GPU that shares the CPU's memory, and 0
/// where it offers neither
#[test]
fn the_deferred_budget_is_a_quarter_of_the_gpus_own_memory() {
    let deferred = u64::from(RGFX_FEATURE_SURFACE_CACHE | RGFX_FEATURE_SURFACE_DEFERRED);
    assert_eq!(super::deferred_budget(deferred, 16 << 30), 4 << 30);
    assert_eq!(
        super::deferred_budget(deferred, 0xcf5c_2c000),
        0xcf5c_2c000 / 4
    );
    let asking = u64::from(RGFX_FEATURE_SURFACE_CACHE_REFUSED | RGFX_FEATURE_SURFACE_DEFERRED_ASK);
    assert_eq!(super::deferred_budget(asking, 16 << 30), 4 << 30);
    let cached = u64::from(RGFX_FEATURE_SURFACE_CACHE);
    assert_eq!(super::deferred_budget(cached, 16 << 30), 0);
    let refused = u64::from(RGFX_FEATURE_SURFACE_CACHE_REFUSED);
    assert_eq!(super::deferred_budget(refused, 16 << 30), 0);
}

/// What the poison-fill tests write over the guest's memory before the host answers into it: a
/// 32-bit word no field the host writes holds.
const POISON: u32 = 0xa5a5_dead;

/// Fills `len` bytes of the region at `offset` with [`POISON`].
fn poison(device: &GraphicsDevice<FakePlatform>, offset: u64, len: u64) {
    let words: Vec<u8> = (0..len / 4).flat_map(|_| POISON.to_le_bytes()).collect();
    // SAFETY: the region is the device's, mapped at this address while it lives, and the span
    // lies in it; the host writes none of it until the guest's next message.
    unsafe {
        std::ptr::copy_nonoverlapping(
            words.as_ptr(),
            device.region_base().add(offset as usize),
            words.len(),
        )
    };
}

/// The offsets of `bytes`' 32-bit words still [`POISON`]: bytes the host answered for and never
/// wrote.
fn unwritten(bytes: &[u8]) -> Vec<usize> {
    let (words, _) = bytes.as_chunks::<4>();
    words
        .iter()
        .enumerate()
        .filter(|(_, w)| u32::from_le_bytes(**w) == POISON)
        .map(|(i, _)| i * 4)
        .collect()
}

/// Every answer on ring 0's completion ring writes every byte of itself: with the ring's data
/// poisoned before each command, no word of a `COMPLETE`, `PIPELINE_REFLECTION`,
/// `SAMPLER_STATE` or `SIZE_AND_ALIGN` is left as the guest's memory held it, and each is as long
/// as its type's struct
#[test]
fn ring_0_s_answers_write_every_byte_they_name() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    let size_query = RgfxHeapTextureSizeAndAlign {
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
    }
    .as_bytes()
    .to_vec();
    let layout_query = {
        let descriptor = RgfxArgumentDescriptor {
            data_type: 60, // MTLDataTypePointer
            index: 0,
            ..RgfxArgumentDescriptor::default()
        };
        let mut bytes = RgfxArgumentLayoutQuery {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ARGUMENT_LAYOUT,
                size: 0,
                seqno: 0,
            },
            obj_id: obj(1, 5),
            reply_offset: BLOCK,
            reply_length: 0x1000,
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
    guest.send(&device, block_grant(1, BLOCK, 0x1000));
    // The queue's rings lie in blocks its process holds, as the kext grants them.
    let ring_span = u64::from(RGFX_RING_HEADER_SIZE) + 0x1_0000;
    guest.send(&device, block_grant(1, SUBMIT_OFF, ring_span));
    guest.send(&device, block_grant(1, COMPLETE_OFF, ring_span));
    let commands: Vec<(Vec<u8>, u32, usize)> = vec![
        (
            library_create(obj(1, 10)),
            RGFX_EVT_COMPLETE,
            size_of::<RgfxComplete>(),
        ),
        (
            render_pipeline_create(obj(1, 11), obj(1, 10)),
            RGFX_EVT_PIPELINE_REFLECTION,
            size_of::<RgfxPipelineReflection>(),
        ),
        (
            compute_pipeline_create(obj(1, 12), obj(1, 10)),
            RGFX_EVT_PIPELINE_REFLECTION,
            size_of::<RgfxPipelineReflection>(),
        ),
        (
            sampler_create(obj(1, 4)),
            RGFX_EVT_SAMPLER_STATE,
            size_of::<RgfxSamplerState>(),
        ),
        (
            size_query,
            RGFX_EVT_SIZE_AND_ALIGN,
            size_of::<RgfxSizeAndAlign>(),
        ),
        (layout_query, RGFX_EVT_COMPLETE, size_of::<RgfxComplete>()),
        (
            queue_create(1, 1, SUBMIT_OFF, COMPLETE_OFF, 0x1_0000, 0),
            RGFX_EVT_COMPLETE,
            size_of::<RgfxComplete>(),
        ),
        (
            context_close(1),
            RGFX_EVT_COMPLETE,
            size_of::<RgfxComplete>(),
        ),
    ];
    for (command, answer, size) in commands {
        let r#type = u32::from_le_bytes(command[..4].try_into().unwrap());
        poison(&device, RGFX_CONTROL_COMPLETE_OFFSET + 256, RING_DATA);
        guest.send(&device, command);
        let bytes = next_event(&mut guest, answer);
        let (header, _) = RgfxMsgHeader::read_from_prefix(&bytes).unwrap();
        assert_eq!(header.seqno, guest.seqno, "command {type:#x}");
        assert_eq!(bytes.len(), size, "command {type:#x}'s answer's length");
        assert_eq!(
            unwritten(&bytes),
            Vec::<usize>::new(),
            "command {type:#x}'s answer left these offsets unwritten"
        );
        // Each answered OK (every status's OK is 0): a refusal would say nothing of what an OK
        // answer writes.
        let status = if answer == RGFX_EVT_PIPELINE_REFLECTION {
            RgfxPipelineReflection::read_from_bytes(&bytes)
                .unwrap()
                .status
        } else {
            u32::from_le_bytes(bytes[16..20].try_into().unwrap())
        };
        assert_eq!(status, 0, "command {type:#x} was not answered OK");
    }
}

/// An argument layout answered OK writes every byte of the answer it says it took (its
/// `needed`), over a block the guest's memory had poisoned; one answered `TOO_SMALL` writes its
/// `needed` alone, as the wire says
#[test]
fn an_argument_layout_writes_every_byte_it_answers() {
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
    let query = |reply_length: u64| {
        let mut bytes = RgfxArgumentLayoutQuery {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ARGUMENT_LAYOUT,
                size: 0,
                seqno: 0,
            },
            obj_id: obj(1, 1),
            reply_offset: BLOCK,
            reply_length,
            source: RGFX_ARGUMENT_SOURCE_DESCRIPTORS,
            count: 2,
            descriptors_offset: size_of::<RgfxArgumentLayoutQuery>() as u32,
            ..RgfxArgumentLayoutQuery::default()
        }
        .as_bytes()
        .to_vec();
        bytes.extend_from_slice(descriptors.as_bytes());
        bytes
    };
    poison(&device, BLOCK, SIZE);
    guest.send(&device, query(SIZE));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_OK);
    // SAFETY: as `Guest`'s helpers; the block lies in the region.
    let needed = unsafe { guest.load_u64(BLOCK as usize) };
    assert!(needed > size_of::<RgfxArgumentLayoutReply>() as u64 && needed <= SIZE);
    // SAFETY: as above.
    let answer = unsafe { guest.read(BLOCK as usize, needed as usize) };
    assert_eq!(
        unwritten(&answer),
        Vec::<usize>::new(),
        "the layout's unwritten offsets"
    );

    poison(&device, BLOCK, SIZE);
    let small = size_of::<RgfxArgumentLayoutReply>() as u64;
    guest.send(&device, query(small));
    let seqno = guest.seqno;
    assert_eq!(answer_of(&mut guest, seqno), RGFX_STATUS_TOO_SMALL);
    // SAFETY: as above.
    let answer = unsafe { guest.read(BLOCK as usize, small as usize) };
    assert_eq!(u64::from_le_bytes(answer[..8].try_into().unwrap()), needed);
    assert_eq!(unwritten(&answer), vec![8, 12], "only `needed` is written");
}

/// A command buffer's `CB_COMPLETE` writes every byte of itself over a completion ring the guest's
/// memory had poisoned: its status, its id and its GPU times
#[test]
fn a_command_buffer_s_completion_writes_every_byte() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const DATA: u32 = 0x1_0000;
    let mut guest = Guest::new(&device);
    assert_eq!(
        create_queue(
            &device,
            &mut guest,
            2,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            DATA,
            0
        ),
        RGFX_STATUS_OK
    );
    poison(&device, COMPLETE_OFF + 256, u64::from(DATA));
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(DATA));
    ring.arm();
    ring.send(&device, command_buffer(obj(2, 1), 100));
    assert!(complete_until_event(&platform, &mut ring), "no CB_COMPLETE");
    let bytes = ring.event().expect("a CB_COMPLETE");
    let done = RgfxCbComplete::read_from_bytes(&bytes).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (100, RGFX_CB_OK));
    assert_eq!(bytes.len(), size_of::<RgfxCbComplete>());
    assert_eq!(
        unwritten(&bytes),
        Vec::<usize>::new(),
        "the completion's unwritten offsets"
    );
}

/// A reset drops everything the guest's processes made, as each one's close does: every object
/// of every context (a queue, a texture, a buffer over a block at a GPU address its process
/// pinned, a library, another context's event), and the platform's objects go with them, so
/// their resource ids and GPU addresses are free again; the library cache goes too. The device
/// is then as one just made: its queues are gone, the same ids make the same objects again, at
/// the same texture resource id and the same pinned GPU address, and a lookup of the library
/// misses
#[test]
fn a_reset_drops_every_object_the_guest_made() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    let world = platform.world();
    let textures_before = world.live("MTLTexture").len();
    let buffers_before = world.live("MTLBuffer").len();
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const SIZE: u64 = 0x10_0000;
    const RING_DATA: u32 = 0x1_0000;
    let (library, key) = keyed_metallib();
    // What a process makes after the device's start, the same before and after the reset.
    let make = |guest: &mut Guest| {
        guest.send(&device, block_grant(1, BLOCK, SIZE));
        assert_eq!(
            answered(
                &device,
                guest,
                queue_create(1, 1, BLOCK, BLOCK + 0x2_0000, RING_DATA, 0)
            ),
            RGFX_STATUS_OK
        );
        guest.send(&device, resource_id_chunk(1, 0));
        guest.send(&device, texture_create(obj(1, 2), 1));
        let pinned = RgfxBufferOverHostMemory {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_OVER_HOST_MEMORY,
                size: 0,
                seqno: 0,
            },
            obj_id: obj(1, 3),
            gpu_address: RGFX_CONTEXT_VA_BASE + 0x8000,
            region_offset: BLOCK + 0x8_0000,
            length: 0x4000,
            ..RgfxBufferOverHostMemory::default()
        };
        guest.send(&device, pinned.as_bytes().to_vec());
        assert_eq!(
            answered(
                &device,
                guest,
                library_lookup(obj(1, 4), library.len(), key)
            ),
            RGFX_STATUS_NOT_FOUND,
            "a lookup misses on a new device"
        );
        assert_eq!(
            answered(
                &device,
                guest,
                keyed_library_create(obj(1, 4), &library, Some(key))
            ),
            RGFX_STATUS_OK
        );
        guest.send(&device, event_create(obj(2, 1)));
        for _ in 0..2000 {
            if device.decoder_object_count() == 5 {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(device.decoder_object_count(), 5);
        assert_eq!(device.graphics_queue_count(), 1);
    };
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    make(&mut guest);
    let texture = world.live("MTLTexture")[textures_before..].to_vec();
    assert_eq!(texture.len(), 1);

    device.reset();
    for _ in 0..2000 {
        if device.decoder_object_count() == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        device.decoder_object_count(),
        0,
        "the reset left the guest's objects"
    );
    assert_eq!(device.graphics_queue_count(), 0);
    assert_eq!(world.live("MTLTexture").len(), textures_before);
    assert_eq!(world.live("MTLBuffer").len(), buffers_before);

    // The guest's driver starts again, from its first command.
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    make(&mut guest);
    let again = world.live("MTLTexture")[textures_before..].to_vec();
    assert_eq!(again.len(), 1);
    assert_ne!(again, texture, "a texture of its own");
}

/// Waits about 2 s for the decoder to hold `n` objects; whether it came to. A queue's handle is
/// made on its service thread, which may start after the queue's create is answered.
fn holds_objects(device: &GraphicsDevice<FakePlatform>, n: usize) -> bool {
    for _ in 0..2000 {
        if device.decoder_object_count() == n {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    false
}

/// Waits about 2 s for the decoder's reset, on the control thread, to have dropped every object
/// of the guest's.
fn reset_done(device: &GraphicsDevice<FakePlatform>) {
    for _ in 0..2000 {
        if device.decoder_object_count() == 0 {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("the reset left the guest's objects");
}

/// Zeroes a queue's ring headers, as the guest's bundle does before its `QUEUE_CREATE`.
fn zero_ring_headers(device: &GraphicsDevice<FakePlatform>, rings: [u64; 2]) {
    for ring in rings {
        // SAFETY: the 256 bytes of each ring header lie in the device's region, mapped while the
        // device lives; no queue of the host's takes these rings now.
        unsafe { std::ptr::write_bytes(device.region_base().add(ring as usize), 0, 256) };
    }
}

/// A command buffer of the guest before a reset whose completion the platform refuses on its own
/// thread fails no context of the rebooted guest, whether it completes before the reset or after
/// it: the kext numbers its contexts from 1 again, and the rebooted guest's context 1 makes its
/// queue
#[test]
fn a_refused_completion_across_a_reset_fails_no_context_of_the_rebooted_guest() {
    const S: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const C: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const DATA: u32 = 0x1_0000;
    for after in [false, true] {
        let raised = Arc::new(AtomicU32::new(0));
        let (device, platform) = headless_on(&raised);
        device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
        let mut guest = Guest::new(&device);
        assert_eq!(
            create_queue(&device, &mut guest, 1, 1, (S, C), DATA, 0),
            RGFX_STATUS_OK
        );
        let mut ring = QueueRing::new(&device, S, C, u64::from(DATA));
        ring.send(&device, command_buffer(obj(1, 1), 100));
        let cb = committed(&platform, 1)[0];
        let refuse = || {
            assert!(platform.world().complete_dying_read(
                cb,
                "a failure none of Metal's",
                CompleteOn::Caller
            ));
        };
        if !after {
            refuse();
        }
        // The guest reboots; the host drops what it made, then the buffer may finish.
        device.reset();
        reset_done(&device);
        if after {
            refuse();
        }

        // The rebooted guest's driver starts, and its first process (context 1) makes its queue.
        device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
        let mut guest = Guest::new(&device);
        zero_ring_headers(&device, [S, C]);
        assert_eq!(
            create_queue(&device, &mut guest, 1, 1, (S, C), DATA, 0),
            RGFX_STATUS_OK,
            "completed after the reset: {after}"
        );
    }
}

/// A command buffer of the guest before a reset counts for nothing in the rebooted guest's
/// presents, whether it completes before the reset or after it: the rebooted guest's queue takes
/// the same id, its seqnos from the start, so a present whose writer is that queue's buffer 5,
/// published and not yet run, waits for it, though the buffer before the reset was at seqno 9
#[test]
fn a_completion_across_a_reset_shows_no_present_of_the_rebooted_guest_early() {
    for after in [false, true] {
        let raised = Arc::new(AtomicU32::new(0));
        let (device, platform) = headless_on(&raised);
        let (_guest, mut ring) = surface_and_queue(&device);
        ring.send(&device, command_buffer_at(obj(1, 1), 100, 9));
        let cb = committed(&platform, 1)[0];
        let world = platform.world();
        let complete = || {
            assert!(world.complete(cb, CommandBufferStatus::COMPLETED, None, CompleteOn::Caller))
        };
        if !after {
            complete();
        }
        device.reset();
        reset_done(&device);
        if after {
            complete();
        }

        // The rebooted guest: the same process numbering, the same queue id, its own seqnos.
        zero_ring_headers(&device, [SUBMIT, COMPLETE]);
        let (mut guest, mut ring) = surface_and_queue(&device);
        ring.publish(command_buffer_at(obj(1, 1), 200, 5));
        write_writers(&device, BLOCK + WRITERS, &[(obj(1, 1), 5)]);
        guest.send(&device, present_surface(0x77));
        presents_waiting(&device, 1);
        assert!(
            quiet(&guest),
            "completed after the reset: {after}: answered"
        );
        assert_eq!(
            device.control.presents.waiting(),
            1,
            "completed after the reset: {after}: shown before its writer ran"
        );
        assert_eq!(shown_width(&device), None);
    }
}

/// A library create taken before a reset whose make finishes after it leaves nothing the
/// rebooted guest meets: no object, and a lookup of its key misses, as on a new device
#[test]
fn a_library_made_across_a_reset_is_not_in_the_new_device_s_cache() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let (bytes, key) = keyed_metallib();
    guest.send(&device, event_create(obj(1, 5)));
    device.control.pool.hold();
    guest.send(&device, keyed_library_create(obj(1, 10), &bytes, Some(key)));
    // Its context's order: once a lookup of its context's is answered, the create is on the
    // compile pool, held.
    carried_out_of(&device, &mut guest, 1);
    assert_eq!(device.decoder_object_count(), 1);

    device.reset();
    reset_done(&device);
    device.control.pool.release();
    for _ in 0..2000 {
        if library_makes(&platform) == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(library_makes(&platform), 1, "the held make never ran");
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        device.decoder_object_count(),
        0,
        "the make stored its object"
    );

    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    assert_eq!(
        answered(
            &device,
            &mut guest,
            library_lookup(obj(1, 10), bytes.len(), key)
        ),
        RGFX_STATUS_NOT_FOUND,
        "the rebooted guest's lookup hits a library made for the guest before"
    );
}

/// Indirect command buffers through the rings: the device offers them, a create on ring 0 is
/// answered with the platform's `gpuResourceID`, and a command buffer posted after an `ICB_WRITE`
/// executes the commands it wrote: the write's setters reach the platform's buffer before the
/// buffer's execute, and the buffer completes.
#[test]
fn an_icb_is_made_written_and_executed_through_the_rings() {
    icb_through_the_rings(false);
}

/// As [`an_icb_is_made_written_and_executed_through_the_rings`], the guest taking slots: the
/// device offers them, and a command buffer posted after an `ICB_SLOTS` executes the commands
/// its slots, in a block of the process's, hold.
#[test]
fn an_icb_is_written_through_slots_and_executed_through_the_rings() {
    icb_through_the_rings(true);
}

/// An indirect command buffer made, its command 1 written by the CPU (as slots where `slots`, as
/// records else), and executed by a command buffer posted after.
fn icb_through_the_rings(slots: bool) {
    use reims_vgpu_graphics::icb::{
        RGFX_ICB_OP_DISPATCH_THREADGROUPS, RGFX_ICB_OP_KERNEL_BUFFER, RGFX_ICB_SLOT_ACTION,
        RGFX_ICB_SLOT_BUFFER_0, RgfxIcbBuffer, RgfxIcbComputeSlot, RgfxIcbDispatch, RgfxIcbRecord,
        RgfxIcbSlotBuffer,
    };
    use reims_vgpu_graphics::ops::compute::{
        RGFX_OP_COMPUTE_EXECUTE_COMMANDS_IN_BUFFER, RgfxComputeOpExecuteCommandsInBuffer,
    };
    use reims_vgpu_graphics::ops::{
        RGFX_OP_ENCODER_BEGIN_COMPUTE, RGFX_OP_ENCODER_END, RgfxComputePassDesc, RgfxOpHeader,
    };
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    assert_ne!(
        device.mmio_read(RGFX_REG_FEATURES_LO.into()) & RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS,
        0,
        "the host offers indirect command buffers"
    );
    assert_ne!(
        device.mmio_read(RGFX_REG_FEATURES_LO.into()) & RGFX_FEATURE_ICB_SLOTS,
        0,
        "the host offers slots"
    );
    let taken = if slots {
        RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS | RGFX_FEATURE_ICB_SLOTS
    } else {
        RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS
    };
    device.mmio_write(RGFX_REG_DRIVER_FEATURES_LO.into(), taken);
    assert_eq!(
        device.mmio_read(RGFX_REG_DRIVER_FEATURES_LO.into()) & taken,
        taken,
        "the guest may take them"
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);

    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const RING_DATA: u32 = 0x1_0000;
    let context = 3;
    let queue = obj(context, 1);
    let mut guest = Guest::new(&device);
    assert_eq!(
        create_queue(
            &device,
            &mut guest,
            context,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            0
        ),
        RGFX_STATUS_OK
    );
    let buffer = obj(context, 2);
    guest.send(
        &device,
        RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: 0,
                seqno: 0,
            },
            obj_id: buffer,
            length: 4096,
            ..RgfxBufferCreate::default()
        }
        .as_bytes()
        .to_vec(),
    );
    let icb = obj(context, 50);
    guest.send(
        &device,
        RgfxIcbCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ICB_CREATE,
                size: 0,
                seqno: 0,
            },
            obj_id: icb,
            max_command_count: 4,
            command_types: 0x20,
            max_kernel_buffer_bind_count: 1,
            ..RgfxIcbCreate::default()
        }
        .as_bytes()
        .to_vec(),
    );
    let bytes = next_event(&mut guest, RGFX_EVT_ICB_STATE);
    let state = RgfxIcbState::read_from_bytes(&bytes).unwrap();
    assert_eq!(
        (state.header.seqno, state.status),
        (guest.seqno, RGFX_STATUS_OK)
    );
    assert_ne!(state.gpu_resource_id, 0);

    // The CPU's encodes of command 1: its kernel buffer, then its dispatch.
    let record = |op: u16, body: &[u8]| {
        let mut r = RgfxIcbRecord {
            op,
            size: (8 + body.len()) as u16,
            command_index: 1,
        }
        .as_bytes()
        .to_vec();
        r.extend_from_slice(body);
        r
    };
    let mut write = RgfxIcbWrite {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_ICB_WRITE,
            size: 0,
            seqno: 0,
        },
        obj_id: icb,
    }
    .as_bytes()
    .to_vec();
    write.extend(record(
        RGFX_ICB_OP_KERNEL_BUFFER,
        RgfxIcbBuffer {
            obj_id: buffer,
            offset: 64,
            index: 0,
            reserved_14: 0,
        }
        .as_bytes(),
    ));
    write.extend(record(
        RGFX_ICB_OP_DISPATCH_THREADGROUPS,
        RgfxIcbDispatch {
            grid: [1, 1, 1],
            threadgroup: [32, 1, 1],
        }
        .as_bytes(),
    ));
    // Command 1's slot, in a block of the process's: its kernel buffer, then its dispatch.
    const SLOTS: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    const STRIDE: u64 = 88 + 24;
    if slots {
        guest.send(&device, block_grant(context, SLOTS, 0x1000));
        let mut slot = RgfxIcbComputeSlot {
            mask: RGFX_ICB_SLOT_BUFFER_0 | RGFX_ICB_SLOT_ACTION,
            op: RGFX_ICB_OP_DISPATCH_THREADGROUPS,
            dispatch: RgfxIcbDispatch {
                grid: [1, 1, 1],
                threadgroup: [32, 1, 1],
            },
            ..RgfxIcbComputeSlot::default()
        }
        .as_bytes()
        .to_vec();
        slot.extend_from_slice(
            RgfxIcbSlotBuffer {
                obj_id: buffer,
                offset: 64,
                stride: 0,
            }
            .as_bytes(),
        );
        // SAFETY: as `Guest`'s helpers: the block lies in the region, granted to the process.
        unsafe { guest.write((SLOTS + STRIDE) as usize, &slot) };
        let mut message = RgfxIcbSlots {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ICB_SLOTS,
                size: 0,
                seqno: 0,
            },
            obj_id: icb,
            run_count: 1,
            reserved_1c: 0,
        }
        .as_bytes()
        .to_vec();
        message.extend_from_slice(
            RgfxIcbSlotRun {
                icb_obj_id: icb,
                region_offset: SLOTS,
                first_command: 0,
                command_count: 4,
                slot_bytes: STRIDE as u32,
                kind: RGFX_ICB_SLOTS_COMPUTE,
                bindings_a: 1,
                bindings_b: 0,
            }
            .as_bytes(),
        );
        guest.send(&device, message);
    } else {
        guest.send(&device, write);
    }

    // A command buffer that executes commands 0 to 3, posted after the write.
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
        RGFX_OP_COMPUTE_EXECUTE_COMMANDS_IN_BUFFER,
        RgfxComputeOpExecuteCommandsInBuffer {
            indirect_command_buffer_obj_id: icb,
            location: 0,
            length: 4,
        }
        .as_bytes(),
    ));
    ops.extend(op(RGFX_OP_ENCODER_END, &[]));
    let mut cb = RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: 0,
            seqno: 0,
        },
        queue_obj_id: queue,
        cb_obj_id: 100,
        op_bytes: ops.len() as u64,
    }
    .as_bytes()
    .to_vec();
    cb.extend(ops);
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    ring.arm();
    ring.send(&device, cb);
    assert!(
        complete_until_event(&platform, &mut ring),
        "the buffer never completed"
    );
    let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (100, RGFX_CB_OK));

    let sels = platform.call_log().sels_since(0);
    let at = |sel: &str| {
        sels.iter()
            .position(|s| *s == sel)
            .unwrap_or_else(|| panic!("no {sel} in {sels:?}"))
    };
    assert!(at("setKernelBuffer:offset:atIndex:") < at("executeCommandsInBuffer:withRange:"));
    assert!(
        at("concurrentDispatchThreadgroups:threadsPerThreadgroup:")
            < at("executeCommandsInBuffer:withRange:")
    );
    if slots {
        // SAFETY: as `Guest`'s helpers.
        let mask = unsafe { guest.load_u64((SLOTS + STRIDE) as usize) };
        assert_eq!(mask, 0, "the host took the slot");
    }
}

/// A headless device whose guest took indirect command buffers, ready.
fn icb_device(raised: &Arc<AtomicU32>) -> (GraphicsDevice<FakePlatform>, Arc<FakePlatform>) {
    let (device, platform) = headless_on(raised);
    device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_LO.into(),
        RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS,
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    (device, platform)
}

/// An `ICB_CREATE` of 4 dispatches, one kernel buffer bound, as indirect command buffer `obj_id`.
fn icb_create(obj_id: u64) -> Vec<u8> {
    RgfxIcbCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_ICB_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        max_command_count: 4,
        command_types: 0x20,
        max_kernel_buffer_bind_count: 1,
        ..RgfxIcbCreate::default()
    }
    .as_bytes()
    .to_vec()
}

/// A display command behind an `ICB_CREATE` on ring 0 is answered while the buffer is made, off
/// ring 0's threads, and the create is answered at its seqno after
#[test]
fn a_display_command_behind_a_slow_icb_create_is_answered() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = icb_device(&raised);
    let mut guest = Guest::new(&device);
    device.control.icb_worker.hold();
    guest.send(&device, icb_create(obj(1, 50)));
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

    device.control.icb_worker.release();
    let bytes = next_event(&mut guest, RGFX_EVT_ICB_STATE);
    let state = RgfxIcbState::read_from_bytes(&bytes).unwrap();
    assert_eq!((state.header.seqno, state.status), (create, RGFX_STATUS_OK));
    assert_ne!(state.gpu_resource_id, 0);
}

/// An `OBJECT_DESTROY` of an indirect command buffer still being made (the kext's destroy behind
/// a create it stopped waiting for) waits for the create, and its context's messages after it wait
/// with it: the create is answered first, then the context's lookup behind the destroy, and the
/// buffer is not kept
#[test]
fn a_destroy_of_an_icb_being_made_waits_for_its_create() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, _platform) = icb_device(&raised);
    let mut guest = Guest::new(&device);
    let before = device.decoder_object_count();
    device.control.icb_worker.hold();
    guest.send(&device, icb_create(obj(1, 50)));
    let create = guest.seqno;
    guest.send(&device, object_destroy(obj(1, 50)));
    guest.send(&device, library_lookup(obj(1, 0xfff), 0, [0xee; 32]));
    let lookup = guest.seqno;
    // The worker goes on a while later, from another thread: the context's lookup behind the
    // destroy is not answered before the create.
    let control = Arc::clone(&device.control);
    let opener = std::thread::Builder::new()
        .spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            control.icb_worker.release();
        })
        .unwrap();
    let bytes = next_event(&mut guest, RGFX_EVT_ICB_STATE);
    let state = RgfxIcbState::read_from_bytes(&bytes).unwrap();
    assert_eq!((state.header.seqno, state.status), (create, RGFX_STATUS_OK));
    let bytes = next_event(&mut guest, RGFX_EVT_COMPLETE);
    let (header, _) = RgfxMsgHeader::read_from_prefix(&bytes).unwrap();
    assert_eq!(header.seqno, lookup);
    opener.join().unwrap();
    assert_eq!(device.decoder_object_count(), before);
}

/// An id the kext's destroy of an indirect command buffer still being made names, made again by a
/// later create (a context's number is given again after its close, and its ids start again from
/// its base), names the later object: the ICB is made, then destroyed, then the buffer made under
/// its id, which is kept
#[test]
fn a_create_reusing_a_pending_icb_s_id_after_its_destroy_keeps_its_object() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = icb_device(&raised);
    let mut guest = Guest::new(&device);
    let before = device.decoder_object_count();
    let id = obj(1, 50);
    device.control.icb_worker.hold();
    guest.send(&device, icb_create(id));
    guest.send(&device, object_destroy(id));
    guest.send(
        &device,
        RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: 0,
                seqno: 0,
            },
            obj_id: id,
            length: 4096,
            ..RgfxBufferCreate::default()
        }
        .as_bytes()
        .to_vec(),
    );
    guest.send(&device, library_lookup(obj(1, 0xfff), 0, [0xee; 32]));
    let control = Arc::clone(&device.control);
    let opener = std::thread::Builder::new()
        .spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            control.icb_worker.release();
        })
        .unwrap();
    next_event(&mut guest, RGFX_EVT_ICB_STATE);
    next_event(&mut guest, RGFX_EVT_COMPLETE);
    opener.join().unwrap();
    let calls = platform.call_log().calls();
    let icb = calls
        .iter()
        .position(|c| c.sel == "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:")
        .expect("the ICB's make");
    let buffer = calls
        .iter()
        .rposition(|c| c.sel.starts_with("newBuffer"))
        .expect("the buffer's make");
    assert!(
        icb < buffer,
        "the buffer was made before the ICB its id named first"
    );
    assert_eq!(
        device.decoder_object_count(),
        before + 1,
        "the destroy removed the later buffer"
    );
}

/// A headless device whose memory map holds `pages` host pages of guest RAM from [`SURFACE_RAM`],
/// which its `SURFACE_PAGES` views map, started; the RAM, and the host's page.
fn surface_device(
    pages: u64,
) -> (
    GraphicsDevice<FakePlatform>,
    Arc<FakePlatform>,
    reims_vgpu_vulkan::core::testkit::GuestRam,
    u64,
) {
    let platform = Arc::new(FakePlatform::new());
    let page = platform.vm().page_size();
    let ram =
        reims_vgpu_vulkan::core::testkit::GuestRam::with_range(SURFACE_RAM, pages * page).unwrap();
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
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    (device, platform, ram, page)
}

/// The guest RAM's first byte in [`surface_device`].
const SURFACE_RAM: u64 = 0x10_0000;
/// The surface's block in the surface tests, of [`SURFACE_BLOCK_SIZE`] bytes, its runs in its
/// second half.
const SURFACE_BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
const SURFACE_BLOCK_SIZE: u64 = 0x2000;

/// A `SURFACE_PAGES` of the block at [`SURFACE_BLOCK`] for context 1's surface of `runs`, given
/// as (page, pages) of guest RAM, written into the block's second half here.
fn surface_pages(device: &GraphicsDevice<FakePlatform>, page: u64, runs: &[(u64, u64)]) -> Vec<u8> {
    let runs: Vec<RgfxPageRun> = runs
        .iter()
        .map(|&(at, n)| RgfxPageRun {
            address: SURFACE_RAM + at * page,
            length: n * page,
        })
        .collect();
    let runs_offset = SURFACE_BLOCK + SURFACE_BLOCK_SIZE / 2;
    // SAFETY: the runs lie in the block, in the device's region, mapped while it lives; nothing of
    // the host's writes them.
    unsafe {
        std::ptr::copy_nonoverlapping(
            runs.as_bytes().as_ptr(),
            device.region_base().add(runs_offset as usize),
            runs.as_bytes().len(),
        );
    }
    RgfxSurfacePages {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SURFACE_PAGES,
            size: 0,
            seqno: 0,
        },
        context: 1,
        run_count: runs.len() as u32,
        block_offset: SURFACE_BLOCK,
        block_size: SURFACE_BLOCK_SIZE,
        runs_offset,
        length: runs.iter().map(|r| r.length).sum(),
    }
    .as_bytes()
    .to_vec()
}

/// A `BUFFER_OVER_SURFACE_PAGES` of `length` bytes over the block at [`SURFACE_BLOCK`].
fn buffer_over_surface_pages(obj_id: u64, length: u64) -> Vec<u8> {
    RgfxBufferOverSurfacePages {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_BUFFER_OVER_SURFACE_PAGES,
            size: 0,
            seqno: 0,
        },
        obj_id,
        block_offset: SURFACE_BLOCK,
        block_size: SURFACE_BLOCK_SIZE,
        length,
        options: 0,
        reserved_34: 0,
    }
    .as_bytes()
    .to_vec()
}

/// Waits, about 2 s at most, for the decoder to hold `n` objects; whether it did.
fn objects_become(device: &GraphicsDevice<FakePlatform>, n: usize) -> bool {
    for _ in 0..2000 {
        if device.decoder_object_count() == n {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    false
}

/// The `COMPLETE` `bytes` is: its seqno and status.
fn complete_of(bytes: &[u8]) -> (u64, u32) {
    let complete = RgfxComplete::read_from_bytes(bytes).unwrap();
    assert_eq!(complete.header.r#type, RGFX_EVT_COMPLETE);
    (complete.header.seqno, complete.status)
}

/// A `SURFACE_PAGES`' view is made off ring 0's threads: while it is being made, ring 0 goes
/// on with a message that does not name the block, and holds the messages that do, a
/// `BUFFER_OVER_SURFACE_PAGES` and an `OBJECT_DESTROY` of its buffer posted right after it, and
/// the block's `BLOCK_RELEASE`. Each is then carried out as right after the `SURFACE_PAGES`,
/// whose answer comes first: the buffer lies over the view, the release is answered after it,
/// and the decoder holds what it would have
#[test]
fn a_surface_s_view_is_made_off_ring_0_and_the_messages_naming_it_wait_for_it() {
    let (device, platform, _ram, page) = surface_device(8);
    let mut guest = Guest::new(&device);
    guest.send(&device, block_grant(1, SURFACE_BLOCK, SURFACE_BLOCK_SIZE));
    let worker = device.control.decoder();
    worker.surface_worker().hold();
    let mark = platform.call_log().mark();

    guest.send(&device, surface_pages(&device, page, &[(5, 2), (1, 1)]));
    let pages_seqno = guest.seqno;
    guest.send(&device, event_create(obj(1, 10)));
    assert!(objects_become(&device, 1), "ring 0 went on past it");
    guest.send(&device, buffer_over_surface_pages(obj(1, 11), 3 * page));
    guest.send(&device, object_destroy(obj(1, 11)));
    guest.send(&device, block_release(1, SURFACE_BLOCK, SURFACE_BLOCK_SIZE));
    let release_seqno = guest.seqno;
    guest.send(&device, event_create(obj(1, 12)));
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        device.decoder_object_count(),
        1,
        "the messages naming the block wait for its view, and their context's behind them"
    );
    let buffers = |sels: &[&str]| {
        sels.iter()
            .filter(|&&s| s == "newBufferWithDescriptor:")
            .count()
    };
    assert_eq!(
        buffers(&platform.call_log().sels_since(mark)),
        0,
        "no view made yet"
    );

    worker.surface_worker().release();
    let first = complete_of(&guest.event().expect("the SURFACE_PAGES' answer"));
    assert_eq!(first, (pages_seqno, RGFX_STATUS_OK), "answered first");
    let second = complete_of(&guest.event().expect("the BLOCK_RELEASE's answer"));
    assert_eq!(second, (release_seqno, RGFX_STATUS_OK));
    assert!(objects_become(&device, 2), "both events, the buffer gone");
    worker.surface_worker().flush();
    let made = platform.call_log().sels_since(mark);
    assert_eq!(
        buffers(&made),
        2,
        "the view's import and the buffer over it: {made:?}"
    );
    assert!(guest.event_now().is_none(), "nothing else answered");
}

/// A `SURFACE_PAGES` the device's reset comes after, before the worker carries it out, makes no
/// view and is not answered; the reset does not wait for it
#[test]
fn a_reset_drops_a_surface_pages_not_yet_carried_out() {
    let (device, platform, _ram, page) = surface_device(8);
    let mut guest = Guest::new(&device);
    let worker = device.control.decoder();
    worker.surface_worker().hold();
    let mark = platform.call_log().mark();
    guest.send(&device, surface_pages(&device, page, &[(1, 1)]));
    std::thread::sleep(Duration::from_millis(20));
    device.reset();
    let mut reset = false;
    for _ in 0..2000 {
        if worker.views_of_surfaces().resets() == 1 {
            reset = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(reset, "the reset waited for the worker");
    worker.surface_worker().release();
    worker.surface_worker().flush();
    assert!(
        platform.call_log().sels_since(mark).is_empty(),
        "no view made"
    );
    assert_eq!(worker.views_of_surfaces().len(), (0, 0));
    assert!(guest.event_now().is_none(), "not answered");
}

/// A `BUFFER_OVER_PAGES` of `length` bytes over the pages block at [`SURFACE_BLOCK`], at no GPU
/// address of the guest's.
fn buffer_over_pages(obj_id: u64, length: u64) -> Vec<u8> {
    RgfxBufferOverPages {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_BUFFER_OVER_PAGES,
            size: 0,
            seqno: 0,
        },
        obj_id,
        gpu_address: 0,
        block_offset: SURFACE_BLOCK,
        block_size: SURFACE_BLOCK_SIZE,
        offset: 0,
        length,
        options: 0,
        reserved_44: 0,
    }
    .as_bytes()
    .to_vec()
}

/// A pages block's view, which its first `BUFFER_OVER_PAGES` makes, is made off ring 0's threads:
/// while it is being made, ring 0 goes on with another context's messages, before and after the
/// buffer's context's next messages, an object's create and the buffer's destroy, and the block's
/// `BLOCK_RELEASE`, which wait for it. Each is then carried out as right after the buffer: the view
/// and the buffer are made first, then the create, the destroy and the release, which is answered
#[test]
fn a_pages_block_s_view_is_made_off_ring_0_and_its_context_waits_for_it() {
    let (device, platform, _ram, page) = surface_device(8);
    let mut guest = Guest::new(&device);
    guest.send(&device, block_grant(1, SURFACE_BLOCK, SURFACE_BLOCK_SIZE));
    let mut pages = surface_pages(&device, page, &[(5, 2), (1, 1)]);
    pages.extend_from_slice(
        RgfxSurfacePagesCache {
            flags: RGFX_SURFACE_PAGES_PRIVATE,
            reserved_04: 0,
        }
        .as_bytes(),
    );
    guest.send(&device, pages);
    let pages_seqno = guest.seqno;
    let answer = complete_of(&guest.event().expect("the SURFACE_PAGES' answer"));
    assert_eq!(answer, (pages_seqno, RGFX_STATUS_OK));
    let worker = device.control.decoder();
    worker.surface_worker().flush();
    worker.surface_worker().hold();
    let mark = platform.call_log().mark();

    guest.send(&device, buffer_over_pages(obj(1, 11), 3 * page));
    guest.send(&device, event_create(obj(2, 10)));
    assert!(objects_become(&device, 1), "ring 0 went on past it");
    guest.send(&device, event_create(obj(1, 12)));
    guest.send(&device, object_destroy(obj(1, 11)));
    guest.send(&device, block_release(1, SURFACE_BLOCK, SURFACE_BLOCK_SIZE));
    let release_seqno = guest.seqno;
    guest.send(&device, event_create(obj(2, 13)));
    assert!(
        objects_become(&device, 2),
        "ring 0 went on with the other context"
    );
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        device.decoder_object_count(),
        2,
        "the buffer's context waits for its view, and no other context"
    );
    let buffers = |sels: &[&str]| {
        sels.iter()
            .filter(|&&s| s == "newBufferWithDescriptor:")
            .count()
    };
    assert_eq!(
        buffers(&platform.call_log().sels_since(mark)),
        0,
        "no view made yet"
    );
    assert!(guest.event_now().is_none(), "nothing answered yet");

    worker.surface_worker().release();
    let release = complete_of(&guest.event().expect("the BLOCK_RELEASE's answer"));
    assert_eq!(release, (release_seqno, RGFX_STATUS_OK));
    assert!(
        objects_become(&device, 3),
        "the three events, the buffer gone"
    );
    worker.surface_worker().flush();
    let made = platform.call_log().sels_since(mark);
    assert_eq!(
        buffers(&made),
        2,
        "the view's import and the buffer over it: {made:?}"
    );
    assert_eq!(worker.views_of_surfaces().counts().private, 1);
    assert!(guest.event_now().is_none(), "nothing else answered");
}

/// A heap over a pages block of several runs, a slab of plain pages, makes the block's view off
/// ring 0's threads as a no-copy buffer does: while it is being made, ring 0 goes on with another
/// context's messages, and the heap's context's next message waits for it; then the view and the
/// heap over it are made, and the message after it. The view is 4 pages, so the heap of 3 rounded
/// up to Metal's 16 KiB page fits it on a host of 4 KiB pages
#[test]
fn a_heap_over_a_pages_block_s_view_is_made_off_ring_0() {
    let (device, platform, _ram, page) = surface_device(8);
    let mut guest = Guest::new(&device);
    guest.send(&device, block_grant(1, SURFACE_BLOCK, SURFACE_BLOCK_SIZE));
    let mut pages = surface_pages(&device, page, &[(5, 2), (1, 2)]);
    pages.extend_from_slice(
        RgfxSurfacePagesCache {
            flags: RGFX_SURFACE_PAGES_PRIVATE,
            reserved_04: 0,
        }
        .as_bytes(),
    );
    guest.send(&device, pages);
    let pages_seqno = guest.seqno;
    let answer = complete_of(&guest.event().expect("the SURFACE_PAGES' answer"));
    assert_eq!(answer, (pages_seqno, RGFX_STATUS_OK));
    let worker = device.control.decoder();
    worker.surface_worker().flush();
    worker.surface_worker().hold();
    let mark = platform.call_log().mark();

    let heap = RgfxHeapOverPages {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_HEAP_OVER_PAGES,
            size: 0,
            seqno: 0,
        },
        obj_id: obj(1, 11),
        gpu_address: 0x2_0000_0000,
        block_offset: SURFACE_BLOCK,
        block_size: SURFACE_BLOCK_SIZE,
        size_bytes: 3 * page,
        options: 0,
        heap_type: 1,
    };
    guest.send(&device, heap.as_bytes().to_vec());
    guest.send(&device, event_create(obj(2, 10)));
    assert!(objects_become(&device, 1), "ring 0 went on past it");
    guest.send(&device, event_create(obj(1, 12)));
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        device.decoder_object_count(),
        1,
        "the heap's context waits for its view"
    );
    let sels = platform.call_log().sels_since(mark);
    assert!(
        !sels.contains(&"newHeapWithDescriptor:"),
        "no heap made yet: {sels:?}"
    );

    worker.surface_worker().release();
    assert!(
        objects_become(&device, 3),
        "the heap, then the event after it"
    );
    worker.surface_worker().flush();
    let made = platform.call_log().sels_since(mark);
    assert!(
        made.contains(&"newHeapWithDescriptor:"),
        "the heap over the view: {made:?}"
    );
    assert_eq!(worker.views_of_surfaces().counts().private, 1);
}

/// A command buffer of a context whose no-copy buffer is being made on the surface worker is not
/// answered at its doorbell, though it asks for its `CB_SUBMITTED` and need wait for nothing
/// else: the doorbell write returns with nothing on its queue's ring, and the queue's thread
/// waits for the buffer's make. Once the worker has made it, the queue's thread begins the
/// command buffer and posts its `CB_SUBMITTED`, and it completes
#[test]
fn a_buffer_of_a_context_making_a_pages_buffer_is_not_answered_at_its_doorbell() {
    let (device, platform, _ram, page) = surface_device(8);
    let mut guest = Guest::new(&device);
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    let (context, queue) = (1, obj(1, 1));
    assert_eq!(
        create_queue(
            &device,
            &mut guest,
            context,
            1,
            (SUBMIT_OFF, COMPLETE_OFF),
            RING_DATA,
            1
        ),
        RGFX_STATUS_OK
    );
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
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
    guest.send(&device, block_grant(1, SURFACE_BLOCK, SURFACE_BLOCK_SIZE));
    let mut pages = surface_pages(&device, page, &[(5, 2), (1, 1)]);
    pages.extend_from_slice(
        RgfxSurfacePagesCache {
            flags: RGFX_SURFACE_PAGES_PRIVATE,
            reserved_04: 0,
        }
        .as_bytes(),
    );
    guest.send(&device, pages);
    let pages_seqno = guest.seqno;
    let answer = complete_of(&guest.event().expect("the SURFACE_PAGES' answer"));
    assert_eq!(answer, (pages_seqno, RGFX_STATUS_OK));
    let worker = device.control.decoder();
    worker.surface_worker().flush();
    worker.surface_worker().hold();
    let mark = platform.call_log().mark();

    guest.send(&device, buffer_over_pages(obj(1, 11), 3 * page));
    let mut handed = false;
    for _ in 0..2000 {
        if device
            .control
            .pages_buffers
            .any(std::iter::once(pages_context(context)))
        {
            handed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(handed, "ring 0 handed the buffer to the worker");
    ring.publish(asking(queue, 100));
    device.mmio_write_taking(RGFX_REG_DOORBELL.into(), SUBMIT_OFF as u32);
    assert!(!ring.has_event(), "not answered at the doorbell");
    std::thread::sleep(Duration::from_millis(50));
    assert!(!ring.has_event(), "the queue's thread waits for the make");
    let buffers = |sels: &[&str]| {
        sels.iter()
            .filter(|&&s| s == "newBufferWithDescriptor:")
            .count()
    };
    assert_eq!(buffers(&platform.call_log().sels_since(mark)), 0);

    worker.surface_worker().release();
    let mut submitted = None;
    for _ in 0..2000 {
        if ring.has_event() {
            submitted = ring.event();
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let posted = RgfxCbSubmitted::read_from_bytes(&submitted.expect("a CB_SUBMITTED")).unwrap();
    assert_eq!(
        (posted.header.r#type, posted.cb_obj_id),
        (RGFX_EVT_CB_SUBMITTED, 100)
    );
    let made = platform.call_log().sels_since(mark);
    assert_eq!(
        buffers(&made),
        2,
        "the view's import and the buffer over it, before the command buffer: {made:?}"
    );
    let mut done = None;
    for _ in 0..3000 {
        platform.world().complete_all(CompleteOn::Caller);
        if ring.has_event() {
            done = ring.event();
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let done = RgfxCbComplete::read_from_bytes(&done.expect("a CB_COMPLETE")).unwrap();
    assert_eq!((done.cb_obj_id, done.status), (100, RGFX_CB_OK));
}

/// A `SURFACE_PAGES` of context `context`'s block at `block` (of [`SURFACE_BLOCK_SIZE`] bytes) of
/// `runs`, given as (page, pages) of guest RAM, written into the block's second half here.
fn surface_pages_of(
    device: &GraphicsDevice<FakePlatform>,
    context: u32,
    block: u64,
    page: u64,
    runs: &[(u64, u64)],
) -> Vec<u8> {
    let runs: Vec<RgfxPageRun> = runs
        .iter()
        .map(|&(at, n)| RgfxPageRun {
            address: SURFACE_RAM + at * page,
            length: n * page,
        })
        .collect();
    let runs_offset = block + SURFACE_BLOCK_SIZE / 2;
    // SAFETY: the runs lie in the block, in the device's region, mapped while it lives; nothing of
    // the host's writes them.
    unsafe {
        std::ptr::copy_nonoverlapping(
            runs.as_bytes().as_ptr(),
            device.region_base().add(runs_offset as usize),
            runs.as_bytes().len(),
        );
    }
    RgfxSurfacePages {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SURFACE_PAGES,
            size: 0,
            seqno: 0,
        },
        context,
        run_count: runs.len() as u32,
        block_offset: block,
        block_size: SURFACE_BLOCK_SIZE,
        runs_offset,
        length: runs.iter().map(|r| r.length).sum(),
    }
    .as_bytes()
    .to_vec()
}

/// A close lets go of its pages blocks' views with no lock held. While the platform takes long
/// to let go of a view's import of the guest's pages (held here, as a driver's unpin of them), a
/// `SURFACE_PAGES` of another context's block is answered; the close is answered once the import
/// is let go of.
#[test]
fn a_close_s_slow_release_of_a_pages_view_holds_up_no_other_context_s_pages() {
    let (device, platform, _ram, page) = surface_device(8);
    let mut guest = Guest::new(&device);
    guest.send(&device, block_grant(1, SURFACE_BLOCK, SURFACE_BLOCK_SIZE));
    let mut pages = surface_pages(&device, page, &[(5, 2), (1, 1)]);
    pages.extend_from_slice(
        RgfxSurfacePagesCache {
            flags: RGFX_SURFACE_PAGES_PRIVATE,
            reserved_04: 0,
        }
        .as_bytes(),
    );
    guest.send(&device, pages);
    let pages_seqno = guest.seqno;
    let answer = complete_of(&guest.event().expect("the SURFACE_PAGES' answer"));
    assert_eq!(answer, (pages_seqno, RGFX_STATUS_OK));
    guest.send(&device, buffer_over_pages(obj(1, 11), 3 * page));
    guest.send(&device, object_destroy(obj(1, 11)));
    guest.send(&device, event_create(obj(1, 12)));
    assert!(objects_become(&device, 1), "the buffer made and gone");
    let worker = device.control.decoder();
    worker.surface_worker().flush();
    assert_eq!(worker.views_of_surfaces().counts().private, 1);
    let world = platform.world();
    // The buffer over the view goes once the walks and workers that saw it let it go.
    let mut imports = world.live("MTLBuffer");
    for _ in 0..2000 {
        if imports.len() == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
        imports = world.live("MTLBuffer");
    }
    assert_eq!(imports.len(), 1, "the view's import alone: {imports:?}");
    let release = HeldCall::new();
    world.hold_at_release(imports[0], release.clone());
    // The other context's first message, before the close: a context's first message waits for
    // every close not yet carried out.
    const OTHER: u64 = SURFACE_BLOCK + 0x10_0000;
    guest.send(&device, block_grant(2, OTHER, SURFACE_BLOCK_SIZE));

    guest.send(&device, context_close(1));
    let close = guest.seqno;
    assert!(
        release.reached(Duration::from_secs(2)),
        "the close let go of the view's import"
    );
    guest.send(
        &device,
        surface_pages_of(&device, 2, OTHER, page, &[(3, 1)]),
    );
    let other = guest.seqno;
    let answer = complete_of(
        &guest
            .event()
            .expect("the other context's SURFACE_PAGES' answer"),
    );
    assert_eq!(answer, (other, RGFX_STATUS_OK));
    assert!(guest.event_now().is_none(), "the close is not answered yet");

    release.open();
    let answer = complete_of(&guest.event().expect("the close's answer"));
    assert_eq!(answer, (close, RGFX_STATUS_OK));
    assert!(
        !world.live("MTLBuffer").contains(&imports[0]),
        "the close let go of the view's import"
    );
}

/// A shared block's `BLOCK_RELEASE` posted behind an earlier holder's `CONTEXT_CLOSE` waits only
/// for the close to take the holder's objects out of the tables: while the platform takes long to
/// let go of one (held here), the release is answered, and so is a later message of the releasing
/// context's, which its lane carries out behind the release. The close is answered once the
/// object is let go of, as the kext gives the holder's texture resource ids and GPU addresses out
/// again at the answer. A compositor that releases a surface's block an app shared with it, right
/// after the app quits, does not wait out the app's releases.
#[test]
fn a_shared_block_s_release_behind_a_holder_s_close_waits_for_none_of_its_releases() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SIZE: u64 = 0x1_0000;
    guest.send(&device, block_share(1, BLOCK, SIZE, true));
    guest.send(&device, block_share(2, BLOCK, SIZE, true));
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(1, 4)),
        RGFX_STATUS_OK
    );
    let world = platform.world();
    let samplers = world.live("MTLSamplerState");
    assert_eq!(samplers.len(), 1, "{samplers:?}");
    let release = HeldCall::new();
    world.hold_at_release(samplers[0], release.clone());

    guest.send(&device, context_close(1));
    let close = guest.seqno;
    assert!(
        release.reached(Duration::from_secs(2)),
        "the close let go of the sampler"
    );
    guest.send(&device, block_release(2, BLOCK, SIZE));
    let block = guest.seqno;
    guest.send(&device, sampler_create(obj(2, 4)));
    let later = guest.seqno;
    let answered: Vec<(u32, u64, u32)> = events(&mut guest, 2)
        .into_iter()
        .map(|(r#type, bytes)| match r#type {
            RGFX_EVT_COMPLETE => {
                let (seqno, status) = complete_of(&bytes);
                (r#type, seqno, status)
            }
            _ => {
                let state = RgfxSamplerState::read_from_bytes(&bytes).unwrap();
                (r#type, state.header.seqno, state.status)
            }
        })
        .collect();
    assert_eq!(
        answered,
        [
            (RGFX_EVT_COMPLETE, block, RGFX_STATUS_OK),
            (RGFX_EVT_SAMPLER_STATE, later, RGFX_STATUS_OK),
        ]
    );
    assert!(quiet(&guest), "the close answered before its release");

    release.open();
    let answer = complete_of(&guest.event().expect("the close's answer"));
    assert_eq!(answer, (close, RGFX_STATUS_OK));
}

/// A reset waits for the closes before it to let go of their objects, as the guest's driver,
/// starting again, may take the texture resource ids and GPU addresses they held at once: while
/// the platform takes long to let go of a closed context's object (held here), the reset drops
/// no other context's objects, and it does once the object is let go of.
#[test]
fn a_reset_waits_for_a_close_s_slow_release() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(1, 4)),
        RGFX_STATUS_OK
    );
    let world = platform.world();
    let samplers = world.live("MTLSamplerState");
    assert_eq!(samplers.len(), 1, "{samplers:?}");
    let release = HeldCall::new();
    world.hold_at_release(samplers[0], release.clone());
    assert_eq!(
        sampler_answered(&device, &mut guest, obj(2, 4)),
        RGFX_STATUS_OK
    );
    guest.send(&device, context_close(1));
    assert!(
        release.reached(Duration::from_secs(2)),
        "the close let go of the sampler"
    );
    assert_eq!(device.decoder_object_count(), 1, "context 2's sampler");

    device.reset();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        device.decoder_object_count(),
        1,
        "the reset went on before the close let go of its sampler"
    );
    release.open();
    for _ in 0..2000 {
        if device.decoder_object_count() == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        device.decoder_object_count(),
        0,
        "the reset left context 2's sampler"
    );
    assert!(world.live("MTLSamplerState").is_empty());
}

/// A colour matrix of `rows` for display 0's transaction 1, in one `DISPLAY_COLOR` part.
fn color_matrix(rows: [[f32; 4]; 3]) -> Vec<u8> {
    let mut bytes = RgfxDisplayColor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_DISPLAY_COLOR,
            size: 40 + 16 * 3,
            seqno: 0,
        },
        display: 0,
        transaction_id: 1,
        stage: RGFX_COLOR_MATRIX,
        entry_count: 3,
        first_entry: 0,
        part_entries: 3,
    }
    .as_bytes()
    .to_vec();
    for row in rows {
        let [x, y, z, w] = row.map(f32::to_bits);
        bytes.extend_from_slice(RgfxColorEntry { x, y, z, w }.as_bytes());
    }
    bytes
}

/// On the Vulkan platform, a display colour of the identity matrix reads as no colour: the
/// scanout's bytes as the guest wrote them, its alpha included (a colour pass would make it
/// opaque), with each of blue, green and red taking all 256 values; a matrix off the identity
/// still draws through the colour, which swaps red and blue here and makes every pixel opaque
#[test]
fn on_vulkan_an_identity_colour_matrix_reads_as_no_colour() {
    let platform = crate::singleton::platform();
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: None,
            ..GraphicsCallbacks::default()
        },
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(platform, config).unwrap();
    let (width, height, stride) = (256usize, 4usize, 1024usize);
    let pixel = |x: usize, y: usize| -> [u8; 4] {
        let x = x as u8;
        let y = y as u8;
        [x, x.wrapping_mul(37).wrapping_add(y), 255 - x, x ^ 0x5a ^ y]
    };
    // SAFETY: the scanout lies in the device's region, mapped while the device lives.
    unsafe {
        let base = device.region_base().add(RGFX_ALLOCATIONS_OFFSET as usize);
        for y in 0..height {
            for x in 0..width {
                let p = pixel(x, y);
                std::ptr::copy_nonoverlapping(p.as_ptr(), base.add(y * stride + x * 4), 4);
            }
        }
    }
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    assert_eq!(
        answered(&device, &mut guest, set_scanout(256, 4, 1024)),
        RGFX_STATUS_OK
    );
    let keep = || {
        let mut present = RgfxPresentSurface::read_from_bytes(&present_surface(0)).unwrap();
        present.flags = RGFX_PRESENT_KEEP;
        present.as_bytes().to_vec()
    };
    let read = || {
        let mut dst = vec![7u8; 1920 * 1080 * 4];
        assert_eq!(device.read_display(&mut dst), ((1920, 1080), Ok(())));
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| <[u8; 4]>::try_from(&dst[(y * 1920 + x) * 4..][..4]).unwrap())
            .collect::<Vec<_>>()
    };
    let guest_pixels: Vec<[u8; 4]> = (0..height)
        .flat_map(|y| (0..width).map(move |x| pixel(x, y)))
        .collect();
    assert_eq!(read(), guest_pixels, "with no colour");

    guest.send(
        &device,
        color_matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]),
    );
    assert_eq!(answered(&device, &mut guest, keep()), RGFX_STATUS_OK);
    assert!(device.control.presented.color().1.is_identity());
    assert_eq!(read(), guest_pixels, "with the identity matrix");

    guest.send(
        &device,
        color_matrix([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]),
    );
    assert_eq!(answered(&device, &mut guest, keep()), RGFX_STATUS_OK);
    let swapped: Vec<[u8; 4]> = guest_pixels
        .iter()
        .map(|&[b, g, r, _]| [r, g, b, 0xff])
        .collect();
    assert_eq!(read(), swapped, "with red and blue swapped");
}

/// `copyFromTexture:...toTexture:...`, a presented surface's copy into a drawable.
const SURFACE_COPY: &str = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:\
                            toTexture:destinationSlice:destinationLevel:destinationOrigin:";

/// A `SET_SCANOUT` of display 0: `width` × `height` pixels, `stride` bytes a row, at the region's
/// allocations.
fn set_scanout(width: u32, height: u32, stride: u32) -> Vec<u8> {
    RgfxSetScanout {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_SCANOUT,
            size: 0,
            seqno: 0,
        },
        display: 0,
        width,
        height,
        stride,
        offset: RGFX_ALLOCATIONS_OFFSET,
    }
    .as_bytes()
    .to_vec()
}

/// The host's read of the display (`read_display`) has the display's mode, 1920 × 1080 without
/// a window: a buffer short of it is refused with nothing written, and the size is given all the
/// same. With no scanout the display shows black, and the read draws nothing. With one, the read
/// blits it into its texture of the mode's size, kept from the read before, as the window draws
/// it, waits for that draw and reads the texture back (the testkit's pixels read as zero). A
/// presented surface is copied into the read's texture in place of the scanout. The guest is told
/// nothing of any read
#[test]
fn a_read_of_the_display_draws_what_it_shows() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    platform.world().set_auto_complete(Some(CompleteOn::Thread));
    let log = platform.call_log();
    let mode = (1920, 1080);
    let length = 1920 * 1080 * 4;

    let mut short = vec![7u8; length - 1];
    assert_eq!(
        device.read_display(&mut short),
        (mode, Err(Error::InvalidArgument))
    );
    assert!(short.iter().all(|b| *b == 7), "a refused read wrote");
    assert_eq!(
        device.read_display(&mut []),
        (mode, Err(Error::InvalidArgument))
    );

    let mut dst = vec![7u8; length + 4];
    let mark = log.mark();
    assert_eq!(device.read_display(&mut dst), (mode, Ok(())));
    assert!(
        dst[..length].chunks(4).all(|p| p == [0, 0, 0, 0xff]),
        "not black"
    );
    assert_eq!(
        dst[length..],
        [7; 4],
        "the read wrote past the mode's pixels"
    );
    let sels = log.sels_since(mark);
    assert!(!sels.contains(&"commit"), "a read of nothing drew");
    assert_eq!(
        sels.iter()
            .filter(|s| **s == "newTextureWithDescriptor:")
            .count(),
        1,
        "the read's texture"
    );

    let (mut guest, _ring) = surface_and_queue(&device);
    assert_eq!(
        answered(&device, &mut guest, set_scanout(64, 32, 256)),
        RGFX_STATUS_OK
    );
    let mark = log.mark();
    dst.fill(7);
    assert_eq!(device.read_display(&mut dst), (mode, Ok(())));
    assert!(
        dst[..length].iter().all(|b| *b == 0),
        "the texture was not read"
    );
    let sels = log.sels_since(mark);
    let count = |sel: &str| sels.iter().filter(|s| **s == sel).count();
    assert_eq!(
        count("newTextureWithDescriptor:"),
        0,
        "a read of the same size made a texture"
    );
    assert_eq!(count(SCANOUT_BLIT), 1);
    assert_eq!(count("commit"), 1);
    assert_eq!(count("waitUntilCompleted"), 1);
    assert!(quiet(&guest), "a read told the guest something");

    assert_eq!(
        answered(&device, &mut guest, present_surface(0x77)),
        RGFX_STATUS_OK
    );
    let surface = Handle::clone_ref(&device.control.presented.shown().unwrap().0);
    let mark = log.mark();
    assert_eq!(device.read_display(&mut dst), (mode, Ok(())));
    let copies: Vec<_> = (log.since(mark).into_iter())
        .filter(|c| c.sel == SURFACE_COPY)
        .collect();
    assert_eq!(copies.len(), 1);
    assert!(copies[0].args.contains(&Arg::obj(&surface)));
    assert!(!log.sels_since(mark).contains(&SCANOUT_BLIT));
    assert!(quiet(&guest), "a read told the guest something");
}

/// On the Vulkan platform, the read gives the guest's pixels: a scanout smaller than the mode
/// shows at its top left, B, G, R and A as the guest wrote them, and the rest of the mode is black
#[test]
fn on_vulkan_a_read_of_the_display_gives_the_scanout_s_pixels() {
    let platform = crate::singleton::platform();
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: None,
            ..GraphicsCallbacks::default()
        },
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(platform, config).unwrap();
    let (width, height, stride) = (64usize, 32usize, 256usize);
    let pixel = |x: usize, y: usize| [x as u8, y as u8, (x ^ y) as u8, 0xff - x as u8];
    // SAFETY: the scanout lies in the device's region, mapped while the device lives.
    unsafe {
        let base = device.region_base().add(RGFX_ALLOCATIONS_OFFSET as usize);
        for y in 0..height {
            for x in 0..width {
                let p = pixel(x, y);
                std::ptr::copy_nonoverlapping(p.as_ptr(), base.add(y * stride + x * 4), 4);
            }
        }
    }
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    assert_eq!(
        answered(&device, &mut guest, set_scanout(64, 32, 256)),
        RGFX_STATUS_OK
    );
    let mut dst = vec![7u8; 1920 * 1080 * 4];
    assert_eq!(device.read_display(&mut dst), ((1920, 1080), Ok(())));
    let at = |x: usize, y: usize| &dst[(y * 1920 + x) * 4..][..4];
    for y in 0..height {
        for x in 0..width {
            assert_eq!(at(x, y), pixel(x, y), "({x}, {y})");
        }
    }
    for (x, y) in [(64, 0), (0, 32), (1919, 1079), (100, 100)] {
        assert_eq!(at(x, y), [0, 0, 0, 0xff], "({x}, {y}) is not black");
    }
}

/// The pixel the odd-width reads put at `(x, y)`, B, G, R and A: no two alike within 7 rows
fn odd_width_pixel(x: usize, y: usize) -> [u8; 4] {
    let v = x * 7 + y * 131;
    [v as u8, (v >> 8) as u8, (x ^ y) as u8, 0xff]
}

/// How a read of `width` × `height` pixels, rows of `width * 4` bytes, differs from `want`: the
/// count and the first, or `None` where every pixel is `want`'s
fn read_differs(
    dst: &[u8],
    (width, height): (usize, usize),
    want: impl Fn(usize, usize) -> [u8; 4],
) -> Option<String> {
    let got = |x: usize, y: usize| <[u8; 4]>::try_from(&dst[(y * width + x) * 4..][..4]).unwrap();
    let differ: Vec<(usize, usize)> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .filter(|&(x, y)| got(x, y) != want(x, y))
        .collect();
    let &(x, y) = differ.first()?;
    Some(format!(
        "{} pixels differ, the first at ({x}, {y}): {:?} for {:?}",
        differ.len(),
        got(x, y),
        want(x, y)
    ))
}

/// On the Vulkan platform, a read at a mode of the scanout's size, as a window that size reads,
/// gives the guest's pixels row for row at widths that are not a multiple of 4, with rows of the
/// width's bytes or padded to 64 or 256 bytes: through no colour (the blit), a matrix that swaps
/// red and blue (the colour's fragment), and a post-gamma ramp that inverts every channel (the
/// table's fragment)
#[test]
fn on_vulkan_a_read_at_an_odd_width_gives_the_scanout_row_for_row() {
    let platform = crate::singleton::platform();
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks::default(),
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(platform, config).unwrap();
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let keep = || {
        let mut present = RgfxPresentSurface::read_from_bytes(&present_surface(0)).unwrap();
        present.flags = RGFX_PRESENT_KEEP;
        present.as_bytes().to_vec()
    };
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let swap = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ];
    // Each colour's name, its messages, and what it makes of a pixel.
    type Colour = (&'static str, Vec<Vec<u8>>, fn([u8; 4]) -> [u8; 4]);
    let colours: [Colour; 3] = [
        ("no colour", vec![], |p| p),
        (
            "red and blue swapped",
            vec![color_matrix(swap)],
            |[b, g, r, _]| [r, g, b, 0xff],
        ),
        (
            "inverted by a ramp",
            vec![
                color_matrix(identity),
                display_color_entries(RGFX_COLOR_POST_GAMMA, &[[1.0; 3], [0.0; 3]]),
            ],
            |[b, g, r, _]| [255 - b, 255 - g, 255 - r, 0xff],
        ),
    ];
    let height = 7usize;
    let mut wrong = Vec::new();
    for (width, stride) in [
        (3662usize, 3662usize * 4),
        (3662, 14656),
        (3662, 14848),
        (1831, 7360),
        (1001, 1001 * 4),
    ] {
        // SAFETY: the scanout lies in the device's region, mapped while the device lives.
        unsafe {
            let base = device.region_base().add(RGFX_ALLOCATIONS_OFFSET as usize);
            for y in 0..height {
                for x in 0..width {
                    let p = odd_width_pixel(x, y);
                    std::ptr::copy_nonoverlapping(p.as_ptr(), base.add(y * stride + x * 4), 4);
                }
            }
        }
        let scanout = set_scanout(width as u32, height as u32, stride as u32);
        assert_eq!(answered(&device, &mut guest, scanout), RGFX_STATUS_OK);
        let mode = Mode {
            width: width as u32,
            height: height as u32,
            refresh_mhz: 60_000,
        };
        let read = device
            .read
            .at_screen(Arc::new(reims_vgpu_graphics::FixedScreen(mode)));
        for (name, messages, want) in &colours {
            if !messages.is_empty() {
                for m in messages {
                    guest.send(&device, m.clone());
                }
                assert_eq!(answered(&device, &mut guest, keep()), RGFX_STATUS_OK);
            }
            let mut dst = vec![7u8; width * height * 4];
            assert_eq!(read.read(&mut dst), ((width as u32, height as u32), Ok(())));
            let expected = |x: usize, y: usize| want(odd_width_pixel(x, y));
            if let Some(differ) = read_differs(&dst, (width, height), expected) {
                wrong.push(format!(
                    "{width} wide, rows of {stride} bytes, {name}: {differ}"
                ));
            }
        }
        // No colour again for the next scanout.
        guest.send(&device, color_matrix(identity));
        guest.send(&device, display_color_entries(RGFX_COLOR_POST_GAMMA, &[]));
        assert_eq!(answered(&device, &mut guest, keep()), RGFX_STATUS_OK);
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// On the Vulkan platform, a presented IOSurface at a width that is not a multiple of 4, its rows
/// padded to 64 bytes as a guest's IOSurface rows are, reads row for row at a mode of its size,
/// rendered to (shadowed) and read only (in place): copied as the guest wrote it with no colour,
/// and drawn through a matrix that swaps red and blue
#[test]
fn on_vulkan_a_presented_surface_at_an_odd_width_reads_row_for_row() {
    const RING_DATA: u32 = 0x1_0000;
    let platform = crate::singleton::platform();
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks::default(),
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(platform, config).unwrap();
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let span = u64::from(RGFX_RING_HEADER_SIZE + RING_DATA);
    guest.send(&device, block_grant(1, SUBMIT, span));
    guest.send(&device, block_grant(1, COMPLETE, span));
    let create = queue_create(1, 1, SUBMIT, COMPLETE, RING_DATA, 0);
    assert_eq!(answered(&device, &mut guest, create), RGFX_STATUS_OK);
    guest.send(&device, resource_id_chunk(1, 0));
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let swap = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ];
    let height = 16usize;
    let sizes = [(3662usize, 14656usize), (1831, 7360), (1001, 4032)];
    // MTLTextureUsageShaderRead | RenderTarget, then ShaderRead alone.
    let cases = [5u32, 1]
        .into_iter()
        .flat_map(|usage| sizes.map(|s| (s, usage)));
    let mut wrong = Vec::new();
    let mut at = BLOCK;
    for (i, ((width, stride), usage)) in cases.enumerate() {
        let n = i as u64;
        let length = ((stride * height) as u64).next_multiple_of(0x1_0000);
        guest.send(&device, block_grant(1, at, length));
        guest.send(
            &device,
            buffer_over_host_memory(obj(1, 10 + 2 * n), at, length),
        );
        // SAFETY: the surface lies in the block just granted, in the device's region.
        unsafe {
            let base = device.region_base().add(at as usize);
            for y in 0..height {
                for x in 0..width {
                    let p = odd_width_pixel(x, y);
                    std::ptr::copy_nonoverlapping(p.as_ptr(), base.add(y * stride + x * 4), 4);
                }
            }
        }
        let (create, _) = RgfxTextureCreate::read_from_prefix(&texture_create(0, 0)).unwrap();
        let id = 0x100 + i as u32;
        let mut texture = RgfxTextureFromIOSurface {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_IOSURFACE,
                size: 0,
                seqno: 0,
            },
            obj_id: obj(1, 11 + 2 * n),
            resource_id: 1 + n,
            buffer_obj_id: obj(1, 10 + 2 * n),
            offset: 0,
            bytes_per_row: stride as u64,
            iosurface_id: id,
            plane: 0,
            desc: RgfxTextureDescriptor {
                width: width as u32,
                height: height as u32,
                usage,
                ..create.desc
            },
            writers_offset: RGFX_SURFACE_NO_WRITERS,
        }
        .as_bytes()
        .to_vec();
        texture.resize(RGFX_SURFACE_FACTS_OFFSET as usize, 0);
        texture.extend_from_slice(
            RgfxSurfaceFacts {
                surface_pixel_format: u32::from_be_bytes(*b"BGRA"),
                plane_count: 0,
                width: width as u32,
                height: height as u32,
                chroma_offset: 0,
                chroma_bytes_per_row: 0,
                ycbcr_matrix: RGFX_YCBCR_MATRIX_NONE,
                reserved_24: 0,
            }
            .as_bytes(),
        );
        guest.send(&device, texture);
        let mode = Mode {
            width: width as u32,
            height: height as u32,
            refresh_mhz: 60_000,
        };
        let read = device
            .read
            .at_screen(Arc::new(reims_vgpu_graphics::FixedScreen(mode)));
        for (name, rows, swapped) in [
            ("no colour", identity, false),
            ("red and blue swapped", swap, true),
        ] {
            guest.send(&device, color_matrix(rows));
            let present = present_surface(id);
            assert_eq!(answered(&device, &mut guest, present), RGFX_STATUS_OK);
            let mut dst = vec![7u8; width * height * 4];
            assert_eq!(read.read(&mut dst), ((width as u32, height as u32), Ok(())));
            let want = |x: usize, y: usize| match odd_width_pixel(x, y) {
                [b, g, r, _] if swapped => [r, g, b, 0xff],
                p => p,
            };
            if let Some(differ) = read_differs(&dst, (width, height), want) {
                wrong.push(format!(
                    "{width} wide, rows of {stride} bytes, usage {usage}, {name}: {differ}"
                ));
            }
        }
        at += length;
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Queue 1 of process 1 on a headless device, as [`one_queue`] makes it, with both its rings of
/// `data` bytes and the device raising through `raise`.
fn one_queue_raising(
    data: u32,
    raise: GraphicsRaiseFn,
) -> (
    GraphicsDevice<FakePlatform>,
    Arc<FakePlatform>,
    Guest,
    QueueRing,
) {
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    let platform = Arc::new(FakePlatform::new());
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(raise),
            ..GraphicsCallbacks::default()
        },
        window: None,
        memory_map: None,
    };
    let device = GraphicsDevice::with_platform(Arc::clone(&platform), config).unwrap();
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let rings = (SUBMIT_OFF, COMPLETE_OFF);
    assert_eq!(
        create_queue(&device, &mut guest, 1, 1, rings, data, 0),
        RGFX_STATUS_OK
    );
    let ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(data));
    (device, platform, guest, ring)
}

/// The threads a device raised on, each with what its raise `found`.
type Raises = Arc<Mutex<Vec<(Option<String>, std::thread::ThreadId, bool)>>>;

/// A raise callback that notes each raise's thread in `raises`, with what `found` answers there.
fn noting_raises(
    raises: &Raises,
    found: impl Fn() -> bool + Send + Sync + 'static,
) -> GraphicsRaiseFn {
    let raises = Arc::clone(raises);
    Arc::new(move |vector| {
        assert_eq!(vector, 0);
        let thread = std::thread::current();
        let seen = found();
        raises
            .lock()
            .push((thread.name().map(str::to_owned), thread.id(), seen));
    })
}

/// A `COMMAND_BUFFER` `cb_obj_id` on queue `queue_obj_id` that waits for shared event `event` to
/// reach `value` and asks for its `CB_SUBMITTED`: its records go at its scheduling, from its
/// submission handler.
fn waiting_and_asking(queue_obj_id: u64, cb_obj_id: u64, event: u64, value: u64) -> Vec<u8> {
    use reims_vgpu_graphics::ops::{RGFX_OP_SUBMITTED, RGFX_OP_WAIT_EVENT, RgfxOpEvent};

    let wait = RgfxOpEvent {
        event_obj_id: event,
        value,
    };
    command_buffer_of(
        queue_obj_id,
        cb_obj_id,
        &[
            (RGFX_OP_WAIT_EVENT, wait.as_bytes()),
            (RGFX_OP_SUBMITTED, &[]),
        ],
    )
}

/// The next `CB_SUBMITTED` on `ring` (none other comes first) and then the `CB_SCHEDULED`, both
/// of command buffer `cb_obj_id`.
fn submitted_then_scheduled(ring: &mut QueueRing, cb_obj_id: u64) {
    let bytes = ring.event().expect("a CB_SUBMITTED");
    let submitted = RgfxCbSubmitted::read_from_bytes(&bytes).unwrap();
    assert_eq!(
        (submitted.header.r#type, submitted.cb_obj_id),
        (RGFX_EVT_CB_SUBMITTED, cb_obj_id)
    );
    let scheduled = eventually(|| {
        ring.has_event();
        ring.scheduled.iter().any(|&(_, cb)| cb == cb_obj_id)
    });
    assert!(scheduled, "no CB_SCHEDULED of {cb_obj_id:#x}");
}

/// A decoder's command buffer whose submission handler runs inside a commit on a thread the
/// header names for no raise, as the Vulkan platform's runtime runs every ready buffer's in a
/// commit, posts nothing there: here the window's render thread commits a draw of its own, holding
/// a lock of its own as a host's thread holds QEMU's BQL, and its commit takes the buffer held
/// before it. The buffer's `CB_SUBMITTED` and `CB_SCHEDULED` reach the guest from the queue's
/// service thread, which raises there and may wait there for the lock the render thread holds
/// across its commit: the raise never comes on the committing thread
#[test]
fn a_foreign_commit_hands_a_decoder_buffer_s_records_to_its_queue_s_thread() {
    use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};

    let caller = Arc::new(parking_lot::Mutex::new(()));
    let raises = Raises::default();
    let waits_for = Arc::clone(&caller);
    let raise = noting_raises(&raises, move || {
        waits_for.try_lock_for(Duration::from_secs(1)).is_some()
    });
    let (device, platform, mut guest, mut ring) = one_queue_raising(0x1_0000, raise);
    guest.send(&device, shared_event_create(obj(1, 7), 0));
    platform.world().hold_submissions();
    ring.send(&device, waiting_and_asking(obj(1, 1), 100, obj(1, 7), 1));
    committed(&platform, 1);
    assert!(!ring.has_event(), "posted before its scheduling");
    ring.arm();

    let draw_queue = platform.device().new_command_queue().expect("a queue");
    platform.world().release_submissions_at_next_commit();
    let (committing, holding) = (Arc::clone(&caller), Arc::clone(&raises));
    let window = std::thread::Builder::new()
        .name("reims-vgpu.window".into())
        .spawn(move || {
            let draw = no_catch(draw_queue.command_buffer(OnFault::Die)).expect("a buffer");
            let held = committing.lock();
            no_catch(draw.commit(OnFault::Die));
            let raised_inside = holding.lock().len();
            drop(held);
            (std::thread::current().id(), raised_inside)
        })
        .unwrap();
    let (window, raised_inside) = window.join().unwrap();

    submitted_then_scheduled(&mut ring, 100);
    assert_eq!(raised_inside, 0, "raised inside the render thread's commit");
    let raises = raises.lock().clone();
    assert_eq!(raises.len(), 1, "{raises:?}");
    let (name, thread, found) = &raises[0];
    assert_ne!(*thread, window, "raised on the committing thread");
    assert!(
        name.as_deref()
            .is_some_and(|n| n.starts_with("reims-vgpu.graphics-queue-")),
        "raised on {name:?}"
    );
    assert!(
        found,
        "the raise could not take the lock the committer let go"
    );
    assert_eq!(device.handed_over(), 2);
    platform.world().complete_all(CompleteOn::Thread);
}

/// The host's read of the display calls nothing back and waits for no room on a completion ring,
/// though its commit, on the host's thread, takes a decoder's buffer held before it whose queue's
/// completion ring is full: the buffer's `CB_SUBMITTED` and `CB_SCHEDULED` are handed to the
/// queue's service thread, which posts them, and raises, once the guest has made room
#[test]
fn a_read_of_the_display_calls_nothing_back_and_waits_for_no_full_ring() {
    use reims_vgpu_graphics::ops::{RGFX_OP_SIGNAL_EVENT, RgfxOpEvent};

    const DATA: u32 = 0x1000;
    const SIGNALS: u64 = 150;
    let raises = Raises::default();
    let (device, platform, mut guest, mut ring) =
        one_queue_raising(DATA, noting_raises(&raises, || true));
    platform
        .world()
        .set_auto_complete(Some(CompleteOn::Detached));
    assert_eq!(
        answered(&device, &mut guest, set_scanout(64, 32, 256)),
        RGFX_STATUS_OK
    );
    guest.send(&device, shared_event_create(obj(1, 7), 0));
    // A buffer that signals the event 150 times: its completion posts each value, 32 bytes, more
    // than the ring holds, and waits on its completion thread for room the guest does not make.
    let signals: Vec<RgfxOpEvent> = (1..=SIGNALS)
        .map(|value| RgfxOpEvent {
            event_obj_id: obj(1, 7),
            value,
        })
        .collect();
    let ops: Vec<(u16, &[u8])> = (signals.iter())
        .map(|s| (RGFX_OP_SIGNAL_EVENT, s.as_bytes()))
        .collect();
    ring.arm();
    ring.send(&device, command_buffer_of(obj(1, 1), 100, &ops));
    // SAFETY: the completion ring's producer count is its header's first field, in the region.
    let produced = |ring: &QueueRing| unsafe { ring.load_u64(ring.complete_off as usize) };
    assert!(
        eventually(|| produced(&ring) > u64::from(DATA) - 40),
        "the ring did not fill: {} of {DATA} bytes",
        produced(&ring)
    );
    std::thread::sleep(Duration::from_millis(50));
    let full = produced(&ring);

    platform.world().hold_submissions();
    ring.send(&device, waiting_and_asking(obj(1, 1), 101, obj(1, 7), 1000));
    committed(&platform, 1);
    platform.world().release_submissions_at_next_commit();
    let device = Arc::new(device);
    let reader = Arc::clone(&device);
    let (read_tx, read_rx) = std::sync::mpsc::channel();
    let host = std::thread::Builder::new()
        .name("host".into())
        .spawn(move || {
            let mut dst = vec![0u8; 1920 * 1080 * 4];
            let read = reader.read_display(&mut dst);
            let _ = read_tx.send((std::thread::current().id(), read.1.is_ok()));
        })
        .unwrap();
    let (host_thread, read) = read_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the read waited for the full ring");
    host.join().unwrap();
    assert!(read, "the read failed");
    assert_eq!(produced(&ring), full, "posted onto a full ring");
    assert_eq!(
        device.handed_over(),
        2,
        "the buffer's records were not handed over"
    );
    assert!(
        raises
            .lock()
            .iter()
            .all(|(_, thread, _)| *thread != host_thread),
        "raised inside the read"
    );

    // The guest makes room: the signals, the first buffer's completion and the second's records
    // all come, the second's from the queue's service thread.
    ring.arm();
    let mut submitted = false;
    let mut signalled = 0;
    while !submitted || signalled < SIGNALS {
        let Some(bytes) = ring.event() else {
            break;
        };
        let header = RgfxMsgHeader::read_from_prefix(&bytes).unwrap().0;
        match header.r#type {
            RGFX_EVT_EVENT_SIGNALLED => signalled += 1,
            RGFX_EVT_CB_SUBMITTED => {
                let posted = RgfxCbSubmitted::read_from_bytes(&bytes).unwrap();
                assert_eq!(posted.cb_obj_id, 101);
                submitted = true;
            }
            RGFX_EVT_CB_COMPLETE => {}
            other => panic!("an event of type {other:#x}"),
        }
    }
    assert!(submitted, "no CB_SUBMITTED of the held buffer");
    assert_eq!(signalled, SIGNALS);
    let scheduled = eventually(|| {
        ring.has_event();
        ring.scheduled.iter().any(|&(_, cb)| cb == 101)
    });
    assert!(scheduled, "no CB_SCHEDULED of the held buffer");
    for (name, thread, _) in raises.lock().iter() {
        assert_ne!(*thread, host_thread, "raised inside the read");
        let name = name.as_deref().unwrap_or_default();
        assert!(
            name.starts_with("reims-vgpu.graphics-") || name == "com.Metal.CompletionQueueDispatch",
            "raised on {name:?}"
        );
    }
    platform.world().complete_all(CompleteOn::Thread);
}

/// A decoder's command buffer whose submission handler runs inside a commit on one of the
/// device's threads, which the header names for the raise, posts its records there, as before:
/// here the queue's next buffer's commit, on its queue's thread, takes the buffer held before it.
/// Nothing is handed over, and the raise comes on that thread
#[test]
fn a_commit_on_the_device_s_thread_posts_a_decoder_buffer_s_records_there() {
    let raises = Raises::default();
    let (device, platform, mut guest, mut ring) =
        one_queue_raising(0x1_0000, noting_raises(&raises, || true));
    guest.send(&device, shared_event_create(obj(1, 7), 0));
    platform.world().hold_submissions();
    ring.send(&device, waiting_and_asking(obj(1, 1), 100, obj(1, 7), 1));
    committed(&platform, 1);
    assert!(!ring.has_event(), "posted before its scheduling");
    ring.arm();
    platform.world().release_submissions_at_next_commit();
    ring.send(&device, command_buffer(obj(1, 1), 101));
    submitted_then_scheduled(&mut ring, 100);
    assert_eq!(device.handed_over(), 0, "a record was handed over");
    let raises = raises.lock().clone();
    assert_eq!(raises.len(), 1, "{raises:?}");
    let name = raises[0].0.as_deref().unwrap_or_default();
    assert!(
        name.starts_with("reims-vgpu.graphics-queue-")
            || name.starts_with("reims-vgpu.graphics-lane-"),
        "raised on {name:?}"
    );
    platform.world().complete_all(CompleteOn::Thread);
}

/// What a decoder buffer's records cost when a commit on a thread the raise is not named for
/// takes the buffer ([`a_foreign_commit_hands_a_decoder_buffer_s_records_to_its_queue_s_thread`])
/// and hands them to the queue's service thread, against one on a device's thread, which posts
/// them inline: the commit call's time on the committing thread, and the time from the commit's
/// start to the buffer's `CB_SCHEDULED` on the ring, with the queue's thread asleep before each, as
/// between frames. Prints the mean, median, 90th and 99th percentiles and the most, in
/// microseconds: a measurement, run by hand (`--run-ignored only --no-capture`, in a release
/// build).
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_handed_over_record_s_cost() {
    use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use std::time::Instant;

    const ROUNDS: usize = 1000;
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, mut guest, mut ring) = one_queue(&raised);
    guest.send(&device, shared_event_create(obj(1, 7), 0));
    // A committer on a thread named `name`: at each word it commits a draw of its own, and
    // answers when the commit began and how long it took.
    let committer = |name: &str| {
        let queue = platform.device().new_command_queue().expect("a queue");
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<(Instant, Duration)>();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                for () in go_rx {
                    let draw = no_catch(queue.command_buffer(OnFault::Die)).expect("a buffer");
                    let start = Instant::now();
                    no_catch(draw.commit(OnFault::Die));
                    let _ = done_tx.send((start, start.elapsed()));
                }
            })
            .unwrap();
        (go_tx, done_rx)
    };
    // (handed over, inline)
    let committers = [
        committer("reims-vgpu.window"),
        committer("reims-vgpu.graphics-lane-bench"),
    ];
    let submit_off = ring.submit_off;
    let mut commit_us: [Vec<f64>; 2] = Default::default();
    let mut scheduled_us: [Vec<f64>; 2] = Default::default();
    for (round, cb) in (0..2 * ROUNDS).zip(5000_u64..) {
        let kind = round % 2;
        platform.world().hold_submissions();
        ring.send(&device, waiting_and_asking(obj(1, 1), cb, obj(1, 7), 1));
        committed(&platform, 1);
        wait_idle(&device, submit_off);
        // A pause as the guest's, so the queue's thread sleeps as it does between frames.
        std::thread::sleep(Duration::from_micros(200));
        ring.arm();
        platform.world().release_submissions_at_next_commit();
        committers[kind].0.send(()).unwrap();
        let (start, took) = committers[kind].1.recv().unwrap();
        while !ring.scheduled.iter().any(|&(_, id)| id == cb) {
            if ring.has_event() {
                let posted = RgfxCbSubmitted::read_from_bytes(&ring.event().unwrap()).unwrap();
                assert_eq!(
                    (posted.header.r#type, posted.cb_obj_id),
                    (RGFX_EVT_CB_SUBMITTED, cb)
                );
            }
            assert!(start.elapsed() < Duration::from_secs(2), "no CB_SCHEDULED");
            std::hint::spin_loop();
        }
        scheduled_us[kind].push(start.elapsed().as_secs_f64() * 1e6);
        commit_us[kind].push(took.as_secs_f64() * 1e6);
        assert!(complete_until_event(&platform, &mut ring), "no completion");
        let complete = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
        assert_eq!((complete.cb_obj_id, complete.status), (cb, RGFX_CB_OK));
        ring.scheduled.clear();
    }
    assert_eq!(device.handed_over(), 2 * ROUNDS as u64);
    let stats = |name: &str, us: &[f64]| {
        let mut us = us.to_vec();
        us.sort_by(f64::total_cmp);
        let n = us.len();
        let mean = us.iter().sum::<f64>() / n as f64;
        println!(
            "{name}: mean {mean:.2} p50 {:.2} p90 {:.2} p99 {:.2} max {:.2} us (n {n})",
            us[n / 2],
            us[n * 9 / 10],
            us[n * 99 / 100],
            us[n - 1]
        );
    };
    stats("commit call, handed over", &commit_us[0]);
    stats("commit call, inline", &commit_us[1]);
    stats("commit to CB_SCHEDULED, handed over", &scheduled_us[0]);
    stats("commit to CB_SCHEDULED, inline", &scheduled_us[1]);
}

/// A host's hardware video for [`video_caps`]: one HEVC decode, landing its pictures in place at
/// a layout of its own, and in device caches where `cached`.
struct CapsVideo {
    cached: bool,
}

impl GraphicsVideo<FakePlatform> for CapsVideo {
    fn caps(&self) -> Vec<RgfxVideoCodecCaps> {
        vec![RgfxVideoCodecCaps {
            codec: RGFX_VIDEO_CODEC_HEVC,
            direction: RGFX_VIDEO_DECODE,
            ..RgfxVideoCodecCaps::default()
        }]
    }

    fn layout(&self, _codec: &RgfxVideoCodecCaps) -> RgfxVideoOutputLayout {
        RgfxVideoOutputLayout {
            granularity: [2, 2],
            alignment: 256,
            reserved_0c: 0,
        }
    }

    fn session(
        &self,
        _codec: u32,
        _direction: u32,
    ) -> Option<Arc<dyn GraphicsVideoSession<FakePlatform>>> {
        None
    }

    fn lands_in_device_caches(&self) -> bool {
        self.cached
    }
}

/// A video that lands pictures in device caches has the guest ask its outputs cached, with no
/// layout in place, on a device whose kext makes caches of every surface it asks one for
/// (`SURFACE_CACHE`) and on one whose GPU shares the CPU's memory and keeps caches of the surfaces
/// whose textures would not lie in place (`SURFACE_CACHE_REFUSED`); a device that makes no cache,
/// or a video that lands none, answers the layout its pictures land in place at. Either way the
/// decode entry carries the codec crate's bounds beyond it, HEVC's narrowest tile column.
#[test]
fn outputs_are_cached_where_the_kext_makes_caches_and_the_video_lands_in_them() {
    let refused = u64::from(RGFX_FEATURE_SURFACE_PAGES | RGFX_FEATURE_SURFACE_CACHE_REFUSED);
    let all = u64::from(RGFX_FEATURE_SURFACE_PAGES | RGFX_FEATURE_SURFACE_CACHE);
    let in_place = u64::from(RGFX_FEATURE_SURFACE_PAGES);
    for (features, lands, cached) in [
        (refused, true, true),
        (all, true, true),
        (in_place, true, false),
        (0, true, false),
        (refused, false, false),
        (all, false, false),
    ] {
        let caps = video_caps(features, &CapsVideo { cached: lands });
        let what = format!("features {features:#x}, landing in caches: {lands}");
        assert_eq!(caps.codec_count, 1, "{what}");
        assert_eq!(caps.codecs[0].codec, RGFX_VIDEO_CODEC_HEVC, "{what}");
        assert_eq!(
            caps.decode[0].min_tile_width,
            reims_vgpu_video::codec::hevc::MIN_TILE_COLUMN_WIDTH,
            "{what}"
        );
        if cached {
            assert_eq!(caps.flags, RGFX_VIDEO_CAPS_OUTPUT_CACHED, "{what}");
            assert_eq!(caps.layouts[0], RgfxVideoOutputLayout::default(), "{what}");
        } else {
            assert_eq!(caps.flags, 0, "{what}");
            assert_eq!(caps.layouts[0].alignment, 256, "{what}");
        }
    }
}
