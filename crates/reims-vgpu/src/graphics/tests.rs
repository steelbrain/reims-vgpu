use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use reims_graphics::ops::{RGFX_COMMAND_BUFFER_OPS_OFFSET, RgfxCommandBuffer};
use reims_graphics::wire::*;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::testkit::FakePlatform;
use reims_vgpu_core::testkit::objects::CompleteOn;
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
    };
    let device = GraphicsDevice::with_platform(Arc::clone(&platform), config).unwrap();
    (device, platform)
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
/// mirroring the GPU device's caps the guest bundle reads before it makes any queue
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
    assert_eq!(info.registry_id, caps.registry_id);
    assert!(info.name_len > 0);
    assert_eq!(
        &info.name[..info.name_len as usize],
        caps.name.as_deref().unwrap().as_bytes()
    );
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
        open_queue(
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
        open_queue(&device, &mut guest, context_close(context)),
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

/// Sends `command` (a `QUEUE_CREATE` or a `CONTEXT_CLOSE`) on ring 0, as the guest's kext does, and
/// waits for its `COMPLETE`: its status.
fn open_queue(device: &GraphicsDevice<FakePlatform>, guest: &mut Guest, command: Vec<u8>) -> u32 {
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
    fn send(&mut self, device: &GraphicsDevice<FakePlatform>, mut command: Vec<u8>) {
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
        device.mmio_write(RGFX_REG_DOORBELL.into(), self.submit_off as u32);
    }

    /// Whether the completion ring has an event the guest has not taken.
    fn has_event(&self) -> bool {
        // SAFETY: as the helpers above; `producer` is at the ring's start.
        unsafe { self.load_u64(self.complete_off as usize) != self.consumed }
    }

    /// The next completion-ring event, waiting about 2 s; `None` where none came.
    fn event(&mut self) -> Option<Vec<u8>> {
        let (complete_off, data) = (self.complete_off, self.data);
        let at = |count: u64| complete_off as usize + 256 + (count % data) as usize;
        for _ in 0..2000 {
            if self.has_event() {
                // SAFETY: as the helpers above; the host published the message before producer.
                let header = unsafe { self.read(at(self.consumed), 8) };
                let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
                let r#type = u32::from_le_bytes(header[0..4].try_into().unwrap());
                let bytes = unsafe { self.read(at(self.consumed), size) };
                self.consumed += size as u64;
                // SAFETY: as the helpers above.
                unsafe { self.store_u64(self.complete_off as usize + 0x40, self.consumed) };
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
        open_queue(
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
