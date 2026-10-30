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
fn complete_until_event(platform: &FakePlatform, ring: &QueueRing) -> bool {
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
        complete_until_event(&platform, &ring),
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
        complete_until_event(&platform, &ring),
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
/// it before the destroy compiles with it, and the library is not stored
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
    assert!(
        guest.event().is_none(),
        "the create after the destroy was answered"
    );
    assert_eq!(device.decoder_object_count(), 1);
}
