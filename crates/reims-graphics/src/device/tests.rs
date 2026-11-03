//! The device as a guest drives it: a stand-in guest writes commands into the control submission
//! ring, rings the doorbell, and reads the completion ring, with a host that records its
//! interrupts and a screen that records what it is told.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use parking_lot::Mutex;
use zerocopy::{FromBytes, FromZeros, IntoBytes};

use super::*;
use crate::display::CursorImage;

/// A refresh rate at which the refresh thread never runs during a test: a period of 1000 s. The
/// tests run each refresh themselves ([`Shared::refresh`]).
const NEVER_MHZ: u32 = 1;

/// What the screen was told, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Told {
    Scanout(Option<Scanout>),
    Present,
    Cursor(Option<CursorImage>),
    Moved(i32, i32, bool),
}

/// A screen with a mode the test sets, recording what it is told.
struct TestScreen {
    mode: Arc<Mutex<Mode>>,
    told: Arc<Mutex<Vec<Told>>>,
}

impl Screen for TestScreen {
    fn mode(&self) -> Mode {
        *self.mode.lock()
    }
    fn scanout(&self, scanout: Option<Scanout>) {
        self.told.lock().push(Told::Scanout(scanout));
    }
    fn present(&self) {
        self.told.lock().push(Told::Present);
    }
    fn cursor_image(&self, image: Option<CursorImage>) {
        self.told.lock().push(Told::Cursor(image));
    }
    fn cursor_moved(&self, x: i32, y: i32, visible: bool) {
        self.told.lock().push(Told::Moved(x, y, visible));
    }
}

/// The surfaces a host was asked to show, each with the CPU generation its present carried.
type Presented = Arc<Mutex<Vec<(u32, Option<u64>)>>>;

/// A host that counts its interrupts and answers whether the scanout was written as the test
/// sets it.
struct TestHost {
    interrupts: Arc<AtomicU32>,
    written: Arc<AtomicBool>,
    asked: Arc<Mutex<Vec<(u64, u64)>>>,
    presented: Presented,
    refreshes: Arc<AtomicU32>,
}

/// The one IOSurface the test host holds a texture over.
const HELD_SURFACE: u32 = 0x77;

impl Host for TestHost {
    fn raise_interrupt(&self, vector: u32) {
        assert_eq!(vector, 0);
        self.interrupts.fetch_add(1, Ordering::SeqCst);
    }
    fn scanout_written(&self, offset: u64, length: u64) -> bool {
        self.asked.lock().push((offset, length));
        self.written.load(Ordering::SeqCst)
    }
    fn refreshed(&self) {
        self.refreshes.fetch_add(1, Ordering::SeqCst);
    }
    fn present_surface(&self, iosurface_id: u32, generation: Option<u64>, answer: PresentAnswer) {
        self.presented.lock().push((iosurface_id, generation));
        answer.answer(if iosurface_id == 0 || iosurface_id == HELD_SURFACE {
            RGFX_STATUS_OK
        } else {
            RGFX_STATUS_INVALID_ARGUMENT
        });
    }
}

/// A device, its stand-in guest, and what its host and screen saw.
struct Rig {
    device: Device,
    mode: Arc<Mutex<Mode>>,
    told: Arc<Mutex<Vec<Told>>>,
    interrupts: Arc<AtomicU32>,
    written: Arc<AtomicBool>,
    asked: Arc<Mutex<Vec<(u64, u64)>>>,
    presented: Presented,
    /// The refreshes the host was told of.
    refreshes: Arc<AtomicU32>,
    /// The guest's count of the submission ring.
    produced: u64,
    /// The guest's count of the completion ring.
    consumed: u64,
    seqno: u64,
}

const MODE: Mode = Mode {
    width: 1280,
    height: 800,
    refresh_mhz: NEVER_MHZ,
};

impl Rig {
    fn new() -> Rig {
        let interrupts = Arc::new(AtomicU32::new(0));
        let written = Arc::new(AtomicBool::new(false));
        let asked = Arc::new(Mutex::new(Vec::new()));
        let presented = Arc::new(Mutex::new(Vec::new()));
        let refreshes = Arc::new(AtomicU32::new(0));
        let host = TestHost {
            interrupts: Arc::clone(&interrupts),
            written: Arc::clone(&written),
            asked: Arc::clone(&asked),
            presented: Arc::clone(&presented),
            refreshes: Arc::clone(&refreshes),
        };
        let mut rig = Rig::with_host(Box::new(host));
        rig.interrupts = interrupts;
        rig.written = written;
        rig.asked = asked;
        rig.presented = presented;
        rig.refreshes = refreshes;
        rig
    }

    /// A rig whose device has `host` (what the rig records of the host's stays empty).
    fn with_host(host: Box<dyn Host>) -> Rig {
        let mode = Arc::new(Mutex::new(MODE));
        let told = Arc::new(Mutex::new(Vec::new()));
        let interrupts = Arc::new(AtomicU32::new(0));
        let written = Arc::new(AtomicBool::new(false));
        let asked = Arc::new(Mutex::new(Vec::new()));
        let presented = Arc::new(Mutex::new(Vec::new()));
        let region = Arc::new(Region::new(REGION_SIZE as usize).unwrap());
        let device = Device::new(
            region,
            host,
            Box::new(TestScreen {
                mode: Arc::clone(&mode),
                told: Arc::clone(&told),
            }),
        )
        .unwrap();
        Rig {
            device,
            mode,
            told,
            interrupts,
            written,
            asked,
            presented,
            refreshes: Arc::new(AtomicU32::new(0)),
            produced: 0,
            consumed: 0,
            seqno: 0,
        }
    }

    fn region(&self) -> &Region {
        &self.device.shared.region
    }

    /// The guest starts the device.
    fn start(&self) {
        self.device
            .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    }

    /// Writes `bytes` (a whole message, its size set) into the submission ring and publishes it,
    /// without a doorbell.
    fn publish(&mut self, bytes: &[u8]) {
        let data = RGFX_CONTROL_RING_DATA_SIZE as u64;
        let at = |count: u64| RGFX_CONTROL_SUBMIT_OFFSET + 256 + (count % data);
        let left = data - self.produced % data;
        let needed = bytes.len() as u64 + if left < bytes.len() as u64 { left } else { 0 };
        let mut waited = 0;
        while self.produced + needed
            - self
                .region()
                .atomic_u64(RGFX_CONTROL_SUBMIT_OFFSET + 0x40)
                .load(Ordering::SeqCst)
            > data
        {
            waited += 1;
            assert!(waited < 5000, "the host took nothing for 5 s");
            std::thread::sleep(Duration::from_millis(1));
        }
        if left < bytes.len() as u64 {
            let mut pad = [0u8; 8];
            pad[4..8].copy_from_slice(&(left as u32).to_le_bytes());
            self.region().write(at(self.produced), &pad);
            self.produced += left;
        }
        self.region().write(at(self.produced), bytes);
        self.produced += bytes.len() as u64;
        self.region()
            .atomic_u64(RGFX_CONTROL_SUBMIT_OFFSET)
            .store(self.produced, Ordering::SeqCst);
    }

    /// Sends a command: its `seqno` the next, its size its length; rings the doorbell where
    /// the host sleeps. The command's `seqno`.
    fn send(&mut self, mut command: Vec<u8>) -> u64 {
        self.seqno += 1;
        command[8..16].copy_from_slice(&self.seqno.to_le_bytes());
        set_size_of(&mut command);
        self.publish(&command);
        std::sync::atomic::fence(Ordering::SeqCst);
        let idle = self
            .region()
            .atomic_u32(RGFX_CONTROL_SUBMIT_OFFSET + 0x80)
            .load(Ordering::SeqCst);
        if idle == 1 {
            self.device
                .mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
        }
        self.seqno
    }

    /// The next event of the completion ring, waiting about 2 s; `None` where none came.
    fn event(&mut self) -> Option<Vec<u8>> {
        let data = RGFX_CONTROL_RING_DATA_SIZE as u64;
        let at = |count: u64| RGFX_CONTROL_COMPLETE_OFFSET + 256 + (count % data);
        for _ in 0..2000 {
            let produced = self
                .region()
                .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET)
                .load(Ordering::SeqCst);
            if produced != self.consumed {
                let start = self.region().read(at(self.consumed), 8).unwrap();
                let size = u32::from_le_bytes(start[4..8].try_into().unwrap()) as usize;
                let r#type = u32::from_le_bytes(start[0..4].try_into().unwrap());
                let bytes = self.region().read(at(self.consumed), size).unwrap();
                self.consumed += size as u64;
                self.region()
                    .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET + 0x40)
                    .store(self.consumed, Ordering::SeqCst);
                if r#type == RGFX_MSG_PAD {
                    continue;
                }
                return Some(bytes);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        None
    }

    /// The next event, as a `COMPLETE`: its seqno and status.
    fn complete(&mut self) -> (u64, u32) {
        let bytes = self.event().expect("an answer");
        let c = RgfxComplete::read_from_bytes(&bytes).unwrap();
        assert_eq!(c.header.r#type, RGFX_EVT_COMPLETE);
        assert_eq!(c.header.size, 24);
        (c.header.seqno, c.status)
    }

    fn arm(&self) {
        self.region()
            .atomic_u32(RGFX_CONTROL_COMPLETE_OFFSET + 0xc0)
            .store(1, Ordering::SeqCst);
    }

    fn told(&self) -> Vec<Told> {
        std::mem::take(&mut *self.told.lock())
    }

    fn refresh(&self) {
        self.device.shared.refresh(1234);
    }
}

fn set_size_of(bytes: &mut [u8]) {
    let len = bytes.len() as u32;
    bytes[4..8].copy_from_slice(&len.to_le_bytes());
}

fn get_displays() -> Vec<u8> {
    let c = RgfxGetDisplays {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_GET_DISPLAYS,
            ..Default::default()
        },
    };
    c.as_bytes().to_vec()
}

fn scanout(width: u32, height: u32, stride: u32, offset: u64) -> Vec<u8> {
    let c = RgfxSetScanout {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_SCANOUT,
            ..Default::default()
        },
        display: 0,
        width,
        height,
        stride,
        offset,
    };
    c.as_bytes().to_vec()
}

fn vblank(enabled: u32) -> Vec<u8> {
    let c = RgfxSetVblank {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_VBLANK,
            ..Default::default()
        },
        display: 0,
        enabled,
    };
    c.as_bytes().to_vec()
}

fn present_surface(display: u32, iosurface_id: u32) -> Vec<u8> {
    let c = RgfxPresentSurface {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_PRESENT_SURFACE,
            ..Default::default()
        },
        display,
        iosurface_id,
        transaction_id: 9,
        reserved_1c: 0,
    };
    c.as_bytes().to_vec()
}

/// `PRESENT_SURFACE` hands display 0's surface to the host and answers whether it shows it: a
/// surface it holds, and 0 (the scanout again), succeed; one it holds no texture over is an
/// invalid argument; another display is refused before the host is asked. A present that carries
/// the surface's CPU generation hands it to the host
#[test]
fn present_surface_asks_the_host_and_answers_its_word() {
    let mut rig = Rig::new();
    rig.start();
    let seqno = rig.send(present_surface(0, HELD_SURFACE));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    let seqno = rig.send(present_surface(0, 5));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_INVALID_ARGUMENT));
    let seqno = rig.send(present_surface(1, HELD_SURFACE));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_NO_DISPLAY));
    let seqno = rig.send(present_surface(0, 0));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    let mut with_generation = present_surface(0, HELD_SURFACE);
    with_generation.extend_from_slice(RgfxPresentSurfaceGen { cpu_gen: 12 }.as_bytes());
    let seqno = rig.send(with_generation);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(
        *rig.presented.lock(),
        [
            (HELD_SURFACE, None),
            (5, None),
            (0, None),
            (HELD_SURFACE, Some(12))
        ]
    );
}

/// A host whose interrupt resets the device, as a host's `raise_interrupt` may (it may call the
/// device), and which answers a present at once, inside [`Host::present_surface`].
struct ResettingHost {
    shared: Arc<OnceLock<Weak<Shared>>>,
    resets: Arc<AtomicU32>,
}

impl Host for ResettingHost {
    fn raise_interrupt(&self, _vector: u32) {
        if let Some(shared) = self.shared.get().and_then(Weak::upgrade) {
            shared.reset();
            self.resets.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn scanout_written(&self, _offset: u64, _length: u64) -> bool {
        false
    }
    fn present_surface(&self, _iosurface_id: u32, _generation: Option<u64>, answer: PresentAnswer) {
        answer.answer(RGFX_STATUS_OK);
    }
}

/// A present the host answers at once raises the interrupt with no lock of the device's held: an
/// interrupt that resets the device returns
#[test]
fn a_present_answered_at_once_raises_with_no_lock_held() {
    let shared = Arc::new(OnceLock::new());
    let resets = Arc::new(AtomicU32::new(0));
    let mut rig = Rig::with_host(Box::new(ResettingHost {
        shared: Arc::clone(&shared),
        resets: Arc::clone(&resets),
    }));
    shared.set(Arc::downgrade(&rig.device.shared)).unwrap();
    rig.start();
    rig.arm();
    rig.send(present_surface(0, HELD_SURFACE));
    for _ in 0..2000 {
        if resets.load(Ordering::SeqCst) != 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        resets.load(Ordering::SeqCst),
        1,
        "the reset inside the interrupt never returned"
    );
    assert_eq!(
        rig.device.mmio_read(RGFX_REG_STATUS.into()),
        RGFX_STATUS_RESET
    );
}

/// The registers read as the wire says, and `STATUS` follows the guest's writes
#[test]
fn the_registers_read_as_the_wire_says() {
    let rig = Rig::new();
    let read = |offset: u32| rig.device.mmio_read(offset.into());
    assert_eq!(read(RGFX_REG_MAGIC), 0x5846_4752);
    assert_eq!(read(RGFX_REG_VERSION), 2);
    assert_eq!(read(RGFX_REG_FEATURES_LO), 0);
    assert_eq!(read(RGFX_REG_FEATURES_HI), 0);
    assert_eq!(read(RGFX_REG_STATUS), 0);
    assert_eq!(
        read(RGFX_REG_REGION_SIZE_LO),
        (REGION_SIZE & 0xffff_ffff) as u32
    );
    assert_eq!(read(RGFX_REG_REGION_SIZE_HI), (REGION_SIZE >> 32) as u32);
    assert_eq!(read(RGFX_REG_DOORBELL), 0);
    assert_eq!(rig.device.region_size(), REGION_SIZE);
    rig.start();
    assert_eq!(read(RGFX_REG_STATUS), 1);
    rig.device.mmio_write(RGFX_REG_STATUS.into(), 7);
    assert_eq!(read(RGFX_REG_STATUS), 1, "other values are ignored");
    rig.device.mmio_write(RGFX_REG_STATUS.into(), 0);
    assert_eq!(read(RGFX_REG_STATUS), 0);
}

/// `GET_DISPLAYS` answers display 0, connected, with the screen's mode as its one, preferred
/// mode; the interrupt is raised only where the guest armed the ring, and clears `armed`
#[test]
fn get_displays_answers_the_screen_s_mode() {
    let mut rig = Rig::new();
    rig.start();
    let seqno = rig.send(get_displays());
    let bytes = rig.event().unwrap();
    assert_eq!(bytes.len(), 48);
    let header = RgfxMsgHeader::read_from_prefix(&bytes).unwrap().0;
    assert_eq!(
        (header.r#type, header.size, header.seqno),
        (RGFX_EVT_DISPLAYS, 48, seqno)
    );
    let words: Vec<u32> = bytes[16..]
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(words, [1, 0, 1, 0, 1, 1280, 800, NEVER_MHZ]);
    assert_eq!(rig.interrupts.load(Ordering::SeqCst), 0, "not armed");
    rig.arm();
    rig.send(get_displays());
    rig.event().unwrap();
    assert_eq!(rig.interrupts.load(Ordering::SeqCst), 1);
    let armed = rig
        .region()
        .atomic_u32(RGFX_CONTROL_COMPLETE_OFFSET + 0xc0)
        .load(Ordering::SeqCst);
    assert_eq!(armed, 0);
}

/// A post whose guest took the record and armed the ring again before the post cleared `armed`
/// (a consumer already awake) raises nothing and leaves the ring armed; with the record not taken
/// the arm is cleared and the vector raised
#[test]
fn a_record_taken_before_the_clear_leaves_the_ring_armed() {
    let mut rig = Rig::new();
    rig.start();
    let take = Arc::new(AtomicBool::new(true));
    let t = Arc::clone(&take);
    rig.device
        .shared
        .completion
        .lock()
        .set_between(move |region, count| {
            if t.load(Ordering::SeqCst) {
                region
                    .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET + 0x40)
                    .store(count, Ordering::SeqCst);
            }
            region
                .atomic_u32(RGFX_CONTROL_COMPLETE_OFFSET + 0xc0)
                .store(1, Ordering::SeqCst);
        });
    let armed = |rig: &Rig| {
        rig.region()
            .atomic_u32(RGFX_CONTROL_COMPLETE_OFFSET + 0xc0)
            .load(Ordering::SeqCst)
    };
    rig.send(get_displays());
    rig.event().unwrap();
    assert_eq!(
        rig.interrupts.load(Ordering::SeqCst),
        0,
        "taken: nothing raised"
    );
    assert_eq!(armed(&rig), 1, "the guest's arm stands");
    take.store(false, Ordering::SeqCst);
    rig.send(get_displays());
    rig.event().unwrap();
    assert_eq!(rig.interrupts.load(Ordering::SeqCst), 1);
    assert_eq!(armed(&rig), 0);
}

/// Before the guest writes `STATUS` 1 nothing is taken; once it does, what it published is
#[test]
fn commands_wait_for_the_driver_s_start() {
    let mut rig = Rig::new();
    rig.send(get_displays());
    assert!(rig.event().is_none());
    rig.start();
    assert!(rig.event().is_some());
}

/// `SET_SCANOUT` shows pixels inside the guest's allocations, blanks with width 0, and refuses
/// a display, a range or a stride the device does not take, changing nothing
#[test]
fn set_scanout_checks_its_range() {
    let mut rig = Rig::new();
    rig.start();
    let alloc = RGFX_ALLOCATIONS_OFFSET;
    let s = rig.send(scanout(640, 480, 2560, alloc));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    let shown = Scanout {
        width: 640,
        height: 480,
        stride: 2560,
        offset: alloc,
    };
    assert_eq!(rig.told(), [Told::Scanout(Some(shown))]);
    for bytes in [
        scanout(640, 480, 2556, alloc),
        scanout(640, 480, 2562, alloc),
        scanout(640, 0, 2560, alloc),
        scanout(640, 480, 2560, alloc - 4096),
        scanout(640, 480, 2560, alloc + 2),
        scanout(640, 480, 2560, REGION_SIZE - 2560 * 479),
        scanout(640, 480, 2560, u64::MAX - 100),
    ] {
        let s = rig.send(bytes);
        assert_eq!(rig.complete(), (s, RGFX_STATUS_INVALID_ARGUMENT));
    }
    let mut bytes = scanout(640, 480, 2560, alloc);
    bytes[16] = 1;
    let s = rig.send(bytes);
    assert_eq!(rig.complete(), (s, RGFX_STATUS_NO_DISPLAY));
    assert_eq!(rig.told(), []);
    assert_eq!(rig.device.shared.display.lock().scanout, Some(shown));
    let s = rig.send(scanout(0, 0, 0, 0));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    assert_eq!(rig.told(), [Told::Scanout(None)]);
}

/// `SET_SCANOUT` keeps to the scanout's room, the `RGFX_SCANOUT_SPAN` bytes before the processes'
/// blocks: a scanout that ends at the room's end is shown, and one that reaches past it, or lies
/// in the blocks, is refused, changing nothing
#[test]
fn set_scanout_keeps_to_the_scanout_s_room() {
    let mut rig = Rig::new();
    rig.start();
    let span = 2560 * 479 + 2560;
    let end = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    let s = rig.send(scanout(640, 480, 2560, end - span));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    let shown = Scanout {
        width: 640,
        height: 480,
        stride: 2560,
        offset: end - span,
    };
    assert_eq!(rig.told(), [Told::Scanout(Some(shown))]);
    for bytes in [
        scanout(640, 480, 2560, end - span + 4),
        scanout(640, 480, 2560, end),
        scanout(640, 480, 2560, end + 0x10_0000),
        scanout(7680, 4400, 30720, RGFX_ALLOCATIONS_OFFSET),
    ] {
        let s = rig.send(bytes);
        assert_eq!(rig.complete(), (s, RGFX_STATUS_INVALID_ARGUMENT));
    }
    assert_eq!(rig.told(), []);
    assert_eq!(rig.device.shared.display.lock().scanout, Some(shown));
}

/// `SET_CURSOR` hands the screen the guest's image as straight RGBA, and `MOVE_CURSOR` its
/// position, unanswered; an image past the largest, or a hot spot outside it, is refused
#[test]
fn the_cursor_reaches_the_screen() {
    let mut rig = Rig::new();
    rig.start();
    let offset = RGFX_ALLOCATIONS_OFFSET + 0x1000;
    rig.region()
        .write(offset, &[0, 0, 128, 128, 10, 20, 30, 255]);
    let cursor = |width, height, hot_x, hot_y| {
        let c = RgfxSetCursor {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SET_CURSOR,
                ..Default::default()
            },
            display: 0,
            width,
            height,
            hot_x,
            hot_y,
            reserved_24: 0,
            offset,
        };
        c.as_bytes().to_vec()
    };
    let s = rig.send(cursor(2, 1, 1, 0));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    let moved = RgfxMoveCursor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_MOVE_CURSOR,
            ..Default::default()
        },
        display: 0,
        x: -3,
        y: 7,
        visible: 1,
    };
    rig.send(moved.as_bytes().to_vec());
    for bad in [cursor(257, 1, 0, 0), cursor(2, 1, 2, 0), cursor(2, 1, 0, 1)] {
        let s = rig.send(bad);
        assert_eq!(rig.complete(), (s, RGFX_STATUS_INVALID_ARGUMENT));
    }
    let s = rig.send(cursor(0, 0, 0, 0));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    assert_eq!(
        rig.told(),
        [
            Told::Cursor(Some(CursorImage {
                width: 2,
                height: 1,
                hot_spot: (1, 0),
                rgba: vec![255, 0, 0, 128, 30, 20, 10, 255],
            })),
            Told::Moved(-3, 7, true),
            Told::Cursor(None),
        ]
    );
}

/// A type the device has no command for, and a message shorter than its command, are answered
/// with their statuses
#[test]
fn unknown_and_short_commands_are_answered() {
    let mut rig = Rig::new();
    rig.start();
    let unknown = RgfxMsgHeader {
        r#type: 0x77,
        ..Default::default()
    };
    let s = rig.send(unknown.as_bytes().to_vec());
    assert_eq!(rig.complete(), (s, RGFX_STATUS_UNKNOWN_COMMAND));
    let short = RgfxMsgHeader {
        r#type: RGFX_CMD_SET_SCANOUT,
        ..Default::default()
    };
    let s = rig.send(short.as_bytes().to_vec());
    assert_eq!(rig.complete(), (s, RGFX_STATUS_INVALID_ARGUMENT));
}

/// A refresh shows the scanout where the host's log says the guest wrote it, asking for the
/// scanout's range; `VBLANK` carries the refresh count while it is on; the host is told of each
/// refresh
#[test]
fn a_refresh_presents_written_pixels_and_sends_vblank() {
    let mut rig = Rig::new();
    rig.start();
    let alloc = RGFX_ALLOCATIONS_OFFSET;
    rig.send(scanout(16, 4, 128, alloc));
    rig.complete();
    rig.told();
    rig.refresh();
    assert_eq!(rig.told(), []);
    assert_eq!(*rig.asked.lock(), [(alloc, 128 * 3 + 64)]);
    assert_eq!(rig.refreshes.load(Ordering::SeqCst), 1);
    rig.written.store(true, Ordering::SeqCst);
    rig.refresh();
    assert_eq!(rig.told(), [Told::Present]);
    let s = rig.send(vblank(1));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    rig.refresh();
    let bytes = rig.event().unwrap();
    let v = RgfxVblank::read_from_bytes(&bytes).unwrap();
    assert_eq!(
        (v.header.r#type, v.header.size, v.header.seqno, v.count),
        (RGFX_EVT_VBLANK, 40, 0, 3)
    );
    assert_eq!(v.host_time_ns, 1234);
    rig.send(vblank(0));
    rig.complete();
    rig.refresh();
    assert!(rig.event().is_none());
    assert_eq!(rig.refreshes.load(Ordering::SeqCst), 4);
}

/// Once the guest has been told the display's mode, a change of the screen's mode is told
/// once with `DISPLAY_CHANGED`, and `GET_DISPLAYS` then answers the new one
#[test]
fn a_new_mode_is_told_with_display_changed() {
    let mut rig = Rig::new();
    rig.start();
    *rig.mode.lock() = Mode { width: 800, ..MODE };
    rig.refresh();
    assert!(rig.event().is_none(), "the guest was told no mode yet");
    rig.send(get_displays());
    rig.event().unwrap();
    *rig.mode.lock() = MODE;
    rig.refresh();
    let bytes = rig.event().unwrap();
    let c = RgfxDisplayChanged::read_from_bytes(&bytes).unwrap();
    assert_eq!(
        (c.header.r#type, c.header.size, c.display),
        (RGFX_EVT_DISPLAY_CHANGED, 24, 0)
    );
    rig.refresh();
    assert!(rig.event().is_none(), "told once");
    rig.send(get_displays());
    let bytes = rig.event().unwrap();
    assert_eq!(u32::from_le_bytes(bytes[36..40].try_into().unwrap()), 1280);
}

/// A reset stops the device, puts both rings back as new and the display with no scanout,
/// cursor or `VBLANK`, and the guest starts again from its first command
#[test]
fn a_reset_puts_the_device_back_as_new() {
    let mut rig = Rig::new();
    rig.start();
    rig.send(scanout(16, 4, 64, RGFX_ALLOCATIONS_OFFSET));
    rig.complete();
    rig.send(vblank(1));
    rig.complete();
    rig.told();
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    assert_eq!(
        rig.told(),
        [
            Told::Scanout(None),
            Told::Cursor(None),
            Told::Moved(0, 0, true)
        ]
    );
    for header in [RGFX_CONTROL_SUBMIT_OFFSET, RGFX_CONTROL_COMPLETE_OFFSET] {
        assert_eq!(rig.region().atomic_u64(header).load(Ordering::SeqCst), 0);
        assert_eq!(
            rig.region()
                .atomic_u64(header + 0x40)
                .load(Ordering::SeqCst),
            0
        );
    }
    let idle = rig
        .region()
        .atomic_u32(RGFX_CONTROL_SUBMIT_OFFSET + 0x80)
        .load(Ordering::SeqCst);
    assert_eq!(idle, 1, "the guest rings for its first message");
    assert_eq!(rig.device.shared.display.lock().scanout, None);
    rig.refresh();
    (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
    rig.start();
    let s = rig.send(get_displays());
    assert_eq!(s, 1);
    let bytes = rig.event().unwrap();
    assert_eq!(
        RgfxMsgHeader::read_from_prefix(&bytes).unwrap().0.r#type,
        RGFX_EVT_DISPLAYS
    );
    assert!(rig.event().is_none(), "no VBLANK after the reset");
}

/// The host's doorbell file wakes the control thread as a `DOORBELL` write does
#[test]
fn the_host_s_doorbell_file_wakes_the_device() {
    let mut rig = Rig::new();
    let bell = Wake::new().unwrap();
    assert!(rig.device.set_doorbell(bell.read_fd()));
    assert!(
        !rig.device.set_doorbell(bell.read_fd()),
        "one doorbell file"
    );
    assert!(!rig.device.set_doorbell(-1));
    rig.start();
    // Let the control thread go to sleep on the empty ring.
    std::thread::sleep(Duration::from_millis(50));
    rig.seqno += 1;
    let mut bytes = get_displays();
    bytes[8..16].copy_from_slice(&rig.seqno.to_le_bytes());
    set_size_of(&mut bytes);
    rig.publish(&bytes);
    bell.wake();
    assert!(rig.event().is_some());
}

/// A message size the protocol refuses breaks the ring: nothing more is taken until a reset
#[test]
fn a_broken_ring_is_left_until_a_reset() {
    let mut rig = Rig::new();
    rig.start();
    let mut bytes = get_displays();
    bytes.extend_from_slice(&[0; 4]);
    rig.seqno += 1;
    bytes[8..16].copy_from_slice(&rig.seqno.to_le_bytes());
    set_size_of(&mut bytes);
    rig.publish(&bytes);
    rig.device
        .mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    assert!(rig.event().is_none());
    rig.send(get_displays());
    assert!(rig.event().is_none(), "still broken");
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
    rig.start();
    rig.send(get_displays());
    assert!(rig.event().is_some());
}

/// While the completion ring is full the device waits, taking no command, and answers the rest
/// in order once the guest takes events
#[test]
fn a_full_completion_ring_holds_the_commands_back() {
    let mut rig = Rig::new();
    rig.start();
    // 24-byte answers: the 32 KiB ring holds 1365 of them.
    let count = 1500u64;
    for _ in 0..count {
        rig.send(vblank(0));
    }
    std::thread::sleep(Duration::from_millis(100));
    let produced = rig
        .region()
        .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET)
        .load(Ordering::SeqCst);
    assert!(produced <= u64::from(RGFX_CONTROL_RING_DATA_SIZE));
    for seqno in 1..=count {
        assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    }
}

/// A post waiting for room on ring 0's full completion ring, driven as the guest's kext drives
/// ring 0, looks again at its poll and stops waiting at a reset: the kext's work loop takes the
/// answers and rings nothing, and its posts ring the doorbell only where `consumer_idle` reads 1,
/// which it never does while the control thread, the waiting poster, is awake. The control thread,
/// holding back the commands behind an answer the full ring has no room for, answers each once the
/// guest has taken an event and the poll comes; with the poll long, waiting at a reset, it answers
/// the guest's first command after it
#[test]
fn a_post_waiting_for_room_looks_again_at_its_poll_and_stops_at_a_reset() {
    let mut rig = Rig::new();
    rig.start();
    // 24-byte answers: the 32 KiB ring holds 1365 of them.
    let count = 1400u64;
    // Full: less room left than one answer's 24 bytes.
    let full = |rig: &Rig| {
        let posted = rig
            .region()
            .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET)
            .load(Ordering::SeqCst);
        posted - rig.consumed + 24 > RGFX_CONTROL_RING_DATA_SIZE as u64
    };
    for _ in 0..count {
        rig.send(vblank(0));
    }
    // The host has answered what the ring holds and waits for room for the next answer.
    std::thread::sleep(Duration::from_millis(100));
    assert!(full(&rig), "the ring never filled");
    for seqno in 1..=count {
        assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    }

    // Full again, the poll long, then a reset: the waiting answer is dropped, and the next command
    // is answered.
    rig.device
        .shared
        .room_poll_ns
        .store(60_000_000_000, Ordering::Relaxed);
    for _ in 0..count {
        rig.send(vblank(0));
    }
    std::thread::sleep(Duration::from_millis(100));
    assert!(full(&rig), "the ring never filled again");
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
    rig.start();
    let seqno = rig.send(vblank(0));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
}

/// A host that records the vectors it raises and the types of the stage-3 messages it is handed,
/// and posts what a decoder would: a command buffer's completion through the queue's poster, and a
/// render pipeline's reflection through ring 0's. A buffer create waits while `gate` is shut: a
/// stand-in for the stage-3 decoder, so the device's rings can be exercised on their own.
struct QueueTestHost {
    vectors: Arc<Mutex<Vec<u32>>>,
    handed: Arc<Mutex<Vec<u32>>>,
    buffers: Arc<Mutex<Vec<Vec<u8>>>>,
    closed: Arc<Mutex<Vec<u32>>>,
    gate: Arc<AtomicBool>,
}

/// What a [`QueueTestHost`] records (the command buffers' whole messages among it), and its gate,
/// open.
#[derive(Default)]
struct QueueTestSeen {
    vectors: Arc<Mutex<Vec<u32>>>,
    handed: Arc<Mutex<Vec<u32>>>,
    buffers: Arc<Mutex<Vec<Vec<u8>>>>,
    closed: Arc<Mutex<Vec<u32>>>,
    gate: Arc<AtomicBool>,
}

impl QueueTestSeen {
    fn new() -> QueueTestSeen {
        let seen = QueueTestSeen::default();
        seen.gate.store(true, Ordering::SeqCst);
        seen
    }

    fn host(&self) -> Box<dyn Host> {
        Box::new(QueueTestHost {
            vectors: Arc::clone(&self.vectors),
            handed: Arc::clone(&self.handed),
            buffers: Arc::clone(&self.buffers),
            closed: Arc::clone(&self.closed),
            gate: Arc::clone(&self.gate),
        })
    }

    /// Waits about 2 s for `done` to hold.
    fn wait(&self, what: &str, done: impl Fn(&QueueTestSeen) -> bool) {
        for _ in 0..2000 {
            if done(self) {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("{what}");
    }
}

impl Host for QueueTestHost {
    fn raise_interrupt(&self, vector: u32) {
        self.vectors.lock().push(vector);
    }
    fn scanout_written(&self, _offset: u64, _length: u64) -> bool {
        false
    }
    fn context_closed(&self, context: u32, answer: crate::device::CompleteAnswer) {
        self.closed.lock().push(context);
        answer.answer(RGFX_STATUS_OK);
    }
    fn surface_download(&self, _block_offset: u64, answer: crate::device::CompleteAnswer) {
        self.handed.lock().push(RGFX_CMD_SURFACE_DOWNLOAD);
        answer.answer(RGFX_STATUS_OK);
    }
    fn stage3_message(&self, completion: &crate::device::CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        self.handed.lock().push(header.r#type);
        match header.r#type {
            RGFX_CMD_BUFFER_CREATE => {
                while !self.gate.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            RGFX_CMD_RENDER_PIPELINE_CREATE => {
                let mut reflection = RgfxPipelineReflection::new_zeroed();
                reflection.header = RgfxMsgHeader {
                    r#type: RGFX_EVT_PIPELINE_REFLECTION,
                    size: size_of::<RgfxPipelineReflection>() as u32,
                    seqno: header.seqno,
                };
                completion.post(reflection.as_bytes());
            }
            RGFX_CMD_COMMAND_BUFFER => {
                self.buffers.lock().push(message.to_vec());
                let (cb, _) = crate::ops::RgfxCommandBuffer::read_from_prefix(message).unwrap();
                let done = RgfxCbComplete {
                    header: RgfxMsgHeader {
                        r#type: RGFX_EVT_CB_COMPLETE,
                        size: size_of::<RgfxCbComplete>() as u32,
                        seqno: header.seqno,
                    },
                    cb_obj_id: cb.cb_obj_id,
                    gpu_start_ns: 0,
                    gpu_end_ns: 0,
                    status: RGFX_CB_OK,
                    error_code: 0,
                };
                completion.post(done.as_bytes());
            }
            _ => {}
        }
    }
}

/// Where the tests' queue rings are, and their size.
const QUEUE_SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
const QUEUE_COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
const QUEUE_RING_DATA: u32 = 0x1_0000;

/// A `QUEUE_CREATE` of queue `obj_id` over the tests' rings, raising MSI-X vector 3.
fn queue_create(obj_id: u64) -> Vec<u8> {
    RgfxQueueCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_QUEUE_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        submit_offset: QUEUE_SUBMIT,
        complete_offset: QUEUE_COMPLETE,
        submit_data_size: QUEUE_RING_DATA,
        complete_data_size: QUEUE_RING_DATA,
        priority: 2,
        vector: 3,
    }
    .as_bytes()
    .to_vec()
}

/// Sends the `QUEUE_CREATE` of queue `obj_id` over the tests' rings as the guest's kext does: each
/// ring's block granted to the queue's process first. The create's seqno.
fn send_queue_create(rig: &mut Rig, obj_id: u64) -> u64 {
    let context = (obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32;
    let span = u64::from(RGFX_RING_HEADER_SIZE + QUEUE_RING_DATA);
    for ring in [QUEUE_SUBMIT, QUEUE_COMPLETE] {
        rig.send(block_message(RGFX_CMD_BLOCK_GRANT, context, ring, span));
    }
    rig.send(queue_create(obj_id))
}

/// A ring-0 message of `type` whose payload is `obj_id`: an `OBJECT_DESTROY`, or a create the
/// test host does not read.
fn object_message(r#type: u32, obj_id: u64) -> Vec<u8> {
    RgfxObjectDestroy {
        header: RgfxMsgHeader {
            r#type,
            size: 0,
            seqno: 0,
        },
        obj_id,
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

/// Publishes an empty `COMMAND_BUFFER` `cb_obj_id` of queue `queue` at the start of the tests'
/// queue submission ring, with seqno `seqno`, ringing its doorbell where `doorbell`.
fn publish_command_buffer(rig: &Rig, queue: u64, cb_obj_id: u64, seqno: u64, doorbell: bool) {
    use crate::ops::{RGFX_COMMAND_BUFFER_OPS_OFFSET, RgfxCommandBuffer};
    let cb = RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: RGFX_COMMAND_BUFFER_OPS_OFFSET,
            seqno,
        },
        queue_obj_id: queue,
        cb_obj_id,
        op_bytes: 0,
    };
    let bytes = cb.as_bytes();
    rig.region().write(QUEUE_SUBMIT + 256, bytes);
    rig.region()
        .atomic_u64(QUEUE_SUBMIT)
        .store(bytes.len() as u64, Ordering::SeqCst);
    if doorbell {
        rig.device
            .mmio_write(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    }
}

/// The first message of the tests' queue completion ring, waiting about 2 s.
fn first_queue_completion(rig: &Rig) -> Option<RgfxCbComplete> {
    for _ in 0..2000 {
        if rig
            .region()
            .atomic_u64(QUEUE_COMPLETE)
            .load(Ordering::SeqCst)
            != 0
        {
            let header = rig.region().read(QUEUE_COMPLETE + 256, 8).unwrap();
            let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
            let msg = rig.region().read(QUEUE_COMPLETE + 256, size).unwrap();
            rig.region()
                .atomic_u64(QUEUE_COMPLETE + 0x40)
                .store(size as u64, Ordering::SeqCst);
            return Some(RgfxCbComplete::read_from_bytes(&msg).unwrap());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    None
}

/// A `COMMAND_BUFFER_IN_REGION` naming `byte_length` bytes at `region_offset`.
fn command_buffer_in_region(region_offset: u64, byte_length: u64) -> Vec<u8> {
    RgfxCommandBufferInRegion {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER_IN_REGION,
            size: size_of::<RgfxCommandBufferInRegion>() as u32,
            seqno: 0,
        },
        region_offset,
        byte_length,
    }
    .as_bytes()
    .to_vec()
}

/// Publishes `message` at the tests' queue submission ring's count `at` with seqno `seqno`, and
/// rings its doorbell; returns the count past it.
fn publish_queue_message(rig: &Rig, at: u64, seqno: u64, mut message: Vec<u8>) -> u64 {
    message[8..16].copy_from_slice(&seqno.to_le_bytes());
    rig.region().write(QUEUE_SUBMIT + 256 + at, &message);
    let end = at + message.len() as u64;
    rig.region()
        .atomic_u64(QUEUE_SUBMIT)
        .store(end, Ordering::SeqCst);
    rig.device
        .mmio_write(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    end
}

/// Waits about 2 s for the tests' queue submission ring to go idle: its thread sleeps, so the
/// guest rings for its next message.
fn wait_queue_idle(rig: &Rig) {
    for _ in 0..2000 {
        if rig
            .region()
            .atomic_u32(QUEUE_SUBMIT + 0x80)
            .load(Ordering::SeqCst)
            == 1
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("the queue's submission ring never went idle");
}

/// A command buffer the guest publishes on a queue's submission ring before the queue is created,
/// with no doorbell since the ring's `consumer_idle` was still the 0 the guest zeroed it to, is
/// still taken and completed: the create keeps the guest's producer count.
#[test]
fn a_command_buffer_published_before_its_queue_create_completes() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (3 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;

    // The guest's first command buffer, published on the queue's zeroed submission ring with no
    // doorbell; then the queue's create.
    publish_command_buffer(&rig, QUEUE, 100, 1, false);
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));

    // The command buffer completes on the queue's completion ring.
    let done = first_queue_completion(&rig)
        .expect("the command buffer published before its queue's create completed");
    assert_eq!(done.header.r#type, RGFX_EVT_CB_COMPLETE);
    assert_eq!(done.cb_obj_id, 100);
    // The queue's create, then the buffer.
    assert_eq!(
        *seen.handed.lock(),
        vec![RGFX_CMD_QUEUE_CREATE, RGFX_CMD_COMMAND_BUFFER]
    );
}

/// A `QUEUE_CREATE` of a context the host refuses (its work failed off its op's thread) is
/// refused and makes no rings; another context's is made
#[test]
fn a_refused_context_s_queue_create_makes_no_rings() {
    struct Refusing;
    impl Host for Refusing {
        fn raise_interrupt(&self, _vector: u32) {}
        fn scanout_written(&self, _offset: u64, _length: u64) -> bool {
            false
        }
        fn context_refused(&self, context: u32) -> bool {
            context == 7
        }
    }
    let mut rig = Rig::with_host(Box::new(Refusing));
    rig.start();
    let seqno = send_queue_create(&mut rig, (7 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_REFUSED));
    assert_eq!(rig.device.queue_count(), 0);
    let seqno = send_queue_create(&mut rig, (8 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(rig.device.queue_count(), 1);
}

/// A `QUEUE_CREATE` is refused, making no rings, where either ring lies whole in no block its
/// process holds: in no block, in another process's, or across two of its own; with each ring in
/// one of its own blocks, it is made
#[test]
fn a_queue_s_rings_lie_in_its_process_s_blocks() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let span = u64::from(RGFX_RING_HEADER_SIZE + QUEUE_RING_DATA);
    let refused = |rig: &mut Rig, what: &str| {
        let seqno = rig.send(queue_create(QUEUE));
        assert_eq!(
            rig.complete(),
            (seqno, RGFX_STATUS_INVALID_ARGUMENT),
            "{what}"
        );
        assert_eq!(rig.device.queue_count(), 0, "{what}");
    };
    refused(&mut rig, "no block");
    // Another process's blocks.
    for ring in [QUEUE_SUBMIT, QUEUE_COMPLETE] {
        rig.send(block_message(RGFX_CMD_BLOCK_GRANT, 3, ring, span));
    }
    refused(&mut rig, "another process's blocks");
    // Its own submission ring, and another's completion ring.
    rig.send(block_message(RGFX_CMD_BLOCK_GRANT, 5, QUEUE_SUBMIT, span));
    refused(&mut rig, "the completion ring in another's block");
    // Its completion ring across two blocks of its own.
    rig.send(block_message(
        RGFX_CMD_BLOCK_GRANT,
        5,
        QUEUE_COMPLETE,
        span / 2,
    ));
    rig.send(block_message(
        RGFX_CMD_BLOCK_GRANT,
        5,
        QUEUE_COMPLETE + span / 2,
        span / 2,
    ));
    refused(&mut rig, "a ring across two blocks");
    // Each in one of its own.
    rig.send(block_message(RGFX_CMD_BLOCK_GRANT, 5, QUEUE_COMPLETE, span));
    let seqno = rig.send(queue_create(QUEUE));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(rig.device.queue_count(), 1);
    // The queue's service thread hands the host its create, which may come after the answer.
    seen.wait("the queue's create was never handed to the host", |seen| {
        !seen.handed.lock().is_empty()
    });
    assert_eq!(*seen.handed.lock().first().unwrap(), RGFX_CMD_QUEUE_CREATE);
}

/// A `QUEUE_CREATE` on ring 0 with valid ring parameters makes a command queue's own submission and
/// completion rings and a service thread before it is answered, and the host is handed the
/// queue's `QUEUE_CREATE` first; a `COMMAND_BUFFER` published on the queue's submission ring
/// reaches [`Host::stage3_message`], whose completion lands on the queue's own completion ring and
/// raises that queue's MSI-X vector; a `QUEUE_CREATE` with no ring parameters is refused; a
/// pipeline create's reflection comes back on ring 0's completion ring; and an `OBJECT_DESTROY` of
/// the queue and a `CONTEXT_CLOSE` end its service, the close answered once the host has dropped
/// the context's objects.
#[test]
fn a_command_queue_gets_its_own_ring_loop() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (7 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;

    // A QUEUE_CREATE with no ring parameters is refused.
    let seqno = rig.send(
        object_message(RGFX_CMD_QUEUE_CREATE, QUEUE)
            .into_iter()
            .chain([0; 32])
            .collect(),
    );
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_INVALID_ARGUMENT));
    assert_eq!(rig.device.queue_count(), 0);

    // A QUEUE_CREATE with valid rings in the guest's allocations and MSI-X vector 3: the queue is
    // there when it is answered.
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(rig.device.queue_count(), 1);
    wait_queue_idle(&rig);

    // Arm the queue's completion ring and publish a COMMAND_BUFFER on its submission ring.
    rig.region()
        .atomic_u32(QUEUE_COMPLETE + 0xc0)
        .store(1, Ordering::SeqCst);
    publish_command_buffer(&rig, QUEUE, 100, 55, true);

    // The completion lands on the queue's completion ring, with the submission seqno.
    let done = first_queue_completion(&rig).expect("a CB_COMPLETE on the queue ring");
    assert_eq!(done.header.r#type, RGFX_EVT_CB_COMPLETE);
    assert_eq!(done.header.seqno, 55);
    assert_eq!(done.cb_obj_id, 100);
    // The queue's create, then the buffer.
    assert_eq!(
        *seen.handed.lock(),
        vec![RGFX_CMD_QUEUE_CREATE, RGFX_CMD_COMMAND_BUFFER]
    );
    // The queue's own vector was raised, not the control vector.
    assert_eq!(*seen.vectors.lock(), vec![3]);

    // A pipeline create's reflection comes back on ring 0's completion ring, with its seqno.
    let seqno = rig.send(object_message(RGFX_CMD_RENDER_PIPELINE_CREATE, QUEUE + 1));
    let reflection = rig.event().expect("a reflection on ring 0");
    let (header, _) = RgfxMsgHeader::read_from_prefix(&reflection).unwrap();
    assert_eq!(header.r#type, RGFX_EVT_PIPELINE_REFLECTION);
    assert_eq!(header.seqno, seqno);

    // An OBJECT_DESTROY of the queue ends its service, and the host drops its handle.
    rig.send(object_message(RGFX_CMD_OBJECT_DESTROY, QUEUE));
    seen.wait("the queue's destroy never reached the host", |seen| {
        seen.handed.lock().last() == Some(&RGFX_CMD_OBJECT_DESTROY)
    });
    assert_eq!(rig.device.queue_count(), 0);

    // CONTEXT_CLOSE ends the context's queues and has the host drop its objects, then is answered.
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(rig.device.queue_count(), 1);
    let seqno = rig.send(context_close(7));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(rig.device.queue_count(), 0);
    assert_eq!(*seen.closed.lock(), vec![7]);
}

/// A `COMMAND_BUFFER_IN_REGION` hands the host the whole `COMMAND_BUFFER` message in the block it
/// names, with the reference's seqno, and the completion names that seqno; a reference to no
/// block its process holds (none granted, another process's, its own once released), or to bytes
/// that are not a `COMMAND_BUFFER` of its length, is refused, and the ring goes on.
#[test]
fn a_command_buffer_in_region_is_taken_from_its_block() {
    use crate::ops::{RGFX_COMMAND_BUFFER_OPS_OFFSET, RgfxCommandBuffer};
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);

    // The buffer's whole message in a block past the rings: its header, then 64 bytes of ops.
    // The kext granted the block to the queue's context, and another to context 6.
    const BLOCK: u64 = QUEUE_COMPLETE + 0x10_0000;
    const OTHER: u64 = BLOCK + 0x10_0000;
    rig.send(block_message(RGFX_CMD_BLOCK_GRANT, 5, BLOCK, 0x1_0000));
    rig.send(block_message(RGFX_CMD_BLOCK_GRANT, 6, OTHER, 0x1_0000));
    let mut message = RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: RGFX_COMMAND_BUFFER_OPS_OFFSET + 64,
            seqno: 0,
        },
        queue_obj_id: QUEUE,
        cb_obj_id: 200,
        op_bytes: 64,
    }
    .as_bytes()
    .to_vec();
    message.extend((0..64u8).map(|i| i ^ 0x5a));
    rig.region().write(BLOCK, &message);
    rig.region().write(OTHER, &message);
    let len = message.len() as u64;

    // Refused: in no granted block, a length past the message, bytes that are not a
    // COMMAND_BUFFER, and another process's block. Then the good one, taken with the reference's
    // seqno.
    let mut at = publish_queue_message(&rig, 0, 70, command_buffer_in_region(0x1000, len));
    at = publish_queue_message(&rig, at, 71, command_buffer_in_region(BLOCK, len + 8));
    at = publish_queue_message(&rig, at, 72, command_buffer_in_region(BLOCK + 8, len - 8));
    at = publish_queue_message(&rig, at, 73, command_buffer_in_region(OTHER, len));
    at = publish_queue_message(&rig, at, 77, command_buffer_in_region(BLOCK, len));

    let done = first_queue_completion(&rig).expect("a CB_COMPLETE for the buffer in its block");
    assert_eq!(done.header.seqno, 77);
    assert_eq!(done.cb_obj_id, 200);
    let mut want = message.clone();
    want[8..16].copy_from_slice(&77u64.to_le_bytes());
    assert_eq!(*seen.buffers.lock(), vec![want.clone()]);

    // Released, the block is no longer the process's to name.
    let seqno = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, 5, BLOCK, 0x1_0000));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    publish_queue_message(&rig, at, 78, command_buffer_in_region(BLOCK, len));
    std::thread::sleep(Duration::from_millis(50));
    wait_queue_idle(&rig);
    assert_eq!(*seen.buffers.lock(), vec![want]);
    assert_eq!(
        *seen.handed.lock(),
        vec![RGFX_CMD_QUEUE_CREATE, RGFX_CMD_COMMAND_BUFFER]
    );
}

/// A `BLOCK_GRANT` is kept where it lies in the processes' blocks, or in the scanout's span (the
/// kext grants it to the context that draws the display into it), and ignored where it reaches
/// the device info before the scanout, crosses from the scanout into the blocks, or reaches the
/// cursor's span. A grant of a span another context holds takes it from that context.
#[test]
fn a_block_grant_is_kept_in_the_blocks_or_the_scanout() {
    let mut rig = Rig::new();
    rig.start();
    const SCANOUT: u64 = RGFX_ALLOCATIONS_OFFSET;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let cursor = REGION_SIZE - RGFX_CURSOR_SPAN;
    let grants = [
        (5, SCANOUT, 0x87_0000),
        (6, BLOCK, 0x1_0000),
        (7, SCANOUT - 0x1000, 0x2000),
        (8, RGFX_CONTEXT_ALLOCATIONS_OFFSET - 0x1000, 0x2000),
        (9, cursor - 0x1000, 0x2000),
    ];
    for (context, offset, size) in grants {
        rig.send(block_message(RGFX_CMD_BLOCK_GRANT, context, offset, size));
    }
    // A grant is not answered: the close after them is, once the ring has taken them.
    let seqno = rig.send(context_close(99));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));

    let region = rig.region();
    assert!(region.held_by(5, SCANOUT, 0x87_0000));
    assert!(region.held_by(5, SCANOUT + 0x1000, 0x8000));
    assert!(!region.held_by(6, SCANOUT, 0x1000));
    assert!(region.held_by(6, BLOCK, 0x1_0000));
    assert!(!region.held_by(7, SCANOUT - 0x1000, 0x1000));
    assert!(!region.held_by(8, RGFX_CONTEXT_ALLOCATIONS_OFFSET - 0x1000, 0x1000));
    assert!(!region.held_by(8, RGFX_CONTEXT_ALLOCATIONS_OFFSET, 0x1000));
    assert!(!region.held_by(9, cursor - 0x1000, 0x1000));

    // A new process drawing the display: the scanout is its context's, no longer the old one's.
    rig.send(block_message(RGFX_CMD_BLOCK_GRANT, 10, SCANOUT, 0x87_0000));
    let seqno = rig.send(context_close(99));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert!(rig.region().held_by(10, SCANOUT, 0x87_0000));
    assert!(!rig.region().held_by(5, SCANOUT, 0x1000));

    // Its context's close ends its grant.
    let seqno = rig.send(context_close(10));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert!(!rig.region().held_by(10, SCANOUT, 0x1000));
}

/// A `BLOCK_SHARE` makes each context it names a holder of the shared block, in the processes'
/// blocks only; a `BLOCK_UNSHARE` ends one context's hold and leaves the others', and the block's
/// `BLOCK_RELEASE` ends every hold.
#[test]
fn a_shared_block_is_held_by_each_context_it_is_shared_with() {
    let mut rig = Rig::new();
    rig.start();
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x40_0000;
    const SCANOUT: u64 = RGFX_ALLOCATIONS_OFFSET;
    for context in [5, 6] {
        rig.send(block_message(
            RGFX_CMD_BLOCK_SHARE,
            context,
            BLOCK,
            0x2_0000,
        ));
    }
    rig.send(block_message(RGFX_CMD_BLOCK_SHARE, 7, SCANOUT, 0x1000));
    let seqno = rig.send(context_close(99));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    let region = rig.region();
    assert!(region.held_by(5, BLOCK, 0x2_0000));
    assert!(region.held_by(6, BLOCK + 0x1_0000, 0x1_0000));
    assert!(!region.held_by(7, SCANOUT, 0x1000));

    rig.send(block_message(RGFX_CMD_BLOCK_UNSHARE, 5, BLOCK, 0x2_0000));
    let seqno = rig.send(context_close(99));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert!(!rig.region().held_by(5, BLOCK, 0x1000));
    assert!(rig.region().held_by(6, BLOCK, 0x1000));

    let seqno = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, 6, BLOCK, 0x2_0000));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert!(!rig.region().held_by(6, BLOCK, 0x1000));
}

/// A `BLOCK_GRANT`, `BLOCK_SHARE`, `BLOCK_UNSHARE` or `BLOCK_RELEASE` (their layouts are one) of
/// `type` for `context`'s block of `size` bytes at `offset`.
fn block_message(r#type: u32, context: u32, offset: u64, size: u64) -> Vec<u8> {
    RgfxBlockGrant {
        header: RgfxMsgHeader {
            r#type,
            size: size_of::<RgfxBlockGrant>() as u32,
            seqno: 0,
        },
        context,
        reserved_14: 0,
        offset,
        size,
    }
    .as_bytes()
    .to_vec()
}

/// A command buffer is decoded only once the creates published on ring 0 before it are carried
/// out, though the control thread is still in one when the buffer's queue takes it.
#[test]
fn a_command_buffer_waits_for_the_creates_before_it() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);

    // A buffer create the host is slow to carry out, then a command buffer that uses it.
    seen.gate.store(false, Ordering::SeqCst);
    rig.send(object_message(RGFX_CMD_BUFFER_CREATE, QUEUE + 1));
    seen.wait("the buffer create never reached the host", |seen| {
        seen.handed.lock().last() == Some(&RGFX_CMD_BUFFER_CREATE)
    });
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        *seen.handed.lock(),
        vec![RGFX_CMD_QUEUE_CREATE, RGFX_CMD_BUFFER_CREATE],
        "the command buffer ran before the create it follows was carried out"
    );

    // Once the create is carried out, the buffer runs.
    seen.gate.store(true, Ordering::SeqCst);
    let done = first_queue_completion(&rig).expect("the command buffer completed");
    assert_eq!(done.cb_obj_id, 100);
    assert_eq!(
        *seen.handed.lock(),
        vec![
            RGFX_CMD_QUEUE_CREATE,
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_COMMAND_BUFFER
        ]
    );
}

/// A command buffer waits for no ring-0 message of another process's: it is decoded while the
/// control thread is still in another process's create published before it.
#[test]
fn a_command_buffer_waits_for_no_other_process_s_creates() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    const OTHER: u64 = (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);

    seen.gate.store(false, Ordering::SeqCst);
    rig.send(object_message(RGFX_CMD_BUFFER_CREATE, OTHER));
    seen.wait("the buffer create never reached the host", |seen| {
        seen.handed.lock().last() == Some(&RGFX_CMD_BUFFER_CREATE)
    });
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    let done = first_queue_completion(&rig).expect("held up by another process's create");
    assert_eq!(done.cb_obj_id, 100);
    assert_eq!(
        *seen.handed.lock(),
        vec![
            RGFX_CMD_QUEUE_CREATE,
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_COMMAND_BUFFER
        ]
    );
    seen.gate.store(true, Ordering::SeqCst);
}

/// A command buffer waits for its own process's messages on ring 0, and so for every message
/// before them: its process's create behind another's the control thread is still in holds it up.
#[test]
fn a_command_buffer_waits_for_its_process_s_creates_behind_another_s() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    const OTHER: u64 = (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);

    seen.gate.store(false, Ordering::SeqCst);
    rig.send(object_message(RGFX_CMD_BUFFER_CREATE, OTHER));
    seen.wait("the buffer create never reached the host", |seen| {
        seen.handed.lock().last() == Some(&RGFX_CMD_BUFFER_CREATE)
    });
    rig.send(object_message(RGFX_CMD_SAMPLER_CREATE, QUEUE + 1));
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        *seen.handed.lock(),
        vec![RGFX_CMD_QUEUE_CREATE, RGFX_CMD_BUFFER_CREATE],
        "the command buffer ran before its process's create"
    );
    seen.gate.store(true, Ordering::SeqCst);
    let done = first_queue_completion(&rig).expect("the command buffer completed");
    assert_eq!(done.cb_obj_id, 100);
    assert_eq!(
        *seen.handed.lock(),
        vec![
            RGFX_CMD_QUEUE_CREATE,
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_SAMPLER_CREATE,
            RGFX_CMD_COMMAND_BUFFER
        ]
    );
}

/// A command buffer waits for no `SURFACE_DOWNLOAD`: one published after a download, behind another
/// process's create the control thread is still in, is handed to the host before the download.
/// So a buffer that writes the surface and is taken first is among the users the download waits
/// for, as one published before it is; only a guest that writes a surface while racing its CPU
/// lock of it can tell.
#[test]
fn a_command_buffer_is_handed_over_before_a_download_published_ahead_of_it() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    const OTHER: u64 = (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);

    seen.gate.store(false, Ordering::SeqCst);
    rig.send(object_message(RGFX_CMD_BUFFER_CREATE, OTHER));
    seen.wait("the buffer create never reached the host", |seen| {
        seen.handed.lock().last() == Some(&RGFX_CMD_BUFFER_CREATE)
    });
    rig.send(
        RgfxSurfaceDownload {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SURFACE_DOWNLOAD,
                size: 0,
                seqno: 0,
            },
            block_offset: RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000,
        }
        .as_bytes()
        .to_vec(),
    );
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    let done = first_queue_completion(&rig).expect("held up by the download");
    assert_eq!(done.cb_obj_id, 100);
    seen.gate.store(true, Ordering::SeqCst);
    seen.wait("the download never reached the host", |seen| {
        seen.handed.lock().last() == Some(&RGFX_CMD_SURFACE_DOWNLOAD)
    });
    assert_eq!(
        *seen.handed.lock(),
        vec![
            RGFX_CMD_QUEUE_CREATE,
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_COMMAND_BUFFER,
            RGFX_CMD_SURFACE_DOWNLOAD
        ]
    );
}

/// Process B learns of process A's surface over IPC after A's messages were posted, which may be
/// before ring 0 has carried them out: B's buffer waits for the share of the surface's block the
/// kext posts for B's context before B learns of it, and so for A's messages before the share.
#[test]
fn a_command_buffer_waits_for_another_process_s_messages_before_its_share() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const B: u32 = 5;
    const QUEUE: u64 = ((B as u64) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    const A_OBJECT: u64 = (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    const SURFACE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);

    seen.gate.store(false, Ordering::SeqCst);
    rig.send(object_message(RGFX_CMD_BUFFER_CREATE, A_OBJECT));
    seen.wait("A's create never reached the host", |seen| {
        seen.handed.lock().last() == Some(&RGFX_CMD_BUFFER_CREATE)
    });
    rig.send(block_message(RGFX_CMD_BLOCK_SHARE, B, SURFACE, 0x1000));
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        *seen.handed.lock(),
        vec![RGFX_CMD_QUEUE_CREATE, RGFX_CMD_BUFFER_CREATE],
        "B's command buffer ran before the share of A's surface"
    );
    seen.gate.store(true, Ordering::SeqCst);
    let done = first_queue_completion(&rig).expect("the command buffer completed");
    assert_eq!(done.cb_obj_id, 100);
    assert!(rig.region().held_by(B, SURFACE, 0x1000));
}

/// A reset wakes a command buffer's wait for the creates before it, though the control thread is
/// still in one: its queue's thread stops waiting and ends, handing the host nothing more of the
/// queue's, neither the buffer nor the queue's destroy, which the host's reset stands for.
#[test]
fn a_reset_wakes_a_command_buffer_waiting_for_the_creates_before_it() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);

    seen.gate.store(false, Ordering::SeqCst);
    rig.send(object_message(RGFX_CMD_BUFFER_CREATE, QUEUE + 1));
    seen.wait("the buffer create never reached the host", |seen| {
        seen.handed.lock().last() == Some(&RGFX_CMD_BUFFER_CREATE)
    });
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        *seen.handed.lock(),
        vec![RGFX_CMD_QUEUE_CREATE, RGFX_CMD_BUFFER_CREATE]
    );

    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    // The queue's thread holds the device's shared state while it lives; the device, its control
    // thread and its refresh thread hold the rest.
    for _ in 0..2000 {
        if Arc::strong_count(&rig.device.shared) == 3 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        Arc::strong_count(&rig.device.shared),
        3,
        "the queue's thread still waits for ring 0 past a reset"
    );
    seen.gate.store(true, Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        *seen.handed.lock(),
        vec![RGFX_CMD_QUEUE_CREATE, RGFX_CMD_BUFFER_CREATE]
    );
}

/// What a [`ResetTellingHost`] records: each message's type, or [`RESET_TOLD`], with its thread.
type TypesTold = Arc<Mutex<Vec<(u32, Option<String>)>>>;

/// A host that records, with each one's thread, the stage-3 messages it is handed and the
/// device's resets it is told of ([`RESET_TOLD`] in place of a type).
struct ResetTellingHost(TypesTold);

/// What a [`ResetTellingHost`] records for a reset.
const RESET_TOLD: u32 = u32::MAX;

impl Host for ResetTellingHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn scanout_written(&self, _offset: u64, _length: u64) -> bool {
        false
    }
    fn stage3_message(&self, _completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        self.0.lock().push((header.r#type, thread_name()));
    }
    fn device_reset(&self) {
        self.0.lock().push((RESET_TOLD, thread_name()));
    }
}

/// A reset has the host drop what it made ([`Host::device_reset`]) once, on the control thread,
/// before the thread hands it anything the guest published after the reset; the reset ends its
/// queues, which hand the host no destroy of theirs, and no block is any process's, the scanout's
/// span's among them.
#[test]
fn a_reset_is_told_to_the_host_before_the_messages_after_it() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut rig = Rig::with_host(Box::new(ResetTellingHost(Arc::clone(&seen))));
    rig.start();
    const QUEUE: u64 = (4 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    const SCANOUT: u64 = RGFX_ALLOCATIONS_OFFSET;
    rig.send(block_message(
        RGFX_CMD_BLOCK_GRANT,
        4,
        QUEUE_SUBMIT,
        QUEUE_COMPLETE + 0x2_0000 - QUEUE_SUBMIT,
    ));
    rig.send(block_message(RGFX_CMD_BLOCK_GRANT, 4, SCANOUT, 0x1000));
    let seqno = rig.send(queue_create(QUEUE));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);
    assert!(rig.region().held_by(4, QUEUE_SUBMIT, 0x1000));
    assert!(rig.region().held_by(4, SCANOUT, 0x1000));

    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    assert_eq!(rig.device.queue_count(), 0);
    assert!(!rig.region().held_by(4, QUEUE_SUBMIT, 0x1000));
    assert!(!rig.region().held_by(4, SCANOUT, 0x1000));
    for _ in 0..2000 {
        if seen.lock().iter().any(|&(t, _)| t == RESET_TOLD) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }

    // The guest starts again, from its first command.
    (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
    rig.start();
    rig.send(object_message(RGFX_CMD_EVENT_CREATE, QUEUE + 1));
    for _ in 0..2000 {
        if seen.lock().iter().any(|&(t, _)| t == RGFX_CMD_EVENT_CREATE) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(Duration::from_millis(50));
    let control = Some("reims-vgpu.graphics-control".to_owned());
    let queue = Some(format!("reims-vgpu.graphics-queue-{QUEUE:#x}"));
    assert_eq!(
        *seen.lock(),
        [
            (RGFX_CMD_QUEUE_CREATE, queue),
            (RESET_TOLD, control.clone()),
            (RGFX_CMD_EVENT_CREATE, control),
        ]
    );
}

/// The control completion ring takes posts from several threads at once (the control thread's
/// answers and the host's compile threads' replies): each lands whole, once, in its thread's order.
#[test]
fn control_completion_posts_from_several_threads_land_whole() {
    const THREADS: u64 = 8;
    const EACH: u64 = 100;
    let mut rig = Rig::new();
    rig.start();
    let posters: Vec<_> = (0..THREADS)
        .map(|t| {
            let poster = rig.device.shared.control_poster(0);
            let thread = std::thread::Builder::new().name(format!("poster-{t}"));
            thread
                .spawn(move || {
                    for i in 0..EACH {
                        let mut reflection = RgfxPipelineReflection::new_zeroed();
                        reflection.header = RgfxMsgHeader {
                            r#type: RGFX_EVT_PIPELINE_REFLECTION,
                            size: size_of::<RgfxPipelineReflection>() as u32,
                            seqno: t * EACH + i + 1,
                        };
                        reflection.obj_id = t;
                        poster.post(reflection.as_bytes());
                    }
                })
                .expect("a poster thread")
        })
        .collect();
    let mut next = [0u64; THREADS as usize];
    for _ in 0..THREADS * EACH {
        let bytes = rig.event().expect("a reply");
        let reflection = RgfxPipelineReflection::read_from_bytes(&bytes).unwrap();
        assert_eq!(reflection.header.r#type, RGFX_EVT_PIPELINE_REFLECTION);
        let t = reflection.obj_id;
        assert_eq!(reflection.header.seqno, t * EACH + next[t as usize] + 1);
        next[t as usize] += 1;
    }
    for poster in posters {
        poster.join().unwrap();
    }
    assert_eq!(next, [EACH; THREADS as usize]);
}

/// The device's feature bits are its host's; a `SURFACE_PAGES` hands the host the runs it lists
/// in its block, read out of the region, and whether the message marks the block cached, and is
/// answered with the host's answer; one whose runs lie past the block, or a block outside the
/// processes' blocks, hands nothing and is refused. A `SURFACE_DOWNLOAD` hands the host its block
/// and is answered with the host's answer
#[test]
fn surface_pages_hand_the_host_their_runs() {
    type Seen = Arc<Mutex<Vec<(RgfxSurfacePages, Vec<RgfxPageRun>, bool)>>>;
    struct Pages(Seen);
    impl Host for Pages {
        fn raise_interrupt(&self, _vector: u32) {}
        fn scanout_written(&self, _offset: u64, _length: u64) -> bool {
            false
        }
        fn features(&self) -> u64 {
            u64::from(RGFX_FEATURE_SURFACE_PAGES) | (1 << 40)
        }
        fn surface_pages(
            &self,
            pages: &RgfxSurfacePages,
            runs: Vec<RgfxPageRun>,
            cached: bool,
        ) -> u32 {
            self.0.lock().push((*pages, runs, cached));
            RGFX_STATUS_OK
        }
        fn surface_download(&self, block_offset: u64, answer: CompleteAnswer) {
            let runs = vec![];
            let pages = RgfxSurfacePages {
                block_offset,
                ..Default::default()
            };
            self.0.lock().push((pages, runs, true));
            answer.answer(RGFX_STATUS_NOT_FOUND);
        }
    }
    let seen: Seen = Arc::default();
    let mut rig = Rig::with_host(Box::new(Pages(Arc::clone(&seen))));
    assert_eq!(
        rig.device.mmio_read(RGFX_REG_FEATURES_LO.into()),
        RGFX_FEATURE_SURFACE_PAGES
    );
    assert_eq!(rig.device.mmio_read(RGFX_REG_FEATURES_HI.into()), 1 << 8);
    rig.start();

    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4000;
    let runs = [
        RgfxPageRun {
            address: 0x9000,
            length: 0x1000,
        },
        RgfxPageRun {
            address: 0x3000,
            length: 0x2000,
        },
    ];
    rig.region().write(BLOCK + 0x1000, runs.as_bytes());
    let pages = |block_offset: u64, runs_offset: u64, run_count: u32| RgfxSurfacePages {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SURFACE_PAGES,
            ..Default::default()
        },
        context: 3,
        run_count,
        block_offset,
        block_size: 0x2000,
        runs_offset,
        length: 0x3000,
    };
    let seqno = rig.send(pages(BLOCK, BLOCK + 0x1000, 2).as_bytes().to_vec());
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    // The same, marked cached.
    let mut cached = pages(BLOCK, BLOCK + 0x1000, 2).as_bytes().to_vec();
    cached.extend_from_slice(
        RgfxSurfacePagesCache {
            flags: RGFX_SURFACE_PAGES_CACHED,
            reserved_04: 0,
        }
        .as_bytes(),
    );
    let seqno = rig.send(cached);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    let seqno = rig.send(
        RgfxSurfaceDownload {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SURFACE_DOWNLOAD,
                ..Default::default()
            },
            block_offset: BLOCK,
        }
        .as_bytes()
        .to_vec(),
    );
    assert_eq!(
        rig.complete(),
        (seqno, RGFX_STATUS_NOT_FOUND),
        "the host's answer"
    );
    // Runs past the block, and a block outside the processes' blocks: refused, nothing handed.
    let seqno = rig.send(pages(BLOCK, BLOCK + 0x1ff0, 2).as_bytes().to_vec());
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_INVALID_ARGUMENT));
    let seqno = rig.send(
        pages(RGFX_ALLOCATIONS_OFFSET, RGFX_ALLOCATIONS_OFFSET, 0)
            .as_bytes()
            .to_vec(),
    );
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_INVALID_ARGUMENT));
    let seen = seen.lock();
    assert_eq!(
        seen.len(),
        3,
        "only the SURFACE_PAGES whose runs lie in its block, and the download"
    );
    assert_eq!(seen[0].0.block_offset, BLOCK);
    assert_eq!(seen[0].0.context, 3);
    assert_eq!(seen[0].1, runs);
    assert!(!seen[0].2, "a message without the cache's flags");
    assert_eq!((seen[1].1.as_slice(), seen[1].2), (&runs[..], true));
    assert_eq!((seen[2].0.block_offset, seen[2].1.len()), (BLOCK, 0));
}

/// What a [`TakingHost`] records, and what it is set to do.
#[derive(Default)]
struct Taking {
    /// Take command buffers on a doorbell's thread ([`Host::take_command_buffer`]).
    takes: AtomicBool,
    /// While set, a `BUFFER_CREATE` or a `COMMAND_BUFFER` handed to the host waits.
    shut: AtomicBool,
    /// Each raise: its vector, and the name of the thread it ran on.
    raised: Mutex<Vec<(u32, Option<String>)>>,
    /// The types of the messages handed to [`Host::stage3_message`], in order.
    handed: Mutex<Vec<u32>>,
    /// Each command buffer carried out, and the name of the thread it was carried out on.
    carried: Mutex<Vec<(u64, Option<String>)>>,
}

/// A host whose command buffers post a `CB_SUBMITTED` and then a `CB_COMPLETE`, as the decoder's
/// do, and which takes them on a doorbell's thread where it is set to: the `CB_SUBMITTED` posted
/// there, the rest carried out on the queue's thread.
struct TakingHost(Arc<Taking>);

/// The calling thread's name.
fn thread_name() -> Option<String> {
    std::thread::current().name().map(str::to_owned)
}

/// Command buffer `cb_obj_id`'s `CB_SUBMITTED` at `seqno`.
fn cb_submitted(seqno: u64, cb_obj_id: u64) -> RgfxCbSubmitted {
    RgfxCbSubmitted {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_CB_SUBMITTED,
            size: size_of::<RgfxCbSubmitted>() as u32,
            seqno,
        },
        cb_obj_id,
    }
}

/// Command buffer `cb_obj_id`'s `CB_COMPLETE` at `seqno`.
fn cb_complete(seqno: u64, cb_obj_id: u64) -> RgfxCbComplete {
    RgfxCbComplete {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_CB_COMPLETE,
            size: size_of::<RgfxCbComplete>() as u32,
            seqno,
        },
        cb_obj_id,
        gpu_start_ns: 0,
        gpu_end_ns: 0,
        status: RGFX_CB_OK,
        error_code: 0,
    }
}

impl Host for TakingHost {
    fn raise_interrupt(&self, vector: u32) {
        self.0.raised.lock().push((vector, thread_name()));
    }
    fn scanout_written(&self, _offset: u64, _length: u64) -> bool {
        false
    }
    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        self.0.handed.lock().push(header.r#type);
        if matches!(
            header.r#type,
            RGFX_CMD_BUFFER_CREATE | RGFX_CMD_COMMAND_BUFFER
        ) {
            while self.0.shut.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        if header.r#type == RGFX_CMD_COMMAND_BUFFER {
            let (cb, _) = RgfxCommandBuffer::read_from_prefix(message).unwrap();
            completion.post(cb_submitted(header.seqno, cb.cb_obj_id).as_bytes());
            self.0.carried.lock().push((cb.cb_obj_id, thread_name()));
            completion.post(cb_complete(header.seqno, cb.cb_obj_id).as_bytes());
        }
    }
    fn take_command_buffer(&self, completion: &CompletionPoster, message: &[u8]) -> Option<Taken> {
        if !self.0.takes.load(Ordering::SeqCst) {
            return None;
        }
        let (cb, _) = RgfxCommandBuffer::read_from_prefix(message).unwrap();
        let (seqno, id) = (cb.header.seqno, cb.cb_obj_id);
        let posted = Arc::new(AtomicBool::new(false));
        let (poster, posted_here) = (completion.clone(), Arc::clone(&posted));
        let (rest_poster, taking) = (completion.clone(), Arc::clone(&self.0));
        Some(Taken {
            post: Some(Box::new(move || {
                let done = poster.try_post(cb_submitted(seqno, id).as_bytes());
                posted_here.store(done, Ordering::SeqCst);
            })),
            rest: Box::new(move || {
                if !posted.load(Ordering::SeqCst) {
                    rest_poster.post(cb_submitted(seqno, id).as_bytes());
                }
                taking.carried.lock().push((id, thread_name()));
                rest_poster.post(cb_complete(seqno, id).as_bytes());
            }),
        })
    }
}

/// An empty `COMMAND_BUFFER` `cb_obj_id` of queue `queue`, with seqno `seqno`.
fn command_buffer(queue: u64, cb_obj_id: u64, seqno: u64) -> Vec<u8> {
    use crate::ops::RGFX_COMMAND_BUFFER_OPS_OFFSET;
    RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: RGFX_COMMAND_BUFFER_OPS_OFFSET,
            seqno,
        },
        queue_obj_id: queue,
        cb_obj_id,
        op_bytes: 0,
    }
    .as_bytes()
    .to_vec()
}

/// Publishes `messages` at the tests' queue submission ring's count `at`, without a doorbell;
/// returns the count past them.
fn publish_unrung(rig: &Rig, mut at: u64, messages: &[Vec<u8>]) -> u64 {
    for message in messages {
        rig.region().write(QUEUE_SUBMIT + 256 + at, message);
        at += message.len() as u64;
    }
    rig.region()
        .atomic_u64(QUEUE_SUBMIT)
        .store(at, Ordering::SeqCst);
    at
}

/// The events on the tests' queue completion ring, each one's type, seqno and the command buffer
/// it names, once `count` of them are there or about 2 s have passed.
fn queue_events(rig: &Rig, count: usize) -> Vec<(u32, u64, u64)> {
    let mut events = Vec::new();
    let mut at = 0;
    for _ in 0..2000 {
        let produced = rig
            .region()
            .atomic_u64(QUEUE_COMPLETE)
            .load(Ordering::SeqCst);
        while at < produced {
            let head = rig.region().read(QUEUE_COMPLETE + 256 + at, 24).unwrap();
            let word = |from: usize| u64::from_le_bytes(head[from..from + 8].try_into().unwrap());
            let size = u64::from(u32::from_le_bytes(head[4..8].try_into().unwrap()));
            events.push((word(0) as u32, word(8), word(16)));
            at += size;
        }
        if events.len() >= count {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    events
}

/// A rig over a [`TakingHost`], started, with queue `queue` made and its completion ring armed.
fn taking_rig(taking: &Arc<Taking>, queue: u64) -> Rig {
    let mut rig = Rig::with_host(Box::new(TakingHost(Arc::clone(taking))));
    rig.start();
    let seqno = send_queue_create(&mut rig, queue);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);
    rig.region()
        .atomic_u32(QUEUE_COMPLETE + 0xc0)
        .store(1, Ordering::SeqCst);
    rig
}

/// A doorbell write through `mmio_write_taking` takes the queue's next command buffer on the
/// writing thread: its `CB_SUBMITTED` is on the queue's completion ring, its vector raised on
/// that thread, before the write returns; the rest of it is carried out on the queue's thread,
/// which then takes the buffer published after it, so the queue's events keep its order.
#[test]
fn a_doorbell_write_takes_a_command_buffer_on_its_thread() {
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let taking = Arc::new(Taking::default());
    taking.takes.store(true, Ordering::SeqCst);
    let rig = taking_rig(&taking, QUEUE);

    publish_unrung(
        &rig,
        0,
        &[command_buffer(QUEUE, 100, 1), command_buffer(QUEUE, 101, 2)],
    );
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    let here = thread_name();
    assert_eq!(
        queue_events(&rig, 0).first(),
        Some(&(RGFX_EVT_CB_SUBMITTED, 1, 100)),
        "the CB_SUBMITTED is posted before the doorbell write returns"
    );
    assert_eq!(taking.raised.lock().first(), Some(&(3, here.clone())));

    assert_eq!(
        queue_events(&rig, 4),
        vec![
            (RGFX_EVT_CB_SUBMITTED, 1, 100),
            (RGFX_EVT_CB_COMPLETE, 1, 100),
            (RGFX_EVT_CB_SUBMITTED, 2, 101),
            (RGFX_EVT_CB_COMPLETE, 2, 101),
        ]
    );
    let carried = taking.carried.lock();
    assert_eq!(
        carried.iter().map(|c| c.0).collect::<Vec<_>>(),
        vec![100, 101]
    );
    assert!(
        carried.iter().all(|c| c
            .1
            .as_deref()
            .is_some_and(|n| n.starts_with("reims-vgpu.graphics-queue-"))),
        "both carried out on the queue's thread: {carried:?}"
    );
    // The first was taken on the doorbell's thread, the second handed over as usual.
    assert_eq!(
        *taking.handed.lock(),
        vec![RGFX_CMD_QUEUE_CREATE, RGFX_CMD_COMMAND_BUFFER]
    );
}

/// A doorbell write through `mmio_write_taking` leaves the buffer on the ring for the queue's
/// thread where the host does not take it, where the queue's thread holds a message of the
/// queue's, and where a ring-0 message of its process's published before it is not yet carried
/// out: the queue's thread hands it to the host as it hands any, in the queue's order and after
/// the ring-0 message.
#[test]
fn a_doorbell_write_leaves_a_command_buffer_to_its_queue_s_thread() {
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let on_queue_thread = |taking: &Taking, ids: &[u64]| {
        let carried = taking.carried.lock();
        assert_eq!(carried.iter().map(|c| c.0).collect::<Vec<_>>(), ids);
        assert!(
            carried.iter().all(|c| {
                c.1.as_deref()
                    .is_some_and(|n| n.starts_with("reims-vgpu.graphics-queue-"))
            }),
            "{carried:?}"
        );
    };

    // The host does not take it.
    let taking = Arc::new(Taking::default());
    let rig = taking_rig(&taking, QUEUE);
    publish_unrung(&rig, 0, &[command_buffer(QUEUE, 100, 1)]);
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    assert_eq!(queue_events(&rig, 2).len(), 2);
    on_queue_thread(&taking, &[100]);
    let raised = taking.raised.lock().clone();
    assert!(
        raised[0]
            .1
            .as_deref()
            .is_some_and(|n| n.starts_with("reims-vgpu.graphics-queue-")),
        "{raised:?}"
    );
    drop(rig);

    // The queue's thread holds a buffer of the queue's.
    let taking = Arc::new(Taking::default());
    taking.takes.store(true, Ordering::SeqCst);
    let rig = taking_rig(&taking, QUEUE);
    taking.shut.store(true, Ordering::SeqCst);
    let at = publish_unrung(&rig, 0, &[command_buffer(QUEUE, 100, 1)]);
    rig.device
        .mmio_write(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    let handed = |taking: &Taking| taking.handed.lock().len() == 2;
    for _ in 0..2000 {
        if handed(&taking) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(handed(&taking), "the first buffer never reached the host");
    publish_unrung(&rig, at, &[command_buffer(QUEUE, 101, 2)]);
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    assert!(queue_events(&rig, 0).is_empty() && taking.carried.lock().is_empty());
    taking.shut.store(false, Ordering::SeqCst);
    assert_eq!(
        queue_events(&rig, 4),
        vec![
            (RGFX_EVT_CB_SUBMITTED, 1, 100),
            (RGFX_EVT_CB_COMPLETE, 1, 100),
            (RGFX_EVT_CB_SUBMITTED, 2, 101),
            (RGFX_EVT_CB_COMPLETE, 2, 101),
        ]
    );
    on_queue_thread(&taking, &[100, 101]);
    drop(rig);

    // A ring-0 create of its process's published before it is not yet carried out.
    let taking = Arc::new(Taking::default());
    taking.takes.store(true, Ordering::SeqCst);
    let mut rig = taking_rig(&taking, QUEUE);
    taking.shut.store(true, Ordering::SeqCst);
    rig.send(object_message(RGFX_CMD_BUFFER_CREATE, QUEUE + 1));
    let created = |taking: &Taking| taking.handed.lock().last() == Some(&RGFX_CMD_BUFFER_CREATE);
    for _ in 0..2000 {
        if created(&taking) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(created(&taking), "the create never reached the host");
    publish_unrung(&rig, 0, &[command_buffer(QUEUE, 100, 1)]);
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    assert!(queue_events(&rig, 0).is_empty() && taking.carried.lock().is_empty());
    taking.shut.store(false, Ordering::SeqCst);
    assert_eq!(queue_events(&rig, 2).len(), 2);
    on_queue_thread(&taking, &[100]);
    assert_eq!(
        *taking.handed.lock(),
        vec![
            RGFX_CMD_QUEUE_CREATE,
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_COMMAND_BUFFER
        ]
    );
}

/// A doorbell write through `mmio_write_taking` takes a command buffer on its thread while the
/// control thread is still in another process's create published before it.
#[test]
fn a_doorbell_write_takes_a_command_buffer_behind_another_process_s_create() {
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    const OTHER: u64 = (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let taking = Arc::new(Taking::default());
    taking.takes.store(true, Ordering::SeqCst);
    let mut rig = taking_rig(&taking, QUEUE);
    taking.shut.store(true, Ordering::SeqCst);
    rig.send(object_message(RGFX_CMD_BUFFER_CREATE, OTHER));
    let created = |taking: &Taking| taking.handed.lock().last() == Some(&RGFX_CMD_BUFFER_CREATE);
    for _ in 0..2000 {
        if created(&taking) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(created(&taking), "the create never reached the host");
    publish_unrung(&rig, 0, &[command_buffer(QUEUE, 100, 1)]);
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    assert_eq!(
        queue_events(&rig, 0).first(),
        Some(&(RGFX_EVT_CB_SUBMITTED, 1, 100)),
        "the CB_SUBMITTED is posted before the doorbell write returns"
    );
    assert_eq!(queue_events(&rig, 2).len(), 2);
    taking.shut.store(false, Ordering::SeqCst);
}

/// `mmio_write_taking` writes every register but `DOORBELL` as `mmio_write` does.
#[test]
fn a_taking_write_starts_and_resets_as_a_plain_one() {
    let rig = Rig::new();
    rig.device
        .mmio_write_taking(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    assert_eq!(
        rig.device.mmio_read(RGFX_REG_STATUS.into()),
        RGFX_STATUS_DRIVER_READY
    );
    rig.device
        .mmio_write_taking(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    assert_eq!(
        rig.device.mmio_read(RGFX_REG_STATUS.into()),
        RGFX_STATUS_RESET
    );
}

/// The wire round trip with a fake peer (`peer.rs`).
mod peer;
