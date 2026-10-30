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

/// A host that counts its interrupts and answers whether the scanout was written as the test
/// sets it.
struct TestHost {
    interrupts: Arc<AtomicU32>,
    written: Arc<AtomicBool>,
    asked: Arc<Mutex<Vec<(u64, u64)>>>,
    presented: Arc<Mutex<Vec<u32>>>,
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
    fn present_surface(&self, iosurface_id: u32, answer: PresentAnswer) {
        self.presented.lock().push(iosurface_id);
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
    presented: Arc<Mutex<Vec<u32>>>,
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
        let host = TestHost {
            interrupts: Arc::clone(&interrupts),
            written: Arc::clone(&written),
            asked: Arc::clone(&asked),
            presented: Arc::clone(&presented),
        };
        let mut rig = Rig::with_host(Box::new(host));
        rig.interrupts = interrupts;
        rig.written = written;
        rig.asked = asked;
        rig.presented = presented;
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
/// invalid argument; another display is refused before the host is asked
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
    assert_eq!(*rig.presented.lock(), [HELD_SURFACE, 5, 0]);
}

/// The registers read as the wire says, and `STATUS` follows the guest's writes
#[test]
fn the_registers_read_as_the_wire_says() {
    let rig = Rig::new();
    let read = |offset: u32| rig.device.mmio_read(offset.into());
    assert_eq!(read(RGFX_REG_MAGIC), 0x5846_4752);
    assert_eq!(read(RGFX_REG_VERSION), 1);
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
/// scanout's range; `VBLANK` carries the refresh count while it is on
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

/// A host that records the vectors it raises and the types of the stage-3 messages it is handed,
/// and posts what a decoder would: a command buffer's completion through the queue's poster, and a
/// render pipeline's reflection through ring 0's. A buffer create waits while `gate` is shut: a
/// stand-in for the stage-3 decoder, so the device's rings can be exercised on their own.
struct QueueTestHost {
    vectors: Arc<Mutex<Vec<u32>>>,
    handed: Arc<Mutex<Vec<u32>>>,
    closed: Arc<Mutex<Vec<u32>>>,
    gate: Arc<AtomicBool>,
}

/// What a [`QueueTestHost`] records, and its gate, open.
#[derive(Default)]
struct QueueTestSeen {
    vectors: Arc<Mutex<Vec<u32>>>,
    handed: Arc<Mutex<Vec<u32>>>,
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
    fn context_closed(&self, context: u32) {
        self.closed.lock().push(context);
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
    let seqno = rig.send(queue_create(QUEUE));
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
    let seqno = rig.send(queue_create(QUEUE));
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
    let seqno = rig.send(queue_create(QUEUE));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(rig.device.queue_count(), 1);
    let seqno = rig.send(context_close(7));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(rig.device.queue_count(), 0);
    assert_eq!(*seen.closed.lock(), vec![7]);
}

/// A command buffer is decoded only once the creates published on ring 0 before it are carried
/// out, though the control thread is still in one when the buffer's queue takes it.
#[test]
fn a_command_buffer_waits_for_the_creates_before_it() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let seqno = rig.send(queue_create(QUEUE));
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
