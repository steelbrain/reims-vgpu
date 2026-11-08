//! The device as a guest drives it: a stand-in guest writes commands into the control submission
//! ring, rings the doorbell, and reads the completion ring, with a host that records its
//! interrupts and a screen that records what it is told.

use std::collections::HashMap;
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

/// What a host was handed on the presents' lane, in order: a present of a surface, a kept
/// present, and a colour part (its stage, first entry, entries carried and the first's red).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seen {
    Present(u32),
    Keep,
    Color {
        stage: u32,
        first: u32,
        entries: usize,
        red: u32,
    },
}

/// A host that counts its interrupts and records the presents and refreshes it is told of.
struct TestHost {
    interrupts: Arc<AtomicU32>,
    presented: Presented,
    refreshes: Arc<AtomicU32>,
    seen: Arc<Mutex<Vec<Seen>>>,
}

/// The one IOSurface the test host holds a texture over.
const HELD_SURFACE: u32 = 0x77;

impl Host for TestHost {
    fn raise_interrupt(&self, vector: u32) {
        assert_eq!(vector, 0);
        self.interrupts.fetch_add(1, Ordering::SeqCst);
    }
    fn refreshed(&self) {
        self.refreshes.fetch_add(1, Ordering::SeqCst);
    }
    fn present_surface(&self, iosurface_id: u32, generation: Option<u64>, answer: PresentAnswer) {
        self.presented.lock().push((iosurface_id, generation));
        self.seen.lock().push(Seen::Present(iosurface_id));
        answer.answer(if iosurface_id == 0 || iosurface_id == HELD_SURFACE {
            RGFX_STATUS_OK
        } else {
            RGFX_STATUS_INVALID_ARGUMENT
        });
    }
    fn present_keep(&self, answer: PresentAnswer) {
        self.seen.lock().push(Seen::Keep);
        answer.answer(RGFX_STATUS_OK);
    }
    fn display_color(&self, color: &RgfxDisplayColor, entries: &[RgfxColorEntry]) {
        self.seen.lock().push(Seen::Color {
            stage: color.stage,
            first: color.first_entry,
            entries: entries.len(),
            red: entries.first().map_or(0, |e| e.x),
        });
    }
}

/// A device, its stand-in guest, and what its host and screen saw.
struct Rig {
    device: Device,
    mode: Arc<Mutex<Mode>>,
    told: Arc<Mutex<Vec<Told>>>,
    interrupts: Arc<AtomicU32>,
    presented: Presented,
    /// The refreshes the host was told of.
    refreshes: Arc<AtomicU32>,
    /// What the host was handed on the presents' lane.
    seen: Arc<Mutex<Vec<Seen>>>,
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
        let presented = Arc::new(Mutex::new(Vec::new()));
        let refreshes = Arc::new(AtomicU32::new(0));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let host = TestHost {
            interrupts: Arc::clone(&interrupts),
            presented: Arc::clone(&presented),
            refreshes: Arc::clone(&refreshes),
            seen: Arc::clone(&seen),
        };
        let mut rig = Rig::with_host(Box::new(host));
        rig.interrupts = interrupts;
        rig.presented = presented;
        rig.refreshes = refreshes;
        rig.seen = seen;
        rig
    }

    /// A rig whose device has `host` (what the rig records of the host's stays empty).
    fn with_host(host: Box<dyn Host>) -> Rig {
        Rig::with_lanes(host, Lanes::default())
    }

    /// [`Self::with_host`], the device's ring 0's lanes `lanes`.
    fn with_lanes(host: Box<dyn Host>, lanes: Lanes) -> Rig {
        let mode = Arc::new(Mutex::new(MODE));
        let told = Arc::new(Mutex::new(Vec::new()));
        let interrupts = Arc::new(AtomicU32::new(0));
        let presented = Arc::new(Mutex::new(Vec::new()));
        let region = Arc::new(Region::new(REGION_SIZE as usize).unwrap());
        let device = Device::with_lanes(
            region,
            host,
            Box::new(TestScreen {
                mode: Arc::clone(&mode),
                told: Arc::clone(&told),
            }),
            lanes,
        )
        .unwrap();
        Rig {
            device,
            mode,
            told,
            interrupts,
            presented,
            refreshes: Arc::new(AtomicU32::new(0)),
            seen: Arc::new(Mutex::new(Vec::new())),
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

    /// Waits about 2 s for the lanes to have carried out every message handed to them: a message
    /// of one context's answered says nothing of another's.
    fn lanes_done(&self) {
        for _ in 0..2000 {
            if self.device.shared.lanes.idle() {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("the lanes never carried out what they were handed");
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
        flags: 0,
    };
    c.as_bytes().to_vec()
}

/// A `PRESENT_SURFACE` of display 0 with `flags`.
fn present_with_flags(iosurface_id: u32, flags: u32) -> Vec<u8> {
    let mut c = RgfxPresentSurface::read_from_bytes(&present_surface(0, iosurface_id)).unwrap();
    c.flags = flags;
    c.as_bytes().to_vec()
}

/// A `DISPLAY_COLOR` of `display`'s `stage`, entries `first` on of `count`, carrying `part`
/// entries, entry `k`'s red `first + k` as bits.
fn display_color(display: u32, stage: u32, count: u32, first: u32, part: u32) -> Vec<u8> {
    let c = RgfxDisplayColor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_DISPLAY_COLOR,
            ..Default::default()
        },
        display,
        transaction_id: 9,
        stage,
        entry_count: count,
        first_entry: first,
        part_entries: part,
    };
    let mut bytes = c.as_bytes().to_vec();
    for k in 0..part {
        let entry = RgfxColorEntry {
            x: first + k,
            ..Default::default()
        };
        bytes.extend_from_slice(entry.as_bytes());
    }
    bytes
}

/// A table's `DISPLAY_COLOR`s reach the host in ring order, unanswered, on the presents' lane
/// before the present after them, and a kept present (`RGFX_PRESENT_KEEP`) is the host's
/// `present_keep`, answered with its word, as a present of a surface is.
#[test]
fn display_colors_reach_the_host_before_the_present_after_them() {
    let mut rig = Rig::new();
    rig.start();
    rig.send(display_color(0, RGFX_COLOR_POST_GAMMA, 300, 0, 256));
    rig.send(display_color(0, RGFX_COLOR_POST_GAMMA, 300, 256, 44));
    let seqno = rig.send(present_with_flags(0, RGFX_PRESENT_KEEP));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    rig.send(display_color(0, RGFX_COLOR_MATRIX, 3, 0, 3));
    let seqno = rig.send(present_surface(0, HELD_SURFACE));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    let color = |stage, first, entries| Seen::Color {
        stage,
        first,
        entries,
        red: first,
    };
    assert_eq!(
        *rig.seen.lock(),
        [
            color(RGFX_COLOR_POST_GAMMA, 0, 256),
            color(RGFX_COLOR_POST_GAMMA, 256, 44),
            Seen::Keep,
            color(RGFX_COLOR_MATRIX, 0, 3),
            Seen::Present(HELD_SURFACE),
        ]
    );
}

/// A kept present that names a surface, and a present with a flag the wire does not name, are
/// refused before the host is asked; a `DISPLAY_COLOR` of another display, one shorter than the
/// entries it names, and one naming more than a part may carry, are dropped unanswered.
#[test]
fn malformed_colors_and_kept_presents_are_refused() {
    let mut rig = Rig::new();
    rig.start();
    let seqno = rig.send(present_with_flags(HELD_SURFACE, RGFX_PRESENT_KEEP));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_INVALID_ARGUMENT));
    let seqno = rig.send(present_with_flags(0, 1 << 1));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_INVALID_ARGUMENT));
    rig.send(display_color(1, RGFX_COLOR_POST_GAMMA, 2, 0, 2));
    let mut short = display_color(0, RGFX_COLOR_POST_GAMMA, 2, 0, 2);
    short.truncate(size_of::<RgfxDisplayColor>() + size_of::<RgfxColorEntry>());
    rig.send(short);
    rig.send(display_color(
        0,
        RGFX_COLOR_POST_GAMMA,
        1024,
        0,
        RGFX_COLOR_PART_ENTRIES + 1,
    ));
    let seqno = rig.send(present_with_flags(0, RGFX_PRESENT_KEEP));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(*rig.seen.lock(), [Seen::Keep]);
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
    // No `SURFACE_DEFERRED`, so no budget.
    assert_eq!(read(RGFX_REG_DEFERRED_BUDGET_LO), 0);
    assert_eq!(read(RGFX_REG_DEFERRED_BUDGET_HI), 0);
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

/// The refreshes' times keep to their grid, a period apart from the first, as a panel's vblanks
/// do: a thread that wakes late for a tick still stamps the tick's time, and one held over
/// periods skips to the grid's last tick before now, then goes on a period at a time
#[test]
fn refresh_times_keep_to_their_grid_under_a_late_wake() {
    const PERIOD: u64 = 8_333_333;
    let start = 1_000_000_000;
    let mut grid = super::RefreshGrid::new(start);
    // On time: the thread asks before the tick, and sleeps to it.
    assert_eq!(grid.next(PERIOD, start + 10), start + PERIOD);
    // Asked 400 us after that tick (it woke late, and the refresh took its time): the next is
    // still a period after the tick, not after the wake.
    assert_eq!(
        grid.next(PERIOD, start + PERIOD + 400_000),
        start + 2 * PERIOD
    );
    // Held 2.5 periods past the tick it slept to: no burst, the grid's last tick before now.
    let now = start + 2 * PERIOD + 5 * PERIOD / 2;
    assert_eq!(grid.next(PERIOD, now), start + 4 * PERIOD);
    assert_eq!(grid.next(PERIOD, now + 1), start + 5 * PERIOD);
    // A new mode's period runs from the last tick.
    assert_eq!(
        grid.next(2 * PERIOD, start + 5 * PERIOD),
        start + 7 * PERIOD
    );
}

/// A refresh shows nothing again before the guest has a scanout, and the scanout at each refresh
/// once it has one; `VBLANK` carries the refresh count while it is on; the host is told of each
/// refresh
#[test]
fn a_refresh_presents_the_scanout_and_sends_vblank() {
    let mut rig = Rig::new();
    rig.start();
    rig.refresh();
    assert_eq!(rig.told(), []);
    assert_eq!(rig.refreshes.load(Ordering::SeqCst), 1);
    rig.send(scanout(16, 4, 128, RGFX_ALLOCATIONS_OFFSET));
    rig.complete();
    rig.told();
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

/// A reset stops the device, puts both rings back as new and the display with no cursor or
/// `VBLANK`, its scanout kept, and the guest starts again from its first command
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
    assert_eq!(rig.told(), [Told::Cursor(None), Told::Moved(0, 0, true)]);
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
    assert_eq!(
        rig.device.shared.display.lock().scanout,
        Some(Scanout {
            width: 16,
            height: 4,
            stride: 64,
            offset: RGFX_ALLOCATIONS_OFFSET,
        })
    );
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

/// A reset keeps the display's scanout, as a display engine goes on scanning out across its
/// GPU driver's reset: the firmware's scanout outlives the kext's reset that takes the device,
/// its start, and its reset on the way out where it then fails, the screen told nothing, and a
/// refresh after each shows what the guest writes there (the OS's fallback to the firmware's
/// framebuffer). Only `SET_SCANOUT` changes it: another scanout, or none for width 0
#[test]
fn a_reset_keeps_the_scanout_and_what_the_guest_writes_there_shows() {
    let mut rig = Rig::new();
    let alloc = RGFX_ALLOCATIONS_OFFSET;
    // The firmware's scanout.
    rig.start();
    rig.send(scanout(16, 4, 64, alloc));
    rig.complete();
    rig.told();
    let restart = |rig: &mut Rig| {
        (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
        rig.start();
    };
    // The kext's take, its start and its GET_DISPLAYS, and its reset on the way out.
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    restart(&mut rig);
    rig.send(get_displays());
    rig.event().expect("DISPLAYS");
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    assert!(
        !rig.told().iter().any(|t| matches!(t, Told::Scanout(_))),
        "a reset told the screen of the scanout"
    );
    rig.refresh();
    assert_eq!(rig.told(), [Told::Present]);
    assert_eq!(
        rig.device
            .shared
            .display
            .lock()
            .scanout
            .map(|s| (s.offset, s.span())),
        Some((alloc, 64 * 3 + 16 * 4))
    );

    // The guest's next driver sets another scanout, and then blanks the display.
    restart(&mut rig);
    rig.send(scanout(8, 2, 64, alloc + 0x1000));
    rig.complete();
    rig.told();
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    rig.told();
    rig.refresh();
    assert_eq!(rig.told(), [Told::Present]);
    assert_eq!(
        rig.device
            .shared
            .display
            .lock()
            .scanout
            .map(|s| (s.offset, s.span())),
        Some((alloc + 0x1000, 64 + 8 * 4))
    );
    restart(&mut rig);
    rig.send(scanout(0, 0, 0, 0));
    rig.complete();
    assert_eq!(rig.told(), [Told::Scanout(None)]);
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    rig.told();
    rig.refresh();
    assert_eq!(rig.told(), []);
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

/// Each queue a [`QueueTestHost`] was told is made, with what its ring had published then.
type Made = Arc<Mutex<Vec<(u64, Option<u64>)>>>;

/// A host that records the vectors it raises and the types of the stage-3 messages it is handed,
/// and posts what a decoder would: a command buffer's completion through the queue's poster, and a
/// render pipeline's reflection through ring 0's. A buffer create waits while `gate` is shut: a
/// stand-in for the stage-3 decoder, so the device's rings can be exercised on their own.
struct QueueTestHost {
    vectors: Arc<Mutex<Vec<u32>>>,
    handed: Arc<Mutex<Vec<u32>>>,
    buffers: Arc<Mutex<Vec<Vec<u8>>>>,
    closed: Arc<Mutex<Vec<u32>>>,
    made: Made,
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
    made: Made,
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
            made: Arc::clone(&self.made),
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
    fn context_closed(&self, context: u32, answer: crate::device::CompleteAnswer) {
        self.closed.lock().push(context);
        answer.answer(RGFX_STATUS_OK);
    }
    fn queue_made(&self, completion: &crate::device::CompletionPoster) {
        let queue = completion.queue_obj_id().expect("a queue's poster");
        self.made.lock().push((queue, completion.published_seqno()));
    }
    fn surface_download(&self, _block_offset: u64, answer: crate::device::CompleteAnswer) {
        self.handed.lock().push(RGFX_CMD_SURFACE_DOWNLOAD);
        while !self.gate.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        answer.answer(RGFX_STATUS_OK);
    }
    fn surface_pages(
        &self,
        _pages: &RgfxSurfacePages,
        _runs: Vec<RgfxPageRun>,
        _flags: u32,
        answer: crate::device::CompleteAnswer,
    ) {
        self.handed.lock().push(RGFX_CMD_SURFACE_PAGES);
        while !self.gate.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        answer.answer(RGFX_STATUS_OK);
    }
    fn stage3_message(&self, completion: &crate::device::CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        self.handed.lock().push(header.r#type);
        match header.r#type {
            r#type if HELD.contains(&r#type) => {
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

/// The host is told of a queue the device makes before its create is answered, so a buffer of
/// another queue that names the queue's buffers from the answer on finds it known, however late
/// the queue's thread hands the host the queue's create: with its ring's poster, which reads what
/// the ring has published, a command buffer published before the create among it. A create the
/// device refuses tells the host of nothing
#[test]
fn the_host_is_told_of_a_queue_before_its_create_is_answered() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;

    // Refused: no block holds the rings.
    let seqno = rig.send(queue_create(QUEUE));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_INVALID_ARGUMENT));
    assert!(
        seen.made.lock().is_empty(),
        "a refused create told the host"
    );

    // The guest's first buffer, published with no doorbell before the create.
    publish_command_buffer(&rig, QUEUE, 100, 1, false);
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(
        *seen.made.lock(),
        [(QUEUE, Some(1))],
        "the host was not told of the queue, and its ring's buffer, by the answer"
    );
    let done = first_queue_completion(&rig).expect("the buffer's CB_COMPLETE");
    assert_eq!(done.cb_obj_id, 100);
    assert_eq!(seen.made.lock().len(), 1, "told twice");
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
    // The grants to the other process are carried out on its lane: the kext grants a block
    // another process holds only once that process's release of it is answered, so they come
    // before the grant below.
    rig.lanes_done();
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
    // A grant is not answered: the close after them is, once the ring has taken them, and their
    // lanes then carry them out.
    let seqno = rig.send(context_close(99));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    rig.lanes_done();

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
    rig.lanes_done();
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
    rig.lanes_done();
    let region = rig.region();
    assert!(region.held_by(5, BLOCK, 0x2_0000));
    assert!(region.held_by(6, BLOCK + 0x1_0000, 0x1_0000));
    assert!(!region.held_by(7, SCANOUT, 0x1000));

    rig.send(block_message(RGFX_CMD_BLOCK_UNSHARE, 5, BLOCK, 0x2_0000));
    let seqno = rig.send(context_close(99));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    rig.lanes_done();
    assert!(!rig.region().held_by(5, BLOCK, 0x1000));
    assert!(rig.region().held_by(6, BLOCK, 0x1000));

    let seqno = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, 6, BLOCK, 0x2_0000));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert!(!rig.region().held_by(6, BLOCK, 0x1000));
}

/// A host that records the holds it is told end ([`Host::block_unshared`]), the releases and the
/// closes, in order, each as its message type, context and offset (0 for a close).
#[derive(Clone, Default)]
struct HoldsHost {
    seen: Arc<Mutex<Vec<(u32, u32, u64)>>>,
    /// While set, a `BUFFER_CREATE` the host carries out waits: its context's lane is in it.
    hold: Arc<AtomicBool>,
    /// Set once such a create is being carried out.
    holding: Arc<AtomicBool>,
}

impl HoldsHost {
    /// Holds `context`'s lane in a buffer create until [`Self::hold`] is cleared, and waits for
    /// the host to be in it.
    fn hold_lane(&self, rig: &mut Rig, context: u32) {
        self.hold.store(true, Ordering::SeqCst);
        rig.send(object_message(
            RGFX_CMD_BUFFER_CREATE,
            (u64::from(context) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 0xfff,
        ));
        for _ in 0..2000 {
            if self.holding.load(Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("the create was not carried out");
    }
}

impl Host for HoldsHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn context_closed(&self, context: u32, answer: CompleteAnswer) {
        self.seen.lock().push((RGFX_CMD_CONTEXT_CLOSE, context, 0));
        answer.answer(RGFX_STATUS_OK);
    }
    fn block_released(&self, release: &RgfxBlockRelease, answer: CompleteAnswer) {
        (self.seen.lock()).push((RGFX_CMD_BLOCK_RELEASE, release.context, release.offset));
        answer.answer(RGFX_STATUS_OK);
    }
    fn block_unshared(&self, unshare: &RgfxBlockGrant) {
        (self.seen.lock()).push((RGFX_CMD_BLOCK_UNSHARE, unshare.context, unshare.offset));
    }
    fn stage3_message(&self, _completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        if header.r#type == RGFX_CMD_BUFFER_CREATE {
            self.holding.store(true, Ordering::SeqCst);
            while self.hold.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// The host learns that each holder of a shared block let it go before the block's release, by
/// whatever came first of its `BLOCK_UNSHARE`, its close and the release, so the release's answer
/// waits for every holder's command buffers: the kext posts an unshare out of its lock, so another
/// holder's release may overtake it, and may let go of a closing context's wire before the close
/// is answered. Once only: an unshare behind the release or the close is not handed on.
#[test]
fn every_holder_s_end_is_told_before_its_shared_block_s_release() {
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x40_0000;
    const OTHER: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x80_0000;
    const ALONE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0xc0_0000;
    let host = HoldsHost::default();
    let mut rig = Rig::with_host(Box::new(host.clone()));
    rig.start();
    fn sync(rig: &mut Rig) {
        let seqno = rig.send(context_close(99));
        assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
        rig.lanes_done();
    }
    for context in [5, 6] {
        rig.send(block_message(
            RGFX_CMD_BLOCK_SHARE,
            context,
            BLOCK,
            0x2_0000,
        ));
    }
    // 6's release overtakes 5's unshare.
    let seqno = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, 6, BLOCK, 0x2_0000));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    rig.send(block_message(RGFX_CMD_BLOCK_UNSHARE, 5, BLOCK, 0x2_0000));
    sync(&mut rig);
    assert_eq!(
        std::mem::take(&mut *host.seen.lock()),
        [
            (RGFX_CMD_BLOCK_UNSHARE, 5, BLOCK),
            (RGFX_CMD_BLOCK_RELEASE, 6, BLOCK),
            (RGFX_CMD_CONTEXT_CLOSE, 99, 0),
        ],
        "a holder's end after the release, or none before it"
    );

    // 7 closes holding a block 8 holds too, and one it alone holds; its unshare comes behind
    // the close.
    for context in [7, 8] {
        rig.send(block_message(
            RGFX_CMD_BLOCK_SHARE,
            context,
            OTHER,
            0x1_0000,
        ));
    }
    rig.send(block_message(RGFX_CMD_BLOCK_SHARE, 7, ALONE, 0x1_0000));
    let seqno = rig.send(context_close(7));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    rig.send(block_message(RGFX_CMD_BLOCK_UNSHARE, 7, OTHER, 0x1_0000));
    let seqno = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, 8, OTHER, 0x1_0000));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(
        std::mem::take(&mut *host.seen.lock()),
        [
            (RGFX_CMD_BLOCK_UNSHARE, 7, OTHER),
            (RGFX_CMD_BLOCK_UNSHARE, 7, ALONE),
            (RGFX_CMD_CONTEXT_CLOSE, 7, 0),
            (RGFX_CMD_BLOCK_RELEASE, 8, OTHER),
        ],
        "a closed holder's end not told before the block's release, or told twice"
    );

    // An unshare in time is told once, and the release by the last holder after it tells none.
    for context in [5, 6] {
        rig.send(block_message(
            RGFX_CMD_BLOCK_SHARE,
            context,
            BLOCK,
            0x2_0000,
        ));
    }
    rig.send(block_message(RGFX_CMD_BLOCK_UNSHARE, 5, BLOCK, 0x2_0000));
    let seqno = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, 6, BLOCK, 0x2_0000));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(
        std::mem::take(&mut *host.seen.lock()),
        [
            (RGFX_CMD_BLOCK_UNSHARE, 5, BLOCK),
            (RGFX_CMD_BLOCK_RELEASE, 6, BLOCK),
        ]
    );
}

/// A close tells the end of its context's hold of every shared block it held, whether or not
/// another holder's `BLOCK_SHARE` was carried out before it: another context's share runs on that
/// context's lane, which a close does not wait for, so a close may find its context the block's
/// only holder while a later holder's share is still on its way, and that holder's release must
/// still wait for the closed context's command buffers.
#[test]
fn a_close_tells_its_hold_s_end_before_another_holder_s_share_is_carried_out() {
    const OTHER: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x80_0000;
    let host = HoldsHost::default();
    let mut rig = Rig::with_host(Box::new(host.clone()));
    rig.start();
    // 8's share of the block waits on its lane, behind a create the host holds.
    host.hold_lane(&mut rig, 8);
    for context in [7, 8] {
        rig.send(block_message(
            RGFX_CMD_BLOCK_SHARE,
            context,
            OTHER,
            0x1_0000,
        ));
    }
    let seqno = rig.send(context_close(7));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    host.hold.store(false, Ordering::SeqCst);
    let seqno = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, 8, OTHER, 0x1_0000));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(
        std::mem::take(&mut *host.seen.lock()),
        [
            (RGFX_CMD_BLOCK_UNSHARE, 7, OTHER),
            (RGFX_CMD_CONTEXT_CLOSE, 7, 0),
            (RGFX_CMD_BLOCK_RELEASE, 8, OTHER),
        ],
        "a close that overtook another holder's share did not tell its hold's end"
    );
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

/// A process's ring-0 messages and command buffers go on while another process's create is
/// carried out: each context's messages are carried out on its own lane, so the other process's
/// create holds up only that process's later messages. The command buffer still waits for its
/// own process's create before it.
#[test]
fn a_process_s_messages_go_on_while_another_s_create_is_carried_out() {
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
    let done = first_queue_completion(&rig);
    seen.gate.store(true, Ordering::SeqCst);
    assert_eq!(
        done.expect("held up by another process's create").cb_obj_id,
        100
    );
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

/// The ring-0 messages a [`QueueTestHost`] holds while its gate is shut: a buffer create, and the
/// object messages past `BUFFER_OVER_PAGES`, a type the host does not know among them.
const HELD: [u32; 6] = [
    RGFX_CMD_BUFFER_CREATE,
    RGFX_CMD_ICB_CREATE,
    RGFX_CMD_ICB_WRITE,
    RGFX_CMD_BUFFER_MANAGED_AT_OFFSET,
    RGFX_CMD_ICB_SLOTS,
    RGFX_CMD_BUFFER_OVER_PAGES + 1,
];

/// A command buffer waits for no object message of another process's past `BUFFER_OVER_PAGES`
/// either: an indirect command buffer's create, write or slots, a Managed buffer placed in a heap,
/// or a type the host does not know, each its object's context's, as the kext takes it from a
/// process only with an object id of the process's. The buffer is decoded while the control thread
/// is still in one published before it.
#[test]
fn a_command_buffer_waits_for_no_other_process_s_later_object_messages() {
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    const OTHER: u64 = (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    for &kind in &HELD[1..] {
        let seen = QueueTestSeen::new();
        let mut rig = Rig::with_host(seen.host());
        rig.start();
        let seqno = send_queue_create(&mut rig, QUEUE);
        assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
        wait_queue_idle(&rig);

        seen.gate.store(false, Ordering::SeqCst);
        rig.send(object_message(kind, OTHER));
        seen.wait("the message never reached the host", |seen| {
            seen.handed.lock().last() == Some(&kind)
        });
        publish_command_buffer(&rig, QUEUE, 100, 1, true);
        let done = first_queue_completion(&rig);
        seen.gate.store(true, Ordering::SeqCst);
        let done = done
            .unwrap_or_else(|| panic!("held up by another process's message of type {kind:#x}"));
        assert_eq!(done.cb_obj_id, 100);
        assert_eq!(
            *seen.handed.lock(),
            vec![RGFX_CMD_QUEUE_CREATE, kind, RGFX_CMD_COMMAND_BUFFER],
            "type {kind:#x}"
        );
    }
}

/// A command buffer waits for no `SURFACE_DOWNLOAD`: one published after a download the host is
/// still carrying out is handed to the host meanwhile. So a buffer that writes the surface and is
/// taken first is among the users the download waits for, as one published before it is; only a
/// guest that writes a surface while racing its CPU lock of it can tell.
#[test]
fn a_command_buffer_is_handed_over_while_a_download_published_ahead_of_it_is_carried_out() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);

    seen.gate.store(false, Ordering::SeqCst);
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
    seen.wait("the download never reached the host", |seen| {
        seen.handed.lock().last() == Some(&RGFX_CMD_SURFACE_DOWNLOAD)
    });
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    let done = first_queue_completion(&rig);
    seen.gate.store(true, Ordering::SeqCst);
    assert_eq!(done.expect("held up by the download").cb_obj_id, 100);
    assert_eq!(
        *seen.handed.lock(),
        vec![
            RGFX_CMD_QUEUE_CREATE,
            RGFX_CMD_SURFACE_DOWNLOAD,
            RGFX_CMD_COMMAND_BUFFER
        ]
    );
}

/// A command buffer waits for its context's `SURFACE_PAGES` published before it, which the
/// block's lane carries out, as it waited when the context's lane did: woken once that lane has
/// carried it out.
#[test]
fn a_command_buffer_waits_for_its_context_s_pages_on_the_block_s_lane() {
    let seen = QueueTestSeen::new();
    let mut rig = Rig::with_host(seen.host());
    rig.start();
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let seqno = send_queue_create(&mut rig, QUEUE);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(&rig);

    seen.gate.store(false, Ordering::SeqCst);
    let pages = rig.send(surface_pages_message(
        5,
        RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000,
        RGFX_SURFACE_PAGES_CACHED,
    ));
    seen.wait("the pages never reached the host", |seen| {
        seen.handed.lock().last() == Some(&RGFX_CMD_SURFACE_PAGES)
    });
    publish_command_buffer(&rig, QUEUE, 100, 1, true);
    settle();
    assert_eq!(
        *seen.handed.lock(),
        vec![RGFX_CMD_QUEUE_CREATE, RGFX_CMD_SURFACE_PAGES],
        "the buffer went before its context's pages"
    );
    seen.gate.store(true, Ordering::SeqCst);
    assert_eq!(rig.complete(), (pages, RGFX_STATUS_OK));
    let done = first_queue_completion(&rig);
    assert_eq!(done.expect("never woken").cb_obj_id, 100);
    assert_eq!(
        *seen.handed.lock(),
        vec![
            RGFX_CMD_QUEUE_CREATE,
            RGFX_CMD_SURFACE_PAGES,
            RGFX_CMD_COMMAND_BUFFER
        ]
    );
}

/// Process B learns of process A's surface over IPC after A's messages were posted, which may be
/// before they are carried out: B's buffer waits for the share of the surface's block the kext
/// posts for B's context before B learns of it, its own context's message, and not for A's
/// messages before it. B's buffer names only B's objects, and what it reads of A's surface the
/// host made at the surface's `SURFACE_PAGES`, which the kext waits for before it posts the share.
#[test]
fn a_command_buffer_waits_for_its_share_of_another_process_s_surface_not_for_that_process() {
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
    let done = first_queue_completion(&rig);
    let shared = rig.region().held_by(B, SURFACE, 0x1000);
    seen.gate.store(true, Ordering::SeqCst);
    assert_eq!(done.expect("held up by A's create").cb_obj_id, 100);
    assert!(shared, "B's buffer ran before its share");
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
    // thread, its refresh thread and the context's lane, in the create, hold the rest.
    for _ in 0..2000 {
        if Arc::strong_count(&rig.device.shared) == 4 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        Arc::strong_count(&rig.device.shared),
        4,
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
    fn stage3_message(&self, _completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        self.0.lock().push((header.r#type, thread_name()));
    }
    fn device_reset(&self) {
        self.0.lock().push((RESET_TOLD, thread_name()));
    }
}

/// A reset has the host drop what it made ([`Host::device_reset`]) once, on the control thread,
/// before the host is handed anything the guest published after the reset (an object's create on
/// its context's lane); the reset ends its
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
    let lane = Some(RING0.to_owned());
    assert_eq!(
        *seen.lock(),
        [
            (RGFX_CMD_QUEUE_CREATE, queue),
            (RESET_TOLD, control),
            (RGFX_CMD_EVENT_CREATE, lane),
        ]
    );
}

/// A reset that comes after the control thread found that none had since it handed a message to
/// its lane, and before it moved `control_done` past that message, leaves the count at the
/// reset's 0: the look and the move are one step to a reset. A count left past the reset's would
/// let a buffer of a queue made after it go to the host before the creates it waits for were
/// handed on (a lane with nothing handed has carried everything out).
#[test]
fn a_reset_between_the_control_thread_s_look_and_its_count_leaves_the_count_at_0() {
    use std::sync::mpsc;
    let mut rig = Rig::new();
    rig.start();
    let (at_stop, stopped) = mpsc::channel();
    let (go, wait_go) = mpsc::channel::<()>();
    let wait_go = Mutex::new(wait_go);
    *rig.device.shared.before_count.lock() = Some(Arc::new(move |through| {
        let _ = at_stop.send(through);
        let _ = wait_go.lock().recv();
    }));
    const EVENT: u64 = (4 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    rig.send(object_message(RGFX_CMD_EVENT_CREATE, EVENT));
    let through = (stopped.recv_timeout(Duration::from_secs(5)))
        .expect("the control thread never handed the create on");
    assert!(through > 0);
    *rig.device.shared.before_count.lock() = None;
    let device = &rig.device;
    std::thread::scope(|scope| {
        let (reset_done, reset_waited) = mpsc::channel();
        scope.spawn(move || {
            device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
            let _ = reset_done.send(());
        });
        // A reset that need not wait for the control thread's step ends well within this.
        let _ = reset_waited.recv_timeout(Duration::from_millis(100));
        go.send(()).unwrap();
    });
    for _ in 0..200 {
        if rig.device.shared.control_done.load(Ordering::Acquire) != 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        rig.device.shared.control_done.load(Ordering::Acquire),
        0,
        "the control thread moved the count past {through} after the reset set it to 0"
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

/// A host that offers `SURFACE_DEFERRED` reads its budget in `DEFERRED_BUDGET_LO` and `_HI`, the
/// same before the driver is ready, after and across a reset, and the driver may take the bit
#[test]
fn the_deferred_budget_reads_as_the_host_gives_it() {
    struct Budget;
    impl Host for Budget {
        fn raise_interrupt(&self, _vector: u32) {}
        fn features(&self) -> u64 {
            u64::from(RGFX_FEATURE_SURFACE_CACHE | RGFX_FEATURE_SURFACE_DEFERRED)
        }
        fn deferred_budget(&self) -> u64 {
            0x1_2345_6000
        }
    }
    let rig = Rig::with_host(Box::new(Budget));
    let read = |offset: u32| rig.device.mmio_read(offset.into());
    let budget = || {
        (
            read(RGFX_REG_DEFERRED_BUDGET_LO),
            read(RGFX_REG_DEFERRED_BUDGET_HI),
        )
    };
    assert_eq!(budget(), (0x2345_6000, 1));
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    rig.device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_LO.into(),
        RGFX_FEATURE_SURFACE_DEFERRED,
    );
    assert_eq!(
        read(RGFX_REG_DRIVER_FEATURES_LO),
        RGFX_FEATURE_SURFACE_DEFERRED
    );
    rig.start();
    assert_eq!(budget(), (0x2345_6000, 1));
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    assert_eq!(budget(), (0x2345_6000, 1));
}

/// A host that offers `PRIVATE_RENDER_STATE` lets the driver take it: the driver's write reads
/// back
#[test]
fn the_driver_takes_private_render_state_where_offered() {
    struct Private;
    impl Host for Private {
        fn raise_interrupt(&self, _vector: u32) {}
        fn features(&self) -> u64 {
            u64::from(RGFX_FEATURE_PRIVATE_RENDER_STATE)
        }
    }
    let rig = Rig::with_host(Box::new(Private));
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    rig.device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_LO.into(),
        RGFX_FEATURE_PRIVATE_RENDER_STATE,
    );
    assert_eq!(
        rig.device.mmio_read(RGFX_REG_DRIVER_FEATURES_LO.into()),
        RGFX_FEATURE_PRIVATE_RENDER_STATE
    );
}

/// A host that offers `OPENCL_MODE` lets the driver take it: the driver's write reads back
#[test]
fn the_driver_takes_opencl_mode_where_offered() {
    struct OpenCl;
    impl Host for OpenCl {
        fn raise_interrupt(&self, _vector: u32) {}
        fn features(&self) -> u64 {
            u64::from(RGFX_FEATURE_OPENCL_MODE)
        }
    }
    let rig = Rig::with_host(Box::new(OpenCl));
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    rig.device
        .mmio_write(RGFX_REG_DRIVER_FEATURES_LO.into(), RGFX_FEATURE_OPENCL_MODE);
    assert_eq!(
        rig.device.mmio_read(RGFX_REG_DRIVER_FEATURES_LO.into()),
        RGFX_FEATURE_OPENCL_MODE
    );
}

/// `HOST_TIME_LO` and `_HI` read the host's monotonic clock, the one each `VBLANK` is stamped on,
/// once the driver took `HOST_TIME`: a read of `HI`, `LO` and `HI` again lies between two of the
/// host's own. They read 0 before the driver took it, and again after a reset
#[test]
fn the_host_time_reads_the_vblank_clock_once_taken() {
    struct Clock;
    impl Host for Clock {
        fn raise_interrupt(&self, _vector: u32) {}
        fn features(&self) -> u64 {
            u64::from(RGFX_FEATURE_HOST_TIME)
        }
    }
    let rig = Rig::with_host(Box::new(Clock));
    let read = |offset: u32| rig.device.mmio_read(offset.into());
    let registers = || (read(RGFX_REG_HOST_TIME_LO), read(RGFX_REG_HOST_TIME_HI));
    let host_time = || {
        let hi = read(RGFX_REG_HOST_TIME_HI);
        let lo = read(RGFX_REG_HOST_TIME_LO);
        let again = read(RGFX_REG_HOST_TIME_HI);
        (u64::from(if lo < 0x8000_0000 { again } else { hi }) << 32) | u64::from(lo)
    };
    assert_eq!(
        rig.device.mmio_read(RGFX_REG_FEATURES_LO.into()),
        RGFX_FEATURE_HOST_TIME
    );
    assert_eq!(registers(), (0, 0));
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    rig.device
        .mmio_write(RGFX_REG_DRIVER_FEATURES_LO.into(), RGFX_FEATURE_HOST_TIME);
    assert_eq!(read(RGFX_REG_DRIVER_FEATURES_LO), RGFX_FEATURE_HOST_TIME);
    rig.start();
    let before = super::now_ns();
    let time = host_time();
    let after = super::now_ns();
    assert!(
        before <= time && time <= after,
        "{before} <= {time} <= {after}"
    );
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    assert_eq!(registers(), (0, 0));
}

/// The device's feature bits are its host's; a `SURFACE_PAGES` hands the host the runs it lists
/// in its block, read out of the region, and the flags the message gives the block, and is
/// answered with the host's answer; one whose runs lie past the block, or a block outside the
/// processes' blocks, hands nothing and is refused. A `SURFACE_DOWNLOAD` hands the host its block
/// and is answered with the host's answer
#[test]
fn surface_pages_hand_the_host_their_runs() {
    type Seen = Arc<Mutex<Vec<(RgfxSurfacePages, Vec<RgfxPageRun>, u32)>>>;
    struct Pages(Seen);
    impl Host for Pages {
        fn raise_interrupt(&self, _vector: u32) {}
        fn features(&self) -> u64 {
            u64::from(RGFX_FEATURE_SURFACE_PAGES) | (1 << 40)
        }
        fn surface_pages(
            &self,
            pages: &RgfxSurfacePages,
            runs: Vec<RgfxPageRun>,
            flags: u32,
            answer: CompleteAnswer,
        ) {
            self.0.lock().push((*pages, runs, flags));
            answer.answer(RGFX_STATUS_OK);
        }
        fn surface_download(&self, block_offset: u64, answer: CompleteAnswer) {
            let runs = vec![];
            let pages = RgfxSurfacePages {
                block_offset,
                ..Default::default()
            };
            self.0.lock().push((pages, runs, u32::MAX));
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
    // A process's pages block: its flags as they are.
    let mut private = pages(BLOCK, BLOCK + 0x1000, 2).as_bytes().to_vec();
    private.extend_from_slice(
        RgfxSurfacePagesCache {
            flags: RGFX_SURFACE_PAGES_PRIVATE,
            reserved_04: 0,
        }
        .as_bytes(),
    );
    let seqno = rig.send(private);
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
        4,
        "only the SURFACE_PAGES whose runs lie in its block, and the download"
    );
    assert_eq!(seen[0].0.block_offset, BLOCK);
    assert_eq!(seen[0].0.context, 3);
    assert_eq!(seen[0].1, runs);
    assert_eq!(seen[0].2, 0, "a message without the flags");
    assert_eq!(
        (seen[1].1.as_slice(), seen[1].2),
        (&runs[..], RGFX_SURFACE_PAGES_CACHED)
    );
    assert_eq!(
        (seen[2].1.as_slice(), seen[2].2),
        (&runs[..], RGFX_SURFACE_PAGES_PRIVATE)
    );
    assert_eq!((seen[3].0.block_offset, seen[3].1.len()), (BLOCK, 0));
}

/// What a [`TakingHost`] records, and what it is set to do.
#[derive(Default)]
struct Taking {
    /// Answer command buffers on a doorbell's thread ([`Host::answer_command_buffer`]).
    takes: AtomicBool,
    /// Each command buffer answered on a doorbell's thread, by seqno: whether its `CB_SUBMITTED`
    /// was posted there.
    answered: Mutex<HashMap<u64, bool>>,
    /// While set, a `BUFFER_CREATE` or a `COMMAND_BUFFER` handed to the host waits.
    shut: AtomicBool,
    /// While set, a `BUFFER_CREATE` handed to the host waits.
    hold_creates: AtomicBool,
    /// Called inside the next answer on a doorbell's thread.
    on_answer: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// Each raise: its vector, and the name of the thread it ran on.
    raised: Mutex<Vec<(u32, Option<String>)>>,
    /// The types of the messages handed to [`Host::stage3_message`], in order.
    handed: Mutex<Vec<u32>>,
    /// Each command buffer carried out, and the name of the thread it was carried out on.
    carried: Mutex<Vec<(u64, Option<String>)>>,
}

/// A host whose command buffers post a `CB_SUBMITTED` and then a `CB_COMPLETE`, as the decoder's
/// do, and which answers them on a doorbell's thread where it is set to: the `CB_SUBMITTED`
/// posted there, the buffer carried out on the queue's thread, which posts the `CB_SUBMITTED`
/// where the doorbell's thread did not.
struct TakingHost(Arc<Taking>);

/// The calling thread's name, any of ring 0's threads' as [`RING0`]: which of them serves a lane
/// is theirs to choose.
fn thread_name() -> Option<String> {
    (std::thread::current().name()).map(|name| match name.strip_prefix(RING0_PREFIX) {
        Some(_) => RING0.to_owned(),
        None => name.to_owned(),
    })
}

/// What [`thread_name`] answers on any of ring 0's threads, which serve the lanes.
const RING0: &str = "reims-vgpu.graphics-ring0-*";

/// The names of ring 0's threads begin so.
const RING0_PREFIX: &str = "reims-vgpu.graphics-ring0-";

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
        while header.r#type == RGFX_CMD_BUFFER_CREATE && self.0.hold_creates.load(Ordering::SeqCst)
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        if header.r#type == RGFX_CMD_COMMAND_BUFFER {
            let (cb, _) = RgfxCommandBuffer::read_from_prefix(message).unwrap();
            if self.0.answered.lock().get(&header.seqno) != Some(&true) {
                completion.post(cb_submitted(header.seqno, cb.cb_obj_id).as_bytes());
            }
            self.0.carried.lock().push((cb.cb_obj_id, thread_name()));
            completion.post(cb_complete(header.seqno, cb.cb_obj_id).as_bytes());
        }
    }
    fn answer_command_buffer(&self, completion: &CompletionPoster, message: &[u8]) -> bool {
        if !self.0.takes.load(Ordering::SeqCst) {
            return false;
        }
        let (cb, _) = RgfxCommandBuffer::read_from_prefix(message).unwrap();
        let seqno = cb.header.seqno;
        let posted = completion.try_post(cb_submitted(seqno, cb.cb_obj_id).as_bytes());
        let first = self.0.answered.lock().insert(seqno, posted).is_none();
        assert!(first, "buffer {seqno} was answered twice");
        if let Some(hook) = self.0.on_answer.lock().take() {
            hook();
        }
        true
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

/// A doorbell write through `mmio_write_taking` answers the queue's next command buffer on the
/// writing thread: its `CB_SUBMITTED` is on the queue's completion ring, its vector raised on
/// that thread, before the write returns; the buffer stays on the ring, and the queue's thread
/// takes it and carries it out, then the buffer published after it, so the queue's events keep
/// its order.
#[test]
fn a_doorbell_write_answers_a_command_buffer_on_its_thread() {
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
    // Both handed over on the queue's thread, the first answered before.
    assert_eq!(
        *taking.handed.lock(),
        vec![
            RGFX_CMD_QUEUE_CREATE,
            RGFX_CMD_COMMAND_BUFFER,
            RGFX_CMD_COMMAND_BUFFER
        ]
    );
    assert_eq!(*taking.answered.lock(), HashMap::from([(1, true)]));
}

/// A command buffer answered on a doorbell's thread is handed over by its queue's thread without
/// waiting for ring 0 again: a create of its process's published after the answer, which the
/// host is still carrying out, does not hold it up, as the doorbell's thread found every ring-0
/// message the buffer waits for carried out.
#[test]
fn an_answered_command_buffer_waits_for_no_later_ring_0_message() {
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let taking = Arc::new(Taking::default());
    taking.takes.store(true, Ordering::SeqCst);
    let mut rig = taking_rig(&taking, QUEUE);
    taking.hold_creates.store(true, Ordering::SeqCst);
    // The create, published on ring 0 inside the answer: after it, and before the queue's thread
    // takes the buffer.
    let mut create = object_message(RGFX_CMD_BUFFER_CREATE, QUEUE + 1);
    let seqno = rig.seqno + 1;
    create[8..16].copy_from_slice(&seqno.to_le_bytes());
    set_size_of(&mut create);
    let data = u64::from(RGFX_CONTROL_RING_DATA_SIZE);
    let at = RGFX_CONTROL_SUBMIT_OFFSET + 256 + rig.produced % data;
    let produced = rig.produced + create.len() as u64;
    let region = Arc::clone(&rig.device.shared.region);
    *taking.on_answer.lock() = Some(Box::new(move || {
        region.write(at, &create);
        (region.atomic_u64(RGFX_CONTROL_SUBMIT_OFFSET)).store(produced, Ordering::SeqCst);
    }));
    publish_unrung(&rig, 0, &[command_buffer(QUEUE, 100, 1)]);
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    assert_eq!(*taking.answered.lock(), HashMap::from([(1, true)]));
    (rig.produced, rig.seqno) = (produced, seqno);
    rig.device
        .mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    let created = |taking: &Taking| taking.handed.lock().contains(&RGFX_CMD_BUFFER_CREATE);
    for _ in 0..2000 {
        if created(&taking) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(created(&taking), "the create never reached the host");
    assert_eq!(
        queue_events(&rig, 2),
        vec![
            (RGFX_EVT_CB_SUBMITTED, 1, 100),
            (RGFX_EVT_CB_COMPLETE, 1, 100),
        ],
        "carried out while the create is held"
    );
    taking.hold_creates.store(false, Ordering::SeqCst);
}

/// A doorbell write through `mmio_write_taking` leaves the buffer to the queue's thread where the
/// host does not answer it, where the queue's thread holds a message of the queue's, and where a
/// ring-0 message of its process's published before it is not yet carried out: the queue's thread
/// hands it to the host as it hands any, in the queue's order and after the ring-0 message.
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

    // The host does not answer it.
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

/// A doorbell write through `mmio_write_taking` answers a command buffer on its thread while the
/// control thread is still in another process's create published before it.
#[test]
fn a_doorbell_write_answers_a_command_buffer_behind_another_process_s_create() {
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    const OTHER: u64 = (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let taking = Arc::new(Taking::default());
    taking.takes.store(true, Ordering::SeqCst);
    let mut rig = taking_rig(&taking, QUEUE);
    taking.hold_creates.store(true, Ordering::SeqCst);
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
    taking.hold_creates.store(false, Ordering::SeqCst);
}

/// A command buffer behind its process's create, which the doorbell leaves to the queue's thread,
/// is offered to the host on the context's lane's thread as the lane carries the create out,
/// before the queue's thread wakes: its `CB_SUBMITTED` is posted and its vector raised there, and
/// the queue's thread then hands it over as one answered, so it is answered once and its events
/// keep their order.
#[test]
fn a_buffer_behind_its_process_s_create_is_answered_on_the_lane_s_thread() {
    const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    let taking = Arc::new(Taking::default());
    taking.takes.store(true, Ordering::SeqCst);
    let mut rig = taking_rig(&taking, QUEUE);
    taking.hold_creates.store(true, Ordering::SeqCst);
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
    assert!(queue_events(&rig, 0).is_empty(), "answered at the doorbell");
    // The queue's thread took it and waits at the gate for the create.
    for _ in 0..2000 {
        if rig.device.shared.gate.counts().1 == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        rig.device.shared.gate.counts().1,
        1,
        "no thread waits at the gate"
    );
    taking.raised.lock().clear();
    taking.hold_creates.store(false, Ordering::SeqCst);
    assert_eq!(
        queue_events(&rig, 2),
        vec![
            (RGFX_EVT_CB_SUBMITTED, 1, 100),
            (RGFX_EVT_CB_COMPLETE, 1, 100),
        ]
    );
    assert_eq!(*taking.answered.lock(), HashMap::from([(1, true)]));
    let raised = taking.raised.lock().clone();
    assert_eq!(
        raised.first(),
        Some(&(3, Some(RING0.to_owned()))),
        "the CB_SUBMITTED is raised on the lane's thread: {raised:?}"
    );
    let carried = taking.carried.lock();
    assert_eq!(carried.len(), 1);
    assert!(
        carried[0]
            .1
            .as_deref()
            .is_some_and(|n| n.starts_with("reims-vgpu.graphics-queue-")),
        "{carried:?}"
    );
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

/// A fuzz of ring 0's lanes (`lanes_fuzz.rs`).
mod lanes_fuzz;
/// The wire round trip with a fake peer (`peer.rs`).
mod peer;
/// What a synchronous create's round trip costs, measured by hand (`round_trip.rs`).
mod round_trip;
/// Ring 0's threads: kept through a steady load's pauses, gone after a burst (`threads.rs`).
mod threads;

// ---------------------------------------------------------------------------------------------
// Ring 0's lanes (`device::lanes`): what crosses contexts waits for the lanes it reads.

/// What an [`OrderHost`] records, in the order it carries each out: a stage-3 message's type, a
/// present (`PRESENT_SURFACE`), a block's release (`BLOCK_RELEASE`), pages (`SURFACE_PAGES`) and
/// download (`SURFACE_DOWNLOAD`), and a context's close (`CONTEXT_CLOSE`), each with the count of
/// the control completion ring's records published as it began; and its gates, open.
#[derive(Default)]
struct OrderSeen {
    order: Arc<Mutex<Vec<(u32, u64)>>>,
    gate: Arc<AtomicBool>,
    /// While shut, a `SURFACE_PAGES` the host carries out waits.
    pages: Arc<AtomicBool>,
    named: Arc<Mutex<Vec<u64>>>,
    region: Arc<OnceLock<Arc<Region>>>,
    /// While set, the host carries a download out where the control thread offers it
    /// ([`Host::try_surface_download`]); else it leaves each to the block's lane.
    here: Arc<AtomicBool>,
    /// The thread each download was carried out on, in order.
    downloaded_on: Arc<Mutex<Vec<Option<String>>>>,
    /// While set, the host has carried out a download offered on a doorbell's thread already
    /// ([`Host::surface_downloaded`]); else it leaves each to the control thread.
    at_doorbell: Arc<AtomicBool>,
    /// While set, a download the control thread offers waits there, the thread awake.
    hold_here: Arc<AtomicBool>,
}

impl OrderSeen {
    fn new() -> OrderSeen {
        let seen = OrderSeen::default();
        seen.gate.store(true, Ordering::SeqCst);
        seen.pages.store(true, Ordering::SeqCst);
        seen
    }

    /// A rig over an [`OrderHost`] of these, started.
    fn rig(&self) -> Rig {
        let host = OrderHost {
            order: Arc::clone(&self.order),
            gate: Arc::clone(&self.gate),
            pages: Arc::clone(&self.pages),
            named: Arc::clone(&self.named),
            region: Arc::clone(&self.region),
            here: Arc::clone(&self.here),
            downloaded_on: Arc::clone(&self.downloaded_on),
            at_doorbell: Arc::clone(&self.at_doorbell),
            hold_here: Arc::clone(&self.hold_here),
        };
        let rig = Rig::with_host(Box::new(host));
        let _ = self.region.set(Arc::clone(&rig.device.shared.region));
        rig.start();
        rig
    }

    /// The types carried out so far, and the resets told ([`RESET_TOLD`]), with no held create's
    /// end ([`HELD_ENDED`]).
    fn types(&self) -> Vec<u32> {
        (self.order.lock().iter())
            .map(|&(t, _)| t)
            .filter(|&t| t != HELD_ENDED)
            .collect()
    }

    /// Waits about 2 s for `type` to be carried out `n` times.
    fn wait_for(&self, r#type: u32, n: usize) {
        for _ in 0..2000 {
            if self.types().iter().filter(|&&t| t == r#type).count() >= n {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!(
            "type {:#x} was not carried out {n} times: {:x?}",
            r#type,
            self.types()
        );
    }

    /// Shuts the gate, sends `context`'s buffer create the host then holds, and waits for the host
    /// to hold it: the context's lane is in it.
    fn hold(&self, rig: &mut Rig, context: u32) {
        self.gate.store(false, Ordering::SeqCst);
        let held = self.types().len();
        rig.send(object_message(
            RGFX_CMD_BUFFER_CREATE,
            (u64::from(context) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 0xfff,
        ));
        for _ in 0..2000 {
            if self.types().len() > held {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("the create was not carried out");
    }

    /// Opens the gate.
    fn release(&self) {
        self.gate.store(true, Ordering::SeqCst);
    }

    /// How many times `type` was carried out so far.
    fn count(&self, r#type: u32) -> usize {
        self.types().iter().filter(|&&t| t == r#type).count()
    }

    /// Shuts the pages' gate, sends `context`'s `SURFACE_PAGES` of `block` the host then holds,
    /// and waits for the host to hold it: the block's lane is in it. Its seqno.
    fn hold_pages(&self, rig: &mut Rig, context: u32, block: u64) -> u64 {
        self.pages.store(false, Ordering::SeqCst);
        let held = self.count(RGFX_CMD_SURFACE_PAGES);
        let seqno = rig.send(surface_pages_message(
            context,
            block,
            RGFX_SURFACE_PAGES_CACHED,
        ));
        self.wait_for(RGFX_CMD_SURFACE_PAGES, held + 1);
        seqno
    }

    /// Opens the pages' gate.
    fn release_pages(&self) {
        self.pages.store(true, Ordering::SeqCst);
    }
}

/// A host that records what it carries out ([`OrderSeen`]), holds a `BUFFER_CREATE` while its gate
/// is shut, answers a render pipeline's create, a present, a release and a close at once, and
/// names an IOSurface by the objects `named` holds.
struct OrderHost {
    order: Arc<Mutex<Vec<(u32, u64)>>>,
    gate: Arc<AtomicBool>,
    pages: Arc<AtomicBool>,
    named: Arc<Mutex<Vec<u64>>>,
    region: Arc<OnceLock<Arc<Region>>>,
    here: Arc<AtomicBool>,
    downloaded_on: Arc<Mutex<Vec<Option<String>>>>,
    at_doorbell: Arc<AtomicBool>,
    hold_here: Arc<AtomicBool>,
}

/// What an [`OrderHost`] records as a held `BUFFER_CREATE` ends.
const HELD_ENDED: u32 = u32::MAX - 1;

impl OrderHost {
    fn record(&self, r#type: u32) {
        let published = self.region.get().map_or(0, |region| {
            region
                .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET)
                .load(Ordering::SeqCst)
        });
        self.order.lock().push((r#type, published));
    }
}

impl Host for OrderHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn context_closed(&self, _context: u32, answer: CompleteAnswer) {
        self.record(RGFX_CMD_CONTEXT_CLOSE);
        answer.answer(RGFX_STATUS_OK);
    }
    fn block_released(&self, _release: &RgfxBlockRelease, answer: CompleteAnswer) {
        self.record(RGFX_CMD_BLOCK_RELEASE);
        answer.answer(RGFX_STATUS_OK);
    }
    fn surface_pages(
        &self,
        _pages: &RgfxSurfacePages,
        _runs: Vec<RgfxPageRun>,
        _flags: u32,
        answer: CompleteAnswer,
    ) {
        self.record(RGFX_CMD_SURFACE_PAGES);
        while !self.pages.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        answer.answer(RGFX_STATUS_OK);
    }
    fn surface_download(&self, _block_offset: u64, answer: CompleteAnswer) {
        self.record(RGFX_CMD_SURFACE_DOWNLOAD);
        let thread = thread_name();
        self.downloaded_on.lock().push(thread);
        answer.answer(RGFX_STATUS_OK);
    }
    fn try_surface_download(
        &self,
        block_offset: u64,
        answer: CompleteAnswer,
    ) -> Result<(), CompleteAnswer> {
        if !self.here.load(Ordering::SeqCst) {
            return Err(answer);
        }
        while self.hold_here.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        self.surface_download(block_offset, answer);
        Ok(())
    }
    fn surface_downloaded(&self, _block_offset: u64) -> bool {
        if !self.at_doorbell.load(Ordering::SeqCst) {
            return false;
        }
        self.record(RGFX_CMD_SURFACE_DOWNLOAD);
        let thread = std::thread::current().name().map(str::to_owned);
        self.downloaded_on.lock().push(thread);
        true
    }
    fn present_surface(&self, _iosurface_id: u32, _generation: Option<u64>, answer: PresentAnswer) {
        self.record(RGFX_CMD_PRESENT_SURFACE);
        answer.answer(RGFX_STATUS_OK);
    }
    fn surface_textures(&self, _iosurface_id: u32) -> Vec<u64> {
        self.named.lock().clone()
    }
    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        self.record(header.r#type);
        match header.r#type {
            RGFX_CMD_BUFFER_CREATE => {
                while !self.gate.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                self.record(HELD_ENDED);
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
            _ => {}
        }
    }
    fn device_reset(&self) {
        self.record(RESET_TOLD);
    }
}

/// A `TEXTURE_FROM_IOSURFACE` of texture `obj_id` over plane `plane` of IOSurface `iosurface_id`.
fn texture_from_iosurface(obj_id: u64, iosurface_id: u32, plane: u32) -> Vec<u8> {
    let mut m = RgfxTextureFromIOSurface::new_zeroed();
    m.header.r#type = RGFX_CMD_TEXTURE_FROM_IOSURFACE;
    m.obj_id = obj_id;
    m.iosurface_id = iosurface_id;
    m.plane = plane;
    m.as_bytes().to_vec()
}

/// A `PRESENT_SURFACE` of IOSurface `iosurface_id` on display 0.
fn present(iosurface_id: u32) -> Vec<u8> {
    let mut m = RgfxPresentSurface::new_zeroed();
    m.header.r#type = RGFX_CMD_PRESENT_SURFACE;
    m.iosurface_id = iosurface_id;
    m.as_bytes().to_vec()
}

/// A `RESOURCE_ID_CHUNK` of chunk `chunk` for `context`.
fn id_chunk(context: u32, chunk: u32) -> Vec<u8> {
    id_grant(RGFX_CMD_RESOURCE_ID_CHUNK, context, chunk)
}

/// A grant of type `type`, `RESOURCE_ID_CHUNK` or `RESOURCE_ID_SMALL_CHUNK`, of chunk `chunk` for
/// `context`.
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

/// A `SURFACE_PAGES` of `block` for `context`, listing no runs, with `flags`.
fn surface_pages_message(context: u32, block: u64, flags: u32) -> Vec<u8> {
    let mut bytes = RgfxSurfacePages {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SURFACE_PAGES,
            size: 0,
            seqno: 0,
        },
        context,
        run_count: 0,
        block_offset: block,
        block_size: 0x1000,
        runs_offset: block,
        length: 0,
    }
    .as_bytes()
    .to_vec();
    bytes.extend_from_slice(
        RgfxSurfacePagesCache {
            flags,
            reserved_04: 0,
        }
        .as_bytes(),
    );
    bytes
}

/// A `SURFACE_DOWNLOAD` of `block`.
fn download_message(block: u64) -> Vec<u8> {
    RgfxSurfaceDownload {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SURFACE_DOWNLOAD,
            size: 0,
            seqno: 0,
        },
        block_offset: block,
    }
    .as_bytes()
    .to_vec()
}

/// A make of `type` over `block` (`BUFFER_OVER_PAGES`, `HEAP_OVER_PAGES` or
/// `BUFFER_OVER_SURFACE_PAGES`) of object `obj_id`.
fn over_block(r#type: u32, obj_id: u64, block: u64) -> Vec<u8> {
    let header = RgfxMsgHeader {
        r#type,
        size: 0,
        seqno: 0,
    };
    match r#type {
        RGFX_CMD_BUFFER_OVER_PAGES => RgfxBufferOverPages {
            header,
            obj_id,
            block_offset: block,
            block_size: 0x1000,
            ..Default::default()
        }
        .as_bytes()
        .to_vec(),
        RGFX_CMD_HEAP_OVER_PAGES => RgfxHeapOverPages {
            header,
            obj_id,
            block_offset: block,
            block_size: 0x1000,
            ..Default::default()
        }
        .as_bytes()
        .to_vec(),
        RGFX_CMD_BUFFER_OVER_SURFACE_PAGES => RgfxBufferOverSurfacePages {
            header,
            obj_id,
            block_offset: block,
            block_size: 0x1000,
            ..Default::default()
        }
        .as_bytes()
        .to_vec(),
        _ => panic!("no make over a block of type {type:#x}"),
    }
}

/// Waits 50 ms, the time a message not held up takes many times over.
fn settle() {
    std::thread::sleep(Duration::from_millis(50));
}

/// A present of a surface waits for another context's `TEXTURE_FROM_IOSURFACE` of the surface's
/// plane 0 published before it, which a lane is still behind: the host shows the newest live
/// texture over the surface, and refuses a present of a surface it holds none over.
#[test]
fn a_present_waits_for_another_context_s_texture_over_its_surface() {
    const A: u32 = 6;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold(&mut rig, A);
    rig.send(texture_from_iosurface(
        (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1,
        7,
        0,
    ));
    rig.send(present(7));
    settle();
    assert_eq!(
        seen.types(),
        [RGFX_CMD_BUFFER_CREATE],
        "the present overtook the texture"
    );
    seen.release();
    seen.wait_for(RGFX_CMD_PRESENT_SURFACE, 1);
    assert_eq!(
        seen.types(),
        [
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_TEXTURE_FROM_IOSURFACE,
            RGFX_CMD_PRESENT_SURFACE
        ]
    );
}

/// A present of a surface waits for another context's destroy of a texture the host names the
/// surface by, and for the close of a context that holds one, each published before it.
#[test]
fn a_present_waits_for_a_destroy_or_a_close_of_a_texture_over_its_surface() {
    const A: u32 = 6;
    const TEXTURE: u64 = (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
    for gone in [
        object_message(RGFX_CMD_OBJECT_DESTROY, TEXTURE),
        context_close(A),
    ] {
        let r#type = u32::from_le_bytes(gone[0..4].try_into().unwrap());
        let seen = OrderSeen::new();
        seen.named.lock().push(TEXTURE);
        let mut rig = seen.rig();
        seen.hold(&mut rig, A);
        rig.send(gone);
        rig.send(present(7));
        settle();
        assert_eq!(
            seen.types(),
            [RGFX_CMD_BUFFER_CREATE],
            "the present overtook type {type:#x}"
        );
        seen.release();
        seen.wait_for(RGFX_CMD_PRESENT_SURFACE, 1);
        assert_eq!(
            seen.types(),
            [RGFX_CMD_BUFFER_CREATE, r#type, RGFX_CMD_PRESENT_SURFACE]
        );
    }
}

/// A present waits for no message that leaves its surface's textures as they are: a destroy of
/// an object the host does not name the surface by, a texture over another surface or another
/// plane, another context's close, or a create; it is carried out while those wait.
#[test]
fn a_present_waits_for_no_message_that_leaves_its_surface_s_textures() {
    const A: u32 = 6;
    const B: u32 = 5;
    let seen = OrderSeen::new();
    seen.named
        .lock()
        .push((u64::from(B) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1);
    let mut rig = seen.rig();
    seen.hold(&mut rig, A);
    let obj = |n: u64| (u64::from(A) << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
    rig.send(object_message(RGFX_CMD_OBJECT_DESTROY, obj(2)));
    rig.send(texture_from_iosurface(obj(3), 8, 0));
    rig.send(texture_from_iosurface(obj(4), 7, 1));
    rig.send(context_close(A));
    rig.send(present(7));
    seen.wait_for(RGFX_CMD_PRESENT_SURFACE, 1);
    assert_eq!(
        seen.types(),
        [RGFX_CMD_BUFFER_CREATE, RGFX_CMD_PRESENT_SURFACE]
    );
    seen.release();
}

/// The release of a block shared with other contexts waits for each other holder's messages
/// published before it (its destroy of an object over the block, its unshare): the host refuses a
/// release while an object of any context lies in the block.
#[test]
fn a_shared_block_s_release_waits_for_its_other_holders() {
    const A: u32 = 6;
    const B: u32 = 5;
    const SURFACE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    rig.send(block_message(RGFX_CMD_BLOCK_SHARE, A, SURFACE, 0x1000));
    rig.send(block_message(RGFX_CMD_BLOCK_SHARE, B, SURFACE, 0x1000));
    seen.hold(&mut rig, A);
    rig.send(object_message(
        RGFX_CMD_OBJECT_DESTROY,
        (u64::from(A) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1,
    ));
    rig.send(block_message(RGFX_CMD_BLOCK_UNSHARE, A, SURFACE, 0x1000));
    let seqno = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, B, SURFACE, 0x1000));
    settle();
    assert_eq!(
        seen.types(),
        [RGFX_CMD_BUFFER_CREATE],
        "the release overtook its other holder's messages"
    );
    seen.release();
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(
        seen.types(),
        [
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_OBJECT_DESTROY,
            RGFX_CMD_BLOCK_RELEASE
        ]
    );
}

/// The release of a block only its context held waits for no other context.
#[test]
fn an_unshared_block_s_release_waits_for_no_other_context() {
    const A: u32 = 6;
    const B: u32 = 5;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    rig.send(block_message(RGFX_CMD_BLOCK_GRANT, B, BLOCK, 0x1000));
    seen.hold(&mut rig, A);
    let seqno = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, B, BLOCK, 0x1000));
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert_eq!(
        seen.types(),
        [RGFX_CMD_BUFFER_CREATE, RGFX_CMD_BLOCK_RELEASE]
    );
    seen.release();
}

/// A grant of the scanout's span waits for every other context's messages published before it: a
/// grant takes the span from whoever held it, so the last one published holds it, as WindowServer's
/// contexts are granted it in turn.
#[test]
fn a_scanout_grant_waits_for_every_context_s_messages_before_it() {
    const A: u32 = 6;
    const B: u32 = 5;
    const SCANOUT: u64 = RGFX_ALLOCATIONS_OFFSET;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold(&mut rig, A);
    rig.send(block_message(RGFX_CMD_BLOCK_GRANT, A, SCANOUT, 0x1000));
    rig.send(block_message(RGFX_CMD_BLOCK_GRANT, B, SCANOUT, 0x1000));
    settle();
    assert!(
        !rig.region().held_by(B, SCANOUT, 0x1000),
        "the later grant overtook the earlier one"
    );
    seen.release();
    rig.lanes_done();
    assert!(rig.region().held_by(B, SCANOUT, 0x1000));
    assert!(!rig.region().held_by(A, SCANOUT, 0x1000));
}

/// A resource id chunk granted to a context after another held it waits for that context's close
/// published before it, a small chunk of it as the whole chunk does, and a new context's first
/// message waits for every close published before it: the kext frees a closed context's chunks
/// and its slot of GPU addresses once it stops waiting for the close's answer, so the close may
/// still hold textures at the chunk's ids and buffers in the slot.
#[test]
fn a_chunk_or_a_new_context_waits_for_the_close_before_it() {
    const A: u32 = 6;
    const B: u32 = 5;
    const C: u32 = 7;
    // B's chunk, of a context with messages before A's (not a new one), then B's small chunk 13,
    // of chunk 3's ids 193 to 256, then C's first message.
    for (late, r#type) in [
        (id_chunk(B, 3), RGFX_CMD_RESOURCE_ID_CHUNK),
        (
            id_grant(RGFX_CMD_RESOURCE_ID_SMALL_CHUNK, B, 13),
            RGFX_CMD_RESOURCE_ID_SMALL_CHUNK,
        ),
        (
            object_message(
                RGFX_CMD_EVENT_CREATE,
                u64::from(C) << RGFX_CONTEXT_OBJ_ID_SHIFT,
            ),
            RGFX_CMD_EVENT_CREATE,
        ),
    ] {
        let seen = OrderSeen::new();
        let mut rig = seen.rig();
        rig.send(object_message(
            RGFX_CMD_EVENT_CREATE,
            u64::from(B) << RGFX_CONTEXT_OBJ_ID_SHIFT,
        ));
        rig.send(id_chunk(A, 3));
        // Each on its own context's lane, in either order.
        seen.wait_for(RGFX_CMD_EVENT_CREATE, 1);
        seen.wait_for(RGFX_CMD_RESOURCE_ID_CHUNK, 1);
        seen.hold(&mut rig, A);
        let held = seen.types();
        rig.send(context_close(A));
        rig.send(late);
        settle();
        assert_eq!(seen.types(), held, "type {type:#x} overtook the close");
        seen.release();
        seen.wait_for(r#type, held.iter().filter(|&&t| t == r#type).count() + 1);
        assert_eq!(seen.types()[held.len()..], [RGFX_CMD_CONTEXT_CLOSE, r#type]);
    }
}

/// A context's first message waits for every close published before it, whichever context has
/// the higher number: the kext numbers contexts at their opens, and a process may post its first
/// message after a process opened later has posted its own.
#[test]
fn a_context_s_first_message_waits_for_the_close_before_it_whatever_its_number() {
    const A: u32 = 6;
    const OPENED_LATER: u32 = 8;
    const OPENED_EARLIER: u32 = 7;
    let obj = |context: u32| u64::from(context) << RGFX_CONTEXT_OBJ_ID_SHIFT;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold(&mut rig, A);
    let held = seen.types();
    rig.send(context_close(A));
    rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(OPENED_LATER)));
    rig.send(object_message(RGFX_CMD_FENCE_CREATE, obj(OPENED_EARLIER)));
    settle();
    assert_eq!(seen.types(), held, "a first message overtook the close");
    seen.release();
    seen.wait_for(RGFX_CMD_EVENT_CREATE, 1);
    seen.wait_for(RGFX_CMD_FENCE_CREATE, 1);
    assert_eq!(seen.types()[held.len()], RGFX_CMD_CONTEXT_CLOSE);
}

/// A context's synchronous create is answered while another context's create is carried out, and
/// before its own context's later messages are: its lane posts the answer before it takes the
/// next.
#[test]
fn a_sync_create_is_answered_before_its_context_s_later_messages_are_carried_out() {
    const A: u32 = 6;
    const B: u32 = 5;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold(&mut rig, A);
    let obj = |n: u64| (u64::from(B) << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
    let seqno = rig.send(object_message(RGFX_CMD_RENDER_PIPELINE_CREATE, obj(1)));
    rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(2)));
    let answer = rig.event().expect("the create was answered");
    let (header, _) = RgfxMsgHeader::read_from_prefix(&answer).unwrap();
    assert_eq!(
        (header.r#type, header.seqno),
        (RGFX_EVT_PIPELINE_REFLECTION, seqno)
    );
    seen.wait_for(RGFX_CMD_EVENT_CREATE, 1);
    let order = seen.order.lock().clone();
    assert_eq!(
        order.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
        [
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_RENDER_PIPELINE_CREATE,
            RGFX_CMD_EVENT_CREATE
        ]
    );
    assert!(
        order[2].1 > order[1].1,
        "the event's create was carried out before the pipeline's answer was posted"
    );
    seen.release();
}

/// Answers of two contexts' messages come in the order each context's are carried out, not ring
/// 0's: the kext matches each answer by its seqno.
#[test]
fn a_context_s_answer_comes_before_one_published_ahead_of_it_by_another_context() {
    const A: u32 = 6;
    const B: u32 = 5;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold(&mut rig, A);
    let first = rig.send(object_message(
        RGFX_CMD_RENDER_PIPELINE_CREATE,
        (u64::from(A) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1,
    ));
    let second = rig.send(object_message(
        RGFX_CMD_RENDER_PIPELINE_CREATE,
        (u64::from(B) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1,
    ));
    let seqno = |bytes: Vec<u8>| RgfxMsgHeader::read_from_prefix(&bytes).unwrap().0.seqno;
    assert_eq!(seqno(rig.event().expect("B's answer")), second);
    seen.release();
    assert_eq!(seqno(rig.event().expect("A's answer")), first);
}

/// A surface's pages the kext hands over from another process's CPU lock, for a maker that has
/// closed, and the download the lock posts right after them without waiting for their answer, are
/// carried out in that order on the block's lane, while the maker's lane is still behind a held
/// message and its close.
#[test]
fn a_closed_maker_s_pages_then_a_download_go_in_order_past_its_held_lane() {
    const MAKER: u32 = 6;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold(&mut rig, MAKER);
    let close = rig.send(context_close(MAKER));
    // A flag the lanes do not read rides along, as the kext's hand-over of put-off pages carries
    // one.
    seen.pages.store(false, Ordering::SeqCst);
    let pages = rig.send(surface_pages_message(
        MAKER,
        BLOCK,
        RGFX_SURFACE_PAGES_CACHED | (1 << 7),
    ));
    let download = rig.send(download_message(BLOCK));
    seen.wait_for(RGFX_CMD_SURFACE_PAGES, 1);
    settle();
    assert_eq!(
        seen.types(),
        [RGFX_CMD_BUFFER_CREATE, RGFX_CMD_SURFACE_PAGES],
        "the download overtook the pages"
    );
    seen.release_pages();
    assert_eq!(rig.complete(), (pages, RGFX_STATUS_OK));
    assert_eq!(rig.complete(), (download, RGFX_STATUS_OK));
    assert_eq!(
        seen.types(),
        [
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_SURFACE_PAGES,
            RGFX_CMD_SURFACE_DOWNLOAD
        ],
        "the maker's lane is still behind its held message"
    );
    seen.release();
    assert_eq!(rig.complete(), (close, RGFX_STATUS_OK));
    assert_eq!(seen.count(RGFX_CMD_CONTEXT_CLOSE), 1);
}

/// Waits about 2 s for the lane of the block at `block` to have nothing left to carry out: a
/// message answered inside the host's call is still running there until the call returns.
fn wait_block_lane_idle(rig: &Rig, block: u64) {
    for _ in 0..2000 {
        if rig.device.shared.lanes.lane_idle(Key::Block(block)) {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("the block's lane did not go idle");
}

/// A download of a block whose lane has nothing left to carry out is offered to the host on the
/// control thread, which carries it out there where the host takes it, and on the block's lane
/// where the host leaves it
#[test]
fn a_download_of_a_block_whose_lane_is_idle_is_carried_out_on_the_control_thread() {
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let seen = OrderSeen::new();
    seen.here.store(true, Ordering::SeqCst);
    let mut rig = seen.rig();
    let pages = rig.send(surface_pages_message(5, BLOCK, RGFX_SURFACE_PAGES_CACHED));
    assert_eq!(rig.complete(), (pages, RGFX_STATUS_OK));
    wait_block_lane_idle(&rig, BLOCK);
    let download = rig.send(download_message(BLOCK));
    assert_eq!(rig.complete(), (download, RGFX_STATUS_OK));
    seen.here.store(false, Ordering::SeqCst);
    let download = rig.send(download_message(BLOCK));
    assert_eq!(rig.complete(), (download, RGFX_STATUS_OK));
    assert_eq!(
        *seen.downloaded_on.lock(),
        [
            Some("reims-vgpu.graphics-control".to_owned()),
            Some(RING0.to_owned())
        ]
    );
}

/// A download of a block whose lane is still carrying out the block's `SURFACE_PAGES` goes to
/// the lane, behind them, though the host would carry it out on the control thread; a download
/// of another block, whose lane is idle, is carried out there meanwhile
#[test]
fn a_download_behind_its_block_s_pages_on_the_lane_is_carried_out_after_them_there() {
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    const OTHER: u64 = BLOCK + 0x10_0000;
    let seen = OrderSeen::new();
    seen.here.store(true, Ordering::SeqCst);
    let mut rig = seen.rig();
    let pages = seen.hold_pages(&mut rig, 5, BLOCK);
    let behind = rig.send(download_message(BLOCK));
    let other = rig.send(download_message(OTHER));
    seen.wait_for(RGFX_CMD_SURFACE_DOWNLOAD, 1);
    settle();
    assert_eq!(
        seen.types(),
        [RGFX_CMD_SURFACE_PAGES, RGFX_CMD_SURFACE_DOWNLOAD],
        "the other block's download alone"
    );
    seen.release_pages();
    let answers = [rig.complete(), rig.complete(), rig.complete()];
    let mut answered: Vec<u64> = answers.iter().map(|&(seqno, _)| seqno).collect();
    answered.sort_unstable();
    assert_eq!(answered, [pages, behind, other]);
    assert!(answers.iter().all(|&(_, status)| status == RGFX_STATUS_OK));
    assert_eq!(
        seen.types(),
        [
            RGFX_CMD_SURFACE_PAGES,
            RGFX_CMD_SURFACE_DOWNLOAD,
            RGFX_CMD_SURFACE_DOWNLOAD
        ]
    );
    assert_eq!(
        *seen.downloaded_on.lock(),
        [
            Some("reims-vgpu.graphics-control".to_owned()),
            Some(RING0.to_owned())
        ],
        "the other block's on the control thread, then this block's on its lane"
    );
}

/// Waits about 2 s for the control thread to sleep with nothing in flight (`Submit::asleep`).
fn wait_control_asleep(rig: &Rig) {
    for _ in 0..2000 {
        if rig.device.shared.submit.lock().asleep {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("the control thread did not go to sleep");
}

/// Publishes `command` on the control ring, its seqno the next, without a doorbell: the seqno.
fn publish_next(rig: &mut Rig, mut command: Vec<u8>) -> u64 {
    rig.seqno += 1;
    command[8..16].copy_from_slice(&rig.seqno.to_le_bytes());
    set_size_of(&mut command);
    rig.publish(&command);
    rig.seqno
}

/// Whether the control completion ring holds a record the guest has not taken.
fn answer_waiting(rig: &Rig) -> bool {
    rig.region()
        .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET)
        .load(Ordering::SeqCst)
        != rig.consumed
}

/// A ring-0 doorbell write through `mmio_write_taking` answers a download at the ring's head on
/// the writing thread, where the host has carried it out already and the control thread sleeps:
/// its `COMPLETE` is on the completion ring before the write returns, and ring 0's count moves
/// past it, so the control thread takes the message after it next, in order;
/// a plain `mmio_write` of the doorbell leaves it to the control thread
#[test]
fn a_landed_download_is_answered_on_its_doorbell_s_thread() {
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let seen = OrderSeen::new();
    seen.at_doorbell.store(true, Ordering::SeqCst);
    let mut rig = seen.rig();
    let pages = rig.send(surface_pages_message(5, BLOCK, RGFX_SURFACE_PAGES_CACHED));
    assert_eq!(rig.complete(), (pages, RGFX_STATUS_OK));
    wait_block_lane_idle(&rig, BLOCK);
    wait_control_asleep(&rig);
    let download = publish_next(&mut rig, download_message(BLOCK));
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    assert!(answer_waiting(&rig), "no answer when the write returned");
    assert_eq!(rig.complete(), (download, RGFX_STATUS_OK));
    assert_eq!(*seen.downloaded_on.lock(), [thread_name()]);
    let through = rig
        .region()
        .atomic_u64(RGFX_CONTROL_SUBMIT_OFFSET + 0x40)
        .load(Ordering::SeqCst);
    assert_eq!(through, rig.produced, "ring 0's count is past the download");
    // The next message goes to the control thread, in order, and so does a download rung by the
    // plain write.
    let present = rig.send(present(0));
    assert_eq!(rig.complete(), (present, RGFX_STATUS_OK));
    wait_control_asleep(&rig);
    let plain = publish_next(&mut rig, download_message(BLOCK));
    rig.device
        .mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    assert_eq!(rig.complete(), (plain, RGFX_STATUS_OK));
    assert_eq!(seen.count(RGFX_CMD_SURFACE_DOWNLOAD), 2);
    assert_ne!(seen.downloaded_on.lock()[1], thread_name());
}

/// A ring-0 doorbell's thread takes no download the control thread must take: one the host has
/// not carried out yet goes to the control thread; so does one behind another message at the
/// ring's head; and one published while the control thread is awake, carrying out a download it
/// was offered, waits for it, though the host would answer it at the doorbell
#[test]
fn a_doorbell_s_thread_leaves_a_download_the_control_thread_must_take() {
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let seen = OrderSeen::new();
    seen.here.store(true, Ordering::SeqCst);
    let mut rig = seen.rig();
    let control = Some("reims-vgpu.graphics-control".to_owned());

    // Not carried out yet.
    wait_control_asleep(&rig);
    let first = publish_next(&mut rig, download_message(BLOCK));
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    assert_eq!(rig.complete(), (first, RGFX_STATUS_OK));
    assert_eq!(*seen.downloaded_on.lock(), std::slice::from_ref(&control));

    // Behind a present at the head.
    seen.at_doorbell.store(true, Ordering::SeqCst);
    wait_control_asleep(&rig);
    let shown = publish_next(&mut rig, present(0));
    let behind = publish_next(&mut rig, download_message(BLOCK));
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    let answers = [rig.complete(), rig.complete()];
    assert!(answers.contains(&(shown, RGFX_STATUS_OK)), "{answers:?}");
    assert!(answers.contains(&(behind, RGFX_STATUS_OK)), "{answers:?}");
    assert_eq!(seen.downloaded_on.lock().last(), Some(&control));

    // The control thread awake in a download it was offered: the next waits for it.
    seen.at_doorbell.store(false, Ordering::SeqCst);
    seen.hold_here.store(true, Ordering::SeqCst);
    wait_control_asleep(&rig);
    let held = rig.send(download_message(BLOCK));
    for _ in 0..2000 {
        if !rig.device.shared.submit.lock().asleep {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    settle();
    seen.at_doorbell.store(true, Ordering::SeqCst);
    let next = publish_next(&mut rig, download_message(BLOCK));
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    assert!(
        !answer_waiting(&rig),
        "answered while the control thread held one before it"
    );
    seen.hold_here.store(false, Ordering::SeqCst);
    assert_eq!(rig.complete(), (held, RGFX_STATUS_OK));
    assert_eq!(rig.complete(), (next, RGFX_STATUS_OK));
    let on = seen.downloaded_on.lock();
    assert_eq!(on[on.len() - 2..], [control.clone(), control]);
}

/// Each message that reads what a block's lane makes of the block waits for the lane through its
/// position, whichever context posts it: a buffer or a heap over a pages block, a buffer over a
/// surface's pages, a download and a release. A message of another context's, or one over another
/// block, goes meanwhile.
#[test]
fn what_reads_a_block_s_pages_waits_for_the_block_s_lane() {
    const MAKER: u32 = 6;
    const OTHER: u32 = 4;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    const ELSEWHERE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x30_0000;
    let obj = |context: u32, n: u64| (u64::from(context) << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold_pages(&mut rig, MAKER, BLOCK);
    // Each on a context's lane of its own, so none waits behind another.
    let kinds = [
        RGFX_CMD_BUFFER_OVER_PAGES,
        RGFX_CMD_HEAP_OVER_PAGES,
        RGFX_CMD_BUFFER_OVER_SURFACE_PAGES,
    ];
    for (n, &kind) in kinds.iter().enumerate() {
        rig.send(over_block(kind, obj(10 + n as u32, 1), BLOCK));
    }
    let release = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, 13, BLOCK, 0x1000));
    let download = rig.send(download_message(BLOCK));
    rig.send(over_block(
        RGFX_CMD_BUFFER_OVER_PAGES,
        obj(OTHER, 1),
        ELSEWHERE,
    ));
    rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(OTHER, 2)));
    seen.wait_for(RGFX_CMD_EVENT_CREATE, 1);
    settle();
    assert_eq!(
        seen.types(),
        [
            RGFX_CMD_SURFACE_PAGES,
            RGFX_CMD_BUFFER_OVER_PAGES,
            RGFX_CMD_EVENT_CREATE
        ],
        "a message that reads the block went before its pages, or another waited"
    );
    seen.release_pages();
    let mut answered: Vec<(u64, u32)> = (0..3).map(|_| rig.complete()).collect();
    answered.sort_unstable();
    assert_eq!(
        answered,
        [
            (1, RGFX_STATUS_OK),
            (release, RGFX_STATUS_OK),
            (download, RGFX_STATUS_OK)
        ]
    );
    rig.lanes_done();
    let types = seen.types();
    let pages_done = types
        .iter()
        .position(|&t| t == RGFX_CMD_EVENT_CREATE)
        .unwrap();
    let mut after: Vec<u32> = types[pages_done + 1..].to_vec();
    after.sort_unstable();
    let mut want = vec![
        RGFX_CMD_BUFFER_OVER_PAGES,
        RGFX_CMD_HEAP_OVER_PAGES,
        RGFX_CMD_BUFFER_OVER_SURFACE_PAGES,
        RGFX_CMD_BLOCK_RELEASE,
        RGFX_CMD_SURFACE_DOWNLOAD,
    ];
    want.sort_unstable();
    assert_eq!(after, want);
}

/// A context's later messages wait for its `SURFACE_PAGES` on the block's lane, as when they
/// shared the context's lane; another context's do not.
#[test]
fn a_context_s_later_messages_wait_for_its_pages_on_the_block_s_lane() {
    const MAKER: u32 = 6;
    const OTHER: u32 = 4;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let obj = |context: u32, n: u64| (u64::from(context) << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold_pages(&mut rig, MAKER, BLOCK);
    rig.send(object_message(RGFX_CMD_SAMPLER_CREATE, obj(MAKER, 1)));
    rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(OTHER, 1)));
    seen.wait_for(RGFX_CMD_EVENT_CREATE, 1);
    settle();
    assert_eq!(
        seen.types(),
        [RGFX_CMD_SURFACE_PAGES, RGFX_CMD_EVENT_CREATE],
        "the maker's later message went before its pages"
    );
    seen.release_pages();
    seen.wait_for(RGFX_CMD_SAMPLER_CREATE, 1);
    assert_eq!(
        seen.types(),
        [
            RGFX_CMD_SURFACE_PAGES,
            RGFX_CMD_EVENT_CREATE,
            RGFX_CMD_SAMPLER_CREATE
        ]
    );
}

/// A lane behind a block's held pages is woken once, by their carrying out: never by the jobs other
/// processes' lanes finish meanwhile, which move nothing it waits for.
#[test]
fn a_lane_behind_a_block_s_pages_is_woken_only_by_their_carrying_out() {
    const MAKER: u32 = 6;
    const OTHER: u32 = 4;
    const OTHERS: usize = 20;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let obj = |context: u32, n: u64| (u64::from(context) << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold_pages(&mut rig, MAKER, BLOCK);
    rig.send(object_message(RGFX_CMD_SAMPLER_CREATE, obj(MAKER, 1)));
    let lanes = Arc::clone(&rig.device.shared.lanes);
    let blocked = || lanes.blocked(Key::Context(MAKER));
    for _ in 0..2000 {
        if blocked().0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let (waits, before) = blocked();
    assert!(waits, "the maker's lane never waited for its pages");
    for n in 0..OTHERS as u64 {
        rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(OTHER, n + 1)));
    }
    seen.wait_for(RGFX_CMD_EVENT_CREATE, OTHERS);
    settle();
    assert_eq!(
        blocked(),
        (true, before),
        "another process's {OTHERS} jobs woke the lane behind the pages"
    );
    seen.release_pages();
    seen.wait_for(RGFX_CMD_SAMPLER_CREATE, 1);
    assert_eq!(blocked(), (false, before + 1));
}

/// A block's `SURFACE_PAGES` waits for its context's messages published before it that name the
/// block, which its context's lane is still behind; not for one over another block, nor for the
/// lane's other messages.
#[test]
fn a_block_s_pages_wait_for_what_their_context_made_over_the_block_before() {
    const MAKER: u32 = 6;
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    const ELSEWHERE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x30_0000;
    let obj = |n: u64| (u64::from(MAKER) << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold(&mut rig, MAKER);
    rig.send(over_block(RGFX_CMD_BUFFER_OVER_PAGES, obj(1), BLOCK));
    let pages = rig.send(surface_pages_message(
        MAKER,
        BLOCK,
        RGFX_SURFACE_PAGES_PRIVATE,
    ));
    let elsewhere = rig.send(surface_pages_message(
        MAKER,
        ELSEWHERE,
        RGFX_SURFACE_PAGES_PRIVATE,
    ));
    assert_eq!(rig.complete(), (elsewhere, RGFX_STATUS_OK));
    settle();
    assert_eq!(
        seen.types(),
        [RGFX_CMD_BUFFER_CREATE, RGFX_CMD_SURFACE_PAGES],
        "the block's pages went before what was made over the block"
    );
    seen.release();
    assert_eq!(rig.complete(), (pages, RGFX_STATUS_OK));
    assert_eq!(
        seen.types(),
        [
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_SURFACE_PAGES,
            RGFX_CMD_BUFFER_OVER_PAGES,
            RGFX_CMD_SURFACE_PAGES
        ]
    );
}

/// A `SURFACE_PAGES` of `block`, of 0x2000 bytes, for `context`, with `flags`, listing `runs`,
/// which it writes into the block's second page in `rig`'s region.
fn pages_over(rig: &Rig, context: u32, block: u64, runs: &[RgfxPageRun], flags: u32) -> Vec<u8> {
    rig.region().write(block + 0x1000, runs.as_bytes());
    let mut bytes = surface_pages_message(context, block, flags);
    let mut pages = RgfxSurfacePages::read_from_prefix(&bytes).unwrap().0;
    pages.run_count = runs.len() as u32;
    pages.block_size = 0x2000;
    pages.runs_offset = block + 0x1000;
    pages.length = runs.iter().map(|run| run.length).sum();
    bytes[..size_of::<RgfxSurfacePages>()].copy_from_slice(pages.as_bytes());
    bytes
}

/// A block's `SURFACE_PAGES` waits for a cached block's `BLOCK_RELEASE` published before it over
/// a page of its own, which its context's lane is still behind, whichever context's lane it is:
/// the release copies the cache into the pages, and the kext makes a surface's new block over
/// them once it has waited 10 s for the release's answer. Not for the release of a block over
/// other pages, nor of one never described cached, which copies nothing.
#[test]
fn a_block_s_pages_wait_for_a_cached_block_s_release_over_their_pages() {
    const OWNER: u32 = 6;
    const MAKER: u32 = 5;
    const OLD: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    const PLAIN: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x30_0000;
    const NEW: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x40_0000;
    const ELSEWHERE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x50_0000;
    let run = |address: u64, length: u64| RgfxPageRun { address, length };
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    let old_runs = [run(0x10_0000, 0x2000), run(0x20_0000, 0x1000)];
    let old = rig.send(pages_over(
        &rig,
        OWNER,
        OLD,
        &old_runs,
        RGFX_SURFACE_PAGES_CACHED,
    ));
    assert_eq!(rig.complete(), (old, RGFX_STATUS_OK));
    let plain = rig.send(pages_over(&rig, OWNER, PLAIN, &[run(0x30_0000, 0x1000)], 0));
    assert_eq!(rig.complete(), (plain, RGFX_STATUS_OK));
    seen.hold(&mut rig, OWNER);
    let released = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, OWNER, OLD, 0x2000));
    let plain_released = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, OWNER, PLAIN, 0x2000));
    // Over the old block's last page, uncached, for another context: the kext's new block of the
    // surface past its wait.
    let new_runs = [run(0x40_0000, 0x1000), run(0x20_0000, 0x1000)];
    let new = rig.send(pages_over(&rig, MAKER, NEW, &new_runs, 0));
    // Over the page right past the old block's first run, and the uncached block's page.
    let other_runs = [run(0x30_0000, 0x1000), run(0x10_2000, 0x1000)];
    let elsewhere = rig.send(pages_over(&rig, MAKER, ELSEWHERE, &other_runs, 0));
    assert_eq!(rig.complete(), (elsewhere, RGFX_STATUS_OK));
    settle();
    assert_eq!(
        seen.types(),
        [
            RGFX_CMD_SURFACE_PAGES,
            RGFX_CMD_SURFACE_PAGES,
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_SURFACE_PAGES
        ],
        "the new block's pages overtook the release of the cached block over them"
    );
    seen.release();
    let mut answered = [rig.complete(), rig.complete(), rig.complete()];
    answered.sort_unstable();
    assert_eq!(
        answered,
        [
            (released, RGFX_STATUS_OK),
            (plain_released, RGFX_STATUS_OK),
            (new, RGFX_STATUS_OK)
        ]
    );
    // The old block's release is the owner's first; the release of the uncached block may come
    // before or after the new block's pages.
    let types = seen.types();
    let first_release = types.iter().position(|&t| t == RGFX_CMD_BLOCK_RELEASE);
    let new_pages = types.iter().rposition(|&t| t == RGFX_CMD_SURFACE_PAGES);
    assert!(
        first_release.zip(new_pages).is_some_and(|(r, p)| r < p) && types.len() == 7,
        "the new block's pages went before the release over them: {types:x?}"
    );
}

/// A shared block's release waits for the lanes of its holders also where the kext hands the
/// block's pages over after the shares: the pages leave the block's holders as they are.
#[test]
fn pages_handed_over_leave_a_shared_block_s_holders() {
    const A: u32 = 6;
    const B: u32 = 5;
    const SURFACE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    rig.send(block_message(RGFX_CMD_BLOCK_SHARE, A, SURFACE, 0x1000));
    rig.send(block_message(RGFX_CMD_BLOCK_SHARE, B, SURFACE, 0x1000));
    seen.hold(&mut rig, A);
    let pages = rig.send(surface_pages_message(B, SURFACE, RGFX_SURFACE_PAGES_CACHED));
    assert_eq!(rig.complete(), (pages, RGFX_STATUS_OK));
    let release = rig.send(block_message(RGFX_CMD_BLOCK_RELEASE, B, SURFACE, 0x1000));
    settle();
    assert_eq!(
        seen.types(),
        [RGFX_CMD_BUFFER_CREATE, RGFX_CMD_SURFACE_PAGES],
        "the release overtook its other holder's messages"
    );
    seen.release();
    assert_eq!(rig.complete(), (release, RGFX_STATUS_OK));
}

/// A reset drops what the lanes have not begun: a lane's message begun before it goes on, the
/// host is told of the reset only once that message has ended, and the messages behind it, in its
/// context's lane or in another lane waiting for it, are never carried out.
#[test]
fn a_reset_drops_what_the_lanes_have_not_begun() {
    const A: u32 = 6;
    let seen = OrderSeen::new();
    let mut rig = seen.rig();
    seen.hold(&mut rig, A);
    rig.send(object_message(
        RGFX_CMD_RENDER_PIPELINE_CREATE,
        (u64::from(A) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1,
    ));
    // The present's lane waits for the texture, in A's.
    rig.send(texture_from_iosurface(
        (u64::from(A) << RGFX_CONTEXT_OBJ_ID_SHIFT) | 2,
        7,
        0,
    ));
    rig.send(present(7));
    settle();
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    settle();
    assert_eq!(
        seen.types(),
        [RGFX_CMD_BUFFER_CREATE],
        "told before the create ended"
    );
    seen.release();
    seen.wait_for(RESET_TOLD, 1);
    settle();
    let order: Vec<u32> = seen.order.lock().iter().map(|&(t, _)| t).collect();
    assert_eq!(order, [RGFX_CMD_BUFFER_CREATE, HELD_ENDED, RESET_TOLD]);
}
