//! The scaler device against hand-built requests, written from the guest
//! kext's builder and Apple's handler, on the CPU stand-in.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::backend::ReferenceBackend;
use super::capabilities::{Capabilities, FormatClass, LENGTH};
use super::format::{Packing, ScalerFormat};
use super::plan::{FormatOf, Matrix, Plan, Siting};
use super::reference::{self, Space};
use super::request::*;
use super::*;
use crate::runtime::StdRuntime;
use crate::surfaces::{GuestSurface, PlaneLayout, SurfaceError, SurfaceLayout};
use crate::sync::Mutex;

/// Guest RAM at `BASE`, `memory.len()` bytes; anything else is not RAM.
struct Ram {
    memory: Mutex<Vec<u8>>,
}

const BASE: u64 = 0x1000_0000;

impl Ram {
    fn new(size: usize) -> Ram {
        Ram {
            memory: Mutex::new(vec![0; size]),
        }
    }

    fn range(&self, address: u64, length: u64) -> Option<std::ops::Range<usize>> {
        let len = self.memory.lock().len() as u64;
        let start = address.checked_sub(BASE)?;
        let end = start.checked_add(length.max(1))?;
        (end <= len).then(|| start as usize..(start + length) as usize)
    }

    fn bytes(&self, address: u64, length: usize) -> Vec<u8> {
        let m = self.memory.lock();
        let start = (address - BASE) as usize;
        m[start..start + length].to_vec()
    }

    fn fill(&self, address: u64, bytes: &[u8]) {
        let mut m = self.memory.lock();
        let start = (address - BASE) as usize;
        m[start..start + bytes.len()].copy_from_slice(bytes);
    }
}

impl GuestMemory for Ram {
    fn is_ram(&self, address: u64, length: u64) -> bool {
        self.range(address, length).is_some()
    }

    fn read(&self, address: u64, bytes: &mut [u8]) -> bool {
        let Some(range) = self.range(address, bytes.len() as u64) else {
            return false;
        };
        let m = self.memory.lock();
        bytes.copy_from_slice(&m[range]);
        true
    }

    fn write(&self, address: u64, bytes: &[u8]) -> bool {
        let Some(range) = self.range(address, bytes.len() as u64) else {
            return false;
        };
        let mut m = self.memory.lock();
        m[range].copy_from_slice(bytes);
        true
    }
}

/// A surface in host memory, and how many holds it has.
struct HostSurface {
    layout: SurfaceLayout,
    bytes: Arc<Mutex<Vec<u8>>>,
    holds: Arc<AtomicUsize>,
    /// Made read-only for its clients by the guest: lent for reading only,
    /// as the glue's platforms lend such a surface.
    read_only: bool,
    /// How many times a device's write of a surface was told of.
    wrote: Arc<AtomicUsize>,
}

impl GuestSurface for HostSurface {
    fn layout(&self) -> SurfaceLayout {
        self.layout.clone()
    }

    fn write(&self, write: &mut dyn FnMut(&mut [u8])) -> Result<(), SurfaceError> {
        if self.read_only {
            return Err(SurfaceError::Unreachable(
                "read-only for its clients".into(),
            ));
        }
        write(&mut self.bytes.lock());
        Ok(())
    }

    fn read(&self, read: &mut dyn FnMut(&[u8])) -> Result<(), SurfaceError> {
        read(&self.bytes.lock());
        Ok(())
    }

    fn device_wrote(&self) {
        self.wrote.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for HostSurface {
    fn drop(&mut self) {
        self.holds.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A surface's layout and bytes, by mapping ID.
type SurfaceTable = HashMap<u64, (SurfaceLayout, Arc<Mutex<Vec<u8>>>)>;

/// What runs, once, inside a surface's retain, before it answers.
type OnRetain = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct HostSurfaces {
    surfaces: Mutex<SurfaceTable>,
    holds: Arc<AtomicUsize>,
    /// How many times a device's write of a surface was told of.
    wrote: Arc<AtomicUsize>,
    /// The IDs the guest made read-only for their clients.
    read_only: Mutex<HashSet<u64>>,
    /// What the next retain of each ID runs first, as the IOSurface
    /// device's interrupt runs the host's code while a retain waits for
    /// that device's queue.
    on_retain: Mutex<HashMap<u64, OnRetain>>,
}

impl HostSurfaces {
    fn add(
        &self,
        id: u64,
        format: ScalerFormat,
        width: u32,
        height: u32,
        picture: &reference::Picture,
    ) -> Arc<Mutex<Vec<u8>>> {
        let (layout, size) = layout(format, width, height);
        let of = FormatOf::of(&layout).unwrap_or_else(|status| panic!("refused: {status:#x}"));
        let mut bytes = vec![0u8; size];
        reference::encode(
            &of,
            &mut bytes,
            picture,
            0,
            0,
            Space::Rgb,
            Matrix::Bt601,
            Siting::CENTRED,
        );
        let bytes = Arc::new(Mutex::new(bytes));
        self.surfaces
            .lock()
            .insert(id, (layout, Arc::clone(&bytes)));
        bytes
    }
}

impl Surfaces for HostSurfaces {
    fn retain(&self, mapping_id: MappingId) -> Result<Box<dyn GuestSurface>, SurfaceError> {
        let on_retain = self.on_retain.lock().remove(&mapping_id.0);
        if let Some(on_retain) = on_retain {
            on_retain();
        }
        let surfaces = self.surfaces.lock();
        let (layout, bytes) = surfaces
            .get(&mapping_id.0)
            .ok_or(SurfaceError::Unknown(mapping_id.0))?;
        self.holds.fetch_add(1, Ordering::SeqCst);
        let read_only = (self.read_only.lock()).contains(&mapping_id.0);
        Ok(Box::new(HostSurface {
            layout: layout.clone(),
            bytes: Arc::clone(bytes),
            holds: Arc::clone(&self.holds),
            read_only,
            wrote: Arc::clone(&self.wrote),
        }))
    }
}

fn layout(format: ScalerFormat, width: u32, height: u32) -> (SurfaceLayout, usize) {
    let row = |bytes: usize| bytes.div_ceil(64) * 64;
    match format.packing() {
        Packing::Biplanar420 => {
            let s = format.bytes_per_sample();
            let (luma, chroma) = (
                row(width as usize * s),
                row(width.div_ceil(2) as usize * 2 * s),
            );
            let luma_size = luma * height as usize;
            let planes = vec![
                PlaneLayout {
                    offset: 0,
                    bytes_per_row: luma,
                    width,
                    height,
                },
                PlaneLayout {
                    offset: luma_size,
                    bytes_per_row: chroma,
                    width: width.div_ceil(2),
                    height: height.div_ceil(2),
                },
            ];
            let layout = SurfaceLayout {
                pixel_format: format.fourcc(),
                width,
                height,
                planes,
                bytes_per_row: 0,
            };
            (layout, luma_size + chroma * height.div_ceil(2) as usize)
        }
        _ => {
            let bytes = if format.is_ycbcr() {
                width.div_ceil(2) as usize * 4
            } else {
                width as usize * 4
            };
            let bpr = row(bytes);
            let layout = SurfaceLayout {
                pixel_format: format.fourcc(),
                width,
                height,
                planes: Vec::new(),
                bytes_per_row: bpr,
            };
            (layout, bpr * height as usize)
        }
    }
}

fn caps() -> Capabilities {
    Capabilities {
        scaler_version: 0,
        formats: vec![FormatClass::TwoPlanes420, FormatClass::Argb8888],
        min_source: 4,
        max_source: (4096, 4096),
        min_destination: 4,
        max_destination: (4096, 4096),
        max_scale: [8, 8, 8, 4],
        xy_offsets_with_tiling: (false, false),
    }
}

fn device(surfaces: &Arc<HostSurfaces>) -> ScalerDevice {
    ScalerDevice::new(
        Arc::clone(surfaces) as Arc<dyn Surfaces>,
        Arc::new(ReferenceBackend {
            capabilities: caps(),
        }),
    )
}

fn capabilities_request(address: u64, length: u64) -> Vec<u8> {
    let mut r = vec![0u8; CAPABILITIES_SIZE];
    put_u32(&mut r, TYPE, TYPE_CAPABILITIES);
    put_u64(&mut r, CAPABILITIES_ADDRESS, address);
    put_u64(&mut r, CAPABILITIES_HOST_POINTER, 0x5555);
    put_u64(&mut r, CAPABILITIES_LENGTH, length);
    r
}

fn transform_request(source: u64, destination: u64) -> Vec<u8> {
    let mut r = vec![0u8; TRANSFORM_SIZE];
    put_u32(&mut r, TYPE, TYPE_TRANSFORM);
    put_u32(&mut r, TRANSFORM, 0x41);
    put_u32(&mut r, TRANSFORM + 4, 0x42);
    put_u64(&mut r, TRANSFORM + 0x20, 1 << 13);
    put_u64(&mut r, SOURCE_MAPPING_ID, source);
    put_u64(&mut r, DESTINATION_MAPPING_ID, destination);
    r
}

fn reply(outcome: Outcome) -> Vec<u8> {
    match outcome {
        Outcome::Reply(r) => r,
        other => {
            assert!(matches!(other, Outcome::Reply(_)), "not a reply: {other:?}");
            Vec::new()
        }
    }
}

#[test]
fn malformed_elements_need_a_reset() {
    let surfaces = Arc::new(HostSurfaces::default());
    let d = device(&surfaces);
    let ram = Ram::new(0x1000);
    assert_eq!(
        d.process(&[], 0, &ram),
        Outcome::NeedsReset("Invalid buffer size")
    );
    assert_eq!(
        d.process(&[0; 8], 7, &ram),
        Outcome::NeedsReset("Write buffer too small")
    );
    assert_eq!(
        d.process(&[1, 0, 0, 0], 4, &ram),
        Outcome::NeedsReset("Invalid header")
    );
    assert_eq!(
        d.process(&[1, 0, 0, 0, 0, 0, 0, 0], 8, &ram),
        Outcome::NeedsReset("Invalid capabilities request")
    );
    let mut large = vec![0u8; MAX_REQUEST_SIZE + 1];
    large[0] = 9;
    assert_eq!(
        d.process(&large, large.len(), &ram),
        Outcome::NeedsReset("Request too large")
    );
    let mut unknown = vec![0u8; MAX_REQUEST_SIZE];
    unknown[0] = 9;
    assert_eq!(
        d.process(&unknown, unknown.len(), &ram),
        Outcome::Reply(unknown.clone()),
        "an unknown type at the bound is replied unchanged"
    );
    let mut short = transform_request(1, 2);
    short.truncate(0x600);
    assert_eq!(
        d.process(&short, 0x600, &ram),
        Outcome::NeedsReset("Invalid transform request")
    );
}

#[test]
fn an_unknown_type_is_replied_unchanged() {
    let surfaces = Arc::new(HostSurfaces::default());
    let d = device(&surfaces);
    let request = [9, 0, 0, 0, 0xaa, 0xbb, 0xcc, 0xdd, 1, 2, 3];
    assert_eq!(
        d.process(&request, 64, &Ram::new(0x10)),
        Outcome::Reply(request.to_vec())
    );
}

#[test]
fn the_capabilities_go_where_the_guest_asked_and_their_size_comes_back() {
    let surfaces = Arc::new(HostSurfaces::default());
    let d = device(&surfaces);
    let ram = Ram::new(0x2000);
    let r = capabilities_request(BASE + 0x100, LENGTH as u64);
    assert_eq!(
        d.process(&r, 0x20, &ram),
        Outcome::NeedsReset("Invalid capabilities request: no accelerator")
    );
    assert_eq!(d.start(), Ok(()));
    let out = reply(d.process(&r, 0x20, &ram));
    assert_eq!(out.len(), 0x20);
    assert_eq!(u32_at(&out, STATUS), 0);
    assert_eq!(u64_at(&out, CAPABILITIES_LENGTH), LENGTH as u64);
    assert_eq!(
        u64_at(&out, CAPABILITIES_HOST_POINTER),
        0x5555,
        "the guest's word, not a host pointer"
    );
    assert_eq!(ram.bytes(BASE + 0x100, LENGTH), caps().to_bytes().to_vec());

    // More room than the capabilities: the capabilities alone; less: as
    // much as there is room for.
    let out = reply(d.process(&capabilities_request(BASE, 0x400), 0x20, &ram));
    assert_eq!(u64_at(&out, CAPABILITIES_LENGTH), LENGTH as u64);
    let ram = Ram::new(0x2000);
    let out = reply(d.process(&capabilities_request(BASE, 0x10), 0x20, &ram));
    assert_eq!(u64_at(&out, CAPABILITIES_LENGTH), 0x10);
    assert_eq!(ram.bytes(BASE + 0x10, 0x10), vec![0; 0x10]);

    // Room that is not all RAM is a reset.
    let r = capabilities_request(BASE + 0x1f00, LENGTH as u64);
    assert_eq!(
        d.process(&r, 0x20, &ram),
        Outcome::NeedsReset("Invalid capabilities request")
    );
    d.reset();
    let r = capabilities_request(BASE, LENGTH as u64);
    assert_eq!(
        d.process(&r, 0x20, &ram),
        Outcome::NeedsReset("Invalid capabilities request: no accelerator")
    );
}

#[test]
fn a_transform_scales_between_two_held_surfaces() {
    let surfaces = Arc::new(HostSurfaces::default());
    let picture = reference::test_picture(64, 32);
    surfaces.add(7, ScalerFormat::Nv12VideoRange, 64, 32, &picture);
    let grey = reference::flat_picture(32, 16, [0.5, 0.5, 0.5, 1.0]);
    let destination = surfaces.add(9, ScalerFormat::Bgra, 32, 16, &grey);
    let d = device(&surfaces);
    assert_eq!(d.start(), Ok(()));
    let r = transform_request(7, 9);
    let out = reply(d.process(&r, TRANSFORM_SIZE, &Ram::new(0)));
    assert_eq!(u32_at(&out, STATUS), io_return::SUCCESS);
    assert_eq!(
        (u32_at(&out, TRANSFORM), u32_at(&out, TRANSFORM + 4)),
        (0x41, 0x42),
        "the guest's IDs stay"
    );
    assert_eq!(
        surfaces.holds.load(Ordering::SeqCst),
        0,
        "both holds end with the request"
    );

    // The destination is the reference's.
    let (sl, _) = layout(ScalerFormat::Nv12VideoRange, 64, 32);
    let (dl, size) = layout(ScalerFormat::Bgra, 32, 16);
    let plan = Plan::new(&Transform::read(&r), &sl, &dl)
        .unwrap_or_else(|status| panic!("refused: {status:#x}"));
    let mut source = vec![0u8; layout(ScalerFormat::Nv12VideoRange, 64, 32).1];
    reference::encode(
        &plan.source_format,
        &mut source,
        &picture,
        0,
        0,
        Space::Rgb,
        Matrix::Bt601,
        Siting::CENTRED,
    );
    let mut expected = vec![0u8; size];
    reference::encode(
        &plan.destination_format,
        &mut expected,
        &grey,
        0,
        0,
        Space::Rgb,
        Matrix::Bt601,
        Siting::CENTRED,
    );
    assert_eq!(reference::run(&plan, &source, &mut expected), Ok(()));
    assert_eq!(*destination.lock(), expected);
}

/// A surface the guest made read-only for its clients is read as a source
/// and never written as a destination: kIOReturnNotWritable, with its bytes
/// as they were. Apple's host hands such a destination to its hardware.
#[test]
fn a_read_only_surface_is_a_source_and_never_a_destination() {
    let surfaces = Arc::new(HostSurfaces::default());
    let picture = reference::test_picture(64, 32);
    surfaces.add(7, ScalerFormat::Nv12VideoRange, 64, 32, &picture);
    let grey = reference::flat_picture(32, 16, [0.5, 0.5, 0.5, 1.0]);
    let destination = surfaces.add(9, ScalerFormat::Bgra, 32, 16, &grey);
    let before = destination.lock().clone();
    surfaces.read_only.lock().insert(7);
    let d = device(&surfaces);
    assert_eq!(d.start(), Ok(()));
    let out = reply(d.process(&transform_request(7, 9), TRANSFORM_SIZE, &Ram::new(0)));
    assert_eq!(
        u32_at(&out, STATUS),
        io_return::SUCCESS,
        "a read-only source"
    );
    let scaled = destination.lock().clone();
    assert_ne!(scaled, before);

    surfaces.read_only.lock().insert(9);
    let out = reply(d.process(&transform_request(7, 9), TRANSFORM_SIZE, &Ram::new(0)));
    assert_eq!(u32_at(&out, STATUS), io_return::NOT_WRITABLE);
    assert_eq!(*destination.lock(), scaled, "the destination is as it was");
    assert_eq!(surfaces.holds.load(Ordering::SeqCst), 0);
}

#[test]
fn a_surface_that_is_not_there_is_not_found() {
    let surfaces = Arc::new(HostSurfaces::default());
    surfaces.add(
        7,
        ScalerFormat::Bgra,
        16,
        16,
        &reference::test_picture(16, 16),
    );
    let d = device(&surfaces);
    assert_eq!(d.start(), Ok(()));
    let out = reply(d.process(&transform_request(7, 8), TRANSFORM_SIZE, &Ram::new(0)));
    assert_eq!(u32_at(&out, STATUS), io_return::NOT_FOUND);
    assert_eq!(surfaces.holds.load(Ordering::SeqCst), 0);
    let mut r = transform_request(7, 7);
    put_u32(&mut r, TRANSFORM + 0x60, 8);
    put_u32(&mut r, TRANSFORM + 0x68, 16);
    put_u32(&mut r, TRANSFORM + 0x6c, 16);
    let out = reply(d.process(&r, TRANSFORM_SIZE, &Ram::new(0)));
    assert_eq!(
        u32_at(&out, STATUS),
        io_return::BAD_ARGUMENT,
        "a window that leaves its surface"
    );
}

#[test]
fn a_ratio_past_the_capabilities_is_a_bad_argument_and_leaves_the_destination() {
    let surfaces = Arc::new(HostSurfaces::default());
    surfaces.add(
        7,
        ScalerFormat::Bgra,
        16,
        64,
        &reference::test_picture(16, 64),
    );
    let grey = reference::flat_picture(16, 8, [0.5, 0.5, 0.5, 1.0]);
    let destination = surfaces.add(9, ScalerFormat::Bgra, 16, 8, &grey);
    let before = destination.lock().clone();
    let d = device(&surfaces);
    assert_eq!(d.start(), Ok(()));
    // 8x down the frame, past caps()'s MaxVDownscale of 4.
    let out = reply(d.process(&transform_request(7, 9), TRANSFORM_SIZE, &Ram::new(0)));
    assert_eq!(u32_at(&out, STATUS), io_return::BAD_ARGUMENT);
    assert_eq!(*destination.lock(), before);
    assert_eq!(surfaces.holds.load(Ordering::SeqCst), 0);
    // A quarter turn makes it 8x down across, within MaxHDownscale's 8.
    let mut r = transform_request(7, 9);
    put_u64(&mut r, TRANSFORM + 0x20, (1 << 13) | 4);
    let destination = surfaces.add(
        9,
        ScalerFormat::Bgra,
        8,
        16,
        &reference::flat_picture(8, 16, [0.5; 4]),
    );
    let out = reply(d.process(&r, TRANSFORM_SIZE, &Ram::new(0)));
    assert_eq!(u32_at(&out, STATUS), io_return::SUCCESS);
    drop(destination);
}

#[test]
fn a_transform_before_driver_ok_needs_a_reset() {
    let surfaces = Arc::new(HostSurfaces::default());
    let d = device(&surfaces);
    assert_eq!(
        d.process(&transform_request(1, 2), TRANSFORM_SIZE, &Ram::new(0)),
        Outcome::NeedsReset("Invalid transform request")
    );
}

/// Has the next retain of `id` run `call` on `device` on another thread,
/// as a host's IOSurface device's interrupt may while the retain waits for
/// that device's queue, and wait 10 s at most for it to return: whether it
/// did, once the retain has run.
fn call_from_inside_a_retain(
    surfaces: &HostSurfaces,
    id: u64,
    device: &Arc<ScalerDevice>,
    call: fn(&ScalerDevice),
) -> Arc<Mutex<Option<bool>>> {
    let returned = Arc::new(Mutex::new(None));
    let on_retain: OnRetain = Box::new({
        let device = Arc::downgrade(device);
        let returned = Arc::clone(&returned);
        move || {
            let (done, came) = std::sync::mpsc::channel();
            let calling = StdRuntime::thread("test-interrupt", move || {
                if let Some(device) = device.upgrade() {
                    call(&device);
                }
                let _ = done.send(());
            });
            assert!(calling.is_ok());
            *returned.lock() = Some(came.recv_timeout(Duration::from_secs(10)).is_ok());
        }
    });
    surfaces.on_retain.lock().insert(id, on_retain);
    returned
}

/// Two 16x16 BGRA surfaces, 7 a picture and 9 grey, and a started device
/// over them; 9's bytes.
fn two_surfaces_started() -> (Arc<HostSurfaces>, Arc<ScalerDevice>, Arc<Mutex<Vec<u8>>>) {
    let surfaces = Arc::new(HostSurfaces::default());
    let picture = reference::test_picture(16, 16);
    surfaces.add(7, ScalerFormat::Bgra, 16, 16, &picture);
    let grey = reference::flat_picture(16, 16, [0.5, 0.5, 0.5, 1.0]);
    let destination = surfaces.add(9, ScalerFormat::Bgra, 16, 16, &grey);
    let d = Arc::new(device(&surfaces));
    assert_eq!(d.start(), Ok(()));
    (surfaces, d, destination)
}

/// The two surfaces are retained outside the accelerator's lock. A host
/// whose IOSurface device's interrupt resets the scaler, as the C API lets
/// it, while a transform waits in a retain for that device's queue, has
/// the reset return: under the lock, the reset would wait for the
/// transform, and the transform for the interrupt, for ever. The reset
/// lands between the retains and the transform, which is then a
/// needs-reset, as a transform after a reset is, with both surfaces let
/// go and the destination as it was.
#[test]
fn a_reset_from_inside_a_surfaces_retain_returns_and_the_transform_needs_a_reset() {
    let (surfaces, d, destination) = two_surfaces_started();
    let before = destination.lock().clone();
    let returned = call_from_inside_a_retain(&surfaces, 9, &d, ScalerDevice::reset);
    assert_eq!(
        d.process(&transform_request(7, 9), TRANSFORM_SIZE, &Ram::new(0)),
        Outcome::NeedsReset("Invalid transform request")
    );
    assert_eq!(*returned.lock(), Some(true), "the reset returned");
    assert_eq!(*destination.lock(), before);
    assert_eq!(surfaces.holds.load(Ordering::SeqCst), 0);
}

/// A reset and a start from inside a retain land between the retains and
/// the transform too: the transform runs on the new accelerator.
#[test]
fn a_restart_from_inside_a_surfaces_retain_transforms_on_the_new_accelerator() {
    let (surfaces, d, destination) = two_surfaces_started();
    let before = destination.lock().clone();
    let returned = call_from_inside_a_retain(&surfaces, 9, &d, |d| {
        d.reset();
        assert_eq!(d.start(), Ok(()));
    });
    let out = reply(d.process(&transform_request(7, 9), TRANSFORM_SIZE, &Ram::new(0)));
    assert_eq!(u32_at(&out, STATUS), io_return::SUCCESS);
    assert_eq!(
        *returned.lock(),
        Some(true),
        "the reset and the start returned"
    );
    assert_ne!(*destination.lock(), before, "the destination was written");
    assert_eq!(surfaces.holds.load(Ordering::SeqCst), 0);
}

#[test]
fn the_completion_and_events_are_cleared_in_the_reply() {
    let surfaces = Arc::new(HostSurfaces::default());
    let d = device(&surfaces);
    assert_eq!(d.start(), Ok(()));
    let mut r = transform_request(1, 2);
    for offset in [8, 0x10, 0x18] {
        put_u64(&mut r, TRANSFORM + offset, 0x1234);
    }
    put_u32(&mut r, TRANSFORM + 0x78, 0x99);
    put_u64(&mut r, TRANSFORM + 0x98, 5);
    let out = reply(d.process(&r, TRANSFORM_SIZE, &Ram::new(0)));
    for offset in [8, 0x10, 0x18, 0x78, 0x98] {
        assert_eq!(u64_at(&out, TRANSFORM + offset), 0, "+{offset:#x}");
    }
}

/// Pair `i` of a request, its lists' first entries at `input` and
/// `output` in RAM.
fn side_buffers(r: &mut [u8], i: usize, input: (i32, u32, u64), output: (i32, u32, u64)) {
    r[SIDE_BUFFERS_USED] = 1;
    let at = SIDE_BUFFERS + SIDE_BUFFER_PAIR_LENGTH * i;
    put_u32(r, at, input.0 as u32);
    put_u64(r, at + 8, 0xdead_0000);
    put_u32(r, at + 0x10, input.1);
    put_u32(r, at + 0x14, output.0 as u32);
    put_u64(r, at + 0x18, 0xbeef_0000);
    put_u32(r, at + 0x20, output.1);
    let list = |k: usize, address: u64, length: u32, r: &mut [u8]| {
        let at = LISTS + LIST_LENGTH * k;
        put_u64(r, at, address);
        put_u32(r, at + 8, length);
        put_u32(r, at + 0x80, u32::from(length != 0));
    };
    list(2 * i, input.2, input.1, r);
    list(2 * i + 1, output.2, output.1, r);
}

#[test]
fn side_buffers_are_gathered_and_scattered_back() {
    let surfaces = Arc::new(HostSurfaces::default());
    surfaces.add(
        7,
        ScalerFormat::Bgra,
        16,
        16,
        &reference::test_picture(16, 16),
    );
    let d = device(&surfaces);
    assert_eq!(d.start(), Ok(()));
    let ram = Ram::new(0x4000);
    ram.fill(BASE, &[0x5a; 0x238]);
    ram.fill(BASE + 0x1000, &[0xff; 0x17c]);
    let mut r = transform_request(7, 7);
    side_buffers(&mut r, 0, (1, 0x238, BASE), (1, 0x17c, BASE + 0x1000));
    let out = reply(d.process(&r, TRANSFORM_SIZE, &ram));
    assert_eq!(u32_at(&out, STATUS), 0);
    assert_eq!(
        ram.bytes(BASE + 0x1000, 0x17c),
        vec![0; 0x17c],
        "the outbound buffer, which this host leaves zero"
    );
    assert_eq!(
        u64_at(&out, SIDE_BUFFERS + 8),
        0xdead_0000,
        "the guest's word, not a host pointer"
    );
}

#[test]
fn side_buffers_over_their_limits_need_a_reset_and_long_lists_end_the_vm() {
    let surfaces = Arc::new(HostSurfaces::default());
    let d = device(&surfaces);
    assert_eq!(d.start(), Ok(()));
    let ram = Ram::new(0x10000);
    let run = |input: (i32, u32, u64), output: (i32, u32, u64)| {
        let mut r = transform_request(1, 2);
        side_buffers(&mut r, 1, input, output);
        d.process(&r, TRANSFORM_SIZE, &ram)
    };
    let reset = Outcome::NeedsReset("Invalid transform request");
    assert_eq!(run((7, 8, BASE), (0, 0, 0)), reset);
    assert_eq!(run((1, 0xbd11, BASE), (0, 0, 0)), reset);
    assert_eq!(run((0, 0, 0), (4, 8, BASE)), reset);
    assert_eq!(run((0, 0, 0), (1, 0x1009, BASE)), reset);
    assert_eq!(
        run((2, 8, BASE + 0x20000), (0, 0, 0)),
        reset,
        "a segment that is not RAM"
    );
    assert!(
        matches!(run((-1, 8, BASE), (0, 0, 0)), Outcome::Reply(_)),
        "a negative kind passes the signed limit"
    );
    let mut r = transform_request(1, 2);
    side_buffers(&mut r, 0, (1, 8, BASE), (0, 0, 0));
    put_u32(&mut r, LISTS + 0x80, 9);
    assert_eq!(
        d.process(&r, TRANSFORM_SIZE, &ram),
        Outcome::EndVm("Security Assertion Hit: ")
    );
    // A list longer than its buffer.
    let mut r = transform_request(1, 2);
    side_buffers(&mut r, 0, (1, 8, BASE), (0, 0, 0));
    put_u32(&mut r, LISTS + 8, 16);
    assert_eq!(d.process(&r, TRANSFORM_SIZE, &ram), reset);
    // The flag other than 1: no side buffers at all.
    let mut r = transform_request(1, 2);
    side_buffers(&mut r, 0, (7, 8, BASE), (0, 0, 0));
    r[SIDE_BUFFERS_USED] = 2;
    assert!(matches!(
        d.process(&r, TRANSFORM_SIZE, &ram),
        Outcome::Reply(_)
    ));
}

/// A backend whose device misses its deadline: it keeps the job's hold in
/// `still_writing`, as a backend keeps it with its work, and fails the
/// transform.
struct StuckBackend {
    still_writing: Arc<Mutex<Vec<GuestHold>>>,
}

impl ScalerBackend for StuckBackend {
    fn create_accelerator(&self) -> Result<Box<dyn Accelerator>, String> {
        Ok(Box::new(StuckBackend {
            still_writing: Arc::clone(&self.still_writing),
        }))
    }
}

impl Accelerator for StuckBackend {
    fn capabilities(&self) -> Capabilities {
        caps()
    }

    fn transform(&mut self, job: &mut TransformJob<'_>) -> io_return::IoReturn {
        self.still_writing.lock().push(job.hold.clone());
        io_return::ERROR
    }
}

/// A transform whose device missed its deadline answers its error, keeps
/// both surfaces held while the device may still write the destination,
/// and tells the host of that write once the device is done, as the
/// backend lets its hold go (`crate::device_wait::GuestHold`).
#[test]
fn a_write_past_the_deadline_is_told_of_once_the_device_is_done() {
    let surfaces = Arc::new(HostSurfaces::default());
    let picture = reference::test_picture(64, 32);
    surfaces.add(7, ScalerFormat::Nv12VideoRange, 64, 32, &picture);
    let grey = reference::flat_picture(32, 16, [0.5, 0.5, 0.5, 1.0]);
    surfaces.add(9, ScalerFormat::Bgra, 32, 16, &grey);
    let still_writing = Arc::new(Mutex::new(Vec::new()));
    let d = ScalerDevice::new(
        Arc::clone(&surfaces) as Arc<dyn Surfaces>,
        Arc::new(StuckBackend {
            still_writing: Arc::clone(&still_writing),
        }),
    );
    assert_eq!(d.start(), Ok(()));
    let out = reply(d.process(&transform_request(7, 9), TRANSFORM_SIZE, &Ram::new(0)));
    assert_eq!(u32_at(&out, STATUS), io_return::ERROR);
    assert_eq!(surfaces.holds.load(Ordering::SeqCst), 2, "both stay held");
    assert_eq!(
        surfaces.wrote.load(Ordering::SeqCst),
        0,
        "told before the device is done"
    );
    still_writing.lock().clear();
    assert_eq!(
        surfaces.wrote.load(Ordering::SeqCst),
        1,
        "the write is told of once"
    );
    assert_eq!(
        surfaces.holds.load(Ordering::SeqCst),
        0,
        "both let go after"
    );
}
