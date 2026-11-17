//! VideoToolbox's paravirtual messages (docs/graphics-kext/design.md, "Decided", 17): a
//! `VT_MESSAGE` reaches the host checked, and is answered `VT_STATUS`; the host's own messages
//! reach a guest that took `VT_MESSAGES`, and no other: in the VideoToolbox ring, each with a
//! `VT_RING` on the control completion ring, where it took `VT_RING` too and the ring has room,
//! else as `VT_MESSAGE` events.

use super::*;

/// What [`VtHost`] was handed: each message, and the link it came with.
type Handed = Arc<Mutex<Vec<(VtMessage, VtLink)>>>;

/// What [`VtHost`] was told of the device's rings.
type Degraded = Arc<Mutex<Vec<RingDegradation>>>;

/// A host that offers `VT_MESSAGES`, sends one message of its own for each `VT_MESSAGE`, then
/// answers it with the message's length, negated.
struct VtHost {
    handed: Handed,
    degraded: Degraded,
}

impl Host for VtHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn features(&self) -> u64 {
        RGFX_FEATURE_VT_MESSAGES
    }
    fn ring_degraded(&self, kind: RingDegradation) {
        self.degraded.lock().push(kind);
    }
    fn vt_message(&self, message: VtMessage, answer: VtAnswer, link: VtLink) {
        let length = message.message.length;
        link.send(b"vtpv-reply");
        self.handed.lock().push((message, link));
        answer.answer(-(length as i32));
    }
}

fn vt_rig() -> (Rig, Handed) {
    let (rig, handed, _) = vt_rig_told();
    (rig, handed)
}

/// [`vt_rig`], and what its host was told of the rings.
fn vt_rig_told() -> (Rig, Handed, Degraded) {
    let handed = Handed::default();
    let degraded = Degraded::default();
    let host = VtHost {
        handed: Arc::clone(&handed),
        degraded: Arc::clone(&degraded),
    };
    (Rig::with_host(Box::new(host)), handed, degraded)
}

/// The guest's take of the VideoToolbox ring from count `from` through `produced`: each
/// `VT_MESSAGE` record's bytes, past the pads; then the ring's `consumer` is `produced`.
fn vt_ring_records(rig: &Rig, from: u64, produced: u64) -> Vec<Vec<u8>> {
    let data = u64::from(RGFX_VT_RING_DATA_SIZE);
    let at = |count: u64| RGFX_VT_RING_OFFSET + 256 + count % data;
    let mut records = Vec::new();
    let mut count = from;
    while count < produced {
        let start = rig.region().read(at(count), 8).unwrap();
        let r#type = u32::from_le_bytes(start[0..4].try_into().unwrap());
        let size = u32::from_le_bytes(start[4..8].try_into().unwrap());
        if r#type != RGFX_MSG_PAD {
            let record = rig.region().read(at(count), size as usize).unwrap();
            let (h, rest) = RgfxVtHostMessage::read_from_prefix(&record).unwrap();
            assert_eq!(
                (h.header.r#type, h.header.size),
                (RGFX_EVT_VT_MESSAGE, size)
            );
            records.push(rest[..h.length as usize].to_vec());
        }
        count += u64::from(size);
    }
    assert_eq!(count, produced, "the records end at the count");
    (rig.region())
        .atomic_u64(RGFX_VT_RING_OFFSET + 0x40)
        .store(produced, Ordering::SeqCst);
    records
}

/// The next event, as a `VT_RING`: the count it carries.
fn vt_ring(rig: &mut Rig) -> u64 {
    let r = RgfxVtRing::read_from_bytes(&rig.event().expect("a VT_RING")).unwrap();
    assert_eq!((r.header.r#type, r.header.size), (RGFX_EVT_VT_RING, 24));
    assert_eq!(r.header.seqno, 0);
    r.produced
}

/// The next event, as a `VT_MESSAGE` event: its message's bytes.
fn vt_event(rig: &mut Rig) -> Vec<u8> {
    let event = rig.event().expect("a VT_MESSAGE");
    let (h, rest) = RgfxVtHostMessage::read_from_prefix(&event).unwrap();
    assert_eq!((h.header.r#type, h.header.seqno), (RGFX_EVT_VT_MESSAGE, 0));
    rest[..h.length as usize].to_vec()
}

/// The guest's driver takes the bits `bits` of `FEATURES_HI` and starts the device.
fn take_and_start(rig: &Rig, bits: u64) {
    rig.device
        .mmio_write(RGFX_REG_DRIVER_FEATURES_HI.into(), (bits >> 32) as u32);
    rig.start();
}

/// A `VT_MESSAGE` of `length` bytes in `runs`, lending a surface of one page for each of the
/// lends `surfaces`.
fn vt_message(length: u64, runs: &[(u64, u64)], surfaces: &[u64]) -> Vec<u8> {
    let mut command = RgfxVtMessage {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_VT_MESSAGE,
            ..Default::default()
        },
        length,
        run_count: runs.len() as u32,
        surface_count: surfaces.len() as u32,
        ..RgfxVtMessage::default()
    };
    command.surfaces[..surfaces.len()].copy_from_slice(surfaces);
    let mut bytes = command.as_bytes().to_vec();
    for &(address, length) in runs {
        bytes.extend_from_slice(RgfxPageRun { address, length }.as_bytes());
    }
    for &lend in surfaces {
        let surface = RgfxVtSurface {
            lend,
            length: 0x4000,
            run_count: 1,
            ..RgfxVtSurface::default()
        };
        bytes.extend_from_slice(surface.as_bytes());
        bytes.extend_from_slice(
            RgfxPageRun {
                address: 0x40000 * lend,
                length: 0x4000,
            }
            .as_bytes(),
        );
    }
    bytes
}

/// The next event, as a `VT_STATUS`: its `seqno` and status.
fn status(rig: &mut Rig) -> (u64, i32) {
    let s = RgfxVtStatus::read_from_bytes(&rig.event().expect("a VT_STATUS")).unwrap();
    assert_eq!((s.header.r#type, s.header.size), (RGFX_EVT_VT_STATUS, 24));
    (s.header.seqno, s.status)
}

/// `VT_MESSAGES` (`FEATURES_HI` bit 20) is offered where the host offers it, and only there can
/// the guest take it; `VT_RING` (bit 22), the device's, with it.
#[test]
fn vt_messages_are_offered_where_the_host_offers_them() {
    let (rig, _) = vt_rig();
    let hi = rig.device.mmio_read(RGFX_REG_FEATURES_HI.into());
    assert_eq!(
        hi,
        ((RGFX_FEATURE_VT_MESSAGES | RGFX_FEATURE_VT_RING) >> 32) as u32
    );
    assert_eq!(RGFX_FEATURE_VT_RING >> 32, 1 << 22);
    let plain = Rig::new();
    assert_eq!(plain.device.mmio_read(RGFX_REG_FEATURES_HI.into()), 0);
}

/// A `VT_MESSAGE` reaches the host with its runs and surfaces, and the host's messages and its
/// answer reach the guest in the order the host gave them: as `VT_MESSAGE` events, for a guest
/// that took `VT_MESSAGES` alone.
#[test]
fn a_vt_message_reaches_the_host_and_is_answered_in_order() {
    let (mut rig, handed) = vt_rig();
    take_and_start(&rig, RGFX_FEATURE_VT_MESSAGES);
    let seqno = rig.send(vt_message(100, &[(0x4000, 64), (0x8000, 0x4000)], &[3, 4]));
    assert_eq!(vt_event(&mut rig), b"vtpv-reply");
    assert_eq!(status(&mut rig), (seqno, -100));
    let handed = handed.lock();
    let (message, _) = &handed[0];
    assert_eq!(message.runs, vec![(0x4000, 64), (0x8000, 0x4000)]);
    let lends: Vec<u64> = message.surfaces.iter().map(|s| s.surface.lend).collect();
    assert_eq!(lends, [3, 4]);
    assert_eq!(message.surfaces[1].runs, vec![(0x100000, 0x4000)]);
}

/// A `VT_MESSAGE` from a guest that did not take `VT_MESSAGES`, or one whose runs do not hold its
/// length, is answered as one the device could not read, and never reaches the host.
#[test]
fn an_unreadable_vt_message_is_answered_malformed() {
    let (mut rig, handed) = vt_rig();
    rig.start();
    let seqno = rig.send(vt_message(64, &[(0x4000, 0x4000)], &[]));
    assert_eq!(status(&mut rig), (seqno, RGFX_VT_STATUS_MALFORMED));
    let (mut taken, taken_handed) = vt_rig();
    take_and_start(&taken, RGFX_FEATURE_VT_MESSAGES);
    let seqno = taken.send(vt_message(0x4001, &[(0x4000, 0x4000)], &[]));
    assert_eq!(status(&mut taken), (seqno, RGFX_VT_STATUS_MALFORMED));
    assert!(handed.lock().is_empty() && taken_handed.lock().is_empty());
}

/// A host message is dropped for a guest that did not take `VT_MESSAGES` and once the device is
/// reset, and a link made before the reset says it is stale.
#[test]
fn a_host_message_reaches_only_a_guest_that_took_vt_messages() {
    let (mut rig, handed) = vt_rig();
    take_and_start(&rig, RGFX_FEATURE_VT_MESSAGES);
    let seqno = rig.send(vt_message(48, &[(0x4000, 48)], &[]));
    rig.event().expect("the host's VT_MESSAGE");
    assert_eq!(status(&mut rig).0, seqno);
    let link = handed.lock()[0].1.clone();
    assert!(!link.stale());
    assert!(link.send(b"again"));
    rig.event().expect("the second VT_MESSAGE");
    rig.device.reset();
    assert!(link.stale());
    assert!(!link.send(b"after the reset"));
}

/// A guest that took `VT_RING` gets the host's messages in the VideoToolbox ring, each followed on
/// the control completion ring by a `VT_RING` of the ring's count through it, before the answer
/// the host gave after it; a reset puts the ring back as new.
#[test]
fn a_guest_that_took_the_vt_ring_gets_the_host_messages_there_in_order() {
    let (mut rig, handed) = vt_rig();
    take_and_start(&rig, RGFX_FEATURE_VT_MESSAGES | RGFX_FEATURE_VT_RING);
    let seqno = rig.send(vt_message(100, &[(0x4000, 0x4000)], &[]));
    // 24 bytes of record and 10 of message, padded to 40.
    let first = vt_ring(&mut rig);
    assert_eq!(first, 40);
    assert_eq!(status(&mut rig), (seqno, -100));
    assert_eq!(vt_ring_records(&rig, 0, first), [b"vtpv-reply".to_vec()]);
    let link = handed.lock()[0].1.clone();
    assert!(link.send(b"again") && link.send(b"and again"));
    let (second, third) = (vt_ring(&mut rig), vt_ring(&mut rig));
    assert_eq!((second, third), (first + 32, first + 72));
    assert_eq!(
        vt_ring_records(&rig, first, third),
        [b"again".to_vec(), b"and again".to_vec()]
    );
    assert!(rig.quiet_for(20));
    rig.device.reset();
    for field in [0, 0x40] {
        let count = rig.region().atomic_u64(RGFX_VT_RING_OFFSET + field);
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }
    assert!(!link.send(b"after the reset"));
}

/// A host message that finds the VideoToolbox ring with no room goes onto the control completion
/// ring as a `VT_MESSAGE` event, after the `VT_RING` of the messages before it, and the spell is
/// told once; once the guest has taken the ring, the next goes there again, past a pad to the
/// ring's end.
#[test]
fn a_host_message_the_vt_ring_has_no_room_for_goes_on_the_control_ring() {
    let (mut rig, handed, degraded) = vt_rig_told();
    take_and_start(&rig, RGFX_FEATURE_VT_MESSAGES | RGFX_FEATURE_VT_RING);
    let seqno = rig.send(vt_message(48, &[(0x4000, 48)], &[]));
    let mut produced = vt_ring(&mut rig);
    assert_eq!(status(&mut rig).0, seqno);
    let link = handed.lock()[0].1.clone();
    // The longest messages, 16408 bytes of record each: 127 fit beside the first's 40 bytes.
    let longest: Vec<u8> = (0..RGFX_VT_HOST_MESSAGE_MAX).map(|i| i as u8).collect();
    for _ in 0..127 {
        assert!(link.send(&longest));
        let next = vt_ring(&mut rig);
        assert_eq!(next, produced + 16408);
        produced = next;
    }
    assert!(degraded.lock().is_empty());
    for _ in 0..2 {
        assert!(link.send(&longest));
        assert_eq!(vt_event(&mut rig), longest);
    }
    assert_eq!(*degraded.lock(), [RingDegradation::VtRingFull]);
    let taken = vt_ring_records(&rig, 0, produced);
    assert_eq!(taken.len(), 128);
    assert!(taken[1..].iter().all(|m| *m == longest));
    // The 13296 bytes left at the ring's end are a pad: the record starts the ring again.
    assert!(link.send(&longest));
    let wrapped = vt_ring(&mut rig);
    assert_eq!(wrapped, produced + 13296 + 16408);
    assert_eq!(
        vt_ring_records(&rig, produced, wrapped),
        std::slice::from_ref(&longest)
    );
    assert_eq!(degraded.lock().len(), 1);
}
