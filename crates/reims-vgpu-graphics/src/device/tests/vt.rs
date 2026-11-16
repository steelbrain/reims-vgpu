//! VideoToolbox's paravirtual messages (docs/graphics-kext/design.md, "Decided", 17): a
//! `VT_MESSAGE` reaches the host checked, and is answered `VT_STATUS`; the host's own messages
//! reach a guest that took `VT_MESSAGES` as `VT_MESSAGE` events, and no other.

use super::*;

/// What [`VtHost`] was handed: each message, and the link it came with.
type Handed = Arc<Mutex<Vec<(VtMessage, VtLink)>>>;

/// A host that offers `VT_MESSAGES`, sends one message of its own for each `VT_MESSAGE`, then
/// answers it with the message's length, negated.
struct VtHost(Handed);

impl Host for VtHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn features(&self) -> u64 {
        RGFX_FEATURE_VT_MESSAGES
    }
    fn vt_message(&self, message: VtMessage, answer: VtAnswer, link: VtLink) {
        let length = message.message.length;
        link.send(b"vtpv-reply");
        self.0.lock().push((message, link));
        answer.answer(-(length as i32));
    }
}

fn vt_rig() -> (Rig, Handed) {
    let handed = Handed::default();
    (
        Rig::with_host(Box::new(VtHost(Arc::clone(&handed)))),
        handed,
    )
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
/// the guest take it.
#[test]
fn vt_messages_are_offered_where_the_host_offers_them() {
    let (rig, _) = vt_rig();
    let hi = rig.device.mmio_read(RGFX_REG_FEATURES_HI.into());
    assert_eq!(hi, (RGFX_FEATURE_VT_MESSAGES >> 32) as u32);
    let plain = Rig::new();
    assert_eq!(plain.device.mmio_read(RGFX_REG_FEATURES_HI.into()), 0);
}

/// A `VT_MESSAGE` reaches the host with its runs and surfaces, and the host's messages and its
/// answer reach the guest in the order the host gave them.
#[test]
fn a_vt_message_reaches_the_host_and_is_answered_in_order() {
    let (mut rig, handed) = vt_rig();
    take_and_start(&rig, RGFX_FEATURE_VT_MESSAGES);
    let seqno = rig.send(vt_message(100, &[(0x4000, 64), (0x8000, 0x4000)], &[3, 4]));
    let reply = rig.event().expect("the host's VT_MESSAGE");
    let (h, rest) = RgfxVtHostMessage::read_from_prefix(&reply).unwrap();
    assert_eq!((h.header.r#type, h.header.seqno), (RGFX_EVT_VT_MESSAGE, 0));
    assert_eq!(&rest[..h.length as usize], b"vtpv-reply");
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
