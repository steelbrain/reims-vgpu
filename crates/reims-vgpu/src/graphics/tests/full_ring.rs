//! A completion ring the guest holds full: the platform's threads that run a decoder buffer's
//! submission and completion handlers wait for no room on a command queue's
//! (`CompletionPoster::post_unwaiting`), nor on the control ring's for the answers they give
//! (`CompleteAnswer::answer`), and the records, the values the buffer signalled, their posts to
//! the other holders of an event's name and the answers reach the guest in their order once it
//! takes them.

use reims_vgpu_graphics::ops::{RGFX_OP_SIGNAL_EVENT, RgfxOpEvent};

use super::*;

/// The completion ring's data: 4 KiB, which holds 128 of a buffer's 32-byte values.
const DATA: u32 = 0x1000;

/// How many values the buffers here signal: more than a [`DATA`] ring holds.
const SIGNALS: u64 = 150;

/// The queue rings' offsets, as [`one_queue_raising`] makes them.
const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;

/// A `COMMAND_BUFFER` `cb_obj_id` on `queue_obj_id` that signals shared event `event` to each of
/// 1 to [`SIGNALS`], at its end: its completion posts each value ahead of its `CB_COMPLETE`.
fn signalling(queue_obj_id: u64, cb_obj_id: u64, event: u64) -> Vec<u8> {
    let signals: Vec<RgfxOpEvent> = (1..=SIGNALS)
        .map(|value| RgfxOpEvent {
            event_obj_id: event,
            value,
        })
        .collect();
    let ops: Vec<(u16, &[u8])> = (signals.iter())
        .map(|s| (RGFX_OP_SIGNAL_EVENT, s.as_bytes()))
        .collect();
    command_buffer_of(queue_obj_id, cb_obj_id, &ops)
}

/// Runs `work` on a thread standing in for the platform's that runs a command buffer's
/// handlers: whether it returned within five seconds.
fn returns_at_once(work: impl FnOnce() + Send + 'static) -> bool {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("platform-handler".into())
        .spawn(move || {
            work();
            let _ = tx.send(());
        })
        .unwrap();
    rx.recv_timeout(Duration::from_secs(5)).is_ok()
}

/// A record on a queue's completion ring: its type, the id it names (a command buffer's, or a
/// shared event's) and, for an `EVENT_SIGNALLED`, the value.
type Record = (u32, u64, u64);

/// Takes every record on `ring` the guest has not taken, pads skipped, in ring order.
fn take_records(ring: &mut QueueRing) -> Vec<Record> {
    let (records, consumed) = read_records(ring);
    ring.consumed = consumed;
    // SAFETY: as `QueueRing`'s helpers; the consumer count is at 0x40 of the ring's header.
    unsafe { ring.store_u64(ring.complete_off as usize + 0x40, consumed) };
    records
}

/// The records on `ring` the guest has not taken, pads skipped, in ring order, taking none; and
/// the guest's count past them.
fn read_records(ring: &QueueRing) -> (Vec<Record>, u64) {
    let (complete_off, data) = (ring.complete_off as usize, ring.data);
    let at = |count: u64| complete_off + 256 + (count % data) as usize;
    let mut records = Vec::new();
    let mut consumed = ring.consumed;
    // SAFETY (this block): as `QueueRing`'s helpers; the host published each message before the
    // producer count.
    unsafe {
        let produced = ring.load_u64(complete_off);
        while consumed != produced {
            let header = ring.read(at(consumed), 8);
            let r#type = u32::from_le_bytes(header[0..4].try_into().unwrap());
            let size = u32::from_le_bytes(header[4..8].try_into().unwrap());
            if r#type != RGFX_MSG_PAD {
                let body = ring.read(at(consumed), size as usize);
                let id = u64::from_le_bytes(body[16..24].try_into().unwrap());
                let value = match r#type {
                    RGFX_EVT_EVENT_SIGNALLED => {
                        u64::from_le_bytes(body[24..32].try_into().unwrap())
                    }
                    _ => 0,
                };
                records.push((r#type, id, value));
            }
            consumed += u64::from(size);
        }
    }
    (records, consumed)
}

/// Takes `ring`'s records until one is `last`, [`eventually`]: all of them, in ring order.
fn take_records_through(ring: &mut QueueRing, last: Record) -> Vec<Record> {
    let mut records = Vec::new();
    let came = eventually(|| {
        records.extend(take_records(ring));
        records.contains(&last)
    });
    assert!(came, "{last:?} never came, after {} records", records.len());
    records
}

/// The values 1 to [`SIGNALS`] of shared event `event`, as its holder's ring has them.
fn values(event: u64) -> impl Iterator<Item = Record> {
    (1..=SIGNALS).map(move |value| (RGFX_EVT_EVENT_SIGNALLED, event, value))
}

/// The completion handler of a buffer whose values and completion are more than its queue's ring
/// holds, and the submission and completion handlers of the queue's next buffer, all run on the
/// platform's thread while the guest takes nothing, return at once: what the ring has no room for
/// is handed to the queue's service thread. Once the guest takes the ring, every record comes in
/// its order: the first buffer's values, then its `CB_COMPLETE`, then the next one's
/// `CB_SUBMITTED`, `CB_SCHEDULED` and `CB_COMPLETE`. Each raise comes on the thread that posted
/// its record, the device's or the platform's
#[test]
fn a_seen_handler_and_a_submission_handler_facing_a_full_completion_ring_return_at_once() {
    let raises = Raises::default();
    let (device, platform, mut guest, mut ring) =
        one_queue_raising(DATA, noting_raises(&raises, || true));
    guest.send(&device, shared_event_create(obj(1, 7), 0));
    ring.arm();
    ring.send(&device, signalling(obj(1, 1), 100, obj(1, 7)));
    committed(&platform, 1);
    // The next buffer waits for the event and asks for its `CB_SUBMITTED`: held by the platform,
    // its records come from its submission handler.
    platform.world().hold_submissions();
    ring.send(&device, waiting_and_asking(obj(1, 1), 101, obj(1, 7), 1000));
    let ids = committed(&platform, 2);
    let handed = device.handed_over();

    let world = Arc::clone(platform.world());
    let returned = returns_at_once(move || {
        let done =
            |id| world.complete(id, CommandBufferStatus::COMPLETED, None, CompleteOn::Caller);
        done(ids[0]);
        world.release_submissions();
        done(ids[1]);
    });
    assert!(
        returned,
        "a handler on the platform's thread waited for room"
    );
    // SAFETY: the completion ring's producer count is its header's first field, in the region.
    let produced = unsafe { ring.load_u64(COMPLETE_OFF as usize) };
    assert!(produced > u64::from(DATA) - 40, "the ring never filled");
    assert!(
        device.handed_over() >= handed + 3,
        "the records the ring had no room for were not handed over"
    );

    let records = take_records_through(&mut ring, (RGFX_EVT_CB_COMPLETE, 101, 0));
    let mut expected = vec![(RGFX_EVT_CB_SCHEDULED, 100, 0)];
    expected.extend(values(obj(1, 7)));
    expected.extend([
        (RGFX_EVT_CB_COMPLETE, 100, 0),
        (RGFX_EVT_CB_SUBMITTED, 101, 0),
        (RGFX_EVT_CB_SCHEDULED, 101, 0),
        (RGFX_EVT_CB_COMPLETE, 101, 0),
    ]);
    assert_eq!(records, expected);
    for (name, _, _) in raises.lock().iter() {
        let name = name.as_deref().unwrap_or_default();
        assert!(
            name.starts_with("reims-vgpu.graphics-") || name == "platform-handler",
            "raised on {name:?}"
        );
    }
}

/// A buffer of one process that signals a shared event another process holds by its name, its
/// values more than its queue's ring holds: its completion handler returns at once, and each value
/// reaches the other process only once the first's own ring holds it, as it did when the handler
/// waited for room, so no process has a value before the one that signalled it
#[test]
fn a_fanned_out_value_follows_its_signaller_s_own_record_through_a_full_ring() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    // The namer's queue (1, 1) with a small ring, its listen queue (1, 2), and the importer's
    // listen queue (2, 1), each a block of its own.
    let rings = |n: u64| {
        let at = RGFX_CONTEXT_ALLOCATIONS_OFFSET + n * 0x4_0000;
        (at, at + 0x2_0000)
    };
    let queues = [(1, 1, DATA), (1, 2, 0x1_0000), (2, 1, 0x1_0000)];
    for (at, &(context, n, data)) in queues.iter().enumerate() {
        let made = create_queue(&device, &mut guest, context, n, rings(at as u64), data, 0);
        assert_eq!(made, RGFX_STATUS_OK);
    }
    let ring = |at: usize| {
        let (submit, complete) = rings(at as u64);
        QueueRing::new(&device, submit, complete, u64::from(queues[at].2))
    };
    let (mut namer, mut importer) = (ring(0), ring(2));
    guest.send(&device, shared_event_create(obj(1, 20), 0));
    let name = shared_event_share(RGFX_CMD_SHARED_EVENT_NAME, obj(1, 20), 7, obj(1, 2));
    assert_eq!(answered(&device, &mut guest, name), RGFX_STATUS_OK);
    let import = shared_event_share(RGFX_CMD_SHARED_EVENT_IMPORT, obj(2, 21), 7, obj(2, 1));
    guest.send(&device, import);
    let value = RgfxSharedEventValue::read_from_bytes(&guest.event().unwrap()).unwrap();
    assert_eq!((value.status, value.value), (RGFX_STATUS_OK, 0));

    namer.send(&device, signalling(obj(1, 1), 100, obj(1, 20)));
    let ids = committed(&platform, 1);
    let world = Arc::clone(platform.world());
    let returned = returns_at_once(move || {
        world.complete(
            ids[0],
            CommandBufferStatus::COMPLETED,
            None,
            CompleteOn::Caller,
        );
    });
    assert!(returned, "the completion handler waited for room");

    // The namer's ring is full and holds a part of the values; the importer has none past them.
    // SAFETY: the completion ring's producer count is its header's first field, in the region.
    let produced = unsafe { namer.load_u64(rings(0).1 as usize) };
    assert!(
        produced > u64::from(DATA) - 40,
        "the namer's ring never filled"
    );
    std::thread::sleep(Duration::from_millis(50));
    let highest = |records: &[Record]| records.iter().map(|r| r.2).max().unwrap_or(0);
    let early = take_records(&mut importer);
    // What the namer's ring holds now, read without taking it.
    let (unheld, _) = read_records(&namer);
    assert!(
        highest(&unheld) < SIGNALS,
        "the namer's ring took every value"
    );
    assert!(
        highest(&early) <= highest(&unheld),
        "the importer had {} before the namer's ring held more than {}",
        highest(&early),
        highest(&unheld)
    );

    // The namer's process takes its ring: every value comes there in order, then the importer has
    // the last too, for its own id.
    let records = take_records_through(&mut namer, (RGFX_EVT_CB_COMPLETE, 100, 0));
    let mut expected: Vec<Record> = values(obj(1, 20)).collect();
    expected.push((RGFX_EVT_CB_COMPLETE, 100, 0));
    assert_eq!(
        records
            .into_iter()
            .filter(|r| r.0 != RGFX_EVT_CB_SCHEDULED)
            .collect::<Vec<_>>(),
        expected
    );
    let last = (RGFX_EVT_EVENT_SIGNALLED, obj(2, 21), SIGNALS);
    let late = take_records_through(&mut importer, last);
    assert!(
        early
            .iter()
            .chain(&late)
            .all(|r| r.0 == RGFX_EVT_EVENT_SIGNALLED && r.1 == obj(2, 21)),
        "{early:?} {late:?}"
    );
}

/// The records a full ring held back when the device is reset are dropped: none reaches the
/// rebooted guest, whose queue over the same rings gets its own buffer's completion first
#[test]
fn a_reset_with_records_handed_over_posts_none_of_them() {
    let (device, platform, mut guest, mut ring) =
        one_queue_raising(DATA, noting_raises(&Raises::default(), || true));
    guest.send(&device, shared_event_create(obj(1, 7), 0));
    ring.send(&device, signalling(obj(1, 1), 100, obj(1, 7)));
    let ids = committed(&platform, 1);
    let world = Arc::clone(platform.world());
    let returned = returns_at_once(move || {
        world.complete(
            ids[0],
            CommandBufferStatus::COMPLETED,
            None,
            CompleteOn::Caller,
        );
    });
    assert!(returned, "the completion handler waited for room");
    // SAFETY: the completion ring's producer count is its header's first field, in the region.
    let produced = || unsafe { ring.load_u64(COMPLETE_OFF as usize) };
    let full = produced();
    assert!(full > u64::from(DATA) - 40, "the ring never filled");

    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    reset_done(&device);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(produced(), full, "posted after the reset");

    // The rebooted guest makes its queue over the same rings and runs a buffer.
    zero_ring_headers(&device, [SUBMIT_OFF, COMPLETE_OFF]);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let made = create_queue(
        &device,
        &mut guest,
        1,
        1,
        (SUBMIT_OFF, COMPLETE_OFF),
        DATA,
        0,
    );
    assert_eq!(made, RGFX_STATUS_OK);
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(DATA));
    ring.send(&device, command_buffer(obj(1, 1), 200));
    assert!(complete_until_event(&platform, &mut ring), "no completion");
    let records = take_records(&mut ring);
    assert_eq!(records, [(RGFX_EVT_CB_COMPLETE, 200, 0)]);
}

/// A `SET_VBLANK` that turns display 0's vblanks off: a ring-0 command the control thread answers
/// as it takes it.
fn vblanks_off() -> Vec<u8> {
    RgfxSetVblank {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_VBLANK,
            size: 0,
            seqno: 0,
        },
        display: 0,
        enabled: 0,
    }
    .as_bytes()
    .to_vec()
}

/// [`Guest::send`] once the control submission ring has room for `command` (and a pad before
/// it), as a guest sends.
fn send_with_room(guest: &mut Guest, device: &GraphicsDevice<FakePlatform>, command: Vec<u8>) {
    let len = command.len() as u64;
    let room = eventually(|| {
        // SAFETY: as `Guest`'s helpers; the consumer count is at 0x40 of the ring's header.
        let taken = unsafe { guest.load_u64(RGFX_CONTROL_SUBMIT_OFFSET as usize + 0x40) };
        guest.produced - taken + 2 * len <= RING_DATA
    });
    assert!(room, "the control thread took no command");
    guest.send(device, command);
}

/// How many full spells of the control completion ring this process told.
fn control_ring_full_spells() -> u64 {
    (reims_vgpu_vulkan::answer::degradation_counts().into_iter())
        .find(|(code, _)| code == "control-ring-full")
        .map_or(0, |(_, n)| n)
}

/// A command buffer's completion handler that ends the drain of its context's close, run on the
/// platform's thread while the guest holds the control completion ring full, returns at once: the
/// close's answer is handed to the control thread, which posts it after the answer it waits to
/// post and before its next one, once the guest takes the ring
#[test]
fn a_seen_handler_that_ends_a_drain_waits_for_no_room_on_the_control_ring() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform, mut guest, mut ring) = one_queue(&raised);
    ring.send(&device, command_buffer(obj(1, 1), 100));
    let ids = committed(&platform, 1);
    guest.send(&device, context_close(1));
    let close = guest.seqno;
    // Answered while the close waits for the buffer: the control thread has taken the close.
    guest.send(&device, vblanks_off());
    let taken = guest.seqno;
    assert_eq!(answer_of(&mut guest, taken), RGFX_STATUS_OK);
    // The close's objects let go of: the buffer's completion holds the drain's last clone.
    device.control.decoder().closes_let_go();

    // More answers than the ring holds, none taken: the control thread waits for room.
    let spells = control_ring_full_spells();
    let first = guest.seqno + 1;
    let count = (RING_DATA / 24) + 40;
    for _ in 0..count {
        send_with_room(&mut guest, &device, vblanks_off());
    }
    let last = guest.seqno;
    assert!(
        eventually(|| control_ring_full_spells() > spells),
        "the control completion ring never filled"
    );
    let world = Arc::clone(platform.world());
    let returned = returns_at_once(move || {
        world.complete(
            ids[0],
            CommandBufferStatus::COMPLETED,
            None,
            CompleteOn::Caller,
        );
    });
    assert!(
        returned,
        "the completion handler waited for room on the control ring"
    );

    let mut seqnos = Vec::new();
    while seqnos.len() < count as usize + 1 {
        let event = next_event(&mut guest, RGFX_EVT_COMPLETE);
        let complete = RgfxComplete::read_from_bytes(&event).unwrap();
        assert_eq!(complete.status, RGFX_STATUS_OK);
        seqnos.push(complete.header.seqno);
    }
    let at = seqnos
        .iter()
        .position(|&s| s == close)
        .expect("the close was never answered");
    let waited = seqnos[at - 1];
    let mut expected: Vec<u64> = (first..=waited).collect();
    expected.push(close);
    expected.extend(waited + 1..=last);
    assert_eq!(seqnos, expected);
}
