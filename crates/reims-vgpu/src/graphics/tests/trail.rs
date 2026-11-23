//! The trail of the holds on the guest's memory (`crate::trail`) at the device's reset: written
//! where the reset dropped live objects, and not where it dropped none.

use reims_vgpu_vulkan::core::testkit::logs::{Capture, Line};

use super::*;

/// The lines `logs` takes up to the reset's own line, which says `dropped` objects were dropped,
/// that one included.
fn up_to_the_reset(logs: &Capture, dropped: usize) -> Vec<Line> {
    let said = format!("the device was reset: {dropped} objects of ");
    let mut lines = Vec::new();
    let reset = eventually(|| {
        lines.extend(logs.take());
        lines
            .iter()
            .any(|(_, target, line)| target == "reims_vgpu::graphics" && line.starts_with(&said))
    });
    assert!(reset, "no reset line: {lines:?}");
    lines
}

/// The trail's lines among `lines`.
fn trail_of(lines: &[Line]) -> Vec<&str> {
    (lines.iter())
        .filter(|(_, target, _)| target == crate::trail::TARGET)
        .map(|(_, _, line)| line.as_str())
        .collect()
}

/// A reset that dropped none of the guest's objects writes none of the trail and keeps it; one
/// that dropped live objects writes it at info before its own line, a line that counts the
/// events and then one an event, the oldest first, each beginning `trail:`, and empties it: a
/// block's release asked and answered OK, with nothing in flight, are written at the first reset
/// that dropped an object after them, and not at the next
#[test]
fn a_reset_writes_the_trail_only_where_it_dropped_objects() {
    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const SIZE: u64 = 0x1_0000;
    let raised = Arc::new(AtomicU32::new(0));
    let device = headless(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    guest.send(&device, block_grant(1, BLOCK, SIZE));
    assert_eq!(
        answered(&device, &mut guest, block_release(1, BLOCK, SIZE)),
        RGFX_STATUS_OK
    );
    let logs = reims_vgpu_vulkan::core::testkit::logs::capture();

    device.reset();
    let lines = up_to_the_reset(&logs, 0);
    assert_eq!(trail_of(&lines), Vec::<&str>::new(), "a clean reset");

    // The guest's driver starts again, and makes an object the next reset drops.
    let unclean = |guest: &mut Guest, n: u64| {
        device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
        *guest = Guest::new(&device);
        guest.send(&device, event_create(obj(2, n)));
        assert!(holds_objects(&device, 1));
        device.reset();
        up_to_the_reset(&logs, 1)
    };
    let lines = unclean(&mut guest, 1);
    assert!(
        (lines.iter())
            .filter(|(_, target, _)| target == crate::trail::TARGET)
            .all(|(level, _, line)| *level == log::Level::Info && line.starts_with("trail: ")),
        "{lines:?}"
    );
    let trail = trail_of(&lines);
    assert!(
        trail
            .first()
            .is_some_and(|line| line.starts_with("trail: the last ")),
        "{trail:?}"
    );
    let at = format!(", context 1, at {BLOCK:#x}: release ");
    let ours: Vec<&str> = (trail.iter())
        .filter_map(|line| line.split_once(at.as_str()).map(|(_, what)| what))
        .collect();
    assert_eq!(ours.len(), 2, "{trail:?}");
    assert_eq!(ours[0], format!("asked, {SIZE:#x} bytes"));
    let nothing = " us; in flight: 0 holds in other contexts, 0 of its queues uncommitted";
    assert!(
        ours[1].starts_with("answered OK after ") && ours[1].ends_with(nothing),
        "{}",
        ours[1]
    );

    let lines = unclean(&mut guest, 2);
    let again = trail_of(&lines);
    assert!(
        again.iter().all(|line| !line.contains(at.as_str())),
        "the trail was emptied: {again:?}"
    );
}

/// A `BLOCK_RELEASE`'s answer says in its entry what was in flight as it was given: no hold of
/// another context's where none of its command buffers was in flight, and one at least where
/// another context's command buffer was committed and not yet complete (its walk's message may
/// hold that context's drain a moment past the commit, a second hold); the releasing context,
/// with no queue, has none uncommitted
#[test]
fn a_release_answer_s_entry_counts_another_context_s_buffer_in_flight() {
    const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
    const RING_DATA: u32 = 0x1_0000;
    const QUIET: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000;
    const BUSY: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x11_0000;
    const SIZE: u64 = 0x1_0000;
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_on(&raised);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let rings = (SUBMIT_OFF, COMPLETE_OFF);
    assert_eq!(
        create_queue(&device, &mut guest, 2, 1, rings, RING_DATA, 0),
        RGFX_STATUS_OK
    );
    assert!(holds_objects(&device, 1), "the queue's handle was not made");
    let release = |guest: &mut Guest, block: u64| {
        guest.send(&device, block_grant(1, block, SIZE));
        let status = answered(&device, guest, block_release(1, block, SIZE));
        assert_eq!(status, RGFX_STATUS_OK);
    };

    // Context 1 releases a block with nothing in flight, then one while context 2's buffer is.
    release(&mut guest, QUIET);
    let mut ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    ring.send(&device, command_buffer(obj(2, 1), 100));
    committed(&platform, 1);
    release(&mut guest, BUSY);
    platform.world().complete_all(CompleteOn::Caller);

    let logs = reims_vgpu_vulkan::core::testkit::logs::capture();
    device.reset();
    let lines = up_to_the_reset(&logs, 1);
    let trail = trail_of(&lines);
    let in_flight = |block: u64| -> (u64, u64) {
        let at = format!(", context 1, at {block:#x}: release answered OK after ");
        let answers: Vec<&str> = (trail.iter())
            .filter_map(|line| line.split_once(at.as_str()).map(|(_, what)| what))
            .collect();
        assert_eq!(answers.len(), 1, "{trail:?}");
        let counts = (answers[0].split_once(" us; in flight: "))
            .and_then(|(_, counts)| counts.strip_suffix(" of its queues uncommitted"))
            .and_then(|counts| counts.split_once(" holds in other contexts, "));
        let Some((others, uncommitted)) = counts else {
            panic!("no counts of what was in flight: {}", answers[0]);
        };
        (others.parse().unwrap(), uncommitted.parse().unwrap())
    };
    assert_eq!(in_flight(QUIET), (0, 0), "with nothing in flight");
    let (others, uncommitted) = in_flight(BUSY);
    assert!(
        others >= 1,
        "context 2's buffer was in flight: {others} holds"
    );
    assert_eq!(uncommitted, 0);
}
