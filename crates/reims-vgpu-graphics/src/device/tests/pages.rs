//! The doorbell pages (`RGFX_FEATURE_DOORBELL_PAGES`): a page bound to a context is that context's
//! own `DOORBELL`, which wakes the rings of the context's command queues alone; every other write
//! to BAR 4 is dropped and counted. A queue is routed on its context's page where it has one. A
//! guest or a host without the feature has the device as before it.

use super::*;

/// A second queue's rings, apart from the tests' queue's.
pub(super) const OTHER_SUBMIT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000;
const OTHER_COMPLETE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x30_0000;

/// The tests' queue, of context 5, over the tests' rings.
const QUEUE: u64 = (5 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
/// A queue of context 6, over the second rings.
const OTHER: u64 = (6 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;

/// Where doorbell page `page`'s doorbell is in BAR 4.
fn page_at(page: u32) -> u64 {
    u64::from(page) * u64::from(RGFX_DOORBELL_PAGE_SIZE)
}

/// Binds doorbell page `page` to `context` (0 unbinds it), as the kext does.
fn bind(rig: &Rig, page: u32, context: u32) {
    rig.device.mmio_write(
        RGFX_REG_DOORBELL_PAGE_BIND.into(),
        page | (context << RGFX_DOORBELL_PAGE_BIND_CONTEXT_SHIFT),
    );
}

/// The contexts the doorbell pages are bound to, as the device holds them.
fn bound(rig: &Rig) -> Vec<u32> {
    (rig.device.shared.pages.iter())
        .map(|page| page.load(Ordering::SeqCst))
        .collect()
}

/// How many doorbell page writes the device dropped.
fn dropped(rig: &Rig) -> u64 {
    rig.device.shared.pages_dropped.load(Ordering::SeqCst)
}

/// Sends the `QUEUE_CREATE` of queue `obj_id` over the second rings, raising vector 4, each ring's
/// block granted to the queue's process first, and waits for its answer and its ring to go idle,
/// its completion ring armed.
fn make_other_queue(rig: &mut Rig, obj_id: u64) {
    let context = (obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32;
    let span = u64::from(RGFX_RING_HEADER_SIZE + QUEUE_RING_DATA);
    for ring in [OTHER_SUBMIT, OTHER_COMPLETE] {
        rig.send(block_message(RGFX_CMD_BLOCK_GRANT, context, ring, span));
    }
    let create = RgfxQueueCreate {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_QUEUE_CREATE,
            size: 0,
            seqno: 0,
        },
        obj_id,
        submit_offset: OTHER_SUBMIT,
        complete_offset: OTHER_COMPLETE,
        submit_data_size: QUEUE_RING_DATA,
        complete_data_size: QUEUE_RING_DATA,
        priority: 2,
        vector: 4,
    };
    let seqno = rig.send(create.as_bytes().to_vec());
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    assert!(
        eventually(|| rig
            .region()
            .atomic_u32(OTHER_SUBMIT + 0x80)
            .load(Ordering::SeqCst)
            == 1),
        "the second queue's ring went idle"
    );
    rig.region()
        .atomic_u32(OTHER_COMPLETE + 0xc0)
        .store(1, Ordering::SeqCst);
}

/// Makes the tests' queue `obj_id` over the tests' rings, its ring idle and completion ring armed.
fn make_queue(rig: &mut Rig, obj_id: u64) {
    let seqno = send_queue_create(rig, obj_id);
    assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    wait_queue_idle(rig);
    rig.region()
        .atomic_u32(QUEUE_COMPLETE + 0xc0)
        .store(1, Ordering::SeqCst);
}

/// Publishes `message` at count 0 of the second queue's submission ring, without a doorbell.
fn publish_other(rig: &Rig, message: &[u8]) {
    rig.region().write(OTHER_SUBMIT + 256, message);
    rig.region()
        .atomic_u64(OTHER_SUBMIT)
        .store(message.len() as u64, Ordering::SeqCst);
}

/// The consumer of the submission ring at `submit`: how far its thread took it.
fn consumer(rig: &Rig, submit: u64) -> u64 {
    rig.region()
        .atomic_u64(submit + 0x40)
        .load(Ordering::SeqCst)
}

/// A rig over a [`TakingHost`] that answers command buffers at the doorbell and offers what
/// `taking` says, whose guest takes `taken` (the `FEATURES_HI` half), started.
fn rig_taking(taking: &Arc<Taking>, taken: u64) -> Rig {
    taking.takes.store(true, Ordering::SeqCst);
    let rig = Rig::with_host(Box::new(TakingHost(Arc::clone(taking))));
    rig.device
        .mmio_write(RGFX_REG_DRIVER_FEATURES_HI.into(), (taken >> 32) as u32);
    rig.start();
    rig
}

/// A write of a ring's id at a page's doorbell wakes the ring where it is one of the page's
/// context's queues', as `DOORBELL` would, and answers its buffer on the writing thread. A write
/// of another context's ring, or of ring 0, is dropped and counted: nothing is answered, and the
/// ring's thread is not woken. Bound again to another context, the page reaches that context's
/// rings alone; unbound, none; and a write at any other offset of a page is dropped too. The
/// buffer a dropped write left on its ring is still there for its own doorbell.
#[test]
fn a_doorbell_page_wakes_only_the_rings_of_the_context_it_is_bound_to() {
    let taking = Arc::new(Taking::default());
    taking.pages.store(true, Ordering::SeqCst);
    let mut rig = rig_taking(&taking, RGFX_FEATURE_DOORBELL_PAGES);
    bind(&rig, 3, 5);
    make_queue(&mut rig, QUEUE);
    make_other_queue(&mut rig, OTHER);

    // Its own context's ring: taken, and answered on the writing thread.
    wait_parked(&rig);
    let first = publish_unrung(&rig, 0, &[command_buffer(QUEUE, 100, 1)]);
    rig.device.doorbell_write(page_at(3), QUEUE_SUBMIT as u32);
    assert_eq!(taking.answered.lock().get(&1), Some(&true));
    assert_eq!(
        queue_events(&rig, 2),
        vec![
            (RGFX_EVT_CB_SUBMITTED, 1, 100),
            (RGFX_EVT_CB_COMPLETE, 1, 100)
        ]
    );
    assert_eq!(dropped(&rig), 0);

    // Context 6's ring and ring 0 through context 5's page: dropped and counted, nothing woken.
    wait_parked(&rig);
    publish_other(&rig, &command_buffer(OTHER, 200, 7));
    assert!(eventually(|| rig.device.parked()));
    rig.device.doorbell_write(page_at(3), OTHER_SUBMIT as u32);
    rig.device.doorbell_write(page_at(3), RGFX_CONTROL_RING_ID);
    assert_eq!(dropped(&rig), 2);
    assert!(
        !taking.answered.lock().contains_key(&7),
        "a foreign ring's buffer was answered"
    );
    assert!(rig.device.parked(), "a foreign write woke a thread");
    assert_eq!(
        consumer(&rig, OTHER_SUBMIT),
        0,
        "a foreign write's ring was taken"
    );

    // Unbound and bound to context 6: the page reaches context 6's ring, and no longer 5's.
    bind(&rig, 3, 0);
    bind(&rig, 3, 6);
    assert!(eventually(|| consumer(&rig, QUEUE_SUBMIT) == first));
    publish_unrung(&rig, first, &[command_buffer(QUEUE, 101, 2)]);
    rig.device.doorbell_write(page_at(3), QUEUE_SUBMIT as u32);
    assert_eq!(dropped(&rig), 3);
    assert!(!taking.answered.lock().contains_key(&2));
    wait_parked(&rig);
    rig.device.doorbell_write(page_at(3), OTHER_SUBMIT as u32);
    assert_eq!(taking.answered.lock().get(&7), Some(&true));
    assert!(eventually(|| consumer(&rig, OTHER_SUBMIT) > 0));

    // Unbound, the page reaches nothing, and an offset past a page's doorbell nothing either.
    bind(&rig, 3, 0);
    rig.device.doorbell_write(page_at(3), QUEUE_SUBMIT as u32);
    bind(&rig, 3, 5);
    rig.device
        .doorbell_write(page_at(3) + 4, QUEUE_SUBMIT as u32);
    rig.device.doorbell_write(page_at(4), QUEUE_SUBMIT as u32);
    assert_eq!(dropped(&rig), 6);
    assert!(!taking.answered.lock().contains_key(&2));
    assert_eq!(
        consumer(&rig, QUEUE_SUBMIT),
        first,
        "buffer 2 is still on the ring"
    );

    // Its own doorbell takes the buffer the dropped writes left.
    wait_parked(&rig);
    rig.device
        .mmio_write_taking(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    assert_eq!(taking.answered.lock().get(&2), Some(&true));
}

/// A page past the pages is no page: a bind of it is ignored, and a write there is dropped. A
/// reset unbinds every page.
#[test]
fn a_reset_unbinds_every_doorbell_page() {
    let taking = Arc::new(Taking::default());
    taking.pages.store(true, Ordering::SeqCst);
    let rig = rig_taking(&taking, RGFX_FEATURE_DOORBELL_PAGES);
    bind(&rig, 0, 9);
    bind(&rig, RGFX_DOORBELL_PAGES - 1, 10);
    let mut expected = vec![0; RGFX_DOORBELL_PAGES as usize];
    expected[0] = 9;
    expected[RGFX_DOORBELL_PAGES as usize - 1] = 10;
    assert_eq!(bound(&rig), expected);
    rig.device
        .doorbell_write(u64::from(RGFX_DOORBELLS_SIZE), QUEUE_SUBMIT as u32);
    assert_eq!(dropped(&rig), 1);
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    assert_eq!(bound(&rig), vec![0; RGFX_DOORBELL_PAGES as usize]);
}

/// A host that does not map BAR 4 offers no `DOORBELL_PAGES`, so a driver's write of it takes
/// nothing, a bind is ignored, and the queues' doorbells are the register's as before the bit,
/// routes among them; and so for a guest that does not take what the host offers.
#[test]
fn without_doorbell_pages_the_doorbells_are_the_register_s() {
    for offered in [false, true] {
        let taking = Arc::new(Taking::default());
        taking.pages.store(offered, Ordering::SeqCst);
        taking.routes_pages.store(offered, Ordering::SeqCst);
        taking.asks.store(true, Ordering::SeqCst);
        // Where the host offers the bit, the guest takes only DOORBELL_ASKS.
        let taken = if offered {
            RGFX_FEATURE_DOORBELL_ASKS
        } else {
            RGFX_FEATURE_DOORBELL_ASKS | RGFX_FEATURE_DOORBELL_PAGES
        };
        let mut rig = rig_taking(&taking, taken);
        let offer = u64::from(rig.device.mmio_read(RGFX_REG_FEATURES_HI.into())) << 32;
        assert_eq!(offer & RGFX_FEATURE_DOORBELL_PAGES != 0, offered);
        let took = u64::from(rig.device.mmio_read(RGFX_REG_DRIVER_FEATURES_HI.into())) << 32;
        assert_eq!(took, RGFX_FEATURE_DOORBELL_ASKS);
        bind(&rig, 3, 5);
        assert_eq!(bound(&rig), vec![0; RGFX_DOORBELL_PAGES as usize]);
        make_queue(&mut rig, QUEUE);
        #[cfg(target_os = "linux")]
        assert_eq!(*taking.routed_at.lock(), vec![DoorbellAt::Register]);

        wait_parked(&rig);
        publish_unrung(&rig, 0, &[command_buffer(QUEUE, 100, 1)]);
        rig.device.mmio_write_taking(
            RGFX_REG_DOORBELL.into(),
            QUEUE_SUBMIT as u32 | RGFX_DOORBELL_ASKS,
        );
        assert_eq!(taking.answered.lock().get(&1), Some(&true));
        assert_eq!(dropped(&rig), 0);
    }
}

/// Where the guest took `DOORBELL_ASKS`, a queue whose context has a page is routed on that page,
/// where the host routes pages, and one whose context has none on `DOORBELL`; a host that routes
/// no page routes no queue of a context with one, whose writes come to the page's write. Its
/// destroy drops the route.
#[cfg(target_os = "linux")]
#[test]
fn a_queue_is_routed_on_its_context_s_doorbell_page() {
    for routes_pages in [true, false] {
        let taking = Arc::new(Taking::default());
        taking.pages.store(true, Ordering::SeqCst);
        taking.routes_pages.store(routes_pages, Ordering::SeqCst);
        taking.asks.store(true, Ordering::SeqCst);
        let mut rig = rig_taking(
            &taking,
            RGFX_FEATURE_DOORBELL_ASKS | RGFX_FEATURE_DOORBELL_PAGES,
        );
        bind(&rig, 7, 5);
        make_queue(&mut rig, QUEUE);
        make_other_queue(&mut rig, OTHER);
        let expected = if routes_pages {
            vec![DoorbellAt::Page(7), DoorbellAt::Register]
        } else {
            vec![DoorbellAt::Register]
        };
        assert_eq!(*taking.routed_at.lock(), expected);
        let rings: Vec<u32> = taking.routes.lock().iter().map(|r| r.1).collect();
        let expected_rings = if routes_pages {
            vec![QUEUE_SUBMIT as u32, OTHER_SUBMIT as u32]
        } else {
            vec![OTHER_SUBMIT as u32]
        };
        assert_eq!(rings, expected_rings);

        // A plain write at the page wakes the queue, as the route's file would.
        publish_unrung(&rig, 0, &[command_buffer(QUEUE, 100, 1)]);
        rig.device.doorbell_write(page_at(7), QUEUE_SUBMIT as u32);
        assert!(
            taking.answered.lock().is_empty(),
            "a plain write was answered"
        );
        assert_eq!(queue_events(&rig, 2).len(), 2);

        rig.send(object_message(RGFX_CMD_OBJECT_DESTROY, QUEUE));
        if routes_pages {
            let route = taking.routes.lock()[0].0;
            assert!(eventually(|| taking.unrouted.lock().contains(&route)));
        }
    }
}

/// A page bound to another context while a queue of its old context is routed on it drops that
/// route, so the host wakes no ring on the page but the new context's; the queue's doorbell is
/// then `DOORBELL`'s, and its page write is dropped as a foreign ring's. A queue of the new context
/// made after is routed on the page.
#[cfg(target_os = "linux")]
#[test]
fn a_page_bound_again_drops_the_routes_of_its_old_context() {
    let taking = Arc::new(Taking::default());
    taking.pages.store(true, Ordering::SeqCst);
    taking.routes_pages.store(true, Ordering::SeqCst);
    taking.asks.store(true, Ordering::SeqCst);
    let mut rig = rig_taking(
        &taking,
        RGFX_FEATURE_DOORBELL_ASKS | RGFX_FEATURE_DOORBELL_PAGES,
    );
    bind(&rig, 3, 5);
    make_queue(&mut rig, QUEUE);
    let route = taking.routes.lock()[0].0;
    assert_eq!(*taking.routed_at.lock(), vec![DoorbellAt::Page(3)]);

    bind(&rig, 3, 6);
    assert_eq!(
        *taking.unrouted.lock(),
        vec![route],
        "the old context's route went"
    );
    assert_eq!(rig.device.shared.routes.load(Ordering::SeqCst), 0);
    rig.device.doorbell_write(page_at(3), QUEUE_SUBMIT as u32);
    assert_eq!(
        dropped(&rig),
        1,
        "the old context's ring is foreign on the page"
    );

    // Its doorbell is the register's now, which still wakes its ring.
    publish_unrung(&rig, 0, &[command_buffer(QUEUE, 100, 1)]);
    rig.device
        .mmio_write(RGFX_REG_DOORBELL.into(), QUEUE_SUBMIT as u32);
    assert_eq!(queue_events(&rig, 2).len(), 2);

    make_other_queue(&mut rig, OTHER);
    assert_eq!(
        *taking.routed_at.lock(),
        vec![DoorbellAt::Page(3), DoorbellAt::Page(3)]
    );
    // Bound to the same context again, nothing moves.
    bind(&rig, 3, 6);
    assert_eq!(taking.unrouted.lock().len(), 1);
}
