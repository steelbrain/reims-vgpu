use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::*;
use crate::runtime::StdRuntime;
use crate::runtime::virtual_time::VirtualRuntime;

/// Work whose device is done once `done` is set, which counts its drops.
struct Work {
    done: Arc<AtomicBool>,
    dropped: Arc<AtomicUsize>,
}

impl Unfinished for Work {
    fn finished(&mut self) -> bool {
        self.done.load(Ordering::SeqCst)
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

fn work() -> (Box<Work>, Arc<AtomicBool>, Arc<AtomicUsize>) {
    let done = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicUsize::new(0));
    (
        Box::new(Work {
            done: Arc::clone(&done),
            dropped: Arc::clone(&dropped),
        }),
        done,
        dropped,
    )
}

/// Waits up to 5 s of real time for `ready`.
fn eventually(mut ready: impl FnMut() -> bool) -> bool {
    let real = StdRuntime;
    poll_until(&real, real.deadline_in(Duration::from_secs(5)), || {
        ready().then_some(())
    })
    .is_some()
}

/// Work done at once is answered without a pause.
#[test]
fn work_already_done_is_answered_at_once() {
    let clock = VirtualRuntime::default();
    let mut asked = 0;
    let answer = poll_until(&clock, clock.deadline_in(DEVICE_WAIT), || {
        asked += 1;
        Some(7)
    });
    assert_eq!((answer, asked), (Some(7), 1));
}

/// Work that never finishes is given up at the deadline, on the runtime's
/// clock, here virtual, with its question asked once more there.
#[test]
fn work_that_never_finishes_is_given_up_at_its_deadline() {
    let clock = Arc::new(VirtualRuntime::default());
    let runtime: Arc<dyn Runtime> = clock.clone();
    let asked = Arc::new(AtomicUsize::new(0));
    let polled = {
        let runtime = Arc::clone(&runtime);
        let asked = Arc::clone(&asked);
        StdRuntime::thread("test-poll", move || {
            poll_until(&*runtime, deadline(&*runtime), || {
                asked.fetch_add(1, Ordering::SeqCst);
                None::<()>
            })
        })
        .expect("a thread")
    };
    assert!(eventually(|| asked.load(Ordering::SeqCst) >= 1));
    clock.advance(DEVICE_WAIT - Duration::from_millis(1));
    let before = asked.load(Ordering::SeqCst);
    assert!(eventually(|| asked.load(Ordering::SeqCst) > before));
    assert!(!polled.is_finished(), "given up before its deadline");
    clock.advance(Duration::from_millis(1));
    let answered = polled.join().unwrap_or(Some(()));
    assert_eq!(answered, None);
}

/// Work done just before the deadline is taken, however late the poll
/// wakes.
#[test]
fn work_done_by_its_deadline_is_taken() {
    let clock = Arc::new(VirtualRuntime::default());
    let runtime: Arc<dyn Runtime> = clock.clone();
    let done = Arc::new(AtomicBool::new(false));
    let polled = {
        let runtime = Arc::clone(&runtime);
        let done = Arc::clone(&done);
        StdRuntime::thread("test-poll", move || {
            poll_until(&*runtime, deadline(&*runtime), || {
                done.load(Ordering::SeqCst).then_some("done")
            })
        })
        .expect("a thread")
    };
    done.store(true, Ordering::SeqCst);
    clock.advance(DEVICE_WAIT * 2);
    assert_eq!(polled.join().unwrap_or(None), Some("done"));
}

/// The pauses are short while a frame's work may be about to finish, and
/// longer after: a device that never finishes is asked 50 times a second
/// after its first 50 ms, not 10,000.
#[test]
fn a_poll_slows_down_after_its_quick_start() {
    assert_eq!(pause_after(Duration::ZERO), QUICK_POLL);
    assert_eq!(
        pause_after(QUICK_POLLS_FOR - Duration::from_nanos(1)),
        QUICK_POLL
    );
    assert_eq!(pause_after(QUICK_POLLS_FOR), SLOW_POLL);
    assert_eq!(pause_after(DEVICE_WAIT), SLOW_POLL);
    let questions = QUICK_POLLS_FOR.as_micros() / QUICK_POLL.as_micros()
        + (DEVICE_WAIT - QUICK_POLLS_FOR).as_micros() / SLOW_POLL.as_micros();
    assert!(questions < 6_000, "{questions} questions in a whole wait");
}

/// Kept work is let go once its device is done, on the list's thread, and
/// work still running stays kept.
#[test]
fn kept_work_is_let_go_once_its_device_is_done() {
    let clock = Arc::new(VirtualRuntime::default());
    let runtime: Arc<dyn Runtime> = clock.clone();
    let lingering = Lingering::new(&runtime);
    let (first, first_done, first_dropped) = work();
    let (second, _second_done, second_dropped) = work();
    lingering.keep("frame 1", first);
    lingering.keep("frame 2", second);
    assert_eq!(lingering.count(), 2);
    clock.advance(LINGER_POLL);
    assert_eq!(first_dropped.load(Ordering::SeqCst), 0);
    first_done.store(true, Ordering::SeqCst);
    let mut ticks = 0;
    assert!(eventually(|| {
        if ticks < 1000 {
            clock.advance(LINGER_POLL);
            ticks += 1;
        }
        first_dropped.load(Ordering::SeqCst) == 1
    }));
    assert_eq!(lingering.count(), 1);
    assert_eq!(second_dropped.load(Ordering::SeqCst), 0, "kept work let go");
}

/// The thread ends once nothing is kept, and the next work kept starts
/// another.
#[test]
fn the_list_asks_again_after_it_has_emptied() {
    let runtime = StdRuntime::shared();
    let lingering = Lingering::new(&runtime);
    for _ in 0..2 {
        let (kept, done, dropped) = work();
        lingering.keep("a frame", kept);
        done.store(true, Ordering::SeqCst);
        assert!(eventually(|| dropped.load(Ordering::SeqCst) == 1));
        assert!(eventually(|| !lingering.shared.lock().running));
        assert_eq!(lingering.count(), 0);
    }
}

/// A target's hold keeps what it holds while a clone of it lives.
#[test]
fn a_hold_keeps_what_it_holds() {
    let held: Arc<dyn Any + Send + Sync> = Arc::new(5u32);
    let weak = Arc::downgrade(&held);
    let hold = GuestHold::new(held);
    let kept = hold.clone();
    drop(hold);
    assert!(weak.upgrade().is_some());
    drop(kept);
    assert!(weak.upgrade().is_none());
    assert_eq!(format!("{:?}", GuestHold::nothing()), "GuestHold(nothing)");
}

/// What a hold holds, which counts its drops in `log`.
struct Payload(Arc<Mutex<Vec<&'static str>>>);

impl Drop for Payload {
    fn drop(&mut self) {
        self.0.lock().push("let go");
    }
}

/// A write a failed call's device may still make is told of once the work
/// a backend kept past its deadline is done and let go, before what the
/// hold holds goes; a hold no one told of a late write tells of none.
#[test]
fn a_late_write_is_told_of_once_kept_work_lets_its_hold_go() {
    struct KeptWork {
        done: Arc<AtomicBool>,
        _hold: GuestHold,
    }
    impl Unfinished for KeptWork {
        fn finished(&mut self) -> bool {
            self.done.load(Ordering::SeqCst)
        }
    }
    let clock = Arc::new(VirtualRuntime::default());
    let runtime: Arc<dyn Runtime> = clock.clone();
    let lingering = Lingering::new(&runtime);
    let log = Arc::new(Mutex::new(Vec::new()));
    let hold = GuestHold::new(Arc::new(Payload(Arc::clone(&log))));
    let done = Arc::new(AtomicBool::new(false));
    lingering.keep(
        "a frame",
        Box::new(KeptWork {
            done: Arc::clone(&done),
            _hold: hold.clone(),
        }),
    );
    let told = Arc::clone(&log);
    hold.report_late_write(Box::new(move || told.lock().push("wrote")));
    drop(hold);
    clock.advance(LINGER_POLL);
    assert!(log.lock().is_empty(), "told before the device was done");
    done.store(true, Ordering::SeqCst);
    let mut ticks = 0;
    assert!(eventually(|| {
        if ticks < 1000 {
            clock.advance(LINGER_POLL);
            ticks += 1;
        }
        log.lock().len() == 2
    }));
    assert_eq!(*log.lock(), ["wrote", "let go"]);

    let quiet = Arc::new(Mutex::new(Vec::new()));
    drop(GuestHold::new(Arc::new(Payload(Arc::clone(&quiet)))));
    assert_eq!(*quiet.lock(), ["let go"]);
    let at_once = Arc::new(AtomicBool::new(false));
    let set = Arc::clone(&at_once);
    GuestHold::nothing().report_late_write(Box::new(move || set.store(true, Ordering::SeqCst)));
    assert!(
        at_once.load(Ordering::SeqCst),
        "a hold of nothing tells at once"
    );
}
