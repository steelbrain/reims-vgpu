//! A display's driver calls, each bounded by the deadline of the frame in
//! hand (`super::wait::current`): the display's call thread, and the gate
//! that keeps the caller's threads out of the driver while that thread
//! makes a call that may wait for the GPU.
//!
//! **What waits**, measured on the RX 7900 XTX and the Radeon 780M
//! (radeonsi, Mesa 26.2.4, libva 2.24.1, 2026-10-26) with the GPU held by
//! a compute shader of the tests' own that spins on the graphics queue, or
//! on 16 compute queues at once (`super::spin`), each call timed on a
//! thread of its own, 4K NV12 surfaces:
//!
//! | call | the graphics ring held | the compute rings held |
//! |---|---|---|
//! | `vaGetImage` | waits | waits |
//! | `vaMapBuffer` of a derived image of a JPEG decode's surface, or a processing run's NV12 target | waits | (its run stalls first) |
//! | `vaSyncSurface` of a processing run's target | returns, the run done | waits |
//! | a decode's or a run's `vaBeginPicture`, `vaRenderPicture`, `vaEndPicture` | return | return |
//! | `vaSyncSurface2` and `vaSyncBuffer` with a timeout of 0 | return | return |
//! | `vaPutImage`, `vaDeriveImage`, `vaMapBuffer` of an image `vaCreateImage` made, `vaExportSurfaceHandle`, creates and destroys | return | return |
//!
//! A run (staged, zero copy) and an encode were held
//! back and waited for by their bounded questions; a decode was not: VCN
//! runs beside both rings. Each call that waited held the display's lock
//! while it did (Mesa's `drv->mutex`): another thread's `vaSyncSurface2`
//! of an idle surface, timeout 0, returned only once it had, 380 ms on.
//! radeonsi copies a surface for `vaGetImage` and for a map on a compute
//! queue, so where one waited with the graphics ring held, it waited on
//! the kernel's page-table updates: a buffer the process destroyed while
//! the ring was held had its unmap's update wait for the spin, and the
//! copy's job wait for that update (**measured** with the GPU scheduler's
//! tracepoints, RX 7900 XTX, 2026-10-26; docs/video/bugs.md, B92, U23).
//! With nothing destroyed, a JPEG decode's derived map ran beside the held
//! graphics ring. A decoder's first `vaEndPicture` waits too: radeonsi
//! begins the context's session there, and waits for that command with
//! the display's lock held, which after an unmap waits for every job then
//! on the GPU; beside a JPEG decode of 8192x8192 on the GPU, a new H.264
//! decoder's first one took 43.5 ms on the RX 7900 XTX and 127.6 ms on the
//! Radeon 780M (`super::buffer`; docs/video/bugs.md, B92, U24).
//!
//! So three kinds of call:
//! - **A call that may wait for the GPU** (`vaGetImage`, `vaMapBuffer` of a
//!   derived image, `vaSyncSurface` where the driver has no
//!   `vaSyncSurface2`, and a decoder's first `vaEndPicture`) is made on
//!   the display's call thread, with the gate held exclusively, and its
//!   caller waits for its answer until the deadline ([`Calls::wait_call`]).
//!   Past it the caller goes on, and the call is left to return on the
//!   thread; what it gives once it does is handed to the caller's late hook
//!   there (a map's unmap). An order not yet begun when its caller left is
//!   never made.
//! - **Every other call** is made on the caller's thread with the gate held
//!   shared ([`Calls::call`]): once the call thread has nothing in hand or
//!   queued, so that no call overtakes one the same thread ordered before
//!   it, and once the gate is had, both by the deadline. So while a call
//!   that waits is in the driver, no other call of ours is, and none waits
//!   in the driver behind it for the display's lock: each waits for the
//!   gate, until its deadline.
//! - **A release** (a destroy, an unmap) is made at once on the caller's
//!   thread where the gate is free and the call thread idle, else ordered
//!   on the call thread behind what it has ([`Calls::release`]); it is
//!   never waited for, and never left out. So nothing a call in hand on
//!   the thread uses (an image, a surface) is destroyed before it returns.
//!
//! The thread starts with the first order, and ends after the display's
//! last, `vaTerminate`.
//!
//! **What the gate costs** is the driver's lock's: a call that waits holds
//! the gate as long as Mesa held its lock for it (`vlVaGetImage` and
//! `vlVaMapBuffer2` map under `drv->mutex`, **read** in Mesa 26.2.4). So
//! the crate makes none on radeonsi where it can help it: the decoders'
//! frames are written into host memory of ours and read there
//! (`super::readback`; docs/video/vaapi.md, "What the gate costs").

use std::collections::VecDeque;
use std::sync::Arc;

use crate::core::runtime::{Bell, MonoTime, Runtime};
use crate::core::sync::{Mutex, RwLock};

use super::ffi::VADisplay;

/// The name of a display's call thread (ours).
const THREAD_NAME: &str = "reims-vgpu-video.va-calls";

/// A display's handle, as the call thread holds it.
#[derive(Clone, Copy)]
struct Raw(VADisplay);

// SAFETY: libva's functions are thread-safe (va.h, "Multithreading
// Guide"); the handle is only passed to them.
unsafe impl Send for Raw {}
// SAFETY: as above.
unsafe impl Sync for Raw {}

/// An order for the call thread: made with the display's handle.
type Order = Box<dyn FnOnce(VADisplay) + Send>;

/// A display's calls (module docs).
pub(crate) struct Calls {
    shared: Arc<Shared>,
}

struct Shared {
    raw: Raw,
    runtime: Arc<dyn Runtime>,
    /// Held shared by each call on a caller's thread, exclusively by each
    /// of the call thread's.
    gate: RwLock<()>,
    queue: Mutex<Queue>,
    /// Rung as an order is queued, and as the thread goes idle.
    bell: Bell,
}

#[derive(Default)]
struct Queue {
    orders: VecDeque<Order>,
    /// Whether the thread is making an order.
    running: bool,
    thread: Thread,
    /// Set as the [`Calls`] drops.
    ending: bool,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Thread {
    #[default]
    NotStarted,
    Started,
    /// None started: orders are made on their caller's thread, with no
    /// bound, as the log says.
    None,
}

/// Why a call was not made, or not answered, by its deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Late {
    /// The gate, or the call thread, was busy with a call that waits until
    /// the deadline passed: the call was not made.
    NotMade,
    /// The call is in the driver on the call thread still.
    InFlight,
}

impl Queue {
    fn idle(&self) -> bool {
        self.orders.is_empty() && !self.running
    }
}

impl Calls {
    /// The calls of the display `raw`, timed on `runtime`'s clock.
    pub(crate) fn new(raw: VADisplay, runtime: Arc<dyn Runtime>) -> Self {
        Self {
            shared: Arc::new(Shared {
                raw: Raw(raw),
                runtime,
                gate: RwLock::new(()),
                queue: Mutex::new(Queue::default()),
                bell: Bell::new(),
            }),
        }
    }

    /// `call`, made on this thread once the call thread is idle and the
    /// gate is had shared, by `deadline` (module docs), taken only where
    /// either is not so now: a deadline set by its first wait starts at
    /// none.
    pub(crate) fn call<T>(
        &self,
        deadline: impl FnOnce() -> MonoTime,
        call: impl FnOnce(VADisplay) -> T,
    ) -> Result<T, Late> {
        let shared = &*self.shared;
        if shared.queue.lock().idle()
            && let Some(_gate) = shared.gate.try_read()
        {
            return Ok(call(shared.raw.0));
        }
        let deadline = deadline();
        let idle = shared.bell.wait_until(&*shared.runtime, deadline, || {
            shared.queue.lock().idle().then_some(())
        });
        idle.ok_or(Late::NotMade)?;
        let left = deadline.saturating_duration_since(shared.runtime.now());
        let _gate = shared.gate.try_read_for(left).ok_or(Late::NotMade)?;
        Ok(call(shared.raw.0))
    }

    /// `call`, made on this thread where the call thread is idle and the
    /// gate free now, with no wait: `None` where not.
    pub(crate) fn try_call<T>(&self, call: impl FnOnce(VADisplay) -> T) -> Option<T> {
        let shared = &*self.shared;
        if !shared.queue.lock().idle() {
            return None;
        }
        let _gate = shared.gate.try_read()?;
        Some(call(shared.raw.0))
    }

    /// `call`, a call that may wait for the GPU, made on the call thread
    /// with the gate held exclusively; its answer, waited for until
    /// `deadline`. Past it, [`Late`]: where the call is in the driver,
    /// what it gives once it returns goes to `late` on the thread.
    pub(crate) fn wait_call<T: Send + 'static>(
        &self,
        deadline: MonoTime,
        call: impl FnOnce(VADisplay) -> T + Send + 'static,
        late: impl FnOnce(VADisplay, T) + Send + 'static,
    ) -> Result<T, Late> {
        let slot = Arc::new(Slot::new());
        let made = Arc::clone(&slot);
        let order: Order = Box::new(move |raw| {
            if !made.begin() {
                return;
            }
            let value = call(raw);
            if let Some(value) = made.give(value) {
                late(raw, value);
            }
        });
        if !self.order(order) {
            return slot.take().ok_or(Late::NotMade);
        }
        slot.wait_until(&*self.shared.runtime, deadline)
    }

    /// `release`, made now on this thread where the call thread is idle
    /// and the gate free, else ordered on the call thread behind what it
    /// has; never waited for (module docs).
    pub(crate) fn release(&self, release: impl FnOnce(VADisplay) + Send + 'static) {
        let shared = &*self.shared;
        let now = shared.queue.lock().idle();
        if now && let Some(_gate) = shared.gate.try_read() {
            release(shared.raw.0);
            return;
        }
        self.order(Box::new(release));
    }

    /// Queues `order` on the call thread, starting it where none runs:
    /// `false` where no thread starts, and `order` was made on this thread
    /// instead.
    fn order(&self, order: Order) -> bool {
        let shared = &self.shared;
        let start = {
            let mut queue = shared.queue.lock();
            match queue.thread {
                Thread::None => {
                    drop(queue);
                    let _gate = shared.gate.write();
                    order(shared.raw.0);
                    return false;
                }
                Thread::Started => false,
                Thread::NotStarted => {
                    queue.thread = Thread::Started;
                    true
                }
            }
        };
        if start {
            let thread = Arc::clone(shared);
            if let Err(error) = shared
                .runtime
                .spawn(THREAD_NAME, Box::new(move || run(&thread)))
            {
                log::warn!(
                    "no thread {THREAD_NAME} starts, so the VA driver's calls that may wait for \
                     the GPU are made on their callers' threads, with no bound: {error}"
                );
                shared.queue.lock().thread = Thread::None;
                let _gate = shared.gate.write();
                order(shared.raw.0);
                return false;
            }
        }
        shared.queue.lock().orders.push_back(order);
        shared.bell.ring();
        true
    }
}

/// The call thread: each order in turn, with the gate held exclusively.
/// It lives as long as the display's last handle and what it ordered.
fn run(shared: &Arc<Shared>) {
    loop {
        let order = shared.bell.wait(&*shared.runtime, || {
            let mut queue = shared.queue.lock();
            match queue.orders.pop_front() {
                Some(order) => {
                    queue.running = true;
                    Some(Some(order))
                }
                // The display's last order is its `vaTerminate`, after
                // which no handle orders more.
                None if queue.ending => Some(None),
                None => None,
            }
        });
        let Some(Some(order)) = order else { return };
        {
            let _gate = shared.gate.write();
            order(shared.raw.0);
        }
        shared.queue.lock().running = false;
        shared.bell.ring();
    }
}

impl Drop for Calls {
    /// Ends the thread once it has made what it was ordered.
    fn drop(&mut self) {
        self.shared.queue.lock().ending = true;
        self.shared.bell.ring();
    }
}

/// Where a [`Calls::wait_call`]'s answer is, as the caller and the thread
/// see it.
enum State<T> {
    Queued,
    Running,
    Given(T),
    /// The caller left: an order not begun is not made, and one begun
    /// hands its answer to the late hook.
    Left,
}

struct Slot<T> {
    state: Mutex<State<T>>,
    bell: Bell,
}

impl<T> Slot<T> {
    fn new() -> Self {
        Self {
            state: Mutex::new(State::Queued),
            bell: Bell::new(),
        }
    }

    /// Whether the order is to be made: not where its caller has left.
    fn begin(&self) -> bool {
        let mut state = self.state.lock();
        if matches!(*state, State::Queued) {
            *state = State::Running;
            return true;
        }
        false
    }

    /// Gives the caller `value`, or gives it back where the caller has
    /// left.
    fn give(&self, value: T) -> Option<T> {
        let mut state = self.state.lock();
        if matches!(*state, State::Left) {
            return Some(value);
        }
        *state = State::Given(value);
        drop(state);
        self.bell.ring();
        None
    }

    /// The answer, where given.
    fn take(&self) -> Option<T> {
        let mut state = self.state.lock();
        match std::mem::replace(&mut *state, State::Left) {
            State::Given(value) => Some(value),
            other => {
                *state = other;
                None
            }
        }
    }

    /// The answer, waited for until `deadline`: past it the caller leaves.
    fn wait_until(&self, runtime: &dyn Runtime, deadline: MonoTime) -> Result<T, Late> {
        if let Some(value) = self.bell.wait_until(runtime, deadline, || self.take()) {
            return Ok(value);
        }
        let mut state = self.state.lock();
        match std::mem::replace(&mut *state, State::Left) {
            State::Given(value) => Ok(value),
            State::Running => Err(Late::InFlight),
            State::Queued | State::Left => Err(Late::NotMade),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    use crate::core::runtime::StdRuntime;

    use super::*;

    fn calls() -> Calls {
        Calls::new(std::ptr::null_mut(), StdRuntime::shared())
    }

    fn soon() -> MonoTime {
        StdRuntime.deadline_in(Duration::from_millis(30))
    }

    fn later() -> MonoTime {
        StdRuntime.deadline_in(Duration::from_secs(5))
    }

    /// A call that waits until the test sends on the channel this gives,
    /// as a driver's call waits for the GPU.
    fn held<T: Send + 'static>(value: T) -> (mpsc::Sender<()>, impl FnOnce(VADisplay) -> T + Send) {
        let (release, released) = mpsc::channel::<()>();
        (release, move |_| {
            let _ = released.recv();
            value
        })
    }

    fn within_5_s(mut ready: impl FnMut() -> bool) -> bool {
        for _ in 0..5000 {
            if ready() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        ready()
    }

    #[test]
    fn a_call_that_waits_past_its_deadline_hands_its_answer_to_its_late_hook() {
        let calls = calls();
        let (release, call) = held(7);
        let late = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&late);
        let answer = calls.wait_call(soon(), call, move |_, value| {
            seen.store(value, Ordering::SeqCst);
        });
        assert_eq!(answer, Err(Late::InFlight));
        release.send(()).expect("the call waits");
        assert!(within_5_s(|| late.load(Ordering::SeqCst) == 7));
    }

    #[test]
    fn a_call_on_this_thread_waits_for_the_call_thread_until_its_deadline() {
        let calls = calls();
        // Made at once, the call takes no deadline.
        let at_once = calls.call(|| unreachable!("a deadline taken"), |_| 0);
        assert_eq!(at_once, Ok(0));
        let (release, call) = held(());
        assert_eq!(
            calls.wait_call(soon(), call, |_, ()| {}),
            Err(Late::InFlight)
        );
        let start = StdRuntime.now();
        assert_eq!(calls.call(soon, |_| 1), Err(Late::NotMade));
        let waited = StdRuntime.elapsed_since(start);
        assert!(waited >= Duration::from_millis(25), "{waited:?}");
        assert!(waited < Duration::from_secs(1), "{waited:?}");
        assert_eq!(calls.try_call(|_| 1), None);
        release.send(()).expect("the call waits");
        assert_eq!(calls.call(later, |_| 2), Ok(2));
        assert_eq!(calls.try_call(|_| 3), Some(3));
    }

    #[test]
    fn an_order_its_caller_left_before_it_began_is_never_made() {
        let calls = calls();
        let (release, call) = held(());
        assert_eq!(
            calls.wait_call(soon(), call, |_, ()| {}),
            Err(Late::InFlight)
        );
        let made = Arc::new(AtomicUsize::new(0));
        let making = Arc::clone(&made);
        let late = Arc::clone(&made);
        let behind = calls.wait_call(
            soon(),
            move |_| making.fetch_add(1, Ordering::SeqCst),
            move |_, _| {
                late.fetch_add(10, Ordering::SeqCst);
            },
        );
        assert_eq!(behind, Err(Late::NotMade));
        release.send(()).expect("the call waits");
        assert_eq!(calls.call(later, |_| ()), Ok(()));
        assert_eq!(made.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_release_behind_a_call_in_hand_is_made_after_it_and_in_order() {
        let calls = calls();
        let order = Arc::new(Mutex::new(Vec::new()));
        let (release, call) = held(());
        let first = Arc::clone(&order);
        assert_eq!(
            calls.wait_call(
                soon(),
                move |raw| {
                    call(raw);
                    first.lock().push("call");
                },
                |_, ()| {}
            ),
            Err(Late::InFlight)
        );
        for name in ["destroy", "and then"] {
            let order = Arc::clone(&order);
            calls.release(move |_| order.lock().push(name));
        }
        std::thread::sleep(Duration::from_millis(20));
        assert!(
            order.lock().is_empty(),
            "released while the call was in hand"
        );
        release.send(()).expect("the call waits");
        assert_eq!(calls.call(later, |_| ()), Ok(()));
        assert_eq!(*order.lock(), ["call", "destroy", "and then"]);
        // Idle, a release is made at once.
        let now = Arc::clone(&order);
        calls.release(move |_| now.lock().push("now"));
        assert_eq!(order.lock().last(), Some(&"now"));
    }

    #[test]
    fn no_call_on_this_thread_is_in_the_driver_beside_the_call_thread() {
        use std::sync::atomic::AtomicBool;
        let calls = Arc::new(calls());
        // Callers inside, one each; the call thread's count as 1000.
        let inside = Arc::new(AtomicUsize::new(0));
        let crossed = Arc::new(AtomicBool::new(false));
        let enter = |inside: &AtomicUsize, crossed: &AtomicBool, exclusive: bool| {
            let weight = if exclusive { 1000 } else { 1 };
            let before = inside.fetch_add(weight, Ordering::SeqCst);
            if (exclusive && before != 0) || (!exclusive && before >= 1000) {
                crossed.store(true, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_micros(200));
            inside.fetch_sub(weight, Ordering::SeqCst);
        };
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let (calls, inside, crossed) = (&calls, &inside, &crossed);
                scope.spawn(move || {
                    for _ in 0..50 {
                        calls
                            .call(later, |_| enter(inside, crossed, false))
                            .expect("made");
                    }
                });
            }
            for _ in 0..50 {
                let (inside, crossed) = (Arc::clone(&inside), Arc::clone(&crossed));
                calls
                    .wait_call(later(), move |_| enter(&inside, &crossed, true), |_, ()| {})
                    .expect("answered");
            }
        });
        assert!(
            !crossed.load(Ordering::SeqCst),
            "a call beside the call thread's"
        );
    }

    #[test]
    fn the_thread_ends_once_its_calls_drop_and_its_orders_are_made() {
        let calls = calls();
        let shared = Arc::downgrade(&calls.shared);
        let (release, call) = held(());
        assert_eq!(
            calls.wait_call(soon(), call, |_, ()| {}),
            Err(Late::InFlight)
        );
        drop(calls);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(shared.strong_count(), 1, "the thread ended in its call");
        release.send(()).expect("the call waits");
        assert!(within_5_s(|| shared.strong_count() == 0));
    }
}
