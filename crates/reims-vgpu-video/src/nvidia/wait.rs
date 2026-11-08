//! The crate's waits on the GPU, each bounded (core's `device_wait`): a
//! picture's decode and its copy are asked after with the driver's
//! non-blocking question, and a call with no bounded form is made on a
//! thread of its own ([`Calls`]) and its answer waited for, until they are
//! done or [`DEVICE_WAIT`] has passed, and work past that is kept, with what
//! it may still write, until the GPU is done with it ([`keep`]).
//!
//! | wait | asked with | in place of |
//! |---|---|---|
//! | a picture's decode | `cuvidGetDecodeStatus` until it is no longer `cuvidDecodeStatus_InProgress` (`nvdec::decoder`) | `cuvidMapVideoFrame64`'s own wait |
//! | its map and copies | `cuStreamQuery` of the decoder's own stream, which the map's post-processing and the two `cuMemcpy2DAsync` run on | `cuMemcpy2D`, which returns once its copy is done |
//! | copies into pageable staging memory | the two `cuMemcpy2DAsync`, which return once copied into pageable memory, made on the decoder's call thread, then the stream | the same copies on the caller's thread |
//! | an encode | the waiting `nvEncLockBitstream`, made on the session's call thread, whose answer is waited for (`nvenc::locks`, which says why `doNotWait` does not serve) | the same lock on the caller's thread |
//!
//! A picture's waits share one deadline ([`Deadline`]), from the first of
//! them.
//!
//! What a decoder or an encoder does once a wait of its has
//! passed its deadline: that frame fails as a malfunction, and each later
//! call on it fails at once the same way ([`crate::nvidia::Error::Stalled`]), so
//! that the guest hears of each frame without a wait of its own; NVDEC's
//! and NVENC's handles are kept until the GPU is done with the work, and
//! destroyed then. A call on a call thread that has not returned is left
//! to return there: nothing it holds (the handle it runs on, the memory it
//! reads or writes) is used, unmapped, freed or destroyed until it has
//! ([`Unreturned`]).
//!
//! The waits run on the clock of the runtime the backend was opened with,
//! the process's first ([`use_runtime`]), and the work kept is asked after
//! on a thread of its ([`Lingering::process`]).

use std::any::Any;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

pub(crate) use crate::core::device_wait::Unfinished;
use crate::core::device_wait::{self, DEVICE_WAIT, Lingering};
use crate::core::runtime::{
    Answer, Bell, MonoTime, Runtime, StdRuntime, Unanswered, answer, park_for_good,
};
use crate::core::sync::Mutex;

/// The runtime the crate's waits run on: the first the backend was opened
/// with, else std's.
static RUNTIME: OnceLock<Arc<dyn Runtime>> = OnceLock::new();

/// Makes `runtime` the one the crate's waits run on, where none was made
/// yet: the backend's opening calls it.
pub(crate) fn use_runtime(runtime: &Arc<dyn Runtime>) {
    RUNTIME.get_or_init(|| Arc::clone(runtime));
}

/// The runtime the crate's waits run on.
pub(crate) fn runtime() -> &'static Arc<dyn Runtime> {
    RUNTIME.get_or_init(StdRuntime::shared)
}

/// How long a frame's work is waited for: [`DEVICE_WAIT`], or what a test
/// of the hardware sets on its thread.
fn wait() -> Duration {
    #[cfg(test)]
    if let Some(wait) = tests::WAIT.get() {
        return wait;
    }
    DEVICE_WAIT
}

/// The deadline of work due now: [`wait`] from now.
pub(crate) fn deadline() -> MonoTime {
    runtime().deadline_in(wait())
}

/// One deadline that several waits share, [`wait`] from the first of them
/// to ask: a picture's, which its decode, map and copies wait until, so
/// that it lands within one deadline of the first wait for it. A clone
/// shares it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Deadline(Arc<OnceLock<MonoTime>>);

impl Deadline {
    /// The deadline: [`wait`] from the first time it was asked for.
    pub(crate) fn at(&self) -> MonoTime {
        *self.0.get_or_init(deadline)
    }
}

/// `check`'s answer, asked until it has one or `deadline` has passed (core's
/// `poll_until`): `None` past it.
pub(crate) fn poll<T>(deadline: MonoTime, check: impl FnMut() -> Option<T>) -> Option<T> {
    device_wait::poll_until(&**runtime(), deadline, check)
}

/// Keeps `work`, which `what` names, until the GPU is done with it.
pub(crate) fn keep(what: String, work: Box<dyn Unfinished>) {
    #[cfg(test)]
    tests::KEPT_EVER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    Lingering::process(runtime()).keep(what, work);
}

/// A thread of a decoder's or an encoder's own that makes its driver calls
/// with no bounded form, one at a time, so that the caller waits for each
/// one's answer until its deadline ([`Calls::call`]): NVENC's locks
/// (`nvenc::locks`), and NVDEC's copies into pageable memory
/// (`nvdec::decoder`).
///
/// A call past its deadline goes on on the thread, which takes the next
/// order only once it has returned. Its caller keeps it, with what it
/// holds, until it has ([`Unreturned`]), and makes no call that would use
/// what it holds meanwhile. The thread ends once its [`Calls`] has dropped
/// and it has made the call it was given.
pub(crate) struct Calls {
    shared: Arc<CallsShared>,
}

struct CallsShared {
    next: Mutex<Next>,
    bell: Bell,
}

/// What the thread takes next.
#[derive(Default)]
struct Next {
    /// The next call: one at a time, since the caller waits for each
    /// answer, or keeps the call that passed its deadline, before it orders
    /// another. An order not yet taken is replaced by the next, and its
    /// answer is then that none is to come.
    call: Option<Box<dyn FnOnce() + Send>>,
    /// Set as the [`Calls`] drops.
    ending: bool,
}

impl Calls {
    /// Starts the thread, named `name`: `None` where no thread starts, and
    /// the caller then makes its calls on its own thread, with no bound, as
    /// the log says.
    pub(crate) fn start(name: &'static str) -> Option<Self> {
        let shared = Arc::new(CallsShared {
            next: Mutex::new(Next::default()),
            bell: Bell::new(),
        });
        let thread = Arc::clone(&shared);
        match runtime().spawn(name, Box::new(move || run_calls(&thread))) {
            Ok(()) => Some(Self { shared }),
            Err(error) => {
                log::warn!("no thread {name} starts, and its calls wait with no bound: {error}");
                None
            }
        }
    }

    /// Orders `call` made on the thread: its answer, which the caller waits
    /// for until the deadline ([`answer_by`]).
    pub(crate) fn call<T: Send + 'static>(
        &self,
        call: impl FnOnce() -> T + Send + 'static,
    ) -> Answer<T> {
        let (answerer, answer) = answer();
        self.shared.next.lock().call = Some(Box::new(move || answerer.give(call())));
        self.shared.bell.ring();
        answer
    }
}

impl Drop for Calls {
    /// Ends the thread once it has made the call it was given.
    fn drop(&mut self) {
        self.shared.next.lock().ending = true;
        self.shared.bell.ring();
    }
}

fn run_calls(shared: &CallsShared) {
    loop {
        let call = shared.bell.wait(&**runtime(), || {
            let mut next = shared.next.lock();
            match next.call.take() {
                Some(call) => Some(Some(call)),
                None if next.ending => Some(None),
                None => None,
            }
        });
        match call {
            Some(Some(call)) => {
                let Some(_step) = runtime().exit_step() else {
                    park_for_good()
                };
                call();
            }
            Some(None) | None => return,
        }
    }
}

/// `answer`'s answer, waited for until `deadline`: `Err` past it, and where
/// none is to come.
pub(crate) fn answer_by<T>(answer: &Answer<T>, deadline: MonoTime) -> Result<T, Unanswered> {
    answer.wait_until(&**runtime(), deadline)
}

/// Whether the call `answer` answers has returned, or will never be made:
/// asked without waiting.
pub(crate) fn returned<T>(answer: &Answer<T>) -> bool {
    !matches!(
        answer_by(answer, MonoTime::from_nanos(0)),
        Err(Unanswered::TimedOut)
    )
}

/// A call made on a [`Calls`] thread that had not returned by its deadline,
/// and what it holds: kept ([`keep`]) until it has returned, then dropped,
/// which lets all of it go.
pub(crate) struct Unreturned<T> {
    answer: Answer<T>,
    /// What the call holds, dropped once it has returned.
    _holds: Vec<Box<dyn Any + Send>>,
}

impl<T> Unreturned<T> {
    /// The call `answer` answers, holding `holds` until it returns.
    pub(crate) fn new(answer: Answer<T>, holds: Vec<Box<dyn Any + Send>>) -> Self {
        Self {
            answer,
            _holds: holds,
        }
    }
}

impl<T: Send> Unfinished for Unreturned<T> {
    fn finished(&mut self) -> bool {
        returned(&self.answer)
    }
}

/// How many works are kept now, process-wide.
#[cfg(test)]
pub(crate) fn kept() -> usize {
    Lingering::process(runtime()).count()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    thread_local! {
        /// A wait a test of the hardware sets in place of
        /// [`super::DEVICE_WAIT`] for the waits on its thread, so that work
        /// in flight passes its deadline.
        pub(crate) static WAIT: Cell<Option<Duration>> = const { Cell::new(None) };
    }

    /// How many works were kept, process-wide, ever.
    pub(crate) static KEPT_EVER: AtomicUsize = AtomicUsize::new(0);

    /// Whether `ready` holds within 5 s, asked every millisecond.
    pub(crate) fn within_5_s(mut ready: impl FnMut() -> bool) -> bool {
        for _ in 0..5000 {
            if ready() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        ready()
    }

    /// A call that returns `value` once the test sends on the channel this
    /// gives, as a driver's call returns once the GPU is done.
    fn held<T: Send + 'static>(value: T) -> (mpsc::Sender<()>, impl FnOnce() -> T + Send) {
        let (release, released) = mpsc::channel::<()>();
        (release, move || {
            let _ = released.recv();
            value
        })
    }

    fn soon() -> MonoTime {
        runtime().deadline_in(Duration::from_millis(20))
    }

    #[test]
    fn a_call_past_its_deadline_is_kept_with_what_it_holds_until_it_returns() {
        let calls = Calls::start("test-calls").expect("a thread");
        let (release, call) = held(7);
        let answer = calls.call(call);
        assert_eq!(answer_by(&answer, soon()).err(), Some(Unanswered::TimedOut));
        let holds = Arc::new(());
        let mut kept = Unreturned::new(answer, vec![Box::new(Arc::clone(&holds))]);
        std::thread::sleep(Duration::from_millis(20));
        assert!(!kept.finished());
        assert_eq!(Arc::strong_count(&holds), 2);
        release.send(()).expect("the call waits");
        assert!(within_5_s(|| kept.finished()));
        drop(kept);
        assert_eq!(Arc::strong_count(&holds), 1);
    }

    #[test]
    fn the_thread_takes_the_next_call_only_once_the_last_has_returned() {
        let calls = Calls::start("test-calls").expect("a thread");
        let (release, call) = held(1);
        let first = calls.call(call);
        assert_eq!(answer_by(&first, soon()).err(), Some(Unanswered::TimedOut));
        let made = Arc::new(AtomicBool::new(false));
        let making = Arc::clone(&made);
        let second = calls.call(move || making.store(true, Ordering::SeqCst));
        std::thread::sleep(Duration::from_millis(20));
        assert!(
            !made.load(Ordering::SeqCst),
            "made while the first was in flight"
        );
        release.send(()).expect("the call waits");
        assert_eq!(answer_by(&first, MonoTime::END), Ok(1));
        assert_eq!(answer_by(&second, MonoTime::END), Ok(()));
        assert!(made.load(Ordering::SeqCst));
    }

    #[test]
    fn an_order_not_yet_taken_is_replaced_and_answers_that_none_is_to_come() {
        let calls = Calls::start("test-calls").expect("a thread");
        let (release, call) = held(());
        let first = calls.call(call);
        assert_eq!(answer_by(&first, soon()).err(), Some(Unanswered::TimedOut));
        let replaced = calls.call(|| 1);
        let next = calls.call(|| 2);
        assert_eq!(answer_by(&replaced, soon()).err(), Some(Unanswered::Gone));
        release.send(()).expect("the call waits");
        assert_eq!(answer_by(&next, MonoTime::END), Ok(2));
    }

    #[test]
    fn the_thread_ends_once_its_calls_drop_and_its_call_returns() {
        let calls = Calls::start("test-calls").expect("a thread");
        let shared = Arc::downgrade(&calls.shared);
        let (release, call) = held(());
        let answer = calls.call(call);
        assert_eq!(answer_by(&answer, soon()).err(), Some(Unanswered::TimedOut));
        drop(calls);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(shared.strong_count(), 1, "the thread ended in its call");
        release.send(()).expect("the call waits");
        assert!(within_5_s(|| shared.strong_count() == 0));
        assert!(returned(&answer));
    }

    /// A shared deadline is the wait from the first time any of its clones
    /// is asked for it, and the same to each after.
    #[test]
    fn a_shared_deadline_is_set_by_the_first_to_ask() {
        let shared = Deadline::default();
        let clone = shared.clone();
        std::thread::sleep(Duration::from_millis(20));
        let before = runtime().now();
        let at = clone.at();
        let after = runtime().now();
        assert!(
            at >= before.saturating_add(DEVICE_WAIT) && at <= after.saturating_add(DEVICE_WAIT)
        );
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(shared.at(), at);
        assert_ne!(Deadline::default().at(), at);
    }
}
