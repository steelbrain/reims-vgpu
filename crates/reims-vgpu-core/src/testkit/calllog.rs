//! Fake: `CallLog`: every platform call in order, with its arguments (architecture 13.2).
//!
//! Every fake platform object, service, host block and seam recorder appends one [`Call`] per call
//! whose position the guest or host observes, so a unit test can assert core's platform call order
//! and arguments (13.2: "Unit tests may assert the call log"). A Metal call whose position is not
//! observable (a getter, a query, a setter no other property aliases) is neither logged nor
//! scripted: the fake answers it from its model. A `CallLog` is a cheap clone of one
//! shared log, so the platform, the fake host and the seam recorders can interleave into one
//! sequence.
//!
//! Beside the calls, a log a test turns events on for with [`CallLog::record_events`] also keeps
//! [`Event`]s: each fake object's creation and last release, and the release with its
//! [`ReleaseContext`]. They are test-only instrumentation. Events are off by default, and a log
//! that records none keeps only its calls.
//!
use std::cell::Cell;
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use crate::sync::Mutex;

use crate::platform::{Handle, ObjId};

/// One argument of a recorded call. Handles are recorded by identity; value structs by their
/// `Debug` text ([`Arg::val`]), so a test compares them with `Arg::val(&expected)`.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    /// `NSUInteger`, `uint64_t` and the raw Metal enums.
    U64(u64),
    /// `NSInteger`, `int64_t`.
    I64(i64),
    /// `uint32_t`.
    U32(u32),
    /// `int32_t`.
    I32(i32),
    /// `uint16_t`.
    U16(u16),
    /// `uint8_t`.
    U8(u8),
    /// `BOOL`.
    Bool(bool),
    /// `float`.
    F32(f32),
    /// `double`.
    F64(f64),
    /// A host address.
    Ptr(usize),
    /// An object argument: `None` is nil.
    Obj(Option<ObjId>),
    /// An array of objects (`const id *`), nil entries kept.
    Objs(Vec<Option<ObjId>>),
    /// A string (`NSString *`, a C string).
    Str(String),
    /// Bytes (`NSData *`, a byte range).
    Bytes(Vec<u8>),
    /// An array of `NSUInteger`s (offsets, indices).
    U64s(Vec<u64>),
    /// An array of `float`s (LOD clamps).
    F32s(Vec<f32>),
    /// A block argument: whether it is non-nil.
    Block(bool),
    /// A value struct, by its `Debug` text.
    Val(String),
}

impl Arg {
    /// A value struct argument (`MTLSize`, `NSRange`, …) by its `Debug` text.
    pub fn val<T: Debug + ?Sized>(v: &T) -> Arg {
        Arg::Val(format!("{v:?}"))
    }

    /// A non-nil object argument.
    pub fn obj<H: Handle>(h: &H) -> Arg {
        Arg::Obj(Some(h.id()))
    }

    /// An object argument that may be nil.
    pub fn opt<H: Handle>(h: Option<&H>) -> Arg {
        Arg::Obj(h.map(Handle::id))
    }

    /// An array of objects that may hold nils.
    pub fn objs<H: Handle>(hs: &[Option<&H>]) -> Arg {
        Arg::Objs(hs.iter().map(|h| h.map(Handle::id)).collect())
    }

    /// The identity of an object argument, if it is one and is non-nil.
    pub fn as_obj(&self) -> Option<ObjId> {
        match self {
            Arg::Obj(o) => *o,
            _ => None,
        }
    }

    /// The value of a `U64` argument.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Arg::U64(v) => Some(*v),
            _ => None,
        }
    }
}

/// One recorded call.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    /// Position in the log, from 0.
    pub seq: u64,
    /// The receiver's identity (`ObjId(0)` for a service or free function).
    pub obj: ObjId,
    /// The receiver's class (`MTLDevice`, `mach_vm`, `IOSurface`).
    pub class: &'static str,
    /// The selector, C function or method name.
    pub sel: &'static str,
    /// The arguments, in order.
    pub args: Vec<Arg>,
    /// The calling thread's name, if it has one.
    pub thread: Option<String>,
}

#[derive(Debug, Default)]
struct Inner {
    calls: Vec<Call>,
    next_seq: u64,
    /// Whether [`Event`]s are kept ([`CallLog::record_events`]).
    recording: bool,
    events: Vec<EventRecord>,
    next_event: u64,
}

/// The shared, ordered log of platform calls. `Clone` shares the log.
#[derive(Debug, Clone)]
pub struct CallLog {
    inner: Arc<Mutex<Inner>>,
}

impl Default for CallLog {
    fn default() -> Self {
        CallLog::new()
    }
}

impl CallLog {
    /// An empty log. It records events from the start when [`record_events_by_default`] is on.
    pub fn new() -> CallLog {
        let log = CallLog {
            inner: Arc::new(Mutex::new(Inner {
                calls: Vec::new(),
                next_seq: 0,
                recording: false,
                events: Vec::new(),
                next_event: 0,
            })),
        };
        if BY_DEFAULT.load(Ordering::SeqCst) {
            log.record_events(true);
        }
        log
    }

    /// Appends one call.
    pub fn push(&self, obj: ObjId, class: &'static str, sel: &'static str, args: Vec<Arg>) {
        let thread = std::thread::current().name().map(str::to_owned);
        let mut g = self.inner.lock();
        let seq = g.next_seq;
        g.next_seq = g.next_seq.wrapping_add(1);
        g.calls.push(Call {
            seq,
            obj,
            class,
            sel,
            args,
            thread,
        });
    }

    /// A copy of every call so far, in order.
    pub fn calls(&self) -> Vec<Call> {
        self.inner.lock().calls.clone()
    }

    /// Every selector so far, in order.
    pub fn sels(&self) -> Vec<&'static str> {
        self.inner.lock().calls.iter().map(|c| c.sel).collect()
    }

    /// The calls of one selector, in order.
    pub fn calls_to(&self, sel: &str) -> Vec<Call> {
        self.filter(|c| c.sel == sel)
    }

    /// The calls received by one object, in order.
    pub fn calls_on(&self, obj: ObjId) -> Vec<Call> {
        self.filter(|c| c.obj == obj)
    }

    /// The calls of one class, in order.
    pub fn calls_of(&self, class: &str) -> Vec<Call> {
        self.filter(|c| c.class == class)
    }

    /// The calls matching `pred`, in order.
    pub fn filter(&self, pred: impl Fn(&Call) -> bool) -> Vec<Call> {
        self.inner
            .lock()
            .calls
            .iter()
            .filter(|c| pred(c))
            .cloned()
            .collect()
    }

    /// How many calls of `sel` so far.
    pub fn count(&self, sel: &str) -> usize {
        self.inner
            .lock()
            .calls
            .iter()
            .filter(|c| c.sel == sel)
            .count()
    }

    /// The number of calls so far.
    pub fn len(&self) -> usize {
        self.inner.lock().calls.len()
    }

    /// Whether nothing was called.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().calls.is_empty()
    }

    /// A position to read from later with [`CallLog::since`].
    pub fn mark(&self) -> u64 {
        self.inner.lock().next_seq
    }

    /// The calls made at or after `mark`.
    pub fn since(&self, mark: u64) -> Vec<Call> {
        self.filter(|c| c.seq >= mark)
    }

    /// The selectors called at or after `mark`.
    pub fn sels_since(&self, mark: u64) -> Vec<&'static str> {
        self.since(mark).into_iter().map(|c| c.sel).collect()
    }

    /// Removes and returns every call so far (the sequence numbers keep counting).
    pub fn take(&self) -> Vec<Call> {
        std::mem::take(&mut self.inner.lock().calls)
    }

    /// Forgets every call so far.
    pub fn clear(&self) {
        self.inner.lock().calls.clear();
    }

    /// Whether `sels` occur in this order (not necessarily adjacent).
    pub fn contains_in_order(&self, sels: &[&str]) -> bool {
        let g = self.inner.lock();
        let mut want = sels.iter().peekable();
        for c in &g.calls {
            if want.peek().is_some_and(|s| **s == c.sel) {
                let _ = want.next();
            }
        }
        want.peek().is_none()
    }

    /// The first call of `sel` at or after position `from` in the current log, with its index.
    pub fn find_from(&self, from: usize, sel: &str) -> Option<(usize, Call)> {
        let g = self.inner.lock();
        g.calls
            .iter()
            .enumerate()
            .skip(from)
            .find(|(_, c)| c.sel == sel)
            .map(|(i, c)| (i, c.clone()))
    }

    /// Starts (`true`) or stops keeping [`Event`]s: this log's objects' creations and last
    /// releases.
    pub fn record_events(&self, on: bool) {
        let was = std::mem::replace(&mut self.inner.lock().recording, on);
        if on && !was {
            RECORDING.lock().push(Arc::downgrade(&self.inner));
            let _ = RECORDING_LIVE.fetch_add(1, Ordering::SeqCst);
        } else if was && !on {
            let mut all = RECORDING.lock();
            all.retain(|w| w.upgrade().is_some_and(|i| !Arc::ptr_eq(&i, &self.inner)));
            RECORDING_LIVE.store(all.len(), Ordering::SeqCst);
        }
    }

    /// Whether this log keeps events.
    pub fn recording_events(&self) -> bool {
        RECORDING_LIVE.load(Ordering::SeqCst) != 0 && self.inner.lock().recording
    }

    /// Records `event` on the current thread, if this log keeps events.
    pub fn event(&self, event: Event) {
        if RECORDING_LIVE.load(Ordering::SeqCst) == 0 {
            return;
        }
        let who = this_thread();
        record(&self.inner, &who, event);
    }

    /// A copy of every event so far, in order.
    pub fn events(&self) -> Vec<EventRecord> {
        self.inner.lock().events.clone()
    }

    /// Removes and returns every event so far (their sequence numbers keep counting).
    pub fn take_events(&self) -> Vec<EventRecord> {
        std::mem::take(&mut self.inner.lock().events)
    }
}

// ---- Events: object lifetimes ----

/// What kind of thread an event happened on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadClass {
    /// A thread core started with a name (`rt::thread::spawn_named`: the FIFO thread).
    Core,
    /// A serial queue's worker (`rt::queue`): a core thread with no name.
    Worker,
    /// The fake's stand-in for Metal's completion thread while it runs completed handlers
    /// (`fake::objects::CompleteOn`), the caller's thread under `CompleteOn::Caller` included.
    Completion,
    /// Any other thread: the test's own (the host's), or a thread of the fake's launcher.
    Other,
}

/// Where a fake object's last release ran: on which kind of thread.
///
/// It holds no queue: the thread's queue chain lives in a thread-local with a destructor
/// (`rt::queue`'s), and reading it would initialise it where the program had not, which moves
/// its destruction at the thread's exit ([`ReleaseContext::here`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseContext {
    /// The kind of thread.
    pub thread: ThreadClass,
}

impl ReleaseContext {
    /// The context of the current thread.
    ///
    /// It reads only thread-locals with no destructor (`const` cells), so recording never
    /// initialises one the program has not touched: Rust destroys a thread's locals last
    /// initialised first, and a local first touched here would outlive, or die before, the ones
    /// the program touches next, and change what the thread's exit does (a release there that
    /// reaches a destroyed local aborts). It can be read at any point of the thread's exit.
    pub fn here() -> ReleaseContext {
        ReleaseContext {
            thread: thread_class(),
        }
    }
}

/// One event a recording log keeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// A fake object was made (`World::new_obj`).
    Created {
        /// Its identity.
        obj: ObjId,
        /// Its class.
        class: &'static str,
    },
    /// A fake object's last reference went: the Metal object is destroyed here.
    Released {
        /// Its identity.
        obj: ObjId,
        /// Its class.
        class: &'static str,
        /// Where the release ran.
        context: ReleaseContext,
    },
}

/// One recorded event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventRecord {
    /// Position among the log's events, from 0.
    pub seq: u64,
    /// The calls logged before it: the `seq` the next [`Call`] gets.
    pub at: u64,
    /// The thread, numbered from 1 in the order threads first recorded an event in this
    /// process.
    pub thread: u64,
    /// The thread's name, if it has one.
    pub thread_name: Option<String>,
    /// What happened.
    pub event: Event,
}

/// Every log that records events.
static RECORDING: Mutex<Vec<Weak<Mutex<Inner>>>> = Mutex::new(Vec::new());
/// How many logs may record: 0 skips every event at once.
static RECORDING_LIVE: AtomicUsize = AtomicUsize::new(0);
/// Whether every new log records from the start ([`record_events_by_default`]).
static BY_DEFAULT: AtomicBool = AtomicBool::new(false);
static NEXT_THREAD: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// This thread's number for events (0: none yet). No destructor, so it can be read at the
    /// thread's exit.
    static THREAD_NO: Cell<u64> = const { Cell::new(0) };
    /// The fake is running completed handlers on this thread ([`as_completion_thread`]).
    static COMPLETION: Cell<bool> = const { Cell::new(false) };
}

/// Makes every log created from now on record events from the start (`true`), or not: for a
/// process that runs one program at a time, whose fake processes are made after it decides.
pub fn record_events_by_default(on: bool) {
    BY_DEFAULT.store(on, Ordering::SeqCst);
}

/// Runs `f` as the fake's completion thread: its releases are [`ThreadClass::Completion`]'s.
pub(crate) fn as_completion_thread<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            COMPLETION.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(COMPLETION.with(|c| c.replace(true)));
    f()
}

fn thread_class() -> ThreadClass {
    if COMPLETION.with(Cell::get) {
        ThreadClass::Completion
    } else if crate::rt::thread::is_core_thread() {
        if std::thread::current().name().is_some() {
            ThreadClass::Core
        } else {
            ThreadClass::Worker
        }
    } else {
        ThreadClass::Other
    }
}

/// This thread's event number and name. `THREAD_NO` has no destructor, and std's handle of the
/// current thread is the one every logged call reads ([`CallLog::push`]); std sets it as a
/// thread it spawns starts.
fn this_thread() -> (u64, Option<String>) {
    let no = THREAD_NO.with(|n| {
        if n.get() == 0 {
            n.set(NEXT_THREAD.fetch_add(1, Ordering::SeqCst));
        }
        n.get()
    });
    (no, std::thread::current().name().map(str::to_owned))
}

fn record(inner: &Mutex<Inner>, who: &(u64, Option<String>), event: Event) {
    let mut g = inner.lock();
    if !g.recording {
        return;
    }
    let seq = g.next_event;
    g.next_event = g.next_event.wrapping_add(1);
    let at = g.next_seq;
    g.events.push(EventRecord {
        seq,
        at,
        thread: who.0,
        thread_name: who.1.clone(),
        event,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    struct H(u64);
    impl Handle for H {
        fn clone_ref(&self) -> Self {
            H(self.0)
        }
        fn id(&self) -> ObjId {
            ObjId(self.0)
        }
    }

    /// Calls are kept in order with their arguments and thread
    #[test]
    fn push_and_read_back() {
        let log = CallLog::new();
        assert!(log.is_empty());
        log.push(ObjId(1), "MTLDevice", "newCommandQueue", vec![]);
        log.push(
            ObjId(2),
            "MTLCommandQueue",
            "setLabel:",
            vec![Arg::Str("reims-vgpu.exec".into())],
        );
        log.push(ObjId(1), "MTLDevice", "newFence", vec![]);
        assert_eq!(log.len(), 3);
        assert_eq!(log.sels(), vec!["newCommandQueue", "setLabel:", "newFence"]);
        assert_eq!(log.calls_on(ObjId(1)).len(), 2);
        assert_eq!(
            log.calls_of("MTLCommandQueue")[0].args[0],
            Arg::Str("reims-vgpu.exec".into())
        );
        assert_eq!(log.calls_to("newFence")[0].seq, 2);
        assert_eq!(log.count("newCommandQueue"), 1);
        let c = log.calls();
        assert_eq!(
            c[0].thread,
            std::thread::current().name().map(str::to_owned)
        );
    }

    /// Marks, since, take and clear
    #[test]
    fn marks_and_take() {
        let log = CallLog::new();
        log.push(ObjId(0), "x", "a", vec![]);
        let m = log.mark();
        log.push(ObjId(0), "x", "b", vec![]);
        log.push(ObjId(0), "x", "c", vec![]);
        assert_eq!(log.sels_since(m), vec!["b", "c"]);
        let taken = log.take();
        assert_eq!(taken.len(), 3);
        assert!(log.is_empty());
        log.push(ObjId(0), "x", "d", vec![]);
        assert_eq!(log.calls()[0].seq, 3);
        log.clear();
        assert_eq!(log.len(), 0);
    }

    /// In-order matching skips unrelated calls; find_from returns the index
    #[test]
    fn order_helpers() {
        let log = CallLog::new();
        for s in ["a", "x", "b", "y", "c"] {
            log.push(ObjId(0), "k", s, vec![]);
        }
        assert!(log.contains_in_order(&["a", "b", "c"]));
        assert!(!log.contains_in_order(&["b", "a"]));
        assert!(log.contains_in_order(&[]));
        assert_eq!(log.find_from(1, "b").map(|(i, _)| i), Some(2));
        assert!(log.find_from(3, "b").is_none());
    }

    /// The argument constructors
    #[test]
    fn arg_constructors() {
        let h = H(7);
        assert_eq!(Arg::obj(&h), Arg::Obj(Some(ObjId(7))));
        assert_eq!(Arg::opt::<H>(None), Arg::Obj(None));
        assert_eq!(
            Arg::objs(&[Some(&h), None]),
            Arg::Objs(vec![Some(ObjId(7)), None])
        );
        assert_eq!(Arg::val(&(1, 2)), Arg::Val("(1, 2)".into()));
        assert_eq!(Arg::obj(&h).as_obj(), Some(ObjId(7)));
        assert_eq!(Arg::U64(3).as_u64(), Some(3));
        assert_eq!(Arg::Bool(true).as_u64(), None);
        assert_eq!(Arg::Bool(true).as_obj(), None);
    }

    /// Events are kept only while recording, in order, each placed among the calls by `at`, with a
    /// thread number stable on its thread and distinct across threads
    #[test]
    fn events_record_only_while_on() {
        let log = CallLog::new();
        let created = |n| Event::Created {
            obj: ObjId(n),
            class: "MTLBuffer",
        };
        log.event(created(1));
        assert!(log.events().is_empty() && !log.recording_events());
        log.record_events(true);
        assert!(log.recording_events());
        log.push(ObjId(0), "k", "a", vec![]);
        log.event(created(2));
        log.push(ObjId(0), "k", "b", vec![]);
        log.event(created(3));
        let l2 = log.clone();
        std::thread::scope(|s| {
            s.spawn(move || l2.event(created(4)));
        });
        log.record_events(false);
        log.event(created(5));
        let all = log.events();
        // Sequence numbers only rise, and `at` places each event among the calls.
        assert!(all.windows(2).all(|w| w[0].seq < w[1].seq));
        assert_eq!(all.first().map(|r| r.seq), Some(0));
        let ev: Vec<&EventRecord> = all
            .iter()
            .filter(|r| matches!(r.event, Event::Created { .. }))
            .collect();
        assert_eq!(ev.iter().map(|r| r.at).collect::<Vec<_>>(), vec![1, 2, 2]);
        assert_eq!(
            ev.iter().map(|r| r.event.clone()).collect::<Vec<_>>(),
            vec![created(2), created(3), created(4)]
        );
        assert_eq!(ev[0].thread, ev[1].thread);
        assert_ne!(ev[0].thread, ev[2].thread);
        assert_eq!(log.take_events().len(), all.len());
        assert!(log.events().is_empty());
    }

    /// A clone shares the log across threads
    #[test]
    fn clones_share() {
        let log = CallLog::new();
        let l2 = log.clone();
        std::thread::scope(|s| {
            s.spawn(move || l2.push(ObjId(0), "k", "t", vec![]));
        });
        assert_eq!(log.sels(), vec!["t"]);
    }
}
