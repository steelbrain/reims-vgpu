//! Pipeline stages whose compile runs outside the VM's process, and what waits for one
//! ([`StageFills`]).
//!
//! A platform whose compiler can hold more memory or time than the VM's process should give one
//! stage refuses, in that process, a stage whose compile is not in its store, naming the stage by a
//! key. The stage's compile then runs elsewhere (on the rt platform, a helper process of its own:
//! `rt_platform::stage`), and fills the store; the refused make, or dispatch, is then made again,
//! and finds the compile there. Each such run is a fill ([`Fill`]).
//!
//! - **One fill a key at a time.** Every refusal of one stage while its fill runs (a command
//!   buffer's make ahead, walks of any queue or context, a pipeline's make) joins that fill
//!   ([`StageFills::fill`]): a stage is never filled twice at once.
//! - **One end a key, for the run.** A fill's end ([`FillEnd`]) is kept for its key, and a later
//!   refusal of that key starts no fill: it is answered that end at once. So no stage is filled
//!   twice in a process's run, whatever its fill ended with, and nothing loops between a refusal
//!   and a fill. A cancelled fill keeps nothing: the next refusal starts its stage's fill again.
//! - **Nothing waits but a hold.** [`StageFills::fill`] returns at once. A walk's dispatch waits
//!   for its fill ([`Fill::wait`]); a make is made again at its fill's end ([`Fill::on_end`]); a
//!   make ahead waits for nothing.
//! - **A held wait goes through its thread's holder.** The thread that runs a walk may install a
//!   holder ([`Holding`]), and a dispatch's wait for its fill ([`held`]) runs inside it: a lane's
//!   thread steps out of the lane's walks while it waits, so the lane's other buffers walk on.
//!   Where none is installed, the wait is the wait alone.
//! - **Cancels.** [`StageFills::cancel_all`] ends every fill running as cancelled, and cancels what
//!   runs it, at once: a device's reset waits for no fill. [`StageFills::close`] does too, and ends
//!   every later fill as cancelled as it is asked for: a process's exit waits for none, and starts
//!   none. Dropping the fills closes them.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::rc::Rc;
use std::sync::Arc;

use crate::core::sync::{Condvar, Mutex};

/// How a stage's fill ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillEnd {
    /// The store has the stage's compile: a make of the stage finds it there.
    Filled,
    /// The compile passed its memory or time cap, or failed, and the platform recorded the breach:
    /// a make of the stage refuses.
    Breached,
    /// The fill ended with nothing recorded (what runs it did not start, ended early, or could not
    /// open its device): a make of the stage refuses as before the fill.
    Failed,
    /// The fill was cancelled (a device's reset, the process's exit). Nothing is kept for its key.
    Cancelled,
}

/// What a fill's end runs, on the thread that ends it.
type OnEnd = Box<dyn FnOnce(FillEnd) + Send>;

/// What cancels whatever runs a fill: it returns at once.
pub type Cancel = Box<dyn FnOnce() + Send>;

/// What starts a fill of a key: it starts whatever runs the fill, which ends it through
/// [`FillDone::end`] once, and returns at once with its [`Cancel`]. Where nothing can start, it
/// ends the fill [`FillEnd::Failed`] itself.
pub type Start<K> = Box<dyn Fn(&K, FillDone) -> Cancel + Send + Sync>;

/// A fill's state.
#[derive(Default)]
struct FillState {
    /// Its end, once it has ended.
    end: Option<FillEnd>,
    /// What its end runs, in the order given.
    on_end: Vec<OnEnd>,
    /// What cancels whatever runs it, until it ends.
    cancel: Option<Cancel>,
}

/// One stage's fill: what a refusal of the stage waits for, or is made again at the end of.
pub struct Fill {
    state: Mutex<FillState>,
    ended: Condvar,
}

impl fmt::Debug for Fill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fill")
            .field("end", &self.end())
            .finish_non_exhaustive()
    }
}

impl Fill {
    /// A fill still running.
    fn running() -> Arc<Fill> {
        Arc::new(Fill {
            state: Mutex::new(FillState::default()),
            ended: Condvar::new(),
        })
    }

    /// A fill that has ended with `end`.
    pub fn ended_with(end: FillEnd) -> Arc<Fill> {
        let fill = Fill::running();
        fill.state.lock().end = Some(end);
        fill
    }

    /// Its end, once it has ended.
    pub fn end(&self) -> Option<FillEnd> {
        self.state.lock().end
    }

    /// Waits for its end.
    pub fn wait(&self) -> FillEnd {
        let mut state = self.state.lock();
        loop {
            if let Some(end) = state.end {
                return end;
            }
            self.ended.wait(&mut state);
        }
    }

    /// Runs `then` with its end: at once, on this thread, where it has ended; else on the thread
    /// that ends it, as it ends.
    pub fn on_end(&self, then: impl FnOnce(FillEnd) + Send + 'static) {
        let mut state = self.state.lock();
        match state.end {
            Some(end) => {
                drop(state);
                then(end);
            }
            None => state.on_end.push(Box::new(then)),
        }
    }

    /// Keeps `cancel` until the fill ends; a fill already cancelled is cancelled now, and one that
    /// ended otherwise drops it.
    fn hold_cancel(&self, cancel: Cancel) {
        let mut state = self.state.lock();
        match state.end {
            None => state.cancel = Some(cancel),
            Some(FillEnd::Cancelled) => {
                drop(state);
                cancel();
            }
            Some(_) => {}
        }
    }

    /// Ends the fill with `end` where it has not ended: wakes its waiters, then runs what its end
    /// runs, here. Returns its cancel, which the caller runs or drops; `None` where it had ended.
    fn finish(&self, end: FillEnd) -> Option<Option<Cancel>> {
        let (on_end, cancel) = {
            let mut state = self.state.lock();
            if state.end.is_some() {
                return None;
            }
            state.end = Some(end);
            (std::mem::take(&mut state.on_end), state.cancel.take())
        };
        self.ended.notify_all();
        for then in on_end {
            then(end);
        }
        Some(cancel)
    }
}

/// The fills' state.
struct FillsState<K> {
    /// The fills running, by key.
    running: HashMap<K, Arc<Fill>>,
    /// How each key's fill ended, for the run; no cancelled one.
    ended: HashMap<K, FillEnd>,
    /// Closed: every fill asked for ends cancelled at once.
    closed: bool,
}

/// What [`StageFills`] and the ends of its fills share.
struct Shared<K> {
    state: Mutex<FillsState<K>>,
    start: Start<K>,
}

/// What ends one fill ([`Start`]), once. One dropped before it has ended the fill ends it
/// [`FillEnd::Failed`]: nothing that loses it leaves a fill that no end would come to.
pub struct FillDone {
    end: Option<Box<dyn FnOnce(FillEnd) + Send>>,
}

impl fmt::Debug for FillDone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FillDone")
            .field("ended", &self.end.is_none())
            .finish_non_exhaustive()
    }
}

impl FillDone {
    /// Ends the fill with `end`, and keeps `end` for its key where the fill still runs (not
    /// cancelled meanwhile) and `end` is not [`FillEnd::Cancelled`].
    pub fn end(mut self, end: FillEnd) {
        if let Some(then) = self.end.take() {
            then(end);
        }
    }
}

impl Drop for FillDone {
    fn drop(&mut self) {
        if let Some(then) = self.end.take() {
            then(FillEnd::Failed);
        }
    }
}

impl<K: Eq + Hash> Shared<K> {
    /// Ends `fill`, the fill of `key`, with `end`: [`FillDone::end`].
    fn end(&self, key: &K, fill: &Fill, end: FillEnd) {
        {
            let mut state = self.state.lock();
            let current = state
                .running
                .get(key)
                .is_some_and(|running| std::ptr::eq(Arc::as_ptr(running), fill));
            let removed = if current {
                state.running.remove_entry(key)
            } else {
                None
            };
            if let Some((key, _)) = removed
                && end != FillEnd::Cancelled
            {
                state.ended.insert(key, end);
            }
        }
        // Its cancel is dropped: what ran the fill has ended.
        let _ = fill.finish(end);
    }
}

/// The stages a device's refusals hand over to be filled, by key: one fill a key at a time, and
/// one end a key for the run (the module doc).
pub struct StageFills<K> {
    shared: Arc<Shared<K>>,
}

impl<K> fmt::Debug for StageFills<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageFills").finish_non_exhaustive()
    }
}

impl<K: Eq + Hash + Clone + Send + 'static> StageFills<K> {
    /// Fills that `start` runs.
    pub fn new(start: Start<K>) -> StageFills<K> {
        StageFills {
            shared: Arc::new(Shared {
                state: Mutex::new(FillsState {
                    running: HashMap::new(),
                    ended: HashMap::new(),
                    closed: false,
                }),
                start,
            }),
        }
    }

    /// The fill of `key`: the one running, else one that has ended with the end its key's fill had
    /// this run, else a fill started now. Returns at once.
    pub fn fill(&self, key: K) -> Arc<Fill> {
        let fill = {
            let mut state = self.shared.state.lock();
            if state.closed {
                return Fill::ended_with(FillEnd::Cancelled);
            }
            if let Some(&end) = state.ended.get(&key) {
                return Fill::ended_with(end);
            }
            if let Some(fill) = state.running.get(&key) {
                return Arc::clone(fill);
            }
            let fill = Fill::running();
            state.running.insert(key.clone(), Arc::clone(&fill));
            fill
        };
        // Started with no lock held: a start that cannot start ends the fill here.
        let shared = Arc::downgrade(&self.shared);
        let (theirs, of) = (Arc::clone(&fill), key.clone());
        let done = FillDone {
            end: Some(Box::new(move |end| match shared.upgrade() {
                Some(shared) => shared.end(&of, &theirs, end),
                // The fills are gone, and closed theirs as they went.
                None => {
                    let _ = theirs.finish(end);
                }
            })),
        };
        let cancel = (self.shared.start)(&key, done);
        fill.hold_cancel(cancel);
        fill
    }

    /// Ends every fill running as cancelled, and cancels what runs each; returns at once.
    pub fn cancel_all(&self) {
        self.shared.cancel(false);
    }

    /// [`Self::cancel_all`], and every fill asked for from now on ends cancelled at once.
    pub fn close(&self) {
        self.shared.cancel(true);
    }
}

impl<K> Shared<K> {
    /// Ends every fill running as cancelled, and cancels what runs each; `closing`, closes the
    /// fills too.
    fn cancel(&self, closing: bool) {
        let running = {
            let mut state = self.state.lock();
            state.closed |= closing;
            std::mem::take(&mut state.running)
        };
        for fill in running.into_values() {
            if let Some(Some(cancel)) = fill.finish(FillEnd::Cancelled) {
                cancel();
            }
        }
    }
}

impl<K> Drop for StageFills<K> {
    /// Closes the fills: what runs each is cancelled, and their waiters go on.
    fn drop(&mut self) {
        self.shared.cancel(true);
    }
}

/// A platform call that a stage's fill holds, a dispatch: `attempt`; where it refuses for a stage
/// whose compile is not in the store (`stage_of` names the stage), the wait for that stage's fill
/// ([`StageFills::fill`]), then, where the fill filled the store, `attempt` once more, whose
/// answer stands. Otherwise the first refusal stands: after a breach, a failed or a cancelled
/// fill, and at once for a stage whose fill ended before.
pub fn held<K, T, E>(
    fills: &StageFills<K>,
    mut attempt: impl FnMut() -> Result<T, E>,
    stage_of: impl FnOnce(&E) -> Option<K>,
) -> Result<T, E>
where
    K: Eq + Hash + Clone + Send + 'static,
{
    let refused = match attempt() {
        Ok(answer) => return Ok(answer),
        Err(refused) => refused,
    };
    let Some(stage) = stage_of(&refused) else {
        return Err(refused);
    };
    let fill = fills.fill(stage);
    if fill.end().is_none() {
        hold(&mut || {
            fill.wait();
        });
    }
    // Ended already, unless the holder did not run the wait.
    match fill.wait() {
        FillEnd::Filled => attempt(),
        FillEnd::Breached | FillEnd::Failed | FillEnd::Cancelled => Err(refused),
    }
}

/// What runs a thread's waits for a fill ([`hold`]): it calls the wait it is given, once, and may
/// do what it needs around it.
pub type Holder = Rc<dyn Fn(&mut dyn FnMut())>;

thread_local! {
    /// The thread's holder, while a [`Holding`] has it installed.
    static HOLDER: RefCell<Option<Holder>> = const { RefCell::new(None) };
}

/// Runs `wait`, a wait for a fill, inside the thread's holder where one is installed
/// ([`Holding`]), else as it is.
pub fn hold(wait: &mut dyn FnMut()) {
    // Cloned out of the cell first: the holder may run what holds again on this thread.
    match HOLDER.with(|holder| holder.borrow().clone()) {
        Some(holder) => holder(wait),
        None => wait(),
    }
}

/// A holder installed on the thread that made it, until it drops, which installs the one it
/// replaced again.
#[must_use = "the holder is installed only until its holding drops"]
pub struct Holding {
    replaced: Option<Holder>,
}

impl Holding {
    /// Installs `holder` on this thread, for this thread's waits for a fill ([`hold`]).
    pub fn install(holder: Holder) -> Holding {
        let replaced = HOLDER.with(|installed| installed.replace(Some(holder)));
        Holding { replaced }
    }
}

impl fmt::Debug for Holding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Holding").finish_non_exhaustive()
    }
}

impl Drop for Holding {
    fn drop(&mut self) {
        let replaced = self.replaced.take();
        HOLDER.with(|installed| *installed.borrow_mut() = replaced);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    /// A fake start: keeps each fill's end for the test, and counts starts and cancels.
    #[derive(Clone, Default)]
    struct Fake {
        dones: Arc<Mutex<Vec<FillDone>>>,
        starts: Arc<AtomicUsize>,
        cancels: Arc<AtomicUsize>,
    }

    impl Fake {
        fn fills(&self) -> StageFills<u32> {
            let fake = self.clone();
            StageFills::new(Box::new(move |_key, done| {
                fake.starts.fetch_add(1, Ordering::SeqCst);
                fake.dones.lock().push(done);
                let cancels = Arc::clone(&fake.cancels);
                Box::new(move || {
                    cancels.fetch_add(1, Ordering::SeqCst);
                })
            }))
        }

        /// Ends the oldest fill not yet ended with `end`.
        fn end(&self, end: FillEnd) {
            let done = self.dones.lock().remove(0);
            done.end(end);
        }

        fn starts(&self) -> usize {
            self.starts.load(Ordering::SeqCst)
        }

        fn cancels(&self) -> usize {
            self.cancels.load(Ordering::SeqCst)
        }
    }

    /// Refusals of one stage while its fill runs join that fill: one start.
    #[test]
    fn refusals_of_one_stage_join_its_fill() {
        let fake = Fake::default();
        let fills = fake.fills();
        let first = fills.fill(1);
        let second = fills.fill(1);
        assert!(Arc::ptr_eq(&first, &second));
        let other = fills.fill(2);
        assert!(!Arc::ptr_eq(&first, &other));
        assert_eq!(fake.starts(), 2);
        assert_eq!(first.end(), None);
    }

    /// A fill's end wakes its waiters and runs what was given to run at its end, once.
    #[test]
    fn a_fill_ends_its_waits_and_runs_its_callbacks() {
        let fake = Fake::default();
        let fills = fake.fills();
        let fill = fills.fill(1);
        let waiter = {
            let fill = Arc::clone(&fill);
            std::thread::Builder::new()
                .spawn(move || fill.wait())
                .unwrap()
        };
        let seen = Arc::new(Mutex::new(Vec::new()));
        let theirs = Arc::clone(&seen);
        fill.on_end(move |end| theirs.lock().push(end));
        fake.end(FillEnd::Filled);
        assert_eq!(waiter.join().unwrap(), FillEnd::Filled);
        assert_eq!(*seen.lock(), [FillEnd::Filled]);
        // Given after the end, run at once.
        let theirs = Arc::clone(&seen);
        fill.on_end(move |end| theirs.lock().push(end));
        assert_eq!(*seen.lock(), [FillEnd::Filled, FillEnd::Filled]);
        assert_eq!(fake.cancels(), 0);
    }

    /// A key whose fill ended is answered that end at once, and starts no other fill this run,
    /// whatever the end.
    #[test]
    fn a_key_whose_fill_ended_starts_no_other() {
        let fake = Fake::default();
        let fills = fake.fills();
        for (key, end) in [
            (1, FillEnd::Filled),
            (2, FillEnd::Breached),
            (3, FillEnd::Failed),
        ] {
            let _ = fills.fill(key);
            fake.end(end);
            let again = fills.fill(key);
            assert_eq!(again.end(), Some(end), "{key}");
        }
        assert_eq!(fake.starts(), 3);
    }

    /// A cancel ends every running fill as cancelled at once, cancels what runs each, and keeps
    /// nothing for their keys: the next refusal starts the fill again, and a late end of the
    /// cancelled one changes nothing.
    #[test]
    fn a_cancelled_fill_ends_at_once_and_keeps_nothing() {
        let fake = Fake::default();
        let fills = fake.fills();
        let fill = fills.fill(1);
        let waiter = {
            let fill = Arc::clone(&fill);
            std::thread::Builder::new()
                .spawn(move || fill.wait())
                .unwrap()
        };
        fills.cancel_all();
        assert_eq!(waiter.join().unwrap(), FillEnd::Cancelled);
        assert_eq!(fake.cancels(), 1);
        // What ran it ends late, breached: nothing is kept.
        fake.end(FillEnd::Breached);
        assert_eq!(fill.end(), Some(FillEnd::Cancelled));
        let again = fills.fill(1);
        assert_eq!(again.end(), None);
        assert_eq!(fake.starts(), 2);
    }

    /// Once closed, every fill asked for ends cancelled at once, and none starts.
    #[test]
    fn closed_fills_start_none() {
        let fake = Fake::default();
        let fills = fake.fills();
        let running = fills.fill(1);
        fills.close();
        assert_eq!(running.end(), Some(FillEnd::Cancelled));
        assert_eq!(fills.fill(2).end(), Some(FillEnd::Cancelled));
        assert_eq!(fills.fill(1).end(), Some(FillEnd::Cancelled));
        assert_eq!((fake.starts(), fake.cancels()), (1, 1));
    }

    /// Dropping the fills cancels what runs each, and their waiters go on.
    #[test]
    fn dropped_fills_cancel_their_fills() {
        let fake = Fake::default();
        let fills = fake.fills();
        let fill = fills.fill(1);
        drop(fills);
        assert_eq!(fill.wait(), FillEnd::Cancelled);
        assert_eq!(fake.cancels(), 1);
        // The late end of what ran it finds no fills.
        fake.end(FillEnd::Filled);
        assert_eq!(fill.end(), Some(FillEnd::Cancelled));
    }

    /// A start that ends its fill before it returns (nothing could start) leaves the fill ended,
    /// its end kept, and its cancel dropped unrun.
    #[test]
    fn a_fill_that_cannot_start_ends_at_once() {
        let cancels = Arc::new(AtomicUsize::new(0));
        let theirs = Arc::clone(&cancels);
        let fills: StageFills<u32> = StageFills::new(Box::new(move |_key, done| {
            done.end(FillEnd::Failed);
            let cancels = Arc::clone(&theirs);
            Box::new(move || {
                cancels.fetch_add(1, Ordering::SeqCst);
            })
        }));
        assert_eq!(fills.fill(1).end(), Some(FillEnd::Failed));
        assert_eq!(fills.fill(1).end(), Some(FillEnd::Failed));
        fills.cancel_all();
        assert_eq!(cancels.load(Ordering::SeqCst), 0);
    }

    /// A fill whose end is dropped unended ends failed, and that end is kept.
    #[test]
    fn a_lost_end_fails_its_fill() {
        let fake = Fake::default();
        let fills = fake.fills();
        let fill = fills.fill(1);
        drop(fake.dones.lock().remove(0));
        assert_eq!(fill.end(), Some(FillEnd::Failed));
        assert_eq!(fills.fill(1).end(), Some(FillEnd::Failed));
        assert_eq!((fake.starts(), fake.cancels()), (1, 0));
    }

    /// A fill cancelled while its start was still starting it is cancelled as the start returns.
    #[test]
    fn a_fill_cancelled_during_its_start_cancels_what_it_started() {
        let fill = Fill::running();
        let _ = fill.finish(FillEnd::Cancelled);
        let cancels = Arc::new(AtomicUsize::new(0));
        let theirs = Arc::clone(&cancels);
        fill.hold_cancel(Box::new(move || {
            theirs.fetch_add(1, Ordering::SeqCst);
        }));
        assert_eq!(cancels.load(Ordering::SeqCst), 1);
        // One that ended otherwise drops it.
        let filled = Fill::running();
        let _ = filled.finish(FillEnd::Filled);
        let theirs = Arc::clone(&cancels);
        filled.hold_cancel(Box::new(move || {
            theirs.fetch_add(1, Ordering::SeqCst);
        }));
        assert_eq!(cancels.load(Ordering::SeqCst), 1);
    }

    /// A call refused for a stage waits for the stage's fill, and is made once more after the
    /// fill filled the store, whose answer stands: a second refusal is not waited for again.
    #[test]
    fn a_held_call_is_made_again_once_after_its_fill() {
        let fake = Fake::default();
        let fills = Arc::new(fake.fills());
        let answers = Arc::new(Mutex::new(vec![Err::<(), _>(Some(7_u32)), Err(Some(7))]));
        let (theirs, asked) = (Arc::clone(&fills), Arc::clone(&answers));
        let call = std::thread::Builder::new()
            .spawn(move || held(&theirs, || asked.lock().remove(0), |&stage| stage))
            .unwrap();
        while fake.starts() == 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!call.is_finished());
        fake.end(FillEnd::Filled);
        assert_eq!(call.join().unwrap(), Err(Some(7)));
        assert!(answers.lock().is_empty());
        assert_eq!(fake.starts(), 1);
    }

    /// A call refused for a stage whose fill ended otherwise than filled answers its refusal,
    /// made once; a refusal that names no stage starts no fill.
    #[test]
    fn a_held_call_answers_its_refusal_after_a_breach() {
        let fake = Fake::default();
        let fills = fake.fills();
        let _ = fills.fill(3);
        fake.end(FillEnd::Breached);
        let mut made = 0;
        let answer: Result<(), Option<u32>> = held(
            &fills,
            || {
                made += 1;
                Err(Some(3))
            },
            |&stage| stage,
        );
        assert_eq!((answer, made), (Err(Some(3)), 1));
        let answer: Result<(), Option<u32>> = held(&fills, || Err(None), |&stage| stage);
        assert_eq!(answer, Err(None));
        assert_eq!(
            held(&fills, || Ok::<_, Option<u32>>(5), |&stage| stage),
            Ok(5)
        );
        assert_eq!(fake.starts(), 1);
    }

    /// A held call's wait for a running fill runs inside its thread's holder while one is
    /// installed, the one it replaced again once its holding drops; a fill ended already is waited
    /// for by none.
    #[test]
    fn a_held_wait_runs_inside_its_thread_s_holder() {
        let fake = Fake::default();
        let fills = fake.fills();
        let seen = Rc::new(RefCell::new(Vec::new()));
        let holder = |name: &'static str| -> Holder {
            let (seen, fake) = (Rc::clone(&seen), fake.clone());
            Rc::new(move |wait: &mut dyn FnMut()| {
                seen.borrow_mut().push(name);
                // The fill ends inside the hold, before its wait.
                fake.end(FillEnd::Filled);
                wait();
            })
        };
        let refusing = |stage: u32| {
            let mut attempts = 0;
            move || {
                attempts += 1;
                if attempts == 1 {
                    Err(stage)
                } else {
                    Ok(attempts)
                }
            }
        };
        let outer = Holding::install(holder("outer"));
        let inner = Holding::install(holder("inner"));
        assert_eq!(held(&fills, refusing(1), |&stage| Some(stage)), Ok(2));
        drop(inner);
        assert_eq!(held(&fills, refusing(2), |&stage| Some(stage)), Ok(2));
        assert_eq!(held(&fills, refusing(1), |&stage| Some(stage)), Ok(2));
        drop(outer);
        assert_eq!(*seen.borrow(), ["inner", "outer"]);
        assert_eq!(fake.starts(), 2);
    }

    /// A wait on a fill that does not end goes on waiting: nothing ends a fill but its end or a
    /// cancel.
    #[test]
    fn a_fill_waits_until_it_ends() {
        let fake = Fake::default();
        let fills = fake.fills();
        let fill = fills.fill(1);
        let (sent, got) = std::sync::mpsc::channel();
        let theirs = Arc::clone(&fill);
        let waiter = std::thread::Builder::new()
            .spawn(move || sent.send(theirs.wait()))
            .unwrap();
        assert!(got.recv_timeout(Duration::from_millis(100)).is_err());
        fake.end(FillEnd::Breached);
        assert_eq!(got.recv().unwrap(), FillEnd::Breached);
        waiter.join().unwrap().unwrap();
    }
}
