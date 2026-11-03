//! Ring-0 work carried out off ring 0's threads, answered later, while the later messages that
//! name its object wait for it: a [`Worker`] thread that runs the jobs in the order they are
//! handed over, and the [`Unsettled`] keys (a block's offset, an object's id) whose job is not
//! yet done.
//!
//! The ring-0 thread that carries the message out (its block's or its context's lane) begins the
//! key's [`Settle`] in that lane's ring order and hands the job over; the job carries the work out, posts the
//! message's answer, then drops the settle, so a message that waited for the key is answered after
//! it. A message that names the key, on ring 0 or a queue's ring, waits for it first
//! ([`Unsettled::wait`]), holding no lock the job takes; a thread that waits for nothing (a
//! doorbell's) asks instead ([`Unsettled::try_any`]) and leaves the message to a thread that may. A
//! job takes `GraphicsControl`'s reset gate shared, recursively, as a queue that holds it may be
//! waiting for the key while the reset waits for the gate, and carries nothing out where the device
//! was reset since its begin (the caller's count of resets, kept in the settle).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, mpsc};

use reims_vgpu_core::platform::os::MonoTime;
use reims_vgpu_core::sync::{Condvar, Mutex};

/// A job of a [`Worker`]'s.
pub(crate) type Job = Box<dyn FnOnce() + Send>;

/// A thread of the library's, named `name`, that runs each job in the order it is handed over,
/// inside a library entry. Started at the first job; where it cannot be, each job runs where it is
/// handed over. Its drop ends the thread once the jobs handed over are done, without waiting for
/// it.
pub(crate) struct Worker {
    name: &'static str,
    jobs: OnceLock<Option<mpsc::Sender<Job>>>,
    /// Holds the thread before each job while shut (tests only).
    #[cfg(test)]
    gate: Arc<Gate>,
}

impl Worker {
    /// A worker whose thread is named `name`.
    pub(crate) fn new(name: &'static str) -> Self {
        Worker {
            name,
            jobs: OnceLock::new(),
            #[cfg(test)]
            gate: Arc::default(),
        }
    }

    /// Hands `job` to the thread.
    pub(crate) fn submit(&self, job: Job) {
        let jobs = self.jobs.get_or_init(|| self.start());
        let job = match jobs {
            Some(jobs) => match jobs.send(job) {
                Ok(()) => return,
                Err(mpsc::SendError(job)) => job,
            },
            None => job,
        };
        crate::entry::run(|_| job());
    }

    /// Lets `value` go on the thread, in a refusal scope, as object `id`'s release
    /// ([`super::drop_refused`]).
    pub(crate) fn drop_later<T: Send + 'static>(&self, id: u64, value: T) {
        self.submit(Box::new(move || super::drop_refused(id, value)));
    }

    /// The thread, and where it takes its jobs.
    fn start(&self) -> Option<mpsc::Sender<Job>> {
        let (send, receive) = mpsc::channel::<Job>();
        #[cfg(test)]
        let gate = Arc::clone(&self.gate);
        let spawned = std::thread::Builder::new()
            .name(self.name.into())
            .spawn(move || {
                while let Ok(job) = receive.recv() {
                    #[cfg(test)]
                    gate.pass();
                    crate::entry::run(|_| job());
                }
            });
        match spawned {
            Ok(_) => Some(send),
            Err(e) => {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "{} could not be started ({e}): its jobs run where they are handed over",
                    self.name
                );
                None
            }
        }
    }

    /// Shuts the gate: the thread holds before its next job (tests only).
    #[cfg(test)]
    pub(crate) fn hold(&self) {
        *self.gate.shut.lock() = true;
    }

    /// Opens the gate (tests only).
    #[cfg(test)]
    pub(crate) fn release(&self) {
        *self.gate.shut.lock() = false;
        self.gate.opened.notify_all();
    }

    /// Waits for every job handed over before it to be done (tests only).
    #[cfg(test)]
    pub(crate) fn flush(&self) {
        let (done, wait) = mpsc::channel();
        self.submit(Box::new(move || {
            let _ = done.send(());
        }));
        let _ = wait.recv();
    }
}

/// A worker's gate (tests only): shut, the thread holds before its next job.
#[cfg(test)]
#[derive(Default)]
struct Gate {
    shut: Mutex<bool>,
    opened: Condvar,
}

#[cfg(test)]
impl Gate {
    fn pass(&self) {
        let mut shut = self.shut.lock();
        while *shut {
            self.opened.wait(&mut shut);
        }
    }
}

/// The keys whose job was handed to a worker and is not yet done, each with how many of them.
#[derive(Default)]
pub(crate) struct Unsettled {
    keys: Mutex<HashMap<u64, u32>>,
    /// Signalled as each settles.
    settled: Condvar,
    /// How many there are, read without the lock where there are none.
    count: AtomicUsize,
}

impl Unsettled {
    /// `key` is unsettled until the returned [`Settle`] drops; `resets` is the caller's count of
    /// the device's resets, and `begun` the time.
    pub(crate) fn begin(self: &Arc<Self>, key: u64, resets: u64, begun: MonoTime) -> Settle {
        *self.keys.lock().entry(key).or_insert(0) += 1;
        self.count.fetch_add(1, Ordering::AcqRel);
        Settle {
            unsettled: Arc::clone(self),
            key,
            resets,
            begun,
        }
    }

    /// Waits for every job of `key` begun before now to be done. Called with no lock held that a
    /// job takes.
    pub(crate) fn wait(&self, key: u64) {
        if self.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut keys = self.keys.lock();
        while keys.contains_key(&key) {
            self.settled.wait(&mut keys);
        }
    }

    /// Whether any of `keys` is unsettled.
    pub(crate) fn any(&self, mut keys: impl Iterator<Item = u64>) -> bool {
        if self.count.load(Ordering::Acquire) == 0 {
            return false;
        }
        let unsettled = self.keys.lock();
        keys.any(|k| unsettled.contains_key(&k))
    }

    /// [`Self::any`], for a thread that waits for nothing and allocates nothing (a doorbell's, the
    /// guest's vCPU): `None` where the keys are held elsewhere, their lock only tried.
    pub(crate) fn try_any(&self, mut keys: impl Iterator<Item = u64>) -> Option<bool> {
        if self.count.load(Ordering::Acquire) == 0 {
            return Some(false);
        }
        let unsettled = self.keys.try_lock()?;
        Some(keys.any(|k| unsettled.contains_key(&k)))
    }
}

/// A job begun and not yet done ([`Unsettled::begin`]); its drop settles its key.
pub(crate) struct Settle {
    unsettled: Arc<Unsettled>,
    key: u64,
    /// The caller's count of the device's resets at the begin.
    pub(crate) resets: u64,
    /// When it was begun.
    pub(crate) begun: MonoTime,
}

impl Drop for Settle {
    fn drop(&mut self) {
        let mut keys = self.unsettled.keys.lock();
        if let Some(n) = keys.get_mut(&self.key) {
            *n -= 1;
            if *n == 0 {
                keys.remove(&self.key);
            }
        }
        self.unsettled.count.fetch_sub(1, Ordering::AcqRel);
        drop(keys);
        self.unsettled.settled.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// A worker runs its jobs in the order they are handed over, on its own thread, and holds
    /// them while its gate is shut
    #[test]
    fn a_worker_runs_its_jobs_in_order_on_its_thread() {
        let worker = Worker::new("reims-vgpu.test-worker");
        let seen = Arc::new(Mutex::new(Vec::new()));
        worker.hold();
        for i in 0..3 {
            let seen = Arc::clone(&seen);
            worker.submit(Box::new(move || {
                let name = std::thread::current().name().map(str::to_owned);
                seen.lock().push((i, name));
            }));
        }
        std::thread::sleep(Duration::from_millis(20));
        assert!(seen.lock().is_empty(), "held");
        worker.release();
        worker.flush();
        let name = Some("reims-vgpu.test-worker".to_owned());
        assert_eq!(
            *seen.lock(),
            [(0, name.clone()), (1, name.clone()), (2, name)]
        );
    }

    /// A key is unsettled from its begin to its settle's drop, each begin counted: a wait for it
    /// returns once the last settles, and one for another key at once
    #[test]
    fn a_wait_for_a_key_returns_once_its_last_settle_drops() {
        let unsettled = Arc::new(Unsettled::default());
        let t = MonoTime::from_nanos(0);
        assert!(!unsettled.any([1, 2].into_iter()));
        let first = unsettled.begin(1, 0, t);
        let second = unsettled.begin(1, 0, t);
        assert!(unsettled.any([2, 1].into_iter()));
        assert!(!unsettled.any([2].into_iter()));
        unsettled.wait(2);
        let waiter = {
            let unsettled = Arc::clone(&unsettled);
            std::thread::Builder::new()
                .name("reims-vgpu.test-waiter".into())
                .spawn(move || unsettled.wait(1))
                .unwrap()
        };
        drop(first);
        std::thread::sleep(Duration::from_millis(20));
        assert!(!waiter.is_finished(), "one begin of the key is left");
        drop(second);
        waiter.join().unwrap();
        assert!(!unsettled.any([1].into_iter()));
    }
}
