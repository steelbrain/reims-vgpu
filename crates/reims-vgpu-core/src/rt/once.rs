//! `ProcessOnce`: a value made once, by its first caller, held by its owner rather than in a Rust
//! static, so each owner has its own and each test starts fresh. Its one user is an isolated
//! task's `GpuTask`, whose XR10 blit pipelines are made once per task. Concurrent callers block
//! until the first finishes.

use std::sync::OnceLock;

/// A once token and the value its block made.
#[derive(Debug)]
pub struct ProcessOnce<T> {
    cell: OnceLock<T>,
}

/// A once token whose block makes no value.
pub type Once = ProcessOnce<()>;

impl<T> Default for ProcessOnce<T> {
    fn default() -> Self {
        ProcessOnce::new()
    }
}

impl<T> ProcessOnce<T> {
    /// An unused token.
    pub const fn new() -> Self {
        ProcessOnce {
            cell: OnceLock::new(),
        }
    }

    /// Runs `f` the first time only, on the calling thread, and answers its value. `f` starts with
    /// no handler, so a raise in `f` with no handler of its own dies at the raise even under the
    /// caller's handler (the service's blit-pipeline once sends to Metal inside it). Its code holds
    /// under [`Hold::NONE`](crate::rt::Hold::NONE), with no holder. `f` is `Send`, as a queue's
    /// block is: it takes nothing that is bound to the caller's thread, its hold included.
    pub fn get_or_init(&self, f: impl FnOnce() -> T + Send) -> &T {
        self.cell.get_or_init(f)
    }

    /// Whether the block has run.
    pub fn is_completed(&self) -> bool {
        self.cell.get().is_some()
    }
}

impl ProcessOnce<()> {
    /// Runs `f` the first time only ([`ProcessOnce::get_or_init`]).
    pub fn call_once(&self, f: impl FnOnce() + Send) {
        self.get_or_init(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Runs exactly once, even under contention
    #[test]
    fn runs_once() {
        let once = Arc::new(Once::new());
        let n = Arc::new(AtomicU32::new(0));
        assert!(!once.is_completed());
        std::thread::scope(|s| {
            for _ in 0..8 {
                let (o, n) = (once.clone(), n.clone());
                s.spawn(move || {
                    o.call_once(|| {
                        n.fetch_add(1, Ordering::SeqCst);
                    })
                });
            }
        });
        assert_eq!(n.load(Ordering::SeqCst), 1);
        assert!(once.is_completed());
        once.call_once(|| n.store(99, Ordering::SeqCst));
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    /// A fatal raise in the block dies at the raise: the block has no handler of its own
    #[test]
    fn fatal_raise_in_block_dies_at_the_raise() {
        use crate::raise::OnFault;
        use crate::raise::death::DeathKind;
        use crate::raise::death::DeathSite;
        use crate::testkit::capture::{expect_death, raise};
        expect_death(
            DeathSite {
                kind: DeathKind::UnhandledRaise,
                file: "gpu_task",
                line: 0,
            },
            || {
                let once = Once::new();
                once.call_once(|| {
                    let _ = raise(OnFault::Die, "gpu_task", "x");
                });
            },
        );
    }

    /// Two tokens are independent
    #[test]
    fn tokens_independent() {
        let (a, b) = (Once::default(), Once::new());
        a.call_once(|| {});
        assert!(a.is_completed());
        assert!(!b.is_completed());
    }
}
