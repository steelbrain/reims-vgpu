//! The queue a display's calls into the host run on: a real seam. The public API gives every
//! display its display thread (`reims-vgpu.display`, one serial queue for the process), a core
//! `SerialQueue`, and the testkit gives its fakes. Core models every display's newFrame source on
//! it (`rt::CoalescingSignal`).

use std::sync::Arc;

use crate::core::rt::queue::Executor;

/// The queue a display's calls into the host run on (the display descriptor's `queue`): each
/// call of a display handler is submitted to it, and the newFrame source targets it. Jobs run one
/// at a time, in submission order.
pub trait HostQueue: Executor {
    /// A reference of the caller's own to the same queue: jobs submitted through either run in one
    /// order.
    fn dispatch_retain(&self) -> Arc<dyn HostQueue>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A HostQueue is an Executor (upcast for dispatch-source targets)
    #[test]
    fn host_queue_upcasts() {
        struct Inline;
        impl Executor for Inline {
            fn exec_async(&self, job: crate::core::rt::Job) {
                job()
            }
        }
        impl HostQueue for Inline {
            fn dispatch_retain(&self) -> Arc<dyn HostQueue> {
                Arc::new(Inline)
            }
        }
        let q: Arc<dyn HostQueue> = Arc::new(Inline);
        drop(q.dispatch_retain());
        let e: Arc<dyn Executor> = q;
        let n = Arc::new(AtomicU32::new(0));
        let n2 = n.clone();
        e.exec_async(Box::new(move || {
            n2.store(1, Ordering::SeqCst);
        }));
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }
}
