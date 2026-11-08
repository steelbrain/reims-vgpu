//! Platform trait: `SharedEvent`; the event kinds a command buffer waits on or signals.

use super::{Handle, Platform};
use crate::core::raise::RaiseResult;

/// `id<MTLSharedEvent>`.
pub trait SharedEvent<P: Platform>: Handle {
    /// sel: setSignaledValue:
    fn set_signaled_value(&self, signaled_value: u64) -> RaiseResult<()>;
    /// sel: signaledValue — the value as the event has it now: what the CPU set and the GPU
    /// signalled it to, the highest of them, as Metal's only rises.
    fn signaled_value(&self) -> u64;
    /// sel: waitUntilSignaledValue:timeoutMS:
    /// (5000 ms). `false` on timeout (measured: `NO` for an unsignalled value after 10 ms). Blocks
    /// the calling thread; the fake runs it on the virtual clock. No path reaches it under a
    /// handler: a raise inside it dies at the send.
    fn wait_until_signaled_value(&self, value: u64, milliseconds: u64) -> bool;
    /// sel: notifyListener:atValue:block:
    /// `block` runs once on `listener`'s thread (Metal's: the listener's dispatch queue), once the
    /// value reaches `value` from whatever raised it (a command buffer's signal as the GPU
    /// reaches it, a CPU set), and at once where it has already. A block the value never reaches
    /// is kept until the event goes. The fake runs it on the thread that raised the value.
    fn notify_listener(
        &self,
        listener: &P::SharedEventListener,
        value: u64,
        block: Notification,
    ) -> RaiseResult<()>;
}

/// A shared event's notification (`MTLSharedEventNotificationBlock`'s work): run once, on its
/// listener's thread, once the event reached the value it was asked for
/// ([`SharedEvent::notify_listener`]).
pub type Notification = Box<dyn FnOnce() + Send>;

/// The event a command buffer waits on or signals: core passes both kinds to
/// `encodeWaitForEvent:value:` and `encodeSignalEvent:value:` (an `MTLEvent` or an
/// `MTLSharedEvent`).
pub enum EventRef<'a, P: Platform> {
    /// An `id<MTLEvent>` (`-[MTLDevice newEvent]`).
    Event(&'a P::Event),
    /// An `id<MTLSharedEvent>`.
    Shared(&'a P::SharedEvent),
}

impl<P: Platform> Clone for EventRef<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: Platform> Copy for EventRef<'_, P> {}

impl<P: Platform> EventRef<'_, P> {
    /// The event's identity (its pointer), for the fake's call log.
    pub fn id(&self) -> super::ObjId {
        match self {
            EventRef::Event(e) => e.id(),
            EventRef::Shared(e) => e.id(),
        }
    }
}
