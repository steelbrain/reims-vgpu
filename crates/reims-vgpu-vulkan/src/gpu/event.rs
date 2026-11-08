//! `id<MTLEvent>`, `id<MTLSharedEvent>`, its handle and its listener: core's `SharedEvent` over
//! the runtime's events.

use std::time::Duration;

use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::event::{Notification, SharedEvent};
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::rt::BlockEnv;

use crate::answer::plain;
use crate::platform::{Obj, VulkanPlatform};

/// `id<MTLEvent>`.
pub type VulkanEvent = Obj<rt::Event>;

/// `id<MTLSharedEvent>`.
pub type VulkanSharedEvent = Obj<rt::SharedEvent>;

/// `MTLSharedEventListener *`.
pub type VulkanSharedEventListener = Obj<rt::SharedEventListener>;

impl SharedEvent<VulkanPlatform> for VulkanSharedEvent {
    /// The value becomes at least `signaled_value`; the runtime refuses no value.
    fn set_signaled_value(&self, signaled_value: u64) -> RaiseResult<()> {
        // evset: event, value: the guest's or the device's host set
        reims_vgpu_core::trace::event(
            "evset",
            &[crate::gpu::command::event_id(self), signaled_value],
        );
        plain(|| rt::SharedEvent::set_signaled_value(self, signaled_value));
        Ok(())
    }

    /// The runtime's value: the event's timeline as the GPU and the CPU have raised it.
    fn signaled_value(&self) -> u64 {
        plain(|| rt::SharedEvent::signaled_value(self))
    }

    /// Blocks up to `milliseconds`.
    fn wait_until_signaled_value(&self, value: u64, milliseconds: u64) -> bool {
        plain(|| {
            rt::SharedEvent::wait_until_signaled_value(
                self,
                value,
                Duration::from_millis(milliseconds),
            )
        })
    }

    /// The runtime's notification: `block` runs once on the listener's thread, once the value
    /// reaches `value`. The runtime sees a command buffer's signal rise as its completion turn
    /// sees the batch that ends at it finish, so a signal encoded before more of the buffer's work
    /// is seen there, before the buffer completes (the runtime's `command_buffer/batches.rs` and
    /// `event.rs`, read from code).
    fn notify_listener(
        &self,
        listener: &VulkanSharedEventListener,
        value: u64,
        block: Notification,
    ) -> RaiseResult<()> {
        #[cfg(test)]
        let slot = reims_vgpu_core::testkit::capture::current_slot();
        let notification: rt::SharedEventNotification =
            Box::new(move |_event: &rt::SharedEvent, _value: u64| {
                // A callout on the runtime's thread: marked ours, so a panic here aborts.
                let _env = BlockEnv::foreign();
                #[cfg(test)]
                reims_vgpu_core::testkit::capture::with_slot(slot, block);
                #[cfg(not(test))]
                block();
            });
        plain(|| rt::SharedEvent::notify_listener(self, listener, value, notification));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::device::GpuDevice;

    use super::*;
    use crate::gpu::device::tests::dev;

    /// setSignaledValue: sets the value; waitUntilSignaledValue:timeoutMS: is YES for a reached
    /// value and NO after the timeout for an unreached one
    #[test]
    fn signal_and_wait() {
        let e = dev().new_shared_event().unwrap();
        SharedEvent::set_signaled_value(&e, 3).unwrap();
        assert_eq!(e.signaled_value(), 3);
        assert!(SharedEvent::wait_until_signaled_value(&e, 3, 0));
        assert!(SharedEvent::wait_until_signaled_value(&e, 2, 0));
        assert!(!SharedEvent::wait_until_signaled_value(&e, 4, 10));
    }

    /// notifyListener:atValue:block: runs the block once the value reaches it, off the asking
    /// thread, and once only; at once for a value reached already
    #[test]
    fn notify_listener_runs_once_the_value_is_reached() {
        use reims_vgpu_core::platform::Platform as _;
        let listener = crate::platform::tests::platform().new_shared_event_listener();
        let e = dev().new_shared_event().unwrap();
        let (ran, runs) = std::sync::mpsc::channel();
        for value in [2, 5] {
            let ran = ran.clone();
            SharedEvent::notify_listener(
                &e,
                &listener,
                value,
                Box::new(move || ran.send((value, std::thread::current().id())).unwrap()),
            )
            .unwrap();
        }
        let wait = Duration::from_millis(100);
        assert!(runs.recv_timeout(wait).is_err(), "ran before the value");
        SharedEvent::set_signaled_value(&e, 3).unwrap();
        let (value, thread) = runs.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(value, 2);
        assert_ne!(thread, std::thread::current().id());
        assert!(runs.recv_timeout(wait).is_err(), "5 ran at 3");
        SharedEvent::set_signaled_value(&e, 5).unwrap();
        assert_eq!(runs.recv_timeout(Duration::from_secs(5)).unwrap().0, 5);
        let ran = ran.clone();
        let block = Box::new(move || ran.send((4, std::thread::current().id())).unwrap());
        SharedEvent::notify_listener(&e, &listener, 4, block).unwrap();
        assert_eq!(runs.recv_timeout(Duration::from_secs(5)).unwrap().0, 4);
        SharedEvent::set_signaled_value(&e, 9).unwrap();
        assert!(runs.recv_timeout(wait).is_err(), "a block ran twice");
    }
}
