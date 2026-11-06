//! Events: core's `SharedEvent`, and the event, handle and listener handle types.

use std::ptr::NonNull;

use block2::RcBlock;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLEvent, MTLSharedEvent, MTLSharedEventHandle, MTLSharedEventListener};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::event::Notification;
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::sync::Mutex;

use crate::msg::{self, Policy};
use crate::platform::{MetalPlatform, Obj};
use crate::spi::methods::SharedEventSpi;

/// `id<MTLEvent>`.
pub type MetalEvent = Obj<ProtocolObject<dyn MTLEvent>>;

/// `id<MTLSharedEvent>`.
pub type MetalSharedEvent = Obj<ProtocolObject<dyn MTLSharedEvent>>;

impl reims_vgpu_core::platform::event::SharedEvent<MetalPlatform> for MetalSharedEvent {
    /// `newSharedEventHandle`: the `+1` handle, or nil.
    fn new_shared_event_handle(&self) -> Option<<MetalPlatform as Platform>::SharedEventHandle> {
        msg::get("newSharedEventHandle", || {
            self.newSharedEventHandle_or_nil()
        })
        .map(Obj::from_retained)
    }
    fn set_signaled_value(&self, signaled_value: u64) -> RaiseResult<()> {
        msg::send("setSignaledValue:", Policy::Caught, || {
            self.setSignaledValue(signaled_value)
        })
    }
    /// `signaledValue`.
    fn signaled_value(&self) -> u64 {
        msg::get("signaledValue", || self.signaledValue())
    }
    /// `waitUntilSignaledValue:timeoutMS:`, which blocks up to `milliseconds`.
    fn wait_until_signaled_value(&self, value: u64, milliseconds: u64) -> bool {
        msg::get("waitUntilSignaledValue:timeoutMS:", || {
            self.waitUntilSignaledValue_timeoutMS(value, milliseconds)
        })
    }
    /// The block runs `block` once when Metal calls it, on the listener's dispatch queue; `block`
    /// is dropped there, when it returns.
    fn notify_listener(
        &self,
        listener: &MetalSharedEventListener,
        value: u64,
        block: Notification,
    ) -> RaiseResult<()> {
        let once = Mutex::new(Some(block));
        let handler = RcBlock::new(
            move |_event: NonNull<ProtocolObject<dyn MTLSharedEvent>>, _value: u64| {
                // A foreign callout: marked ours, so a panic here aborts before it unwinds into
                // Metal.
                let _env = reims_vgpu_core::rt::BlockEnv::foreign();
                // Metal calls a notification once; a second call would find nothing to run.
                if let Some(block) = once.lock().take() {
                    block();
                }
            },
        );
        let handler = RcBlock::as_ptr(&handler);
        // SAFETY: a live listener and a live block; Metal copies the block.
        msg::send("notifyListener:atValue:block:", Policy::Caught, || unsafe {
            self.notifyListener_atValue_block(listener.get(), value, handler)
        })
    }
}

/// `MTLSharedEventHandle *`.
pub type MetalSharedEventHandle = Obj<MTLSharedEventHandle>;

/// `MTLSharedEventListener *`.
pub type MetalSharedEventListener = Obj<MTLSharedEventListener>;

#[cfg(test)]
mod tests {

    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::event::SharedEvent;

    use super::*;
    use crate::gpu::resource::testkit::*;

    /// setSignaledValue: sets the value; waitUntilSignaledValue:timeoutMS: is YES for a reached
    /// value and NO after the timeout for an unreached one
    #[test]
    fn signal_and_wait() {
        let d = dev();
        let e = d.new_shared_event().unwrap();
        e.set_signaled_value(3).unwrap();
        assert_eq!(e.get().signaledValue(), 3);
        assert!(e.wait_until_signaled_value(3, 0));
        assert!(e.wait_until_signaled_value(2, 0));
        assert!(!e.wait_until_signaled_value(4, 10));
    }

    /// notifyListener:atValue:block: runs the block once the value reaches it, off the asking
    /// thread, and once only; at once for a value reached already
    #[test]
    fn notify_listener_runs_once_the_value_is_reached() {
        use reims_vgpu_core::platform::Platform as _;
        let d = dev();
        let listener = crate::platform::tests::platform().new_shared_event_listener();
        let e = d.new_shared_event().unwrap();
        let (ran, runs) = std::sync::mpsc::channel();
        for value in [2, 5] {
            let ran = ran.clone();
            e.notify_listener(
                &listener,
                value,
                Box::new(move || ran.send((value, std::thread::current().id())).unwrap()),
            )
            .unwrap();
        }
        let wait = std::time::Duration::from_millis(100);
        assert!(runs.recv_timeout(wait).is_err(), "ran before the value");
        e.set_signaled_value(3).unwrap();
        let (value, thread) = runs
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert_eq!(value, 2);
        assert_ne!(thread, std::thread::current().id());
        assert!(runs.recv_timeout(wait).is_err(), "5 ran at 3");
        e.set_signaled_value(5).unwrap();
        assert_eq!(
            runs.recv_timeout(std::time::Duration::from_secs(5))
                .unwrap()
                .0,
            5
        );
        let ran = ran.clone();
        e.notify_listener(
            &listener,
            4,
            Box::new(move || ran.send((4, std::thread::current().id())).unwrap()),
        )
        .unwrap();
        assert_eq!(
            runs.recv_timeout(std::time::Duration::from_secs(5))
                .unwrap()
                .0,
            4
        );
        e.set_signaled_value(9).unwrap();
        assert!(runs.recv_timeout(wait).is_err(), "a block ran twice");
    }

    /// newSharedEventHandle gives a +1 handle that opens the same event
    #[test]
    fn new_shared_event_handle() {
        let d = dev();
        let e = d.new_shared_event().unwrap();
        e.set_signaled_value(9).unwrap();
        let h = e.new_shared_event_handle().unwrap();
        let e2 = d.open_shared_event(Some(&h)).unwrap();
        assert_eq!(e2.get().signaledValue(), 9);
    }
}
