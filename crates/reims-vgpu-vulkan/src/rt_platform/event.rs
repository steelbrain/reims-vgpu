//! The rt platform's shared events: rt's `SharedEvent`, whose value is the higher of what the GPU
//! signalled and what the host set, and only rises.
//!
//! **Notifications** run on rt's completion thread, the one that runs the device's completed
//! handlers, in turn with them, where Metal runs each on its listener's dispatch queue: the
//! listener ([`super::Listener`]) names no thread of its own. A notification for a value reached
//! already runs there too, soon after the ask, never on the asking thread. A wait in one for a
//! value that thread has not yet seen is refused, counted, and answers false.

use std::time::Duration;

use metal2vulkan_runtime as rt;

use crate::core::platform::Platform;
use crate::core::platform::event::{Notification, SharedEvent};
use crate::core::raise::RaiseResult;
use crate::core::rt::BlockEnv;

use super::unbuilt::rt_refused;
use super::{RtObj, RtPlatform};

impl SharedEvent<RtPlatform> for RtObj<rt::SharedEvent> {
    /// rt's: raises the host's value; a value not above it changes nothing.
    fn set_signaled_value(&self, signaled_value: u64) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!("setSignaledValue:", self, signaled_value);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            rt::SharedEvent::set_signaled_value(self, signaled_value);
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's value. In a process that records, it is answered once the recorder compared what the
    /// command buffers that signal the event to that value could have written
    /// ([`crate::recorder::event_reached`]).
    fn signaled_value(&self) -> u64 {
        let value = rt::SharedEvent::signaled_value(self);
        if let Some(id) = crate::recorder::event_key(self) {
            crate::recorder::event_reached(id, value);
        }
        value
    }

    /// rt's wait. False, counted, where rt refuses it: in a scheduled handler, or on rt's
    /// completion thread (a completed handler or a notification) for a value that thread has not
    /// yet seen, which a wait there would never see. In a process that records, a wait that saw
    /// the value reached returns once the recorder compared what the command buffers that signal
    /// the event could have written ([`crate::recorder::event_reached`]).
    fn wait_until_signaled_value(&self, value: u64, milliseconds: u64) -> bool {
        let recorded_call = crate::recorder::rec_call!(
            "waitUntilSignaledValue:timeoutMS:",
            self,
            value,
            milliseconds
        );
        let recorded_answer = crate::recorder::run(|| -> bool {
            let timeout = Duration::from_millis(milliseconds);
            match rt::SharedEvent::wait_until_signaled_value(self, value, timeout) {
                Ok(reached) => {
                    if reached && let Some(id) = crate::recorder::event_key(self) {
                        crate::recorder::event_reached(id, value);
                    }
                    reached
                }
                Err(error) => {
                    rt_refused("waitUntilSignaledValue:timeoutMS:", &error);
                    false
                }
            }
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// `block` runs once on rt's completion thread (the module's docs), once the value reaches
    /// `value`. Refused, counted, after the device was quiesced, when `block` never runs. In a
    /// process that records, `block` runs once the recorder compared what the command buffers
    /// that signal the event to `value` could have written ([`crate::recorder::event_reached`]).
    fn notify_listener(
        &self,
        listener: &<RtPlatform as Platform>::SharedEventListener,
        value: u64,
        block: Notification,
    ) -> RaiseResult<()> {
        let _ = listener;
        #[cfg(test)]
        let slot = crate::core::testkit::capture::current_slot();
        let recorded = crate::recorder::event_key(self);
        let notification = move |_value: u64| {
            // A callout on rt's thread: marked ours, so a panic here aborts.
            let _env = BlockEnv::foreign();
            if let Some(id) = recorded {
                crate::recorder::event_reached(id, value);
            }
            #[cfg(test)]
            crate::core::testkit::capture::with_slot(slot, block);
            #[cfg(not(test))]
            block();
        };
        if let Err(error) = rt::SharedEvent::notify_listener(self, value, notification) {
            rt_refused("notifyListener:atValue:block:", &error);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread::{self, ThreadId};

    use super::*;
    use crate::core::platform::Handle as _;
    use crate::core::platform::device::GpuDevice as _;
    use crate::rt_platform::RtRefusal;

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// A shared event on the GPU the test holds: set from the host, its value only rises; a wait
    /// answers true for a value reached and false at its timeout for one not; a notification
    /// runs off the asking thread once the value reaches its own, and soon for one reached
    /// already; a wait in a notification for a value not reached refused, counted, false.
    #[test]
    fn shared_events_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let event = device.new_shared_event().expect("a shared event");
        assert_eq!(event.signaled_value(), 0);
        assert_eq!(event.set_signaled_value(3), Ok(()));
        assert_eq!(event.set_signaled_value(2), Ok(()));
        assert_eq!(event.signaled_value(), 3);
        assert!(event.wait_until_signaled_value(3, 0));
        assert!(!event.wait_until_signaled_value(4, 10));

        let listener = platform.new_shared_event_listener();
        let asker = thread::current().id();
        let (tell, told) = mpsc::channel::<ThreadId>();
        let notify = |tell: mpsc::Sender<ThreadId>| -> Notification {
            Box::new(move || {
                let _ = tell.send(thread::current().id());
            })
        };
        assert_eq!(
            event.notify_listener(&listener, 5, notify(tell.clone())),
            Ok(())
        );
        assert!(told.recv_timeout(Duration::from_millis(50)).is_err());
        assert_eq!(event.set_signaled_value(5), Ok(()));
        let ran = told.recv_timeout(Duration::from_secs(10));
        assert_ne!(ran.expect("the notification ran"), asker);
        assert_eq!(event.notify_listener(&listener, 4, notify(tell)), Ok(()));
        let ran = told.recv_timeout(Duration::from_secs(10));
        assert_ne!(ran.expect("the notification of a value reached ran"), asker);

        let metal = RtRefusal::Refused.code();
        let before = counted(metal);
        let (tell, told) = mpsc::channel::<bool>();
        let again = event.clone_ref();
        let wait = move || {
            let _ = tell.send(again.wait_until_signaled_value(100, 1000));
        };
        assert_eq!(event.notify_listener(&listener, 5, Box::new(wait)), Ok(()));
        let waited = told.recv_timeout(Duration::from_secs(10));
        assert_eq!(waited, Ok(false));
        assert_eq!(counted(metal), before + 1);
    }
}
