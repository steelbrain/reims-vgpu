//! Fake: the `platform/event.rs` traits over the fake models (architecture 13.2).
//!
//! Each method logs its selector and arguments (`fake::calllog`) and consults the script
//! (`fake::script`) before its model; see `fake::impls` for the outcome channels.

use super::super::calllog::Arg;
use super::super::objects::{Fk, Model, kind};
use super::super::platform::FakePlatform;
use super::fatal;
use crate::platform::Platform;
use crate::platform::event::*;
use crate::raise::RaiseResult;
use crate::testkit::script::Policy;

impl SharedEvent<FakePlatform> for Fk<kind::SharedEvent> {
    fn new_shared_event_handle(&self) -> Option<<FakePlatform as Platform>::SharedEventHandle> {
        let target = self.erase();
        fatal(
            self.make(Policy::Fatal, "newSharedEventHandle", vec![], |w| {
                Some(w.new_obj::<kind::SharedEventHandle>(Model::Target(target)))
            }),
        )
    }

    fn set_signaled_value(&self, signaled_value: u64) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "setSignaledValue:",
            vec![Arg::U64(signaled_value)],
        )?;
        super::super::objects::signal_event(self.obj(), signaled_value);
        Ok(())
    }

    fn signaled_value(&self) -> u64 {
        let step = self.call("signaledValue", vec![]);
        fatal(step.throws(Policy::Fatal, "signaledValue"));
        if let Some(v) = step.get::<u64>() {
            return v;
        }
        super::super::objects::event_value(self.obj())
    }

    fn wait_until_signaled_value(&self, value: u64, milliseconds: u64) -> bool {
        let step = self.call(
            "waitUntilSignaledValue:timeoutMS:",
            vec![Arg::U64(value), Arg::U64(milliseconds)],
        );
        fatal(step.throws(Policy::Fatal, "waitUntilSignaledValue:timeoutMS:"));
        if let Some(b) = step.get::<bool>() {
            return b;
        }
        super::super::objects::wait_event(self.obj(), value, milliseconds)
    }

    fn notify_listener(
        &self,
        listener: &<FakePlatform as Platform>::SharedEventListener,
        value: u64,
        block: Notification,
    ) -> RaiseResult<()> {
        use crate::platform::Handle as _;
        self.act(
            Policy::Caught,
            "notifyListener:atValue:block:",
            vec![Arg::Obj(Some(listener.id())), Arg::U64(value)],
        )?;
        super::super::objects::notify_event(self.obj(), value, block);
        Ok(())
    }
}
