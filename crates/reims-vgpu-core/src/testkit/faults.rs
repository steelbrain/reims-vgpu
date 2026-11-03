//! `Faults`: nil, an `NSError`, a raise or a crash at the nth call of one selector sent to one
//! receiver class.
//!
//! Core's tests key a fault by the receiver's class as well as the selector, so two classes that
//! share a selector (`endEncoding` on the three encoders, `setLabel:` on a queue and on a
//! resource) are told apart. The class is the fake's (`Kind::CLASS`: `MTLComputeCommandEncoder`,
//! `MTLDevice`, …). Faults ride the platform's [`Script`]: a class's entry and a selector's entry
//! are consulted in the order they were added.

use super::script::{HeldCall, Key, Script, Step};
use crate::platform::ObjId;
use crate::platform::types::NsErrorText;

/// What happens at the faulted call.
#[derive(Debug, Clone)]
pub enum Fault {
    /// Metal returns nil (a creator) or zero (a getter).
    Nil,
    /// Metal returns nil with this `NSError` (a creator with an error out-parameter).
    Error(NsErrorText),
    /// Metal raises `name` with `reason` at the send.
    Raise {
        /// The exception's name.
        name: String,
        /// Its reason.
        reason: String,
    },
    /// Metal, IOGPU or the driver crashes (the fake's marker death site).
    Crash,
    /// The call takes until the test lets it go ([`HeldCall`]), then answers as the model does.
    Hold(HeldCall),
}

impl Fault {
    fn step(self) -> Step {
        match self {
            Fault::Nil => Step::Nil,
            Fault::Error(e) => Step::Error(e),
            Fault::Raise { name, reason } => Step::Raise { name, reason },
            Fault::Crash => Step::crash(),
            Fault::Hold(held) => Step::Hold(held),
        }
    }
}

/// The faults of one fake world, keyed by receiver class and selector. `Clone` shares them.
#[derive(Debug, Clone)]
pub struct Faults {
    script: Script,
}

impl Faults {
    /// Faults that ride `script` (the platform's: [`super::FakePlatform::faults`]).
    pub fn new(script: Script) -> Faults {
        Faults { script }
    }

    /// The call of `sel` on `class` after `skip` such calls that run the model gets `fault`:
    /// `skip` 0 is the next call.
    pub fn at(&self, class: &'static str, sel: &'static str, skip: u64, fault: Fault) {
        self.script
            .add_key(Key::Call(class, sel), None, skip, Some(1), fault.step());
    }

    /// The next call of `sel` on `obj`, of `class`, gets `fault`.
    pub fn on(&self, obj: ObjId, class: &'static str, sel: &'static str, fault: Fault) {
        self.script
            .add_key(Key::Call(class, sel), Some(obj), 0, Some(1), fault.step());
    }

    /// Every call of `sel` on `class` gets `fault` (until the script is cleared).
    pub fn always(&self, class: &'static str, sel: &'static str, fault: Fault) {
        self.script
            .add_key(Key::Call(class, sel), None, 0, None, fault.step());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::Handle;
    use crate::platform::blit::{BlitCmd, BlitEncoder};
    use crate::platform::command::{CommandBuffer, CommandQueue};
    use crate::platform::compute::{ComputeCmd, ComputeEncoder};
    use crate::platform::device::GpuDevice;
    use crate::raise::OnFault;

    use crate::testkit::FakePlatform;

    /// A fault keyed by one class leaves another class's same selector alone, and lets `skip`
    /// calls of its own pass first
    #[test]
    fn keyed_by_class_not_selector() {
        let p = FakePlatform::new();
        let d = p.device();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let compute = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let faults = p.faults();
        faults.at(
            "MTLComputeCommandEncoder",
            "endEncoding",
            1,
            Fault::Raise {
                name: "N".into(),
                reason: "r".into(),
            },
        );
        assert!(blit.encode(OnFault::Return, BlitCmd::EndEncoding).is_ok());
        assert!(
            compute
                .encode(OnFault::Return, ComputeCmd::EndEncoding)
                .is_ok()
        );
        assert!(blit.encode(OnFault::Return, BlitCmd::EndEncoding).is_ok());
        assert!(
            compute
                .encode(OnFault::Return, ComputeCmd::EndEncoding)
                .is_err()
        );
        assert!(
            compute
                .encode(OnFault::Return, ComputeCmd::EndEncoding)
                .is_ok()
        );
        let ends: Vec<_> = p
            .call_log()
            .calls_to("endEncoding")
            .into_iter()
            .map(|c| (c.obj == blit.id(), c.class))
            .collect();
        assert_eq!(
            ends,
            vec![
                (true, "MTLBlitCommandEncoder"),
                (false, "MTLComputeCommandEncoder"),
                (true, "MTLBlitCommandEncoder"),
                (false, "MTLComputeCommandEncoder"),
                (false, "MTLComputeCommandEncoder"),
            ]
        );
    }

    /// A nil fault on a creator, and a fault on one object
    #[test]
    fn nil_and_on() {
        let p = FakePlatform::new();
        let d = p.device();
        p.faults().at("MTLDevice", "newCommandQueue", 0, Fault::Nil);
        assert!(d.new_command_queue().is_none());
        let q = d.new_command_queue().unwrap();
        p.faults()
            .on(q.id(), "MTLCommandQueue", "commandBuffer", Fault::Nil);
        assert!(q.command_buffer(OnFault::Die).unwrap().is_none());
        assert!(q.command_buffer(OnFault::Die).unwrap().is_some());
    }
}
