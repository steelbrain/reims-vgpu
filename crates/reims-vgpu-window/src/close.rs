//! A close the person asks for: a display's window closed (its close button, or the keyboard's or
//! the window system's close), or Quit on macOS. Each ends the VM: the window tells the host
//! through its close callback, and the host powers the VM off (QEMU's `reims-graphics-pci`: a
//! shutdown request with the power-off action, which the guest is not asked about). So the window
//! asks first, in a question of the host OS's own over the window ([`QUESTION`], [`DETAIL`]), with
//! two answers: [`CANCEL`], the default, keeps the VM running, and [`TURN_OFF`] tells the host.
//!
//! **When it asks** ([`Asks::request`]): only where the host would hear of the close, while the
//! window lives (the display's release takes the host's callbacks back) and the host gave a close
//! callback (without one a close does nothing). One question is open at a time in the process: a
//! close asked for while one is open adds none. Once the person has chosen Turn Off, a later close
//! tells the host at once, as the VM is ending. Where the OS cannot show the question
//! ([`Answer::Unshown`]), the close goes ahead as it did before the question, and the OS's side
//! logs why.
//!
//! Before the question shows, the window's input lets go as a close's does: every key and button
//! held goes up and capture ends, the keys going to the guest first. A cancelled close leaves it
//! so: nothing held in the guest, the input released, as after a loss of the keyboard.
//!
//! The question runs on the thread the windows run on (the main thread on macOS, the library's
//! `reims-vgpu.ui` on Linux and Windows), which keeps the process's [`Asks`].

use std::cell::RefCell;

/// The question, the dialog's title (an alert's message on macOS).
pub(crate) const QUESTION: &str = "Turn off the VM?";

/// What the question says the close does: the host powers the VM off with no word to the guest.
pub(crate) const DETAIL: &str =
    "The VM turns off at once, as if its power were cut.\nUnsaved work in it is lost.";

/// The answer that tells the host.
pub(crate) const TURN_OFF: &str = "Turn Off";

/// The answer that keeps the VM running, the default.
pub(crate) const CANCEL: &str = "Cancel";

/// What a close the person asked for does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// Nothing: the host would not hear of it, or a question is already open.
    Nothing,
    /// The person is asked; the answer comes to [`Asks::answered`].
    Ask,
    /// The host is told at once: the person already chose Turn Off.
    Tell,
}

/// The person's answer to the question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Answer {
    /// Turn Off: the host is told.
    TurnOff,
    /// Cancel, or the question dismissed: nothing happens.
    Cancel,
    /// The OS could not show the question: the host is told, as before the question.
    Unshown,
}

/// The process's questions: whether one is open, and whether the person has chosen Turn Off.
#[derive(Debug, Default)]
pub(crate) struct Asks {
    /// A question is open.
    open: bool,
    /// The person chose Turn Off, or no question could be shown: the VM is ending.
    ending: bool,
}

impl Asks {
    /// What a close the person asked for does, where `host_hears` says whether the host would
    /// hear of it (the module doc). An [`Step::Ask`] opens the question.
    pub(crate) fn request(&mut self, host_hears: bool) -> Step {
        if !host_hears {
            Step::Nothing
        } else if self.ending {
            Step::Tell
        } else if self.open {
            Step::Nothing
        } else {
            self.open = true;
            Step::Ask
        }
    }

    /// The person answered the open question: whether the host is told.
    pub(crate) fn answered(&mut self, answer: Answer) -> bool {
        self.open = false;
        if answer != Answer::Cancel {
            self.ending = true;
        }
        answer != Answer::Cancel
    }
}

thread_local! {
    /// The process's questions, on the thread the windows run on (the module doc).
    static ASKS: RefCell<Asks> = RefCell::new(Asks::default());
}

/// What a close the person asked for does ([`Asks::request`]), on the windows' thread.
pub(crate) fn request(host_hears: bool) -> Step {
    ASKS.with(|asks| asks.borrow_mut().request(host_hears))
}

/// The person answered the question ([`Asks::answered`]), on the windows' thread: whether the
/// host is told. The answer is logged.
pub(crate) fn answered(answer: Answer) -> bool {
    let tell = ASKS.with(|asks| asks.borrow_mut().answered(answer));
    let said = match answer {
        Answer::TurnOff => "the person chose Turn Off: the host turns the VM off",
        Answer::Cancel => "the close is cancelled, or its window ended: the host is not told",
        Answer::Unshown => "no question showed: the host turns the VM off",
    };
    log::info!(target: "reims_vgpu_window", "{said}");
    tell
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A close asks only where the host would hear of it, and while a question is open another
    /// close asks nothing
    #[test]
    fn a_close_asks_only_where_the_host_would_hear_of_it() {
        let mut asks = Asks::default();
        assert_eq!(asks.request(false), Step::Nothing);
        assert_eq!(asks.request(true), Step::Ask);
        assert_eq!(asks.request(true), Step::Nothing);
        assert_eq!(asks.request(false), Step::Nothing);
    }

    /// Cancel tells the host nothing, and the next close asks again; Turn Off, or a question the
    /// OS could not show, tells the host, and a later close tells it at once
    #[test]
    fn only_turn_off_tells_the_host() {
        let mut asks = Asks::default();
        assert_eq!(asks.request(true), Step::Ask);
        assert!(!asks.answered(Answer::Cancel));
        assert_eq!(asks.request(true), Step::Ask);
        assert!(asks.answered(Answer::TurnOff));
        assert_eq!(asks.request(true), Step::Tell);
        assert_eq!(asks.request(false), Step::Nothing);

        let mut unshown = Asks::default();
        assert_eq!(unshown.request(true), Step::Ask);
        assert!(unshown.answered(Answer::Unshown));
        assert_eq!(unshown.request(true), Step::Tell);
    }
}
