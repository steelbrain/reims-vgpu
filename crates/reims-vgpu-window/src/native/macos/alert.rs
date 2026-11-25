//! The question a close asks before it ends the VM (`crate::close`), as macOS asks it: an
//! `NSAlert` run as a sheet on the window, for AppKit's window and SDL3's alike. A sheet runs no
//! loop of its own: `beginSheetModalForWindow:completionHandler:` returns at once, the window
//! takes no input while the sheet is on it, and the handler runs once the person answers, or once
//! the window's end takes the sheet off ([`end`]), which answers Cancel. The handler posts the
//! answer to the main thread, so it reaches the windows outside whatever AppKit was running.
//! Before the sheet begins, the application comes forward and the window to the front, out of the
//! Dock where it was minimized: Quit may come from the Dock while another application is active.
//!
//! The alert: the question as its message and what the close does as its informative text, a
//! warning; Cancel first, so the default (Return) and the trailing button, and Turn Off beside it,
//! marked destructive where AppKit has the mark (macOS 11).

use std::cell::Cell;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::{MainThreadMarker, sel};
use objc2_app_kit::{NSAlert, NSAlertSecondButtonReturn, NSAlertStyle, NSModalResponse, NSWindow};
use reims_vgpu_vulkan::core::rt::BlockEnv;
use reims_vgpu_vulkan::os::msg;

use super::app::activate;
use super::main_thread::{ns_text, post_main, responds, shared_app};
use crate::close::{self, Answer};

/// Where an answer goes, on the main thread: `(ids, answer, mtm)`.
pub(crate) type Answered = fn(&[u64], Answer, MainThreadMarker);

/// Asks the person whether to turn the VM off for windows `ids`, in an alert as a sheet on
/// `window` (the module doc): the answer is posted to the main thread, to `answered`. Where
/// AppKit refuses the sheet, that is logged, and the answer is Unshown: the VM runs on.
pub(crate) fn ask(window: &NSWindow, ids: Vec<u64>, answered: Answered, mtm: MainThreadMarker) {
    let alert = msg::get("new", || NSAlert::new(mtm));
    msg::get("setAlertStyle:", || {
        alert.setAlertStyle(NSAlertStyle::Warning)
    });
    msg::get("setMessageText:", || {
        alert.setMessageText(&ns_text(close::QUESTION));
    });
    msg::get("setInformativeText:", || {
        alert.setInformativeText(&ns_text(close::DETAIL));
    });
    msg::get("addButtonWithTitle:", || {
        alert.addButtonWithTitle(&ns_text(close::CANCEL))
    });
    let turn_off = msg::get("addButtonWithTitle:", || {
        alert.addButtonWithTitle(&ns_text(close::TURN_OFF))
    });
    if responds(&*turn_off, sel!(setHasDestructiveAction:)) {
        msg::get("setHasDestructiveAction:", || {
            turn_off.setHasDestructiveAction(true);
        });
    }
    activate(&shared_app(mtm));
    if msg::get("isMiniaturized", || window.isMiniaturized()) {
        msg::get("deminiaturize:", || window.deminiaturize(None));
    }
    msg::get("makeKeyAndOrderFront:", || {
        window.makeKeyAndOrderFront(None)
    });
    // The handler keeps the alert until it runs, and runs its answer once.
    let held: Cell<Option<(Retained<NSAlert>, Vec<u64>)>> =
        Cell::new(Some((alert.clone(), ids.clone())));
    let handler = RcBlock::new(move |response: NSModalResponse| {
        let _env = BlockEnv::foreign();
        if let Some((_alert, ids)) = held.take() {
            let answer = if response == NSAlertSecondButtonReturn {
                Answer::TurnOff
            } else {
                Answer::Cancel
            };
            post_main(move |mtm| answered(&ids, answer, mtm));
        }
    });
    if let Err(caught) = msg::catch("beginSheetModalForWindow:completionHandler:", || {
        alert.beginSheetModalForWindow_completionHandler(window, Some(&handler));
    }) {
        log::warn!(
            "the question before turning the VM off does not show ({}: {}): the VM runs on",
            caught.name,
            caught.reason
        );
        post_main(move |mtm| answered(&ids, Answer::Unshown, mtm));
    }
}

/// Takes the question's sheet off `window` as the window ends: its handler answers Cancel, and
/// the question is no longer open.
pub(crate) fn end(window: &NSWindow) {
    if let Some(sheet) = msg::get("attachedSheet", || window.attachedSheet()) {
        msg::get("endSheet:", || window.endSheet(&sheet));
    }
}
