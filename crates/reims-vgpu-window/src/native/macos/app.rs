//! The application the windows belong to: its start, once per process, its menu, and its end.
//!
//! **The start** ([`start`]), at the first window's attach, is posted to the main run loop alone
//! ([`main_thread`]), and runs there:
//! - when the host already made an `NSApplication`, the windows just open in it. Unless it is
//!   ours, a local event monitor stands in for our `sendEvent:` (below);
//! - otherwise [`ReimsApplication`] becomes the shared application, with our delegate, and
//!   `[NSApp run]` runs, for good: it never returns, and the run loop the host parked the main
//!   thread in runs inside it. Once it has launched, the application:
//!   - becomes a regular one, with a Dock icon and a ⌘Tab entry (a process with no bundle starts
//!     as one that may not);
//!   - gets its main menu: the application's (Hide, Hide Others, Show All, Quit), titled with
//!     the process's name, and Window (Minimize, Zoom, Enter Full Screen, Release Input);
//!   - opens the windows posted so far;
//!   - activates. Launched from Terminal, macOS may refuse; the person's click on a window then
//!     activates it;
//!   - holds a user-initiated activity for the process's life, so App Nap never slows the VM
//!     while its windows are hidden.
//!
//! **Events.** The application's `sendEvent:` gives each event to the key window's input first
//! ([`super::input::InputState::app_send_event`]), before AppKit dispatches it: so the input sees
//! key-ups under ⌘ and ⌘ shortcuts, which AppKit sends to no view, and a captured keyboard's ⌘Q
//! reaches the guest rather than the menu. In a host's application the monitor does the same for
//! key events. SDL3's windows (`REIMS_VGPU_WINDOW_IMPL=sdl`) take every event of theirs there,
//! their mouse and scroll events too, as SDL's view gives them no input of its own
//! (`native/macos/sdl/input.rs`); SDL's video starts as the application launches, before the
//! windows open.
//!
//! **The end.** Quit (⌘Q uncaptured, or the Dock's) tells the host through every window's close
//! callback, and cancels AppKit's own termination, which would `exit()` behind the host's back:
//! the host ends the VM itself, and the windows close as their displays are released.

use std::cell::RefCell;
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject, Sel};
use objc2::{ClassType, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate,
    NSApplicationDidChangeScreenParametersNotification, NSApplicationTerminateReply, NSEvent,
    NSEventModifierFlags, NSMenu, NSMenuItem, NSMenuItemValidation, NSResponder,
};
use objc2_foundation::{
    NSActivityOptions, NSNotification, NSNotificationCenter, NSObject, NSObjectProtocol,
    NSProcessInfo, NSString, ns_string,
};
use reims_vgpu_vulkan::core::rt::BlockEnv;
use reims_vgpu_vulkan::os::msg;

use super::main_thread::{self, MainOnly, ns_text, responds, shared_app};
use super::view;
use crate::native;

#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {
    /// AppKit's shared application: nil until one is made.
    #[link_name = "NSApp"]
    static mut NS_APP: *mut AnyObject;
}

/// What the application keeps for the process's life, on the main thread.
struct AppState {
    /// Our delegate, which is also the menu's target and the screens' observer: the
    /// application's reference to it is weak.
    delegate: Retained<AppDelegate>,
    /// The user-initiated activity that keeps App Nap away.
    activity: Option<Retained<ProtocolObject<dyn NSObjectProtocol>>>,
    /// The local event monitor, in a host's application.
    monitor: Option<Retained<AnyObject>>,
}

/// The application's state, once it has started.
static APP: MainOnly<RefCell<Option<AppState>>> = MainOnly::new(RefCell::new(None));

define_class!(
    /// The shared application when the library makes it: its `sendEvent:` gives the key
    /// window's input each event first (module doc).
    // SAFETY:
    // - NSApplication may be subclassed, and `sendEvent:` overridden, calling `super`.
    // - The application does not implement `Drop`.
    #[unsafe(super(NSApplication, NSResponder, NSObject))]
    #[thread_kind = MainThreadOnly]
    pub(crate) struct ReimsApplication;

    impl ReimsApplication {
        #[unsafe(method(sendEvent:))]
        fn send_event(&self, event: &NSEvent) {
            let _env = BlockEnv::foreign();
            if native::input_takes(event, self.mtm()) {
                return;
            }
            // SAFETY: `sendEvent:` takes the event and returns nothing.
            view::pass("sendEvent:", || unsafe { msg_send![super(self), sendEvent: event] });
        }
    }

    unsafe impl NSObjectProtocol for ReimsApplication {}
);

define_class!(
    /// The application's delegate, the menu's target and the screens' observer (module doc).
    // SAFETY:
    // - NSObject has no subclassing requirements.
    // - The delegate does not implement `Drop`.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    pub(crate) struct AppDelegate;

    impl AppDelegate {
        /// Window ▸ Release Input: the key window's input lets the keyboard and the pointer go.
        #[unsafe(method(releaseInput:))]
        fn release_input(&self, _sender: Option<&AnyObject>) {
            let _env = BlockEnv::foreign();
            native::release_key_input(self.mtm());
        }

        /// The screens were added, removed, moved or resized.
        #[unsafe(method(screensChanged:))]
        fn screens_changed(&self, _notification: &NSNotification) {
            let _env = BlockEnv::foreign();
            native::screens_changed(self.mtm());
        }
    }

    unsafe impl NSObjectProtocol for AppDelegate {}

    unsafe impl NSApplicationDelegate for AppDelegate {
        #[unsafe(method(applicationDidFinishLaunching:))]
        fn did_finish_launching(&self, _notification: &NSNotification) {
            let _env = BlockEnv::foreign();
            launched(self.mtm());
        }

        /// Quit tells the host, and cancels AppKit's own termination (module doc).
        #[unsafe(method(applicationShouldTerminate:))]
        fn should_terminate(&self, _sender: &NSApplication) -> NSApplicationTerminateReply {
            let _env = BlockEnv::foreign();
            native::quit_requested(self.mtm());
            NSApplicationTerminateReply::TerminateCancel
        }
    }

    unsafe impl NSMenuItemValidation for AppDelegate {
        /// Release Input is on while the key window's keyboard is captured.
        #[unsafe(method(validateMenuItem:))]
        fn validate_menu_item(&self, _item: &NSMenuItem) -> bool {
            let _env = BlockEnv::foreign();
            native::key_input_captured(self.mtm())
        }
    }
);

impl AppDelegate {
    /// A new delegate.
    fn new(mtm: MainThreadMarker) -> Retained<AppDelegate> {
        // SAFETY: `init` is NSObject's designated initializer.
        msg::get("init", || unsafe {
            msg_send![super(AppDelegate::alloc(mtm).set_ivars(())), init]
        })
    }
}

/// Starts the application, once per process (module doc). Waits for nothing.
pub(crate) fn start() {
    main_thread::start(launch);
}

/// The application's start, on the main thread (module doc): the application to run, when the
/// library made it.
fn launch(mtm: MainThreadMarker) -> Option<Retained<NSApplication>> {
    let delegate = AppDelegate::new(mtm);
    observe_screens(&delegate);
    *APP.get(mtm).borrow_mut() = Some(AppState {
        delegate: delegate.clone(),
        activity: None,
        monitor: None,
    });
    if let Some(app) = host_application() {
        if !msg::get("isKindOfClass:", || {
            app.isKindOfClass(ReimsApplication::class())
        }) {
            install_monitor(mtm);
        }
        native::launched(mtm);
        main_thread::drain(mtm);
        return None;
    }
    // SAFETY: `sharedApplication` makes the receiving class's instance the shared application
    // and returns it; no application exists yet.
    let app: Retained<NSApplication> = msg::get("sharedApplication", || unsafe {
        msg_send![ReimsApplication::class(), sharedApplication]
    });
    msg::get("setDelegate:", || {
        app.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    });
    log::info!("the windows' application runs on the main thread");
    Some(app)
}

/// The shared application, when one exists already.
fn host_application() -> Option<Retained<NSApplication>> {
    // SAFETY: a read of AppKit's global by value, on the main thread, the only one that sets it.
    let app = unsafe { (&raw const NS_APP).read() };
    // SAFETY: a non-nil `NSApp` is the shared NSApplication, which lives for the process.
    unsafe { Retained::retain(app.cast::<NSApplication>()) }
}

/// The application has launched: its policy, menu, windows, activation and activity (module
/// doc).
fn launched(mtm: MainThreadMarker) {
    let app = shared_app(mtm);
    let regular = msg::get("setActivationPolicy:", || {
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular)
    });
    if !regular {
        log::warn!("the application could not become a regular one");
    }
    let target = APP
        .get(mtm)
        .borrow()
        .as_ref()
        .map(|state| state.delegate.clone());
    let (main_menu, window_menu) = menus(target.as_deref(), mtm);
    msg::get("setMainMenu:", || app.setMainMenu(Some(&main_menu)));
    msg::get("setWindowsMenu:", || app.setWindowsMenu(Some(&window_menu)));
    let activity = hold_activity();
    if let Some(state) = APP.get(mtm).borrow_mut().as_mut() {
        state.activity = Some(activity);
    }
    native::launched(mtm);
    main_thread::drain(mtm);
    activate(&app);
}

/// Brings the application forward: `activate` (macOS 14), else `activateIgnoringOtherApps:`.
#[expect(
    deprecated,
    reason = "activateIgnoringOtherApps: is the way before macOS 14, which has no activate"
)]
fn activate(app: &NSApplication) {
    if responds(app, sel!(activate)) {
        msg::get("activate", || app.activate());
    } else {
        msg::get("activateIgnoringOtherApps:", || {
            app.activateIgnoringOtherApps(true);
        });
    }
}

/// A user-initiated activity, which keeps App Nap from slowing the process. It still lets the
/// host sleep when idle, as the host's own settings say.
fn hold_activity() -> Retained<ProtocolObject<dyn NSObjectProtocol>> {
    let info = msg::get("processInfo", NSProcessInfo::processInfo);
    let reason = ns_string!("reims-vgpu: a virtual machine's display");
    msg::get("beginActivityWithOptions:reason:", || {
        info.beginActivityWithOptions_reason(
            NSActivityOptions::UserInitiatedAllowingIdleSystemSleep,
            reason,
        )
    })
}

/// The main menu and its Window menu (module doc). Release Input's target is `target`; the
/// others go to the first responder.
fn menus(
    target: Option<&AppDelegate>,
    mtm: MainThreadMarker,
) -> (Retained<NSMenu>, Retained<NSMenu>) {
    let name = msg::get("processName", || {
        msg::get("processInfo", NSProcessInfo::processInfo).processName()
    });
    let main_menu = new_menu(ns_string!(""), mtm);

    let app_menu = new_menu(&name, mtm);
    let hide = ns_text(&format!("Hide {name}"));
    add_item(&app_menu, &hide, sel!(hide:), "h", None, mtm);
    let others = add_item(
        &app_menu,
        ns_string!("Hide Others"),
        sel!(hideOtherApplications:),
        "h",
        None,
        mtm,
    );
    msg::get("setKeyEquivalentModifierMask:", || {
        others.setKeyEquivalentModifierMask(
            NSEventModifierFlags::Option | NSEventModifierFlags::Command,
        );
    });
    add_item(
        &app_menu,
        ns_string!("Show All"),
        sel!(unhideAllApplications:),
        "",
        None,
        mtm,
    );
    add_separator(&app_menu, mtm);
    let quit = ns_text(&format!("Quit {name}"));
    add_item(&app_menu, &quit, sel!(terminate:), "q", None, mtm);
    add_submenu(&main_menu, &app_menu, mtm);

    let window_menu = new_menu(ns_string!("Window"), mtm);
    add_item(
        &window_menu,
        ns_string!("Minimize"),
        sel!(performMiniaturize:),
        "m",
        None,
        mtm,
    );
    add_item(
        &window_menu,
        ns_string!("Zoom"),
        sel!(performZoom:),
        "",
        None,
        mtm,
    );
    let full_screen = add_item(
        &window_menu,
        ns_string!("Enter Full Screen"),
        sel!(toggleFullScreen:),
        "f",
        None,
        mtm,
    );
    msg::get("setKeyEquivalentModifierMask:", || {
        full_screen.setKeyEquivalentModifierMask(
            NSEventModifierFlags::Control | NSEventModifierFlags::Command,
        );
    });
    add_separator(&window_menu, mtm);
    add_item(
        &window_menu,
        ns_string!("Release Input"),
        sel!(releaseInput:),
        "",
        target.map(|t| -> &AnyObject { t.as_ref() }),
        mtm,
    );
    add_submenu(&main_menu, &window_menu, mtm);
    (main_menu, window_menu)
}

/// A menu titled `title`.
fn new_menu(title: &NSString, mtm: MainThreadMarker) -> Retained<NSMenu> {
    msg::get("initWithTitle:", || {
        NSMenu::initWithTitle(NSMenu::alloc(mtm), title)
    })
}

/// Adds an item titled `title` sending `action` to `target` (the first responder for `None`),
/// with ⌘`key` for a `key` that is not empty.
fn add_item(
    menu: &NSMenu,
    title: &NSString,
    action: Sel,
    key: &str,
    target: Option<&AnyObject>,
    mtm: MainThreadMarker,
) -> Retained<NSMenuItem> {
    let key = ns_text(key);
    // SAFETY: the action is a selector the target (or a responder along the chain) answers,
    // taking the sender.
    let item = msg::get("initWithTitle:action:keyEquivalent:", || unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            title,
            Some(action),
            &key,
        )
    });
    if let Some(target) = target {
        // SAFETY: the target lives for the process (the application's state holds it), and
        // answers the action.
        msg::get("setTarget:", || unsafe { item.setTarget(Some(target)) });
    }
    msg::get("addItem:", || menu.addItem(&item));
    item
}

/// Adds a separator.
fn add_separator(menu: &NSMenu, mtm: MainThreadMarker) {
    let separator = msg::get("separatorItem", || NSMenuItem::separatorItem(mtm));
    msg::get("addItem:", || menu.addItem(&separator));
}

/// Adds `submenu` to `menu` as an item of its own.
fn add_submenu(menu: &NSMenu, submenu: &NSMenu, mtm: MainThreadMarker) {
    let item = msg::get("new", || NSMenuItem::new(mtm));
    msg::get("setSubmenu:", || item.setSubmenu(Some(submenu)));
    msg::get("addItem:", || menu.addItem(&item));
}

/// Has `delegate` told when the screens change.
fn observe_screens(delegate: &AppDelegate) {
    let center = msg::get("defaultCenter", NSNotificationCenter::defaultCenter);
    // SAFETY: a constant AppKit defines.
    let name = unsafe { NSApplicationDidChangeScreenParametersNotification };
    // SAFETY: the observer answers `screensChanged:`, taking the notification, and lives for the
    // process (the application's state holds it), so it is never removed.
    msg::get("addObserver:selector:name:object:", || unsafe {
        center.addObserver_selector_name_object(
            delegate.as_ref(),
            sel!(screensChanged:),
            Some(name),
            None,
        );
    });
}

/// In a host's application, a local monitor gives the key window's input the key events first
/// (and SDL3's windows their mouse and scroll events), as [`ReimsApplication`]'s `sendEvent:`
/// would.
fn install_monitor(mtm: MainThreadMarker) {
    let block = RcBlock::new(|event: NonNull<NSEvent>| -> *mut NSEvent {
        let _env = BlockEnv::foreign();
        let Some(mtm) = MainThreadMarker::new() else {
            return event.as_ptr();
        };
        // SAFETY: AppKit hands the monitor a live event for the call.
        let taken = native::input_takes(unsafe { event.as_ref() }, mtm);
        if taken {
            std::ptr::null_mut()
        } else {
            event.as_ptr()
        }
    });
    let mask = native::monitored_events();
    // SAFETY: the block returns the event it was given, or nil to drop it.
    let monitor = msg::get("addLocalMonitorForEventsMatchingMask:handler:", || unsafe {
        NSEvent::addLocalMonitorForEventsMatchingMask_handler(mask, &block)
    });
    if let Some(state) = APP.get(mtm).borrow_mut().as_mut() {
        state.monitor = monitor;
    }
}
