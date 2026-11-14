//! The GPU the window's compositor composites on, where SDL's window is on Wayland: the main
//! device of the window's surface's dma-buf feedback (`zwp_linux_dmabuf_feedback_v1`, which
//! `zwp_linux_dmabuf_v1` gives from its version 4), as a DRM node's device number ([`main_device`]).
//! The compositor names there the device it would rather import the window's buffers on, which is
//! the one it composites with (linux-dmabuf-v1's own words: "the main device is the device used by
//! the compositor for compositing").
//!
//! It is read once a presenter asks ([`crate::VulkanTarget::compositor_device`]), through an event
//! queue of this module's own on SDL's display, so SDL's own queue sees none of its events, and
//! with a proxy wrapper of the display on that queue, as libwayland-client asks of a second user
//! of a display. The library is the one SDL opened (`libwayland-client.so.0`, opened again by its
//! name, which answers the same library), its calls looked up by name; no protocol binding is
//! linked. The feedback's events are taken through round trips of that queue until its `done`,
//! [`ROUND_TRIPS`] of them at most. `None` where any step has no answer: SDL's window is not on
//! Wayland, the library or a call is missing, the compositor lists no `zwp_linux_dmabuf_v1` of
//! version 4 or later, or sends no main device; the window then presents as before.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;

use sdl3_sys::everything::{
    SDL_GetPointerProperty, SDL_GetWindowProperties, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER,
    SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, SDL_Window,
};

/// How many round trips of the queue the feedback's `done` is waited for.
const ROUND_TRIPS: usize = 4;

/// The lowest version of `zwp_linux_dmabuf_v1` that gives a surface's feedback.
const FEEDBACK_VERSION: u32 = 4;

/// `wl_message`.
#[repr(C)]
struct Message {
    name: *const c_char,
    signature: *const c_char,
    types: *const *const Interface,
}

/// `wl_interface`.
#[repr(C)]
struct Interface {
    name: *const c_char,
    version: c_int,
    method_count: c_int,
    methods: *const Message,
    event_count: c_int,
    events: *const Message,
}

/// `wl_array`.
#[repr(C)]
struct Array {
    size: usize,
    alloc: usize,
    data: *mut c_void,
}

/// `union wl_argument`.
#[repr(C)]
#[derive(Clone, Copy)]
union Argument {
    u: u32,
    s: *const c_char,
    o: *mut c_void,
}

/// A table of the protocol's own, which libwayland-client reads only: shared between threads as
/// it is.
#[repr(transparent)]
struct Table<T>(T);

// SAFETY: the tables hold pointers to static strings and to other tables, never written.
unsafe impl<T> Sync for Table<T> {}

/// Types for the arguments of a message that names no interface of its own.
static NO_TYPES: Table<[*const Interface; 4]> = Table([std::ptr::null(); 4]);

/// `zwp_linux_dmabuf_v1`'s requests (only `destroy` and `get_surface_feedback` are made).
static DMABUF_METHODS: Table<[Message; 4]> = Table([
    Message {
        name: c"destroy".as_ptr(),
        signature: c"".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
    Message {
        name: c"create_params".as_ptr(),
        signature: c"n".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
    Message {
        name: c"get_default_feedback".as_ptr(),
        signature: c"4n".as_ptr(),
        types: FEEDBACK_TYPES.0.as_ptr(),
    },
    Message {
        name: c"get_surface_feedback".as_ptr(),
        signature: c"4no".as_ptr(),
        types: FEEDBACK_TYPES.0.as_ptr(),
    },
]);

/// `zwp_linux_dmabuf_v1`'s events, which a version 4 binding is never sent.
static DMABUF_EVENTS: Table<[Message; 2]> = Table([
    Message {
        name: c"format".as_ptr(),
        signature: c"u".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
    Message {
        name: c"modifier".as_ptr(),
        signature: c"3uuu".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
]);

/// `zwp_linux_dmabuf_v1`.
static DMABUF: Table<Interface> = Table(Interface {
    name: c"zwp_linux_dmabuf_v1".as_ptr(),
    version: FEEDBACK_VERSION as c_int,
    method_count: 4,
    methods: DMABUF_METHODS.0.as_ptr(),
    event_count: 2,
    events: DMABUF_EVENTS.0.as_ptr(),
});

/// The types of the feedback requests' arguments: the new feedback, and the surface (any).
static FEEDBACK_TYPES: Table<[*const Interface; 2]> =
    Table([&raw const FEEDBACK.0, std::ptr::null()]);

/// `zwp_linux_dmabuf_feedback_v1`'s one request.
static FEEDBACK_METHODS: Table<[Message; 1]> = Table([Message {
    name: c"destroy".as_ptr(),
    signature: c"".as_ptr(),
    types: NO_TYPES.0.as_ptr(),
}]);

/// `zwp_linux_dmabuf_feedback_v1`'s events, in their order.
static FEEDBACK_EVENTS: Table<[Message; 7]> = Table([
    Message {
        name: c"done".as_ptr(),
        signature: c"".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
    Message {
        name: c"format_table".as_ptr(),
        signature: c"hu".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
    Message {
        name: c"main_device".as_ptr(),
        signature: c"a".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
    Message {
        name: c"tranche_done".as_ptr(),
        signature: c"".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
    Message {
        name: c"tranche_target_device".as_ptr(),
        signature: c"a".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
    Message {
        name: c"tranche_formats".as_ptr(),
        signature: c"a".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
    Message {
        name: c"tranche_flags".as_ptr(),
        signature: c"u".as_ptr(),
        types: NO_TYPES.0.as_ptr(),
    },
]);

/// `zwp_linux_dmabuf_feedback_v1`.
static FEEDBACK: Table<Interface> = Table(Interface {
    name: c"zwp_linux_dmabuf_feedback_v1".as_ptr(),
    version: FEEDBACK_VERSION as c_int,
    method_count: 1,
    methods: FEEDBACK_METHODS.0.as_ptr(),
    event_count: 7,
    events: FEEDBACK_EVENTS.0.as_ptr(),
});

/// The calls of libwayland-client this module makes.
struct Client {
    create_queue: ProxyToProxy,
    queue_destroy: OfProxy,
    create_wrapper: ProxyToProxy,
    wrapper_destroy: OfProxy,
    set_queue: SetQueue,
    marshal: Marshal,
    add_listener: AddListener,
    roundtrip_queue: RoundtripQueue,
    proxy_destroy: OfProxy,
    proxy_version: ProxyVersion,
    registry: *const Interface,
}

/// `wl_display_create_queue` and `wl_proxy_create_wrapper`.
type ProxyToProxy = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
/// `wl_event_queue_destroy`, `wl_proxy_wrapper_destroy` and `wl_proxy_destroy`.
type OfProxy = unsafe extern "C" fn(*mut c_void);
/// `wl_proxy_set_queue`.
type SetQueue = unsafe extern "C" fn(*mut c_void, *mut c_void);
/// `wl_proxy_marshal_array_flags`.
type Marshal = unsafe extern "C" fn(
    *mut c_void,
    u32,
    *const Interface,
    u32,
    u32,
    *mut Argument,
) -> *mut c_void;
/// `wl_proxy_add_listener`.
type AddListener = unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void) -> c_int;
/// `wl_display_roundtrip_queue`.
type RoundtripQueue = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
/// `wl_proxy_get_version`.
type ProxyVersion = unsafe extern "C" fn(*mut c_void) -> u32;

// SAFETY: function pointers and the library's own `wl_registry_interface`, never written.
unsafe impl Send for Client {}
// SAFETY: as above.
unsafe impl Sync for Client {}

/// `wl_display.get_registry`'s opcode.
const GET_REGISTRY: u32 = 1;
/// `wl_registry.bind`'s opcode.
const BIND: u32 = 0;
/// `zwp_linux_dmabuf_v1.get_surface_feedback`'s opcode.
const GET_SURFACE_FEEDBACK: u32 = 3;
/// The `destroy` requests' opcode, of both interfaces.
const DESTROY: u32 = 0;
/// `WL_MARSHAL_FLAG_DESTROY`.
const MARSHAL_DESTROY: u32 = 1;

/// libwayland-client's calls, looked up once: `None` where the library or any call is missing.
fn client() -> Option<&'static Client> {
    static CLIENT: OnceLock<Option<Client>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            // SAFETY: the library SDL opened for Wayland, opened again by its name; it stays
            // loaded for the process, as SDL's own opening keeps it.
            let lib = unsafe {
                libc::dlopen(
                    c"libwayland-client.so.0".as_ptr(),
                    libc::RTLD_NOW | libc::RTLD_LOCAL,
                )
            };
            if lib.is_null() {
                return None;
            }
            let sym = |name: &CStr| {
                // SAFETY: a live handle and a symbol name.
                let p = unsafe { libc::dlsym(lib, name.as_ptr()) };
                (!p.is_null()).then_some(p)
            };
            // SAFETY: each symbol is libwayland-client's call of that name, of the C type it is
            // read as here (wayland-client-core.h).
            unsafe {
                Some(Client {
                    create_queue: std::mem::transmute::<*mut c_void, ProxyToProxy>(sym(
                        c"wl_display_create_queue",
                    )?),
                    queue_destroy: std::mem::transmute::<*mut c_void, OfProxy>(sym(
                        c"wl_event_queue_destroy",
                    )?),
                    create_wrapper: std::mem::transmute::<*mut c_void, ProxyToProxy>(sym(
                        c"wl_proxy_create_wrapper",
                    )?),
                    wrapper_destroy: std::mem::transmute::<*mut c_void, OfProxy>(sym(
                        c"wl_proxy_wrapper_destroy",
                    )?),
                    set_queue: std::mem::transmute::<*mut c_void, SetQueue>(sym(
                        c"wl_proxy_set_queue",
                    )?),
                    marshal: std::mem::transmute::<*mut c_void, Marshal>(sym(
                        c"wl_proxy_marshal_array_flags",
                    )?),
                    add_listener: std::mem::transmute::<*mut c_void, AddListener>(sym(
                        c"wl_proxy_add_listener",
                    )?),
                    roundtrip_queue: std::mem::transmute::<*mut c_void, RoundtripQueue>(sym(
                        c"wl_display_roundtrip_queue",
                    )?),
                    proxy_destroy: std::mem::transmute::<*mut c_void, OfProxy>(sym(
                        c"wl_proxy_destroy",
                    )?),
                    proxy_version: std::mem::transmute::<*mut c_void, ProxyVersion>(sym(
                        c"wl_proxy_get_version",
                    )?),
                    registry: sym(c"wl_registry_interface")?.cast(),
                })
            }
        })
        .as_ref()
}

/// What the listeners gather, through their `data`.
#[derive(Default)]
struct Gathered {
    /// `zwp_linux_dmabuf_v1`'s global name and version, once listed.
    dmabuf: Option<(u32, u32)>,
    /// The feedback's main device, once sent.
    main_device: Option<u64>,
    /// The feedback's `done` came.
    done: bool,
}

/// `wl_registry_listener`.
#[repr(C)]
struct RegistryListener {
    global: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const c_char, u32),
    global_remove: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
}

/// `zwp_linux_dmabuf_feedback_v1_listener`.
#[repr(C)]
struct FeedbackListener {
    done: unsafe extern "C" fn(*mut c_void, *mut c_void),
    format_table: unsafe extern "C" fn(*mut c_void, *mut c_void, i32, u32),
    main_device: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut Array),
    tranche_done: unsafe extern "C" fn(*mut c_void, *mut c_void),
    tranche_target_device: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut Array),
    tranche_formats: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut Array),
    tranche_flags: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
}

/// A global listed: `zwp_linux_dmabuf_v1`'s name and version kept.
unsafe extern "C" fn global(
    data: *mut c_void,
    _registry: *mut c_void,
    name: u32,
    interface: *const c_char,
    version: u32,
) {
    // SAFETY: `data` is the `Gathered` the listener was added with, which outlives the queue's
    // dispatch; `interface` is the event's string.
    let (gathered, interface) =
        unsafe { (&mut *data.cast::<Gathered>(), CStr::from_ptr(interface)) };
    if interface == c"zwp_linux_dmabuf_v1" {
        gathered.dmabuf = Some((name, version));
    }
}

/// A global gone: nothing to do.
unsafe extern "C" fn global_remove(_data: *mut c_void, _registry: *mut c_void, _name: u32) {}

/// The feedback is complete.
unsafe extern "C" fn feedback_done(data: *mut c_void, _feedback: *mut c_void) {
    // SAFETY: as `global`'s.
    unsafe { (*data.cast::<Gathered>()).done = true };
}

/// The format table's file, which is ours to close.
unsafe extern "C" fn format_table(_data: *mut c_void, _feedback: *mut c_void, fd: i32, _size: u32) {
    // SAFETY: the event hands the file over.
    unsafe { libc::close(fd) };
}

/// The main device: a `dev_t`.
unsafe extern "C" fn main_device(data: *mut c_void, _feedback: *mut c_void, device: *mut Array) {
    // SAFETY: as `global`'s; the array is the event's, of `size` bytes at `data`.
    unsafe {
        let array = &*device;
        if array.size >= size_of::<u64>() && !array.data.is_null() {
            (*data.cast::<Gathered>()).main_device =
                Some(array.data.cast::<u64>().read_unaligned());
        }
    }
}

/// A tranche's end: nothing to do.
unsafe extern "C" fn tranche_done(_data: *mut c_void, _feedback: *mut c_void) {}

/// A tranche's array: nothing to do.
unsafe extern "C" fn tranche_array(_data: *mut c_void, _feedback: *mut c_void, _array: *mut Array) {
}

/// A tranche's flags: nothing to do.
unsafe extern "C" fn tranche_flags(_data: *mut c_void, _feedback: *mut c_void, _flags: u32) {}

static REGISTRY_LISTENER: RegistryListener = RegistryListener {
    global,
    global_remove,
};

static FEEDBACK_LISTENER: FeedbackListener = FeedbackListener {
    done: feedback_done,
    format_table,
    main_device,
    tranche_done,
    tranche_target_device: tranche_array,
    tranche_formats: tranche_array,
    tranche_flags,
};

/// The DRM node's major and minor numbers of `device`, a `dev_t` (glibc's encoding).
pub(crate) fn major_minor(device: u64) -> (u32, u32) {
    let major = ((device >> 32) & 0xffff_f000) | ((device >> 8) & 0x0fff);
    let minor = ((device >> 12) & 0xffff_ff00) | (device & 0xff);
    (
        u32::try_from(major).unwrap_or(u32::MAX),
        u32::try_from(minor).unwrap_or(u32::MAX),
    )
}

/// The main device of the dma-buf feedback of `window`'s Wayland surface, as its DRM node's
/// major and minor numbers (the module doc); `None` where there is none to read.
///
/// # Safety
///
/// `window` is SDL's live window.
pub(crate) unsafe fn main_device_of(window: *mut SDL_Window) -> Option<(u32, u32)> {
    let client = client()?;
    // SAFETY: SDL's live window; its properties are read only.
    let (display, surface) = unsafe {
        let props = SDL_GetWindowProperties(window);
        (
            SDL_GetPointerProperty(
                props,
                SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER,
                std::ptr::null_mut(),
            ),
            SDL_GetPointerProperty(
                props,
                SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER,
                std::ptr::null_mut(),
            ),
        )
    };
    if display.is_null() || surface.is_null() {
        return None;
    }
    // SAFETY: SDL's live display and surface, with an event queue and proxies of this call's own,
    // each destroyed before it returns; the listeners' data outlives every dispatch of the queue.
    unsafe {
        let queue = (client.create_queue)(display);
        if queue.is_null() {
            return None;
        }
        let wrapper = (client.create_wrapper)(display);
        if wrapper.is_null() {
            (client.queue_destroy)(queue);
            return None;
        }
        (client.set_queue)(wrapper, queue);
        let mut gathered = Gathered::default();
        let data = (&raw mut gathered).cast::<c_void>();
        let mut found = None;
        let mut args = [Argument {
            o: std::ptr::null_mut(),
        }];
        let registry = (client.marshal)(
            wrapper,
            GET_REGISTRY,
            client.registry,
            (client.proxy_version)(wrapper),
            0,
            args.as_mut_ptr(),
        );
        if !registry.is_null() {
            (client.add_listener)(registry, (&raw const REGISTRY_LISTENER).cast(), data);
            if (client.roundtrip_queue)(display, queue) >= 0
                && let Some((name, version)) = gathered.dmabuf
                && version >= FEEDBACK_VERSION
            {
                let mut bind = [
                    Argument { u: name },
                    Argument { s: DMABUF.0.name },
                    Argument {
                        u: FEEDBACK_VERSION,
                    },
                    Argument {
                        o: std::ptr::null_mut(),
                    },
                ];
                let dmabuf = (client.marshal)(
                    registry,
                    BIND,
                    &raw const DMABUF.0,
                    FEEDBACK_VERSION,
                    0,
                    bind.as_mut_ptr(),
                );
                if !dmabuf.is_null() {
                    let mut ask = [
                        Argument {
                            o: std::ptr::null_mut(),
                        },
                        Argument { o: surface },
                    ];
                    let feedback = (client.marshal)(
                        dmabuf,
                        GET_SURFACE_FEEDBACK,
                        &raw const FEEDBACK.0,
                        FEEDBACK_VERSION,
                        0,
                        ask.as_mut_ptr(),
                    );
                    if !feedback.is_null() {
                        (client.add_listener)(
                            feedback,
                            (&raw const FEEDBACK_LISTENER).cast(),
                            data,
                        );
                        for _ in 0..ROUND_TRIPS {
                            if gathered.done || (client.roundtrip_queue)(display, queue) < 0 {
                                break;
                            }
                        }
                        found = gathered.done.then_some(gathered.main_device).flatten();
                        (client.marshal)(
                            feedback,
                            DESTROY,
                            std::ptr::null(),
                            FEEDBACK_VERSION,
                            MARSHAL_DESTROY,
                            std::ptr::null_mut(),
                        );
                    }
                    (client.marshal)(
                        dmabuf,
                        DESTROY,
                        std::ptr::null(),
                        FEEDBACK_VERSION,
                        MARSHAL_DESTROY,
                        std::ptr::null_mut(),
                    );
                }
            }
            (client.proxy_destroy)(registry);
        }
        (client.wrapper_destroy)(wrapper);
        (client.queue_destroy)(queue);
        found.map(major_minor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// glibc's `makedev` of (226, 128), a render node, and of (226, 1), a primary one, and of a
    /// number past 12 bits of major and 8 of minor, read back.
    #[test]
    fn device_numbers_read_back_as_glibc_makes_them() {
        let makedev = |major: u64, minor: u64| {
            ((major & 0xffff_f000) << 32)
                | ((major & 0x0fff) << 8)
                | ((minor & 0xffff_ff00) << 12)
                | (minor & 0xff)
        };
        assert_eq!(major_minor(makedev(226, 128)), (226, 128));
        assert_eq!(major_minor(makedev(226, 1)), (226, 1));
        assert_eq!(major_minor(makedev(0x12345, 0x6789a)), (0x12345, 0x6789a));
    }

    /// The tables name the protocol's interfaces and messages as linux-dmabuf-v1 does, the
    /// requests at the opcodes the module makes them by.
    #[test]
    fn the_tables_follow_the_protocol() {
        // SAFETY: static strings.
        let name = |m: &Message| unsafe { CStr::from_ptr(m.name) };
        assert_eq!(
            name(&DMABUF_METHODS.0[GET_SURFACE_FEEDBACK as usize]),
            c"get_surface_feedback"
        );
        assert_eq!(name(&DMABUF_METHODS.0[DESTROY as usize]), c"destroy");
        assert_eq!(name(&FEEDBACK_METHODS.0[DESTROY as usize]), c"destroy");
        let events: Vec<_> = FEEDBACK_EVENTS.0.iter().map(name).collect();
        assert_eq!(
            events,
            [
                c"done",
                c"format_table",
                c"main_device",
                c"tranche_done",
                c"tranche_target_device",
                c"tranche_formats",
                c"tranche_flags"
            ]
        );
    }
}
