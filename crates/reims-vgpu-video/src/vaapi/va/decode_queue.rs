//! One video decode queue for each Direct3D 12 device, on Windows: every ask on a device of ours
//! for a command queue of type `D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE` gets the same queue.
//!
//! Mesa's VA driver on Direct3D 12 (`vaon12`) makes a video decode queue for each decoder, which
//! is one for each VA context (`d3d12_video_decoder_create_command_objects`,
//! `d3d12_video_dec.cpp`, Mesa 26.2.7). It makes them on the process's one device on the
//! adapter: Mesa asks for it with `ALLOW_RETURNING_EXISTING_DEVICE`, and the device is the same
//! one `super::d3d12` makes the pools' textures on. Intel's Direct3D 12 video driver for the UHD
//! Graphics 630 (`igd12dxva64.dll` 31.0.101.2145, Windows 10.0.26300) decodes wrong pictures on
//! a device's later video decode queue while an earlier one lives. **Measured** on 2026-11-20
//! (potluck job m3-max-62074, twice), each run decoding a 4K H.264 stream's first 8 access units
//! on a lone VA decoder and comparing them with a decoder's alone:
//! - 8 of 8 pictures wrong when only an idle video decode queue was made on the device first;
//! - 8 of 8 wrong with that queue plus an `ID3D12VideoDecoder` and an `ID3D12VideoDecoderHeap`;
//! - none wrong with the decoder or the heap, or both, made first without a queue;
//! - none wrong with nothing made first.
//!
//! FFmpeg's own D3D12 decoders, each on a queue of its own, show the same: of two or three
//! decoding at once in one process, one decodes right. Its D3D11 decoders and two processes are
//! right (docs/video/bugs.md, U33). So no second video decode queue is made on a device of ours.
//! The vendor does not decide: every device gets the one queue.
//!
//! The device's `ID3D12Device::CreateCommandQueue` (its vtable's slot 8, `d3d12.h`) is replaced
//! by [`create_command_queue`] while a [`Hold`] lives on a device with that vtable. The vtable
//! belongs to the Direct3D 12 runtime's device class, which every device of the process with
//! that vtable shares, so the slot is set for all of them. [`create_command_queue`] calls the
//! slot's own method for every other queue type, and for a device no [`Hold`] holds. For a held
//! device, it makes the device's first video decode queue of a given priority, flags and node
//! mask, keeps it, and answers each later ask with that queue, a reference added for the asker.
//! A queue answered this way lives while any VA context holds it, and while its device's holds
//! live. The module's lock is held across the queue's making, so two first asks on one device
//! make one queue.
//!
//! A device's [`Hold`] is taken in `super::d3d12::Device::new`, which runs before any VA context
//! is made: a pool's textures are made on the device first. When the last hold on a device
//! goes, its kept queues are released. When the last hold on any device with a vtable goes, the
//! slot gets its own method back, unless something else set the slot after us. The method is
//! kept with its vtable for good, so a call that read the slot before it was set back still
//! finds it.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::sync::Mutex;

use super::d3d12::{
    Com, CommandQueueDesc, DEVICE_CREATE_COMMAND_QUEUE, Guid, Hresult, IID_ID3D12_COMMAND_QUEUE,
    guid,
};

/// `D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE`.
const VIDEO_DECODE: u32 = 4;

/// `IUnknown`'s IID, whose interface is a COM object's identity.
const IID_IUNKNOWN: Guid = guid(
    0x0000_0000,
    0x0000,
    0x0000,
    [0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46],
);

/// `E_FAIL`'s bits.
const E_FAIL: Hresult = 0x8000_4005_u32.cast_signed();

/// `PAGE_READWRITE`.
const PAGE_READWRITE: u32 = 0x04;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualProtect(address: *mut c_void, size: usize, protect: u32, old: *mut u32) -> i32;
}

/// `ID3D12Device::CreateCommandQueue`.
type CreateCommandQueue = unsafe extern "system" fn(
    *mut c_void,
    *const CommandQueueDesc,
    *const Guid,
    *mut *mut c_void,
) -> Hresult;

/// `IUnknown::QueryInterface`, every COM object's method 0.
type QueryInterface =
    unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> Hresult;

/// What the module keeps: the vtables it set, and the devices held.
struct State {
    vtables: Vec<Vtable>,
    devices: Vec<Held>,
}

/// A device class's vtable whose `CreateCommandQueue` slot the module set.
struct Vtable {
    /// The vtable's address.
    at: usize,
    /// The slot's own method, kept for good.
    original: CreateCommandQueue,
    /// The holds on devices with this vtable.
    holds: usize,
}

/// A device held, and its video decode queues.
struct Held {
    /// The device's identity: its `IUnknown` pointer.
    identity: usize,
    holds: usize,
    /// Each queue made, by its description's priority, flags and node mask.
    queues: Vec<((i32, u32, u32), Com)>,
}

static STATE: Mutex<State> = Mutex::new(State {
    vtables: Vec::new(),
    devices: Vec::new(),
});

/// A hold on a device's one video decode queue: while it lives, the device answers every ask for
/// a video decode queue with one queue. Let go of, it lets the device's queue and the vtable's
/// slot go as the module's doc says.
pub(super) struct Hold {
    identity: usize,
    vtable: usize,
}

/// The address of `device`'s vtable: a COM object's first field.
fn vtable_of(device: *mut c_void) -> usize {
    // SAFETY: `device` is a live COM object, whose first field is its vtable's address.
    unsafe { *device.cast::<usize>() }
}

/// The address of `vtable`'s `CreateCommandQueue` slot, as an atomic.
fn slot_of(vtable: usize) -> &'static AtomicUsize {
    let slot = (vtable as *mut usize).wrapping_add(DEVICE_CREATE_COMMAND_QUEUE);
    // SAFETY: a device class's vtable lives as long as its module, which the process does not
    // unload while a device lives, and its slot holds a function's address, aligned as a usize.
    unsafe { AtomicUsize::from_ptr(slot) }
}

/// Sets `vtable`'s slot to `to`, under the module's lock: its old value.
fn set_slot(vtable: usize, to: usize) -> Result<usize, String> {
    let slot = slot_of(vtable);
    let at = slot.as_ptr().cast::<c_void>();
    let mut old = 0;
    // SAFETY: the slot is a usize of the vtable's, which the call makes writable.
    let made = unsafe { VirtualProtect(at, size_of::<usize>(), PAGE_READWRITE, &mut old) };
    if made == 0 {
        return Err(format!(
            "VirtualProtect of the device's vtable failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let was = slot.swap(to, Ordering::SeqCst);
    let mut unused = 0;
    // SAFETY: as above, given back its protection.
    unsafe { VirtualProtect(at, size_of::<usize>(), old, &mut unused) };
    Ok(was)
}

/// The identity of the COM object `object`: its `IUnknown` pointer, released at once (the
/// caller holds the object).
fn identity_of(object: *mut c_void) -> Result<usize, String> {
    let mut out = null_mut();
    // SAFETY: `object` is a live COM object, whose vtable's method 0 is `QueryInterface`.
    let result = unsafe {
        let vtable = *object.cast::<*const *const c_void>();
        let query = std::mem::transmute::<*const c_void, QueryInterface>(*vtable);
        query(object, &IID_IUNKNOWN, &mut out)
    };
    let unknown = Com::made("QueryInterface (IUnknown)", result, out)?;
    Ok(unknown.raw() as usize)
}

/// The address of [`create_command_queue`], which a slot the module set holds.
fn ours() -> usize {
    create_command_queue as *const () as usize
}

impl Hold {
    /// A hold on `device`'s one video decode queue, setting its vtable's slot where no hold did.
    pub(super) fn new(device: &Com) -> Result<Hold, String> {
        let identity = identity_of(device.raw())?;
        let vtable = vtable_of(device.raw());
        let mut state = STATE.lock();
        match state.vtables.iter_mut().find(|kept| kept.at == vtable) {
            Some(kept) if kept.holds > 0 => kept.holds += 1,
            Some(kept) => {
                let was = set_slot(vtable, ours())?;
                if was != ours() {
                    // SAFETY: the slot held `CreateCommandQueue`'s method, of that signature.
                    kept.original =
                        unsafe { std::mem::transmute::<usize, CreateCommandQueue>(was) };
                }
                kept.holds = 1;
            }
            None => {
                let was = set_slot(vtable, ours())?;
                // SAFETY: as above; this vtable was not set before, so the slot held its own.
                let original = unsafe { std::mem::transmute::<usize, CreateCommandQueue>(was) };
                state.vtables.push(Vtable {
                    at: vtable,
                    original,
                    holds: 1,
                });
            }
        }
        match state
            .devices
            .iter_mut()
            .find(|held| held.identity == identity)
        {
            Some(held) => held.holds += 1,
            None => state.devices.push(Held {
                identity,
                holds: 1,
                queues: Vec::new(),
            }),
        }
        Ok(Hold { identity, vtable })
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        let released;
        {
            let mut state = STATE.lock();
            let identity = self.identity;
            let Some(at) = state
                .devices
                .iter()
                .position(|held| held.identity == identity)
            else {
                // Every hold's device is held while the hold lives.
                return;
            };
            let held = &mut state.devices[at];
            held.holds -= 1;
            released = (held.holds == 0).then(|| state.devices.swap_remove(at));
            let vtable = self.vtable;
            if let Some(kept) = state.vtables.iter_mut().find(|kept| kept.at == vtable) {
                kept.holds -= 1;
                if kept.holds == 0 && slot_of(vtable).load(Ordering::SeqCst) == ours() {
                    // Set back only where the slot still holds ours: something that set it after
                    // us calls ours, which passes through.
                    let _ = set_slot(vtable, kept.original as usize);
                }
            }
        }
        // The queues are released outside the lock: a release calls into the runtime.
        drop(released);
    }
}

/// `ID3D12Device::CreateCommandQueue` on a device whose vtable the module set: its own method,
/// but for a video decode queue on a held device, which gets the device's one.
///
/// # Safety
///
/// The arguments are `CreateCommandQueue`'s: a live device, and the description, IID and out
/// pointer its caller gives it.
unsafe extern "system" fn create_command_queue(
    device: *mut c_void,
    desc: *const CommandQueueDesc,
    iid: *const Guid,
    out: *mut *mut c_void,
) -> Hresult {
    let vtable = vtable_of(device);
    let mut state = STATE.lock();
    let Some(original) = state
        .vtables
        .iter()
        .find(|kept| kept.at == vtable)
        .map(|kept| kept.original)
    else {
        // The slot is set only in vtables the module keeps.
        return E_FAIL;
    };
    // SAFETY: `desc` is the caller's description, read where it is not null.
    let key = match unsafe { desc.as_ref() } {
        Some(desc) if desc.type_ == VIDEO_DECODE => (desc.priority, desc.flags, desc.node_mask),
        _ => {
            drop(state);
            // SAFETY: the slot's own method, given the caller's arguments.
            return unsafe { original(device, desc, iid, out) };
        }
    };
    let Ok(identity) = identity_of(device) else {
        drop(state);
        // SAFETY: as above.
        return unsafe { original(device, desc, iid, out) };
    };
    let Some(held) = state
        .devices
        .iter_mut()
        .find(|held| held.identity == identity)
    else {
        drop(state);
        // SAFETY: as above.
        return unsafe { original(device, desc, iid, out) };
    };
    let at = match held.queues.iter().position(|(made, _)| *made == key) {
        Some(at) => at,
        None => {
            let mut made = null_mut();
            // SAFETY: the slot's own method, given the caller's description, for the queue's
            // ID3D12CommandQueue, which the module keeps.
            let result = unsafe { original(device, desc, &IID_ID3D12_COMMAND_QUEUE, &mut made) };
            let Ok(queue) = Com::made("CreateCommandQueue (video decode)", result, made) else {
                return result;
            };
            held.queues.push((key, queue));
            held.queues.len() - 1
        }
    };
    let queue = &held.queues[at].1;
    // SAFETY: `QueryInterface` is every COM object's method 0, given the caller's IID and out
    // pointer: it adds the reference the caller is given.
    unsafe {
        let query: QueryInterface = queue.method(0);
        query(queue.raw(), iid, out)
    }
}

/// Makes a command queue of `type_` on `device` through its vtable, as Mesa does: its identity.
#[cfg(test)]
fn queue_identity(device: &Com, type_: u32) -> (Com, usize) {
    let desc = CommandQueueDesc {
        type_,
        priority: 0,
        flags: 0,
        node_mask: 0,
    };
    let mut out = null_mut();
    // SAFETY: the method is ID3D12Device's `CreateCommandQueue`, given a valid description and
    // ID3D12CommandQueue's IID.
    let result = unsafe {
        let create: CreateCommandQueue = device.method(DEVICE_CREATE_COMMAND_QUEUE);
        create(device.raw(), &desc, &IID_ID3D12_COMMAND_QUEUE, &mut out)
    };
    let queue = Com::made("CreateCommandQueue", result, out).expect("a queue");
    let identity = identity_of(queue.raw()).expect("an identity");
    (queue, identity)
}

/// A device of ours answers two asks for a video decode queue with one queue, and two asks for a
/// direct queue with two; let go of, it answers the next video decode asks with a queue each.
#[test]
#[ignore = "needs a Direct3D 12 device with video decode: run on the Windows host"]
fn a_held_devices_video_decode_queues_are_one_and_its_direct_queues_its_own() {
    const DIRECT: u32 = 0;
    let device = super::d3d12::Device::new(None).expect("a Direct3D 12 device");
    let (a, first) = queue_identity(&device.device, VIDEO_DECODE);
    let (b, second) = queue_identity(&device.device, VIDEO_DECODE);
    assert_eq!(
        first, second,
        "two video decode queues on a held device are one"
    );
    let (c, direct) = queue_identity(&device.device, DIRECT);
    let (d, other) = queue_identity(&device.device, DIRECT);
    assert_ne!(direct, other, "two direct queues are their own");
    drop((a, b, c, d));
    // SAFETY: `AddRef` is every COM object's method 1: the reference added is `unheld`'s.
    unsafe {
        let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = device.device.method(1);
        add_ref(device.device.raw());
    }
    let unheld = Com::made("the device", 0, device.device.raw()).expect("the device");
    drop(device);
    let (e, first) = queue_identity(&unheld, VIDEO_DECODE);
    let (f, second) = queue_identity(&unheld, VIDEO_DECODE);
    assert_ne!(
        first, second,
        "a device let go of answers a video decode queue each"
    );
    drop((e, f));
}
