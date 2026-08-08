//! Global MTLDevice, per-thread command queues, native color format TLS, host buffers.
//!
//! `new_buffer_from_host` is the only `newBufferWithBytesNoCopy` left. Every
//! caller it has today passes **host** allocations — the CPU-staged
//! vertex/fragment/compute byte vectors this crate owns.
//!
//! # Aliasing guest RAM here is permitted, and is how this arm reaches the rail
//!
//! `newBufferWithBytesNoCopy` is what MoltenVK implements
//! `VK_EXT_external_memory_host` *over*, so it is the Metal-direct spelling of
//! the one primitive that spans Linux, Windows and macOS. Guest RAM reaching
//! the GPU through it is the intended design, not a violation: see
//! [`crate::runtime::guest_ram`], whose `GuestRamImport`/`GuestSlice` pair is
//! the bound, sized to one RAMBlock with a single bounds-checking constructor.
//!
//! An earlier type-11 attachment cache did alias guest pages here
//! (`mach_vm_remap` view → no-copy MTLBuffer → linear texture view) and was
//! deleted. It is worth being exact about why, because the reason is not the
//! aliasing: that cache retained a remap view per fragmented map until
//! teardown, leaking a VA reservation each time. A RAMBlock-wide import has no
//! remap view to retain.
//!
//! **No importer is wired here yet.** Nothing in this module turns a
//! `GuestSlice` into an `MTLBuffer`, because no caller would use one — the
//! staged `Vec` is filled from guest RAM further up, and the seam worth cutting
//! is there rather than here. Landing an importer without that consumer would
//! be untestable dead code on an arm no Linux host can run.

use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::cell::RefCell;

use crate::backend::metal::util::Status;

static DEVICE: OnceCell<Device> = OnceCell::new();
static DEFAULT_SAMPLER: Mutex<Option<metal::SamplerState>> = Mutex::new(None);
thread_local! {
    static QUEUE: RefCell<Option<CommandQueue>> = const { RefCell::new(None) };
}

pub fn system_device() -> Option<&'static Device> {
    DEVICE
        .get_or_try_init(|| Device::system_default().ok_or(()))
        .ok()
}

/// The selected `MTLDevice`'s name.
///
/// Gated with the test that reads it, because **no product path on this arm
/// reports which GPU was selected.** The Vulkan arm emits it in its `vk_caps`
/// line (`caps::Snapshot::selection_line`); this arm has the string available
/// and never says it, so a Metal boot's log cannot answer "which device did we
/// pick" at all. That is an observability gap on a pathway no Linux host can
/// boot — recorded here rather than filled, since adding an emission is a
/// behaviour change that would land unverified.
#[cfg(test)]
pub fn system_device_name() -> Option<String> {
    system_device().map(|d| d.name().to_string())
}

/// Per-worker-thread command queue (never one shared process-global queue).
pub fn thread_queue(device: &Device) -> CommandQueue {
    QUEUE.with(|q| {
        if let Some(existing) = q.borrow().as_ref() {
            return existing.clone();
        }
        let queue = device.new_command_queue();
        *q.borrow_mut() = Some(queue.clone());
        queue
    })
}

/// Apple Silicon's virtual page. `newBufferWithBytesNoCopy` measures both the
/// base pointer and the length against it and returns nil for anything else,
/// which [`new_buffer_from_host`] reports as `None`.
///
/// Spelled out rather than read from `sysconf` because this is a compile-time
/// claim about the *guest* geometry below, and `sysconf` cannot be one. The
/// runtime path still asks `sysconf` for the page it actually aligns against —
/// that is the host's answer for the host's pointer, and it stays a query.
const APPLE_SILICON_PAGE: u32 = 16 * 1024;

/// A guest arm64e page satisfies Metal's no-copy alignment on its own.
///
/// This is the fact that makes the Metal-direct arm of the guest-RAM import
/// cheap: an offset that is guest-page-aligned is already Apple-page-aligned,
/// so nothing has to round a `GuestSlice` outward to bind it, and no rounding
/// means no chance of a bound that reaches past the RAMBlock.
///
/// Two independently-derived values, which is the only thing a `const`
/// assertion is worth: the left comes from `reims-vgpu-wire`'s `ARM64E`
/// descriptor, decoded from Apple's own page-table geometry; the right is the
/// Apple Silicon platform page. Neither is written in terms of the other, so if
/// either moves — a guest page shift that is no longer 14, or an Apple page
/// that is no longer 16 KiB — this stops compiling instead of silently handing
/// Metal a base it will refuse.
///
/// A `#[test]` cannot hold this. Nothing under `backend/metal/` runs its tests
/// on a non-Apple host; they are `cfg`-ed out and the green count does not say
/// so. `rustc` evaluates a `const` assertion on every arm that *compiles* the
/// file, including the cross-compiled `--target aarch64-apple-darwin` clippy
/// run, so this one is checked from Linux and a `#[test]` beside it would not
/// be.
const _: () = assert!(
    crate::contract::gva::PAGE_SIZE_ARM64E.is_multiple_of(APPLE_SILICON_PAGE),
    "a guest arm64e page must be a whole number of Apple Silicon pages, or a \
     guest-page-aligned base is not a valid newBufferWithBytesNoCopy base"
);

fn no_copy_buffer_length_status(requested_len: usize, actual_len: u64) -> Status {
    if actual_len == requested_len as u64 {
        Status::OK
    } else {
        Status::execute("metal_buffer_no_copy_length_mismatch")
            .field("requested_len", requested_len)
            .field("actual_len", actual_len)
    }
}

/// Prefer no-copy when host pointer+length are page-aligned (Metal contract).
/// Fall back to a copy otherwise. Caller owns host bytes for command-buffer lifetime.
pub fn new_buffer_from_host(device: &Device, data: *const u8, len: usize) -> Option<Buffer> {
    if data.is_null() || len == 0 {
        return None;
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    let addr = data as usize;
    if page != 0 && addr.is_multiple_of(page) && len.is_multiple_of(page) {
        // A nil here is the device refusing the allocation, and it arrives as
        // `None` rather than as a `Buffer`. This used to read
        // `device.new_buffer_with_bytes_no_copy(..)`, whose return type is
        // `metal::Buffer` — a `NonNull` — so the nil became an invalid value
        // before anything could test it, and the comment that stood here said a
        // null Metal result "behaves as a zero-length Objective-C receiver". The
        // *messaging* does; the Rust wrapper does not, and `.length()` returning
        // zero was reading a value that already had no legal representation.
        let allocated = unsafe {
            crate::backend::metal::raw_metal::new_buffer_no_copy(
                device,
                data as *mut _,
                len as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        if let Some(buf) = allocated {
            let actual_len = buf.length();
            let status = no_copy_buffer_length_status(len, actual_len);
            if status.is_ok() {
                return Some(buf);
            }
            // A length Metal did not honour is not an allocation failure, so the
            // copy fallback below is still correct. Make the performance
            // degradation visible once per rejected requested/actual pair.
            if let Some(emit) = crate::observe::Emit::refusal("metal_buffer_copy_fallback", &status)
            {
                emit.fail_once((len as u64) ^ actual_len.rotate_left(32));
            }
        }
    }
    // The copying constructor can refuse too, and for the reason that matters
    // most: it is the one that has to find `len` bytes. `None` reaches the
    // caller as a refusal instead of a buffer that is not one.
    unsafe {
        crate::backend::metal::raw_metal::new_buffer_with_data(
            device,
            data as *const _,
            len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

pub fn cached_default_sampler(device: &Device) -> metal::SamplerState {
    let mut guard = DEFAULT_SAMPLER.lock();
    if let Some(s) = guard.as_ref() {
        return s.clone();
    }
    let desc = metal::SamplerDescriptor::new();
    desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
    desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
    desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
    desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
    desc.set_normalized_coordinates(true);
    let sampler = device.new_sampler(&desc);
    *guard = Some(sampler.clone());
    sampler
}

#[cfg(test)]
mod no_copy_buffer_tests {
    use super::*;
    use crate::observe::{Emit, Refusal as _};

    #[test]
    fn rejected_no_copy_buffer_names_the_copy_fallback() {
        let status = no_copy_buffer_length_status(0x4000, 0);
        assert_eq!(
            status.refusal(),
            Some("metal_buffer_no_copy_length_mismatch")
        );
        assert_eq!(
            Emit::refusal("metal_buffer_copy_fallback", &status)
                .expect("length mismatch must be a refusal")
                .render(),
            "metal_buffer_copy_fallback reason=metal_buffer_no_copy_length_mismatch \
             class=execute requested_len=16384 actual_len=0"
        );
    }
}
