//! The M2 scaler device's host side (virtio-PCI 106b:1a0b), apart from its
//! transport: each request the guest's AppleM2ScalerParavirtDriver puts on
//! the device's one queue, checked, answered and transformed as Apple's
//! `AvpM2ScalerDevice` does.
//!
//! The transport hands [`ScalerDevice::process`] each element's readable
//! bytes, the size of its writable part and the guest's memory, and does
//! what the [`Outcome`] says: write the reply, set DEVICE_NEEDS_RESET, or
//! end the VM. [`ScalerDevice::start`] is DRIVER_OK, which makes the
//! backend's accelerator; [`ScalerDevice::reset`] drops it.
//!
//! - [`request`] lays out the requests; [`capabilities`] the 0x288 bytes a
//!   type 1 request fills; [`plan`] resolves a transform against its two
//!   surfaces, which the device holds by mapping ID through the same
//!   [`Surfaces`] the VideoToolbox device uses.
//! - [`backend`] is what transforms; [`reference`](mod@reference) is a
//!   CPU scaler for the tests and for checking a backend's output.

pub mod backend;
pub mod capabilities;
pub mod coverage;
pub mod format;
pub mod plan;
pub mod reference;
pub mod request;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use crate::device_wait::GuestHold;
use crate::num::{ToU64, ToUsize};
use crate::scaler::backend::{Accelerator, ScalerBackend, TransformJob};
use crate::scaler::plan::Plan;
use crate::scaler::request::{ScatterGatherList, SideBufferPair, Transform};
use crate::surfaces::{GuestSurface, MappingId, Surfaces};
use crate::sync::Mutex;

/// IOKit's return codes the scaler answers with, from the SDK's
/// `IOKit/IOReturn.h` (`iokit_common_err`).
pub mod io_return {
    /// An `IOReturn`, as the request's status word carries it.
    pub type IoReturn = u32;
    /// `kIOReturnSuccess`.
    pub const SUCCESS: IoReturn = 0;
    /// `kIOReturnError`.
    pub const ERROR: IoReturn = 0xe000_02bc;
    /// `kIOReturnNoMemory`.
    pub const NO_MEMORY: IoReturn = 0xe000_02bd;
    /// `kIOReturnBadArgument`.
    pub const BAD_ARGUMENT: IoReturn = 0xe000_02c2;
    /// `kIOReturnUnsupported`: what Apple's scaler answers a format it
    /// does not convert.
    pub const UNSUPPORTED: IoReturn = 0xe000_02c7;
    /// `kIOReturnNotReadable`.
    pub const NOT_READABLE: IoReturn = 0xe000_02ce;
    /// `kIOReturnNotWritable`.
    pub const NOT_WRITABLE: IoReturn = 0xe000_02cf;
    /// `kIOReturnNotFound`: a mapping ID with no surface. Ours; the guest
    /// kext answers its own failed surface lookups so.
    pub const NOT_FOUND: IoReturn = 0xe000_02f0;
}

/// The guest's memory, as the device reaches it for the capabilities and
/// the side buffers' scatter-gather lists: guest-physical addresses.
pub trait GuestMemory {
    /// Whether all of `[address, address + length)` is guest RAM, as
    /// Apple's `get_memory` asks (a length of 0: the address alone).
    fn is_ram(&self, address: u64, length: u64) -> bool;

    /// Reads `bytes.len()` bytes at `address`; false, with nothing read,
    /// unless all of them are RAM.
    fn read(&self, address: u64, bytes: &mut [u8]) -> bool;

    /// Writes `bytes` at `address`; false, with nothing written, unless
    /// all of them are RAM.
    fn write(&self, address: u64, bytes: &[u8]) -> bool;
}

/// What the transport does with an element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Write these bytes, the request rewritten, into the writable part,
    /// with a used length of their size.
    Reply(Vec<u8>),
    /// Set DEVICE_NEEDS_RESET and raise the configuration interrupt, with
    /// no reply; the reason is Apple's log's.
    NeedsReset(&'static str),
    /// End the VM, as Apple's host ends its process.
    EndVm(&'static str),
}

/// The largest element the device takes, its readable part: ours. Apple's
/// host copies a readable part of any size into a buffer of its own before
/// it checks one, so a guest could have the host allocate and copy
/// gigabytes; the guest kext's largest request is a transform's 0x608
/// bytes. Past this, the element is a needs-reset, and a transport checks
/// it before it copies the element.
pub const MAX_REQUEST_SIZE: usize = 0x1_0000;

/// The largest inbound side buffer.
pub const MAX_INPUT_SIZE: u32 = 0xbd10;
/// The highest inbound kind.
pub const MAX_INPUT_KIND: i32 = 6;
/// The largest outbound side buffer.
pub const MAX_OUTPUT_SIZE: u32 = 0x1008;
/// The highest outbound kind.
pub const MAX_OUTPUT_KIND: i32 = 3;

/// Why a transform request fails as a whole, before or after the
/// transform: Apple's needs-reset, or its abort.
enum Refusal {
    Reset,
    End,
}

/// A transform's two surfaces, held: what a backend whose device has not
/// finished writing the destination by its deadline keeps
/// ([`TransformJob::hold`]), so that the guest sees both in use until the
/// device is done.
struct SurfacesHeld {
    _source: Arc<dyn GuestSurface>,
    _destination: Arc<dyn GuestSurface>,
}

impl SurfacesHeld {
    /// A hold on `source` and `destination`, while it lives.
    fn hold(source: &Arc<dyn GuestSurface>, destination: &Arc<dyn GuestSurface>) -> GuestHold {
        GuestHold::new(Arc::new(SurfacesHeld {
            _source: Arc::clone(source),
            _destination: Arc::clone(destination),
        }))
    }
}

/// The scaler device's host side (module doc).
pub struct ScalerDevice {
    surfaces: Arc<dyn Surfaces>,
    backend: Arc<dyn ScalerBackend>,
    /// From DRIVER_OK to the reset.
    accelerator: Mutex<Option<Box<dyn Accelerator>>>,
}

impl ScalerDevice {
    /// A device whose surfaces are reached through `surfaces` and whose
    /// transforms run on `backend`'s accelerators.
    pub fn new(surfaces: Arc<dyn Surfaces>, backend: Arc<dyn ScalerBackend>) -> ScalerDevice {
        ScalerDevice {
            surfaces,
            backend,
            accelerator: Mutex::new(None),
        }
    }

    /// The guest's DRIVER_OK: a new accelerator. A failure is a
    /// needs-reset, as Apple's DRIVER_OK hook makes one.
    pub fn start(&self) -> Result<(), &'static str> {
        let made = self.backend.create_accelerator();
        let mut accelerator = self.accelerator.lock();
        match made {
            Ok(made) => {
                *accelerator = Some(made);
                Ok(())
            }
            Err(why) => {
                log::warn!("scaler: no accelerator: {why}");
                *accelerator = None;
                Err("IOSurfaceAcceleratorCreate failed")
            }
        }
    }

    /// The device's reset or stop: the accelerator goes.
    pub fn reset(&self) {
        self.accelerator.lock().take();
    }

    /// One element: `request` its readable bytes, `writable` the size of
    /// its writable part.
    pub fn process(&self, request: &[u8], writable: usize, memory: &dyn GuestMemory) -> Outcome {
        let n = request.len();
        if n == 0 {
            return Outcome::NeedsReset("Invalid buffer size");
        }
        if n > MAX_REQUEST_SIZE {
            // Ours (MAX_REQUEST_SIZE), as is the reason.
            return Outcome::NeedsReset("Request too large");
        }
        if writable < n {
            return Outcome::NeedsReset("Write buffer too small");
        }
        if n < 8 {
            return Outcome::NeedsReset("Invalid header");
        }
        let mut r = request.to_vec();
        match request::u32_at(&r, request::TYPE) {
            request::TYPE_TRANSFORM => {
                if n != request::TRANSFORM_SIZE {
                    return Outcome::NeedsReset("Invalid transform request");
                }
                match self.transform(&mut r, memory) {
                    Ok(()) => {}
                    Err(Refusal::Reset) => return Outcome::NeedsReset("Invalid transform request"),
                    Err(Refusal::End) => return Outcome::EndVm("Security Assertion Hit: "),
                }
            }
            request::TYPE_CAPABILITIES => {
                if n != request::CAPABILITIES_SIZE {
                    return Outcome::NeedsReset("Invalid capabilities request");
                }
                if let Err(reason) = self.capabilities(&mut r, memory) {
                    return Outcome::NeedsReset(reason);
                }
            }
            other => log::warn!("scaler: Invalid payload type: {other}"),
        }
        Outcome::Reply(r)
    }

    /// Type 1: the capabilities, written at `R+8`, `R+0x18` bytes of room.
    /// The bytes written are the host's capabilities' or the room's,
    /// whichever is fewer, and their count goes back at `R+0x18`, as
    /// Apple's scaler answers its selector 11.
    fn capabilities(&self, r: &mut [u8], memory: &dyn GuestMemory) -> Result<(), &'static str> {
        let address = request::u64_at(r, request::CAPABILITIES_ADDRESS);
        let length = request::u64_at(r, request::CAPABILITIES_LENGTH);
        if !memory.is_ram(address, length) {
            log::warn!("scaler: Invalid capabilities address {address:#x}, {length:#x} bytes");
            return Err("Invalid capabilities request");
        }
        let accelerator = self.accelerator.lock();
        let Some(accelerator) = accelerator.as_ref() else {
            // Apple's host calls its null accelerator here and dies: ours
            // resets.
            log::warn!("scaler: a capabilities request before DRIVER_OK");
            return Err("Invalid capabilities request: no accelerator");
        };
        let caps = accelerator.capabilities().to_bytes();
        let count = caps
            .len()
            .min(usize::try_from(length).unwrap_or(usize::MAX));
        if !memory.write(address, caps.get(..count).unwrap_or_default()) {
            return Err("Invalid capabilities request");
        }
        request::put_u64(r, request::CAPABILITIES_LENGTH, count.to_u64());
        request::put_u32(r, request::STATUS, io_return::SUCCESS);
        log::debug!("scaler: capabilities, {count:#x} bytes at {address:#x}");
        Ok(())
    }

    /// Type 2, in Apple's order: the two surfaces held, the side buffers
    /// checked and gathered, the transform, the side buffers scattered back.
    ///
    /// The surfaces are retained outside the accelerator's lock, between a
    /// check under it that there is an accelerator and a second once both
    /// are held, and let go outside it once the request is answered: over
    /// reims-vgpu's mapper a retain waits for the IOSurface device's queue,
    /// whose interrupt the C API lets the host answer with a call on this
    /// device, which takes the lock. A reset that lands between the two
    /// checks makes the transform a needs-reset, as a transform after the
    /// reset is; a start, a transform on its new accelerator.
    fn transform(&self, r: &mut [u8], memory: &dyn GuestMemory) -> Result<(), Refusal> {
        if self.accelerator.lock().is_none() {
            log::warn!("scaler: a transform before DRIVER_OK");
            return Err(Refusal::Reset);
        }
        let source_id = MappingId(request::u64_at(r, request::SOURCE_MAPPING_ID));
        let destination_id = MappingId(request::u64_at(r, request::DESTINATION_MAPPING_ID));
        let hold = |id: MappingId| match self.surfaces.retain(id) {
            Ok(surface) => Some(Arc::<dyn GuestSurface>::from(surface)),
            Err(error) => {
                log::warn!("scaler: surface {:#x}: {error}", id.0);
                None
            }
        };
        let source = hold(source_id);
        let destination = hold(destination_id);
        let answered = self.transform_held(
            r,
            memory,
            (source_id, source.as_ref()),
            (destination_id, destination.as_ref()),
        );
        drop(destination);
        drop(source);
        answered
    }

    /// The rest of a transform, under the accelerator's lock, its source
    /// and destination held: `None` for one with no surface.
    fn transform_held(
        &self,
        r: &mut [u8],
        memory: &dyn GuestMemory,
        (source_id, source): (MappingId, Option<&Arc<dyn GuestSurface>>),
        (destination_id, destination): (MappingId, Option<&Arc<dyn GuestSurface>>),
    ) -> Result<(), Refusal> {
        let mut accelerator = self.accelerator.lock();
        let Some(accelerator) = accelerator.as_mut() else {
            log::warn!("scaler: a transform whose device was reset while it held its surfaces");
            return Err(Refusal::Reset);
        };

        let used = r.get(request::SIDE_BUFFERS_USED) == Some(&1);
        let mut inputs: [Option<(i32, Vec<u8>)>; 4] = Default::default();
        let mut outputs: [Option<(i32, Vec<u8>)>; 4] = Default::default();
        if used {
            for (i, (input, output)) in inputs.iter_mut().zip(&mut outputs).enumerate() {
                let pair = SideBufferPair::read(r, i);
                if pair.input_size > MAX_INPUT_SIZE || pair.input_kind > MAX_INPUT_KIND {
                    return Err(Refusal::Reset);
                }
                if pair.input_size != 0 && pair.input_kind != 0 {
                    let mut bytes = vec![0u8; pair.input_size.to_usize()];
                    gather(&ScatterGatherList::read(r, 2 * i), &mut bytes, memory)?;
                    *input = Some((pair.input_kind, bytes));
                }
                if pair.output_size > MAX_OUTPUT_SIZE || pair.output_kind > MAX_OUTPUT_KIND {
                    return Err(Refusal::Reset);
                }
                if pair.output_size != 0 && pair.output_kind != 0 {
                    *output = Some((pair.output_kind, vec![0u8; pair.output_size.to_usize()]));
                }
            }
        }

        request::clear_completion_and_events(r);
        let transform = Transform::read(r);
        let status = match (source, destination) {
            (Some(source), Some(destination)) => {
                match Plan::new(&transform, &source.layout(), &destination.layout()) {
                    Ok(plan) => match plan.ratio_past(accelerator.capabilities().max_scale) {
                        Some(past) => {
                            // Apple's scaler's answer (Plan::ratio_past).
                            log::warn!("scaler: refused, as the transform goes past {past}");
                            io_return::BAD_ARGUMENT
                        }
                        None => {
                            log::debug!(
                                "scaler: transform {:#x} {} {}x{} to {:#x} {} {}x{}: {:?} into {:?}, {:?}, flags {:#x}",
                                source_id.0,
                                plan.source_format.format.fourcc(),
                                plan.source_format.width,
                                plan.source_format.height,
                                destination_id.0,
                                plan.destination_format.format.fourcc(),
                                plan.destination_format.width,
                                plan.destination_format.height,
                                plan.source_rect,
                                plan.destination_rect,
                                plan.orientation,
                                transform.flags.0
                            );
                            let mut job = TransformJob {
                                transform: &transform,
                                plan,
                                source: source.as_ref(),
                                destination: destination.as_ref(),
                                hold: SurfacesHeld::hold(source, destination),
                                inputs: std::mem::take(&mut inputs),
                                outputs: std::mem::take(&mut outputs),
                            };
                            let status = accelerator.transform(&mut job);
                            if status != io_return::SUCCESS {
                                // A device that missed its deadline may still write the
                                // destination: told of once it is done (`GuestHold`).
                                let destination = Arc::clone(destination);
                                job.hold.report_late_write(Box::new(move || {
                                    destination.device_wrote();
                                }));
                            }
                            outputs = std::mem::take(&mut job.outputs);
                            status
                        }
                    },
                    Err(status) => status,
                }
            }
            _ => io_return::NOT_FOUND,
        };
        if status != io_return::SUCCESS {
            log::warn!(
                "scaler: transform {:#x} to {:#x} answered {status:#x}",
                source_id.0,
                destination_id.0
            );
        }
        request::put_u32(r, request::STATUS, status);

        if used {
            for (i, output) in outputs.iter().enumerate() {
                let pair = SideBufferPair::read(r, i);
                if pair.output_kind != 0 && pair.output_size != 0 {
                    let bytes = output
                        .as_ref()
                        .map_or(&[][..], |(_, bytes)| bytes.as_slice());
                    scatter(&ScatterGatherList::read(r, 2 * i + 1), bytes, memory)?;
                }
            }
        }
        Ok(())
    }
}

/// The segments of `list` the guest filled. A count over 8 ends the VM, as
/// it ends Apple's host.
fn segments(list: &ScatterGatherList) -> Result<&[(u64, u32)], Refusal> {
    list.entries
        .get(..list.count.to_usize())
        .ok_or(Refusal::End)
}

/// Gathers `list` into `buffer`, as Apple's host does: a count over 8 ends
/// the VM; a running length over the buffer's, or a segment that is not
/// RAM, fails.
fn gather(
    list: &ScatterGatherList,
    buffer: &mut [u8],
    memory: &dyn GuestMemory,
) -> Result<(), Refusal> {
    let mut offset = 0usize;
    for &(address, length) in segments(list)? {
        let Some(part) = offset
            .checked_add(length.to_usize())
            .and_then(|end| buffer.get_mut(offset..end))
        else {
            return Err(Refusal::Reset);
        };
        if length == 0 {
            if !memory.is_ram(address, 0) {
                return Err(Refusal::Reset);
            }
            continue;
        }
        if !memory.read(address, part) {
            return Err(Refusal::Reset);
        }
        offset += part.len();
    }
    Ok(())
}

/// Scatters `buffer` to `list`, as Apple's host does.
fn scatter(
    list: &ScatterGatherList,
    buffer: &[u8],
    memory: &dyn GuestMemory,
) -> Result<(), Refusal> {
    let mut offset = 0usize;
    for &(address, length) in segments(list)? {
        let Some(part) = offset
            .checked_add(length.to_usize())
            .and_then(|end| buffer.get(offset..end))
        else {
            return Err(Refusal::Reset);
        };
        let ok = if length == 0 {
            memory.is_ram(address, 0)
        } else {
            memory.write(address, part)
        };
        if !ok {
            return Err(Refusal::Reset);
        }
        offset += part.len();
    }
    Ok(())
}
