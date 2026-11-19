//! The sampler IDs of the pipeline states made on metal2vulkan-rt.
//!
//! rt's creation of a pipeline state (`Device::new_compute_pipeline_state`,
//! `Device::new_render_pipeline_state`) lowers the state's functions, then hands its consumer the
//! samplers the state numbers, in Metal's order (`NumberedSampler`: each function's constexpr
//! samplers, then `Read` where it reads a texture and `Write` where it writes one; a render
//! state's vertex function's, then its fragment function's), and takes one sampler ID for each.
//! A constexpr sampler's ID is to be an index of a live `IndexClass::Sampler` grant, where rt
//! places its device samplers; rt answers `Refusal::IndicesNotGranted` for any other. A read's
//! and a write's take nothing at their IDs.
//!
//! [`SamplerIds`] grants rt the device's whole sampler range, [`FIRST_SAMPLER_ID`] on, once, and
//! numbers each state's samplers ([`Numbering`]): a sampler equal to one a live state holds takes
//! its ID, as Metal's numbering does (the translator's `NumberedSampler`: "unless an equal one of
//! the device holds one"), and any other the lowest ID no sampler holds. A state's IDs go back
//! when its [`Numbered`] handle goes, after rt's state; a creation that fails gives back what it
//! took. The handle is to live as long as the guest's object, which the guest's command buffers
//! retain while their work runs.
//!
//! Beside a state's samplers, Metal numbers what metal2vulkan-runtime's numbering models
//! (`crate::gpu::states`, `metal2vulkan-runtime`'s `resource_ids`), which this does not: its
//! plain allocation from the smallest free range of a pool that doubles, the device's own
//! sampler ID, the sampler states' IDs from the same range, and the function objects and
//! libraries that hold a state's IDs beside it.

use std::collections::{BTreeSet, HashMap};
use std::mem::ManuallyDrop;
use std::sync::Arc;

use metal2vulkan_rt::{
    ComputePipelineDescriptor, ComputePipelineState, Device, Error, IndexClass, IndexGrant,
    NumberedSampler, Refusal, RenderPipelineDescriptor, RenderPipelineState,
};
use parking_lot::Mutex;

/// The first sampler ID a pipeline state's sampler takes: 0 is nil.
pub const FIRST_SAMPLER_ID: u32 = 1;

/// How many sampler IDs the pipeline states' samplers take, from [`FIRST_SAMPLER_ID`]: 1 to
/// 3,999, rt's whole `IndexClass::Sampler` range (`IndexClass::range`).
pub const SAMPLER_ID_COUNT: u32 = 3999;

/// A pipeline state's creation that failed, holding no sampler ID.
#[derive(Debug)]
pub enum PipelineError {
    /// rt's answer.
    Runtime(Error),
    /// The state numbers a sampler that takes an ID of its own while live states hold every one.
    SamplerIds,
}

/// The sampler IDs of one rt device's pipeline states: the device's grant of its sampler range,
/// and the numbering of the states made through it.
pub struct SamplerIds {
    device: Device,
    grant: IndexGrant,
    numbering: Arc<Mutex<Numbering>>,
}

impl SamplerIds {
    /// Grants `device`'s sampler range, [`SAMPLER_ID_COUNT`] IDs from [`FIRST_SAMPLER_ID`], for
    /// the device's life (rt's grants go with its device), with no sampler numbered.
    ///
    /// Err: rt's refusal of the grant: `Refusal::IndicesGranted` where any ID of the range is
    /// granted already, as for a second numbering of one device.
    pub fn new(device: &Device) -> Result<SamplerIds, Error> {
        let grant =
            device.grant_indices(IndexClass::Sampler, FIRST_SAMPLER_ID, SAMPLER_ID_COUNT)?;
        Ok(SamplerIds {
            device: device.clone(),
            grant,
            numbering: Arc::new(Mutex::new(Numbering::new())),
        })
    }

    /// The device's grant.
    pub fn grant(&self) -> IndexGrant {
        self.grant
    }

    /// How many IDs live states' samplers hold.
    pub fn held(&self) -> usize {
        self.numbering.lock().held.len()
    }

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` on rt, without
    /// reflection: rt's state of `descriptor`, its samplers numbered here.
    ///
    /// Err: [`PipelineError::SamplerIds`]; rt's answer ([`PipelineError::Runtime`]).
    pub fn new_compute_pipeline_state(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> Result<Numbered<ComputePipelineState>, PipelineError> {
        let mut taken = Taken::Nothing;
        let made = (self.device)
            .new_compute_pipeline_state(descriptor, |samplers| self.take(samplers, &mut taken));
        self.numbered(made, taken)
    }

    /// `newRenderPipelineStateWithDescriptor:error:` on rt: rt's state of `descriptor`, its
    /// vertex function's samplers and then its fragment function's numbered here.
    ///
    /// Err: [`PipelineError::SamplerIds`]; rt's answer ([`PipelineError::Runtime`]).
    pub fn new_render_pipeline_state(
        &self,
        descriptor: &RenderPipelineDescriptor,
    ) -> Result<Numbered<RenderPipelineState>, PipelineError> {
        let mut taken = Taken::Nothing;
        let made = (self.device)
            .new_render_pipeline_state(descriptor, |samplers| self.take(samplers, &mut taken));
        self.numbered(made, taken)
    }

    /// rt's `sampler_ids`: an ID for each of `samplers`, held by the lease left in `taken`. Where
    /// none is free for one, it holds nothing, says so in `taken`, and answers rt what rt answers
    /// an ID past the grant.
    fn take(&self, samplers: &[NumberedSampler], taken: &mut Taken) -> Result<Vec<u32>, Error> {
        let ids = self.numbering.lock().take(samplers);
        match ids {
            Some(ids) => {
                *taken = Taken::Ids(Lease {
                    numbering: Arc::clone(&self.numbering),
                    samplers: samplers.to_vec(),
                    ids: ids.clone(),
                });
                Ok(ids)
            }
            None => {
                *taken = Taken::Exhausted;
                Err(Error::Refused(Refusal::IndicesNotGranted {
                    class: IndexClass::Sampler,
                    index: FIRST_SAMPLER_ID.saturating_add(SAMPLER_ID_COUNT),
                }))
            }
        }
    }

    /// The handle of a state rt `made`, or its error, with the IDs `taken` for it: a failed
    /// creation drops them, as rt holds none of them then.
    fn numbered<S>(
        &self,
        made: Result<S, Error>,
        taken: Taken,
    ) -> Result<Numbered<S>, PipelineError> {
        let lease = match taken {
            Taken::Exhausted => return Err(PipelineError::SamplerIds),
            Taken::Ids(lease) => lease,
            Taken::Nothing => Lease {
                numbering: Arc::clone(&self.numbering),
                samplers: Vec::new(),
                ids: Vec::new(),
            },
        };
        match made {
            Ok(state) => Ok(Numbered {
                state: ManuallyDrop::new(state),
                lease,
            }),
            Err(error) => Err(PipelineError::Runtime(error)),
        }
    }
}

/// What a creation's `sampler_ids` took.
enum Taken {
    /// rt did not ask: the creation failed before it numbered the samplers.
    Nothing,
    /// The IDs it answered.
    Ids(Lease),
    /// No ID was free for a sampler, and it took none.
    Exhausted,
}

/// A pipeline state rt made, with the sampler IDs its samplers hold.
///
/// Dropping it drops rt's state, which gives up a constexpr sampler's device samplers at its ID
/// as the last state numbering it there goes, and then gives the IDs back, so no sampler takes
/// an ID before rt lets go of it.
pub struct Numbered<S> {
    state: ManuallyDrop<S>,
    lease: Lease,
}

impl<S> Numbered<S> {
    /// rt's state. A clone of it is not to outlive this handle, which frees the state's IDs.
    pub fn state(&self) -> &S {
        &self.state
    }

    /// The samplers the state numbers, in rt's order.
    pub fn samplers(&self) -> &[NumberedSampler] {
        &self.lease.samplers
    }

    /// The ID of each of [`samplers`](Self::samplers).
    pub fn sampler_ids(&self) -> &[u32] {
        &self.lease.ids
    }
}

impl<S> Drop for Numbered<S> {
    fn drop(&mut self) {
        // SAFETY: the state is dropped here, once, and nothing reads it after.
        unsafe { ManuallyDrop::drop(&mut self.state) };
        self.lease.give_back();
    }
}

/// The IDs one state's samplers hold, given back to the numbering once.
struct Lease {
    numbering: Arc<Mutex<Numbering>>,
    /// The samplers that hold them; empty once given back.
    samplers: Vec<NumberedSampler>,
    ids: Vec<u32>,
}

impl Lease {
    fn give_back(&mut self) {
        let samplers = std::mem::take(&mut self.samplers);
        if !samplers.is_empty() {
            self.numbering.lock().give_back(&samplers);
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.give_back();
    }
}

/// The IDs of the samplers live states hold.
struct Numbering {
    /// The IDs no sampler holds.
    free: BTreeSet<u32>,
    /// Each sampler a live state holds: its ID, and how many of the live states' samplers hold
    /// it.
    held: HashMap<NumberedSampler, Held>,
}

/// A sampler's ID and its holders.
struct Held {
    id: u32,
    holders: usize,
}

impl Numbering {
    /// Every ID free.
    fn new() -> Numbering {
        Numbering {
            free: (FIRST_SAMPLER_ID..FIRST_SAMPLER_ID + SAMPLER_ID_COUNT).collect(),
            held: HashMap::new(),
        }
    }

    /// An ID for each of `samplers`, in their order, held until [`give_back`](Self::give_back)
    /// of the same samplers: a sampler equal to one held takes its ID, and any other the lowest
    /// free. `None`, holding nothing, where one needs an ID of its own and none is free.
    fn take(&mut self, samplers: &[NumberedSampler]) -> Option<Vec<u32>> {
        let mut ids = Vec::with_capacity(samplers.len());
        for (at, sampler) in samplers.iter().enumerate() {
            let id = match self.held.get_mut(sampler) {
                Some(held) => {
                    held.holders = held.holders.saturating_add(1);
                    held.id
                }
                None => {
                    let Some(id) = self.free.pop_first() else {
                        self.give_back(samplers.get(..at).unwrap_or_default());
                        return None;
                    };
                    self.held.insert(*sampler, Held { id, holders: 1 });
                    id
                }
            };
            ids.push(id);
        }
        Some(ids)
    }

    /// Gives back one hold of each of `samplers`: an ID whose last holder goes is free.
    fn give_back(&mut self, samplers: &[NumberedSampler]) {
        for sampler in samplers {
            let Some(held) = self.held.get_mut(sampler) else {
                continue;
            };
            held.holders = held.holders.saturating_sub(1);
            if held.holders == 0 {
                let id = held.id;
                self.held.remove(sampler);
                self.free.insert(id);
            }
        }
    }
}

#[cfg(test)]
mod tests;
