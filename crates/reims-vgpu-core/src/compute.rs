//! Backend-independent resolved compute commands and their typed results.

use reims_vgpu_memory::{GuestImageSource, GuestPageTarget, GuestRunSource};
use reims_vgpu_protocol::{SampledImageFormat, StorageImageFormat};
pub use reims_vgpu_protocol::{
    SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerFilter, SamplerMipFilter,
};
use std::sync::Arc;

/// One memory dependency declared between dispatches in a compute encoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeBarrier {
    Resources(Arc<[crate::BarrierResource]>),
    Scope(crate::MemoryBarrierScope),
    /// A satisfied `waitForFence:` immediately before this dispatch.
    ///
    /// Unlike a scope barrier, an encoder fence names all writes before the
    /// matching update. Keeping the fence as its own semantic command avoids
    /// inventing a resource class while allowing a backend to realize the
    /// required memory dependency conservatively.
    Fence,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SamplerResource {
    pub binding: u32,
    pub source: SamplerSource,
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub mip_filter: SamplerMipFilter,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub border_color: SamplerBorderColor,
    pub compare_function: SamplerCompareFunction,
    /// Floating-point limits retained as bits so complete sampler state is hashable.
    pub lod_min: u32,
    pub lod_max: u32,
    pub max_anisotropy: u32,
    pub unnormalized_coordinates: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SamplerSource {
    State,
    Null,
}

impl SamplerResource {
    pub fn normalized_default(binding: u32) -> Self {
        Self {
            binding,
            source: SamplerSource::State,
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Linear,
            mip_filter: SamplerMipFilter::NotMipmapped,
            address_mode_u: SamplerAddressMode::ClampToEdge,
            address_mode_v: SamplerAddressMode::ClampToEdge,
            address_mode_w: SamplerAddressMode::ClampToEdge,
            border_color: SamplerBorderColor::TransparentBlack,
            compare_function: SamplerCompareFunction::Never,
            lod_min: 0.0f32.to_bits(),
            lod_max: f32::MAX.to_bits(),
            max_anisotropy: 1,
            unnormalized_coordinates: false,
        }
    }

    pub fn null(binding: u32) -> Self {
        let mut sampler = Self::normalized_default(binding);
        sampler.source = SamplerSource::Null;
        sampler
    }

    pub fn lod_min_f32(&self) -> f32 {
        f32::from_bits(self.lod_min)
    }

    pub fn lod_max_f32(&self) -> f32 {
        f32::from_bits(self.lod_max)
    }
}

/// Fully resolved inputs for one compute dispatch.
///
/// Every guest reference has already been resolved. The executor receives
/// a prepared shader identity, semantic resource descriptions, and bounded
/// memory sources; it never receives backend-native shader words, a wire tag,
/// or an object-table ordinal.
#[derive(Debug, Default)]
pub struct ComputeRequest {
    pub program: crate::PreparedShaderStage,
    pub entry: String,
    /// Workgroup counts **and** the exact Metal thread grid they round up to.
    ///
    /// Both halves are needed to record one dispatch: `vkCmdDispatch` takes the
    /// counts, and the translated entry point culls its surplus invocations
    /// against the thread grid. Carrying the plan rather than a bare `[u32; 3]`
    /// is what stops a backend recording the first without the second — which
    /// is not a missing optimization but the whole of Metal's exact-grid
    /// contract, and was silently unmet for as long as this field was the
    /// quotient alone.
    pub dispatch: reims_vgpu_protocol::dispatch::WorkgroupPlan,
    /// Memory barriers immediately before this dispatch in encoder order.
    pub barriers: Vec<ComputeBarrier>,
    pub storage_buffers: Vec<ComputeBufferResource>,
    pub sampled_images: Vec<ComputeSampledImageResource>,
    pub samplers: Vec<SamplerResource>,
    pub storage_images: Vec<ComputeStorageImageResource>,
}

#[derive(Debug, Default)]
pub struct ComputeOutput {
    /// Writable-buffer results only. Read-only descriptors never cross back.
    pub buffers: Vec<ComputeBufferOutput>,
    /// One result per storage image, in request order.
    pub images: Vec<ComputeImageResult>,
}

/// The one destination selected for a storage image's post-dispatch texels.
///
/// This is a sum type because a host readback and a direct guest-page landing
/// are mutually exclusive execution plans. A request cannot accidentally carry
/// both and leave the executor to choose which guest-visible result wins.
#[derive(Default)]
pub enum ComputeImageDestination {
    #[default]
    Host,
    /// The GPU copy lands in a bounded guest window. `pages` is the matching
    /// guest-physical footprint used by the completion ledger; it cannot be
    /// reconstructed from the host references inside `target`.
    GuestPages {
        target: Box<GuestPageTarget>,
        pages: Vec<u64>,
    },
}

impl std::fmt::Debug for ComputeImageDestination {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host => f.write_str("Host"),
            Self::GuestPages { target, pages } => write!(
                f,
                "GuestPages({}x{}, {} runs, {} pages)",
                target.width,
                target.height,
                target.runs.len(),
                pages.len()
            ),
        }
    }
}

#[derive(Debug)]
pub enum ComputeImageResult {
    /// Tight host bytes for a host-readback destination.
    Bytes(Vec<u8>),
    /// A queued copy already targets guest RAM; no placeholder bytes exist.
    Landed { bytes: u64 },
}

impl ComputeImageResult {
    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            Self::Landed { .. } => None,
        }
    }
}

#[derive(Debug)]
pub struct ComputeBufferResource {
    pub binding: u32,
    pub backing: ComputeBufferBacking,
    pub writable: bool,
}

#[derive(Debug)]
pub enum ComputeBufferBacking {
    /// Host-owned bytes for the staging policy.
    Bytes(Vec<u8>),
    /// The exact retained guest allocation and the footprint a writable bind
    /// publishes on successful completion.
    GuestPages {
        source: GuestRunSource,
        write_pages: Vec<u64>,
    },
}

impl ComputeBufferBacking {
    pub fn len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::GuestPages { source, .. } => source.total_len as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug)]
pub struct ComputeBufferOutput {
    pub binding: u32,
    pub result: ComputeBufferResult,
}

impl ComputeBufferOutput {
    pub fn bytes(&self) -> Option<&[u8]> {
        self.result.bytes()
    }
}

#[derive(Debug)]
pub enum ComputeBufferResult {
    /// Host-visible result bytes.
    Bytes(Vec<u8>),
    /// The shader wrote the imported guest allocation directly.
    Landed { bytes: u64 },
}

impl ComputeBufferResult {
    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            Self::Landed { .. } => None,
        }
    }
}

#[derive(Debug)]
pub struct ComputeStorageImageResource {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub format: StorageImageFormat,
    pub width: u32,
    pub height: u32,
    /// Prior contents selected before execution.
    pub seed: ComputeStorageImageSeed,
    /// Exactly one post-dispatch destination.
    pub destination: ComputeImageDestination,
    /// Semantic resident identity and content generations, when retained.
    pub residency: Option<ComputeStorageResidency>,
}

#[derive(Debug)]
pub enum ComputeStorageImageSeed {
    /// Tight host bytes.
    Bytes(Vec<u8>),
    /// Guest allocation with its physical row pitch.
    GuestPages(GuestRunSource),
    /// The matching resident already holds the requested generation.
    Resident,
}

#[derive(Clone, Copy, Debug)]
pub struct ComputeResidentSampleBind {
    /// Semantic storage lifetime, not a host image handle.
    pub identity: crate::ComputeStorageResidencyKey,
    /// Generation the resolver proved current.
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeStorageResidency {
    pub identity: crate::ComputeStorageResidencyKey,
    pub seed_generation: u32,
    pub output_generation: u32,
}

#[derive(Debug)]
pub struct ComputeSampledImageResource {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub format: SampledImageFormat,
    pub width: u32,
    pub height: u32,
    /// Whether the shader binding requires a multisampled 2D image.
    pub multisampled: bool,
    pub source: ComputeSampledImageSource,
    pub content: Option<crate::ContentStamp>,
}

#[derive(Debug)]
pub enum ComputeSampledImageSource {
    /// A serialized texture slot containing no object.
    Null,
    /// Tight host bytes.
    Bytes(Vec<u8>),
    /// Exact bounded guest allocation/window.
    GuestPages(GuestRunSource),
    /// Complete guest image allocation and the mip/layer view selected from it.
    GuestImage(GuestImageSource),
    /// A render/blit target whose authoritative image is already resident.
    TargetResident(crate::TargetIdentity),
    /// Prior storage output copied device-locally for this sampled bind.
    Resident(ComputeResidentSampleBind),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_page_results_do_not_masquerade_as_empty_readbacks() {
        assert_eq!(ComputeImageResult::Landed { bytes: 64 }.bytes(), None);
        assert_eq!(ComputeBufferResult::Landed { bytes: 64 }.bytes(), None);
        assert_eq!(
            ComputeImageResult::Bytes(vec![1, 2]).bytes(),
            Some(&[1, 2][..])
        );
    }
}
