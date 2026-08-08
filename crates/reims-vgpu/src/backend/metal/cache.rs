//! Process-global content-hash caches.

use crate::backend::blob::{BlobIdentity, BlobKey};
use crate::backend::hash::hash_u64;
use crate::backend::metal::abi::{
    ReimsVgpuComputeStageInputDescriptor, ReimsVgpuComputeTextureUsage, ReimsVgpuDepthStencilState,
    ReimsVgpuSampler,
};
use crate::backend::render_pso_key::{RenderPsoIdentity, RenderPsoLookup};
use crate::contract::fnv::FNV_OFFSET_BASIS;
use crate::model::content_cache::{CacheEntry, ContentCache};
use metal::{ComputePipelineState, DepthStencilState, Function, RenderPipelineState, SamplerState};
use parking_lot::Mutex;

pub struct FnEntry {
    pub blob: BlobIdentity,
    pub function: Function,
}

/// Where the identity of a cached `MTLRenderPipelineState` lives, and why it is
/// not here.
///
/// [`crate::backend::render_pso_key`] owns `RenderPsoKey`, `RenderPsoLookup` and
/// `RenderPsoIdentity`. It is out of this module because nothing in it names the
/// `metal` crate and this module does not compile on a Linux host, so its tests
/// — the ones asking whether two different pipelines can be served as one — ran
/// nowhere while they sat here. This type is what is left: the identity beside
/// the objects Metal built from it.
pub struct RenderPsoEntry {
    pub id: RenderPsoIdentity,
    pub pso: RenderPipelineState,
    pub frag_sampler_mask: u32,
    pub vert_sampler_mask: u32,
}

/// What decides `MTLComputePipelineState` identity: the kernel blob, plus the
/// stage-input descriptor the PSO is specialized against.
///
/// Both halves are compared by content. The descriptor used to travel as a
/// `stage_hash` beside a `has_stage_input` flag and nothing retained it, so two
/// descriptors whose digests collided specialized one PSO — the same hole
/// [`crate::backend::blob`] describes for the blob, over 1520 bytes of decoded
/// guest record. It is `Copy` and this cache holds one per distinct pipeline, so
/// retaining it costs less than the flag saved.
#[derive(Clone, Copy)]
pub struct ComputePsoKey<'a> {
    pub mtlb: BlobKey<'a>,
    /// Buckets with the blob's digest and decides nothing.
    pub stage_hash: u64,
    pub stage_input: Option<&'a ReimsVgpuComputeStageInputDescriptor>,
}

pub struct ComputePsoEntry {
    pub mtlb: BlobIdentity,
    pub stage_hash: u64,
    pub stage_input: Option<ReimsVgpuComputeStageInputDescriptor>,
    pub pso: ComputePipelineState,
}

impl ComputePsoEntry {
    fn stage_input_is(&self, key: &ComputePsoKey<'_>) -> bool {
        match (&self.stage_input, key.stage_input) {
            (None, None) => true,
            (Some(mine), Some(theirs)) => {
                crate::backend::metal::util::bytes_of(mine)
                    == crate::backend::metal::util::bytes_of(theirs)
            }
            _ => false,
        }
    }
}

/// Every `MTLSamplerDescriptor` property this device sets, and nothing else, as
/// the words that decide `MTLSamplerState` identity.
///
/// One list, because it used to be four: the same fourteen fields were
/// transcribed into `SamplerCacheEntry`, again into its `matches`, again into
/// `sampler_key_hash`, and again into the entry's construction. A property
/// added to the descriptor in [`super::samplers::make_explicit_sampler`] and
/// forgotten in any one of them is a cache *hit* on a state built from
/// different words, which nothing reports and which shows up only as a texture
/// filtered the way some earlier bind asked for.
///
/// The three `ReimsVgpuSampler` words that are absent are absent by rule rather
/// than by omission: `has_lod_clamp` and the two `clamp_lod_*` words are the
/// encoder call `setSamplerState:lodMinClamp:lodMaxClamp:atIndex:`, applied per
/// bind and never baked into the state, so two binds differing only there share
/// one state and must hit. `binding` is per bind for the same reason.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SamplerDescriptorKey {
    /// First so the derived comparison rejects on it before the words, which is
    /// the prefilter the separate `key_hash` field used to be.
    hash: u64,
    words: [u32; 14],
}

impl SamplerDescriptorKey {
    pub fn new(s: &ReimsVgpuSampler) -> Self {
        let words = [
            s.unnormalized,
            s.min_filter,
            s.mag_filter,
            s.mip_filter,
            s.s_address_mode,
            s.t_address_mode,
            s.r_address_mode,
            s.border_color,
            s.compare_function,
            s.lod_min_bits,
            s.lod_max_bits,
            s.max_anisotropy,
            s.lod_average,
            s.support_argument_buffers,
        ];
        let hash = words
            .iter()
            .fold(FNV_OFFSET_BASIS, |h, &w| hash_u64(h, w as u64));
        Self { hash, words }
    }
}

pub struct SamplerCacheEntry {
    pub key: SamplerDescriptorKey,
    pub state: SamplerState,
}

/// What decides `MTLDepthStencilState` identity.
///
/// `hash` is a prefilter for the byte compare, the same role `hash` plays in
/// [`SamplerDescriptorKey`] — the descriptor is the identity and the hash only
/// rejects early.
#[derive(Clone, Copy)]
pub struct DepthStencilKey {
    pub hash: u64,
    pub desc: ReimsVgpuDepthStencilState,
}

pub struct DepthStencilEntry {
    pub key: DepthStencilKey,
    pub state: DepthStencilState,
}

pub struct ReflectEntry {
    pub blob: BlobIdentity,
    pub usages: Vec<ReimsVgpuComputeTextureUsage>,
}

impl CacheEntry for FnEntry {
    type Key<'a> = BlobKey<'a>;
    fn lookup_key(&self) -> BlobKey<'_> {
        self.blob.as_key()
    }
    /// The blob's bytes, not its digest — see [`crate::backend::blob`].
    fn matches(&self, key: &BlobKey<'_>) -> bool {
        self.blob.is(key)
    }
    fn bucket(key: &BlobKey<'_>) -> u64 {
        key.hash
    }
}

impl CacheEntry for ComputePsoEntry {
    type Key<'a> = ComputePsoKey<'a>;
    fn lookup_key(&self) -> ComputePsoKey<'_> {
        ComputePsoKey {
            mtlb: self.mtlb.as_key(),
            stage_hash: self.stage_hash,
            stage_input: self.stage_input.as_ref(),
        }
    }
    fn matches(&self, key: &ComputePsoKey<'_>) -> bool {
        self.mtlb.is(&key.mtlb) && self.stage_input_is(key)
    }
    /// The kernel blob's hash folded with the stage-input hash, so two PSOs
    /// specialized from one blob against different stage inputs do not pile
    /// into one bucket. Both are prefilters; `matches` compares both records.
    fn bucket(key: &ComputePsoKey<'_>) -> u64 {
        hash_u64(key.mtlb.hash, key.stage_hash)
    }
}

impl CacheEntry for RenderPsoEntry {
    type Key<'a> = RenderPsoLookup<'a>;
    fn lookup_key(&self) -> RenderPsoLookup<'_> {
        self.id.as_lookup()
    }
    fn matches(&self, key: &RenderPsoLookup<'_>) -> bool {
        self.id.is(key)
    }
    fn bucket(key: &RenderPsoLookup<'_>) -> u64 {
        key.bucket()
    }
}

impl CacheEntry for SamplerCacheEntry {
    type Key<'a> = SamplerDescriptorKey;
    fn lookup_key(&self) -> SamplerDescriptorKey {
        self.key
    }
    fn matches(&self, key: &SamplerDescriptorKey) -> bool {
        self.key == *key
    }
    fn bucket(key: &SamplerDescriptorKey) -> u64 {
        key.hash
    }
}

impl CacheEntry for DepthStencilEntry {
    type Key<'a> = DepthStencilKey;
    fn lookup_key(&self) -> DepthStencilKey {
        self.key
    }
    fn matches(&self, key: &DepthStencilKey) -> bool {
        self.key.hash == key.hash && depth_stencil_eq(&self.key.desc, &key.desc)
    }
    fn bucket(key: &DepthStencilKey) -> u64 {
        key.hash
    }
}

impl CacheEntry for ReflectEntry {
    type Key<'a> = BlobKey<'a>;
    fn lookup_key(&self) -> BlobKey<'_> {
        self.blob.as_key()
    }
    fn matches(&self, key: &BlobKey<'_>) -> bool {
        self.blob.is(key)
    }
    fn bucket(key: &BlobKey<'_>) -> u64 {
        key.hash
    }
}

struct GlobalCaches {
    fn_cache: ContentCache<FnEntry>,
    render_pso: ContentCache<RenderPsoEntry>,
    compute_pso: ContentCache<ComputePsoEntry>,
    sampler: ContentCache<SamplerCacheEntry>,
    depth_stencil: ContentCache<DepthStencilEntry>,
    reflect: ContentCache<ReflectEntry>,
}

impl GlobalCaches {
    const fn new() -> Self {
        Self {
            fn_cache: ContentCache::new(),
            render_pso: ContentCache::new(),
            compute_pso: ContentCache::new(),
            sampler: ContentCache::new(),
            depth_stencil: ContentCache::new(),
            reflect: ContentCache::new(),
        }
    }
}

static CACHES: Mutex<Option<GlobalCaches>> = Mutex::new(None);

fn with_caches<R>(f: impl FnOnce(&mut GlobalCaches) -> R) -> R {
    let mut guard = CACHES.lock();
    f(guard.get_or_insert_with(GlobalCaches::new))
}

/// Live entries in each cache, in the order
/// `(functions, render_pso, compute_pso, samplers, depth_stencil, reflections)`.
///
/// The Metal counterpart of the Vulkan engine's `object_cache_levels`, and the
/// reading that closes a gap this arm has carried since the caps came off:
/// [`crate::model::content_cache`] argues these tables settle at the guest's
/// distinct object set, and cites `pipelines=92` measured on the Vulkan arm
/// against the 64-slot render-PSO table this arm used to hold. That is the other
/// arm's count for the same command stream. This is how an Apple host reads its
/// own.
pub fn cache_levels() -> [usize; 6] {
    with_caches(|c| {
        [
            c.fn_cache.len(),
            c.render_pso.len(),
            c.compute_pso.len(),
            c.sampler.len(),
            c.depth_stencil.len(),
            c.reflect.len(),
        ]
    })
}

pub fn fn_cache_lookup(key: &BlobKey<'_>) -> Option<Function> {
    with_caches(|c| c.fn_cache.find(key).map(|e| e.function.clone()))
}

pub fn fn_cache_insert(key: &BlobKey<'_>, function: Function) -> Function {
    with_caches(|c| {
        c.fn_cache
            .insert_unique(FnEntry {
                blob: BlobIdentity::of(key),
                function,
            })
            .function
            .clone()
    })
}

pub fn compute_pso_lookup(key: &ComputePsoKey<'_>) -> Option<ComputePipelineState> {
    with_caches(|c| c.compute_pso.find(key).map(|e| e.pso.clone()))
}

pub fn compute_pso_insert(
    key: &ComputePsoKey<'_>,
    pso: ComputePipelineState,
) -> ComputePipelineState {
    with_caches(|c| {
        c.compute_pso
            .insert_unique(ComputePsoEntry {
                mtlb: BlobIdentity::of(&key.mtlb),
                stage_hash: key.stage_hash,
                stage_input: key.stage_input.copied(),
                pso,
            })
            .pso
            .clone()
    })
}

pub fn render_pso_lookup(key: &RenderPsoLookup<'_>) -> Option<(RenderPipelineState, u32, u32)> {
    with_caches(|c| {
        c.render_pso
            .find(key)
            .map(|e| (e.pso.clone(), e.vert_sampler_mask, e.frag_sampler_mask))
    })
}

pub fn render_pso_insert(
    key: &RenderPsoLookup<'_>,
    pso: RenderPipelineState,
    vert_mask: u32,
    frag_mask: u32,
) -> (RenderPipelineState, u32, u32) {
    with_caches(|c| {
        let entry = c.render_pso.insert_unique(RenderPsoEntry {
            id: RenderPsoIdentity::of(key),
            pso,
            frag_sampler_mask: frag_mask,
            vert_sampler_mask: vert_mask,
        });
        (
            entry.pso.clone(),
            entry.vert_sampler_mask,
            entry.frag_sampler_mask,
        )
    })
}

pub fn sampler_lookup(key: &SamplerDescriptorKey) -> Option<SamplerState> {
    with_caches(|c| c.sampler.find(key).map(|e| e.state.clone()))
}

pub fn sampler_insert(key: SamplerDescriptorKey, state: SamplerState) -> SamplerState {
    with_caches(|c| {
        c.sampler
            .insert_unique(SamplerCacheEntry { key, state })
            .state
            .clone()
    })
}

pub fn depth_stencil_lookup(key: &DepthStencilKey) -> Option<DepthStencilState> {
    with_caches(|c| c.depth_stencil.find(key).map(|e| e.state.clone()))
}

pub fn depth_stencil_insert(key: DepthStencilKey, state: DepthStencilState) -> DepthStencilState {
    with_caches(|c| {
        c.depth_stencil
            .insert_unique(DepthStencilEntry { key, state })
            .state
            .clone()
    })
}

fn depth_stencil_eq(a: &ReimsVgpuDepthStencilState, b: &ReimsVgpuDepthStencilState) -> bool {
    crate::backend::metal::util::bytes_of(a) == crate::backend::metal::util::bytes_of(b)
}

pub fn reflect_lookup(key: &BlobKey<'_>) -> Option<Vec<ReimsVgpuComputeTextureUsage>> {
    with_caches(|c| c.reflect.find(key).map(|e| e.usages.clone()))
}

pub fn reflect_insert(key: &BlobKey<'_>, usages: Vec<ReimsVgpuComputeTextureUsage>) {
    with_caches(|c| {
        c.reflect.insert_unique(ReflectEntry {
            blob: BlobIdentity::of(key),
            usages,
        });
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_stencil_cache_key_covers_both_faces() {
        let base = ReimsVgpuDepthStencilState::default();
        let mut changed = base;
        assert!(depth_stencil_eq(&base, &changed));
        changed.back_face.write_mask = 0xff;
        assert!(!depth_stencil_eq(&base, &changed));
    }
}
