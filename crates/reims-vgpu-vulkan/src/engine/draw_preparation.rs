//! Vulkan specialization of the core-owned draw-preparation vocabulary.

/// Semantic draw preparation with Vulkan translation failures in its one
/// executor-specific payload.
pub type DrawPreparationDecline =
    reims_vgpu_core::DrawPreparationDecline<crate::m2v_cache::M2vCacheDecline>;
