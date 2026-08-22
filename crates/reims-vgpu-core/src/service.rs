//! Backend-neutral service results used beside the submission port.

use crate::{ResidentContentBacking, ResourceLifetimeRef, TargetIdentity};

/// One semantic resident selected for host presentation.
///
/// A display transaction names exactly one surface, so this is one identity
/// rather than a candidate list. It remains a request instead of a resolved
/// backend slot: only the executor can decide whether the identity is resident
/// and presentable at this geometry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresentationSource {
    width: u32,
    height: u32,
    identity: TargetIdentity,
}

impl PresentationSource {
    pub fn new(identity: TargetIdentity, width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            identity,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn identity(&self) -> &TargetIdentity {
        &self.identity
    }
}

/// A presentation source accepted by the executor's resident registry and
/// window-presenter policy.
///
/// This is deliberately distinct from [`PresentationSource`]: composition may
/// construct a request, but only the executor preparation transition returns a
/// value the native window can offer for direct presentation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedPresentation {
    source: PresentationSource,
}

impl PreparedPresentation {
    /// Construct the result of a successful executor-side preparation.
    #[doc(hidden)]
    pub fn accepted(source: PresentationSource) -> Self {
        Self { source }
    }

    pub fn source(&self) -> &PresentationSource {
        &self.source
    }
}

/// Source that actually reached native presentation completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentationRoute {
    Resident,
    CpuBgra,
}

/// What an executor can prove about outstanding writes into a page window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestWriteReach {
    /// Nothing outstanding lands in any page asked about.
    Disjoint,
    /// An outstanding write lands in at least one page asked about.
    Overlap,
    /// The executor cannot name the write footprint precisely.
    Unnamed,
}

/// What a resident registry says about one target's content stamp.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentContent {
    /// No resident exists under this identity.
    Absent,
    /// A resident exists but no content epoch currently vouches for it.
    Unstamped,
    /// A resident contains content from the stated semantic epoch.
    Epoch(u32),
}

/// Why a previously known resident no longer exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentReclaim {
    /// Allocation pressure reclaimed a replica whose current bytes survive.
    AllocationReclaimed,
    /// The backend recreated the resident representation for the same identity.
    Recreated,
    /// The owning semantic resource lifetime ended.
    ResourceReleased,
}

/// One atomic executor reading of mutable resident content state.
///
/// Lifetime retention is deliberately absent: a semantic resource acquires its
/// lease separately, while this snapshot answers readiness, content currency,
/// and the reason an absent representation disappeared without three registry
/// transactions observing three different moments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentReadPlan {
    pub backing: ResidentContentBacking,
    pub content_epoch: Option<u32>,
    pub absent_after_reclaim: Option<(ResidentReclaim, u64)>,
}

impl Default for ResidentReadPlan {
    fn default() -> Self {
        Self {
            backing: ResidentContentBacking::NotReady,
            content_epoch: None,
            absent_after_reclaim: None,
        }
    }
}

impl ResidentReclaim {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::AllocationReclaimed => "allocation_reclaimed",
            Self::Recreated => "recreated",
            Self::ResourceReleased => "resource_released",
        }
    }
}

/// Why the direct host-window presentation route cannot carry a frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentDecline {
    WindowNotAttached,
    NoResident,
    ContentNotReady,
    ScanoutOrder,
    Geometry,
}

impl PresentDecline {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::WindowNotAttached => "winpub_window_not_attached",
            Self::NoResident => "winpub_no_resident",
            Self::ContentNotReady => "winpub_content_not_ready",
            Self::ScanoutOrder => "winpub_scanout_order",
            Self::Geometry => "winpub_geometry",
        }
    }
}

/// A resident target's pixels and their physical channel order.
#[derive(Debug, Eq, PartialEq)]
pub struct TargetReadback {
    pub pixels: Vec<u8>,
    /// The bytes are BGRA8 when true and semantic RGBA8 otherwise.
    pub bgra: bool,
}

/// Borrowed readback bytes whose backend allocation remains pinned until drop.
pub trait ReadbackLease: Send {
    fn bytes(&self) -> &[u8];
    fn is_bgra(&self) -> bool;
}

impl TargetReadback {
    /// Return semantic RGBA8, exchanging red and blue only when required.
    pub fn into_rgba8(mut self) -> Vec<u8> {
        if self.bgra {
            swap_red_blue(&mut self.pixels);
        }
        self.pixels
    }

    /// Return guest scanout order (BGRA8), exchanging only when required.
    pub fn into_bgra8(mut self) -> Vec<u8> {
        if !self.bgra {
            swap_red_blue(&mut self.pixels);
        }
        self.pixels
    }
}

fn swap_red_blue(pixels: &mut [u8]) {
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
}

/// Executor service for semantic resident-content state.
pub trait ResidentService: std::fmt::Debug + Send + Sync {
    fn resident_read_plan(&self, _identity: &TargetIdentity) -> ResidentReadPlan {
        ResidentReadPlan::default()
    }

    fn resident_content_state(&self, _identity: &TargetIdentity) -> ResidentContent {
        ResidentContent::Absent
    }

    fn stamp_resident_content_epoch(&self, _identity: &TargetIdentity, _epoch: u32) -> bool {
        false
    }

    /// Publish a guest CPU write to a resident that directly imports the
    /// guest's canonical allocation. This updates both content currency and
    /// the synchronization state for its next GPU consumer.
    fn note_resident_guest_write(&self, _identity: &TargetIdentity, _epoch: u32) -> bool {
        false
    }

    fn note_resident_content_copied_out(&self, _identity: &TargetIdentity) -> bool {
        false
    }

    /// Retain `identity` inside this executor for exactly `owner`'s semantic
    /// lifetime and report the resident's current backing class.
    ///
    /// The weak lifetime proof is deliberately the only ownership token that
    /// crosses this port. Backend leases remain executor-local and are reaped
    /// after the semantic resource is deleted.
    fn retain_resident_resource(
        &self,
        _owner: ResourceLifetimeRef,
        _identity: &TargetIdentity,
    ) -> ResidentContentBacking {
        ResidentContentBacking::NotReady
    }
}

/// Executor service for synchronizing writes that target guest memory.
pub trait GuestWriteService: std::fmt::Debug + Send + Sync {
    fn guest_writes_outstanding(&self) -> bool {
        false
    }

    fn guest_writes_reaching(&self, _pages: &[u64]) -> GuestWriteReach {
        GuestWriteReach::Disjoint
    }

    fn quiesce_guest_writes(&self) {}
}

/// Pixel readback service over semantic resident identities.
pub trait ReadbackService: std::fmt::Debug + Send + Sync {
    type Error;

    fn read_target(&self, identity: &TargetIdentity) -> Result<TargetReadback, Self::Error>;

    fn read_target_leased(
        &self,
        _identity: &TargetIdentity,
    ) -> Result<Option<Box<dyn ReadbackLease>>, Self::Error> {
        Ok(None)
    }

    fn read_resident_bgra(&self, _identity: &TargetIdentity, _need: usize) -> Option<Vec<u8>> {
        None
    }
}

/// Host-window presentation service over semantic resident identities.
pub trait PresentationService: std::fmt::Debug + Send + Sync {
    fn resident_presentable(&self, _identity: &TargetIdentity, _width: u32, _height: u32) -> bool {
        false
    }

    fn prepare_window_resident_present(
        &self,
        _source: &PresentationSource,
    ) -> Result<PreparedPresentation, PresentDecline> {
        Err(PresentDecline::WindowNotAttached)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PreparedPresentation, PresentDecline, PresentationSource, TargetIdentity, TargetReadback,
    };

    #[test]
    fn accepted_presentation_is_a_distinct_value_with_the_exact_checked_source() {
        let source = PresentationSource::new(TargetIdentity::default(), 1440, 900);
        let prepared = PreparedPresentation::accepted(source.clone());
        assert_eq!(prepared.source(), &source);
    }

    #[test]
    fn presentation_completion_names_the_two_payload_routes() {
        assert_ne!(
            super::PresentationRoute::Resident,
            super::PresentationRoute::CpuBgra
        );
    }

    #[test]
    fn readback_converts_only_when_the_requested_order_differs() {
        let rgba = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            TargetReadback {
                pixels: rgba.clone(),
                bgra: false,
            }
            .into_rgba8(),
            rgba
        );
        assert_eq!(
            TargetReadback {
                pixels: rgba,
                bgra: false,
            }
            .into_bgra8(),
            vec![3, 2, 1, 4, 7, 6, 5, 8]
        );
    }

    #[test]
    fn presentation_declines_have_stable_observation_names() {
        assert_eq!(
            PresentDecline::WindowNotAttached.slug(),
            "winpub_window_not_attached"
        );
        assert_eq!(PresentDecline::NoResident.slug(), "winpub_no_resident");
        assert_eq!(
            PresentDecline::ContentNotReady.slug(),
            "winpub_content_not_ready"
        );
        assert_eq!(PresentDecline::ScanoutOrder.slug(), "winpub_scanout_order");
        assert_eq!(PresentDecline::Geometry.slug(), "winpub_geometry");
    }
}
