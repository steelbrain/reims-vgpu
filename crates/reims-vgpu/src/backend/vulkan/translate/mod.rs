//! The **only** boundary where decoded Metal state becomes Vulkan state.
//!
//! # Why this module exists
//!
//! Before it, there was no single place that turned a Metal enum into a Vulkan
//! enum. The decision got re-made at whichever call site needed it first, each
//! invented its own vocabulary and its own failure style, and a choice made
//! once on one code path silently disagreed with the same choice made later on
//! another. The sRGB defect was the proof: the identical "sRGB is just linear"
//! fold appeared at twelve independent sites, none referencing a shared rule
//! and none recording that a choice had been made at all.
//!
//! [`super::caps`] already proved the fix for *device capability* — one module
//! classifies, everything else consumes, a source gate stops erosion. This is
//! the same shape for *format and pipeline state*.
//!
//! # The four properties
//!
//! 1. **Total, and typed.** Every entry point returns
//!    `Result<VkThing, TranslateReason>`. There is no `_ =>` arm reaching a
//!    default. An unmappable Metal value declines **by name**, which is what
//!    "unknown wire format stays unknown" already demands — applied uniformly
//!    instead of at whichever site remembered.
//! 2. **One arm per Metal value, sRGB included.** Where a path genuinely cannot
//!    honour a qualifier, that is a *named decline* recorded once, not a silent
//!    fold repeated wherever the format is touched.
//! 3. **Co-located invariants.** A format's Vulkan spelling and its byte size
//!    live in one table so they cannot drift, and each table asserts itself in
//!    unit tests that need no GPU — the same "testable without a device"
//!    property that makes the `caps` tests useful.
//! 4. **Capability-aware.** Translation returns the *desired* Vulkan value; a
//!    device-facing layer checks it against the bound GPU and either confirms
//!    it or declines by name. Which host GPU is in the machine never changes
//!    what a Metal value *means*, only whether this device can do it.
//!
//! # Rules for adding a translation
//!
//! * Put it here, not at the call site. A bare `vk::Format` return type is fine
//!   anywhere; spelling a specific *variant* outside this module and `caps` is
//!   not. A source scan used to enforce this and is gone, so it is a review
//!   rule now — a variant spelled at a call site is a translation that will be
//!   missed the next time this table changes.
//! * Make it total. No catch-all arm; add a [`TranslateReason`] variant with
//!   its own slug instead.
//! * Keep every co-varying property (byte size, texel size, aspect) in the same
//!   table as the format it belongs to.
//! * Never gate on a driver name, a vendor id, or an API version — that is
//!   [`super::caps`]'s job and it has its own rules.

pub mod blend;
pub mod pixel;
pub mod raster;
pub mod reason;
pub mod sampler;
pub mod support;
pub mod vertex;

pub(crate) use reason::TranslateReason;
pub(crate) use support::VertexFormatSupport;
