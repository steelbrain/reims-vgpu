//! Always-on declines for the type-8 view-swizzle bug class.
//!
//! # The class
//!
//! A guest texture view (type-8, opcode `0x1b`) can remap which channel each
//! output component reads. Vulkan performs exactly that for free through the
//! image view's `VkComponentMapping`. Doing it any other way is a defect with
//! two shapes, and these declines tell them apart:
//!
//! * **Dropped.** The bind refuses because the swizzle is not identity, so the
//!   draw loses its sampled input entirely and renders without the texture.
//!   This is the shape the Vulkan pathway had, and it was *silent* — the
//!   refusal returned `None` with nothing in the fail log, indistinguishable
//!   from a texture that simply was not there.
//! * **CPU-remapped.** The bind rewrites every texel on the CPU. Correct, but
//!   it forces the texture onto the upload path and **costs it the zero-copy
//!   property** — the guest→GPU crossing that the whole present path is built
//!   to avoid. It is the one that needs saying out loud, precisely because it
//!   is invisible in the output.
//!
//! # Reading it
//!
//! `/tmp/reims-vgpu-fail.log`, always-on, one line per (reason, texture ref):
//!
//! * `view_swizzle_cpu_remap reason=swizzle_cpu_remap ref=<n>`
//! * `view_swizzle_declined reason=<slug> ref=<n>`
//!
//! **Zero lines is the invariant.** Any `swizzle_cpu_remap` means some rail is
//! remapping texels by hand again; any `swizzle_resident_direct_bind` means a
//! swizzled bind was dropped. The healthy path — the swizzle riding a
//! `VkComponentMapping` — is silent, because a bind that worked is not news.
//!
//! Measure-only: nothing here gates decode, execute or present.

use std::collections::BTreeSet;
use std::sync::Mutex;

use crate::observe;

/// `(reason slug, texture ref)` pairs already reported, so a per-draw rail
/// fires once per distinct pair rather than once per bind. Bounded by the live
/// swizzled-texture set.
static SEEN: Mutex<BTreeSet<(&'static str, u32)>> = Mutex::new(BTreeSet::new());

fn first_sight(slug: &'static str, texture_ref: u32) -> bool {
    SEEN.lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((slug, texture_ref))
}

/// A non-identity view swizzle was performed by rewriting texels on the CPU.
///
/// Always fail-visible on first sight, because this is the regression the GPU
/// component mapping replaced: it is *correct* and therefore invisible in the
/// output, while costing the texture its zero-copy crossing.
pub fn note_cpu_remap(texture_ref: u32) {
    use crate::observe::Decline as _;
    let reason = SwizzleDecline::CpuRemap;
    if first_sight(reason.slug(), texture_ref) {
        observe::Emit::decline("view_swizzle_cpu_remap", &reason)
            .field("ref", texture_ref)
            .fail();
    }
}

/// A swizzled bind was refused, naming the specific check.
pub fn note_declined(reason: SwizzleDecline, texture_ref: u32) {
    use crate::observe::Decline as _;
    if first_sight(reason.slug(), texture_ref) {
        observe::Emit::decline("view_swizzle_declined", &reason)
            .field("ref", texture_ref)
            .fail();
    }
}

/// The way a non-identity view swizzle fails to reach the GPU as a component
/// mapping.
///
/// It is a refusal of the zero-copy path and not of the bind: a CPU remap
/// renders correctly and is therefore invisible in the output, which is exactly
/// why it needs a name in the log.
///
/// This replaced a `pub mod decline` of bare `&str` constants. The slug is
/// `swizzle_`-prefixed because `cpu_remap`, bare, names nothing about which rail
/// wrote it — the same argument that prefixed the slate reasons.
///
/// A second variant, `ResidentDirectBind`, is gone. It named a resident bound
/// through the registry's own image view, which the engine once created one of
/// per target and could not re-decorate — so that arm dropped the bind. It does
/// not drop it any more, and no longer copies to avoid it either: the registry
/// keys a view on the format *and* the mapping, so a swizzled resident binds
/// directly and the hardware performs the mapping at sample time. That is a rail
/// taken and not a swizzle lost, so it does not belong in this enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwizzleDecline {
    /// Every texel was rewritten by hand. Correct output, zero-copy lost.
    CpuRemap,
}

impl crate::observe::Decline for SwizzleDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::CpuRemap => "swizzle_cpu_remap",
        }
    }
}

/// Drop the first-sight set. Test-only: it is process-global, so a test that
/// asserts a line was emitted must start from a known point.
#[cfg(test)]
pub fn reset_for_tests() {
    SEEN.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every slug names the rail that wrote it.
    ///
    /// Bare, `cpu_remap` says nothing about which subsystem refused — the same
    /// argument that prefixed the slate reasons. This asserts the prefix only;
    /// crate-wide distinctness is unchecked, and naming the rail is what keeps a
    /// slug distinct for a reason rather than by luck.
    ///
    /// The `match` is what makes this exhaustive: a variant added to
    /// [`SwizzleDecline`] and not named here fails to compile, which a loop over
    /// a hand-written list would not.
    #[test]
    fn every_swizzle_slug_names_its_rail() {
        use crate::observe::Decline as _;
        let every: &[SwizzleDecline] = &[SwizzleDecline::CpuRemap];
        for r in every {
            match r {
                SwizzleDecline::CpuRemap => {}
            }
            assert!(
                r.slug().starts_with("swizzle_"),
                "{} is not namespaced to this rail",
                r.slug()
            );
        }
    }

    /// A hot rail must cost one line per distinct (reason, ref), not one per
    /// bind — the dedup is what makes it safe to leave on forever.
    #[test]
    fn each_reason_and_ref_pair_reports_once() {
        reset_for_tests();
        assert!(first_sight("swizzle_cpu_remap", 7));
        assert!(!first_sight("swizzle_cpu_remap", 7));
        assert!(
            first_sight("swizzle_cpu_remap", 8),
            "a new ref is a new event"
        );
        assert!(
            first_sight("swizzle_resident_direct_bind", 7),
            "a new reason on the same ref is a new event"
        );
        reset_for_tests();
    }
}
