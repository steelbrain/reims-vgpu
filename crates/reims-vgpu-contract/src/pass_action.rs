//! The `MTLLoadAction` and `MTLStoreAction` ordinals a render-pass attachment
//! prefix carries, and the closed sets this device implements.
//!
//! Two adjacent 16-bit words of every colour, depth and stencil attachment. The
//! guest writes the Metal SDK's own ordinals into them, so the numbers here are
//! simultaneously a wire fact and an SDK fact — which is why the conversion on
//! the Metal encode path is an identity: nothing is mapped, only widened.
//!
//! # Why they live here rather than in the decoder or the backend
//!
//! They were declared twice. `runtime::decode::render` had them as `u16` under
//! `PASS_*`, `backend::metal::abi` as `u32` under `REIMS_VGPU_MTL_*`, five
//! ordinals each, and nothing in the toolchain compared the two — the only
//! thing that touched both was an identity `match` in `crate::runtime::draw`
//! that read as a translation table. A value that arrives on the wire and is
//! consumed by both backends belongs beside the other wire/SDK numbers, per this
//! module tree's own doc; `backend::metal::abi` keeps its spelling because that
//! file is a mirror of an archived C header and the mirror is its provenance,
//! with `const` assertions there pinning the two equal on every arm that
//! compiles it.
//!
//! # The widths differ, and that is the whole conversion
//!
//! The attachment prefix spells an action in 16 bits; the Metal C shim takes an
//! `MTLLoadAction`/`MTLStoreAction` as `uint32_t`. So the declaration that
//! crosses to C is `u32` and this one is `u16`, and everything between them is a
//! widening — see `crate::runtime::draw`'s `map_load_action`, which returns
//! DontCare for a value outside the set and widens every value inside it.

/// `MTLLoadActionDontCare` — the attachment's prior contents may be discarded.
pub const MTL_LOAD_ACTION_DONT_CARE: u16 = 0;
/// `MTLLoadActionLoad` — the pass composites onto the attachment's contents.
pub const MTL_LOAD_ACTION_LOAD: u16 = 1;
/// `MTLLoadActionClear` — the attachment starts at the record's clear value.
pub const MTL_LOAD_ACTION_CLEAR: u16 = 2;

/// `MTLStoreActionDontCare` — the pass's result for this attachment is dropped.
pub const MTL_STORE_ACTION_DONT_CARE: u16 = 0;
/// `MTLStoreActionStore` — the pass's result is written back to the attachment.
pub const MTL_STORE_ACTION_STORE: u16 = 1;
/// `MTLStoreActionMultisampleResolve` — resolve into the attachment's
/// single-sample destination and discard the multisample source.
pub const MTL_STORE_ACTION_MULTISAMPLE_RESOLVE: u16 = 2;
/// `MTLStoreActionStoreAndMultisampleResolve` — preserve both images.
pub const MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE: u16 = 3;

/// Whether `raw` is one of the three `MTLLoadAction` values.
///
/// The set is *closed* in the same sense as [`crate::dispatch`]'s:
/// `MTLLoadAction` has exactly these three, so a fourth ordinal is a corrupt
/// record or a wrong wire offset rather than a guest feature this device has no
/// contract for yet.
#[must_use]
pub fn is_declared_load_action(raw: u16) -> bool {
    raw <= MTL_LOAD_ACTION_CLEAR
}

/// A colour attachment's load action in this device's own vocabulary rather than
/// the guest's ordinal.
///
/// The ordinal is a guest value: arbitrary, and parsed once at the boundary into
/// something **total**. Every value outside the closed set becomes
/// [`Self::DontCare`], which is the substitution every caller already made by
/// hand after asking [`is_declared_load_action`] — the *reporting* of an
/// out-of-contract value stays with that caller, because only it knows which
/// pipeline sent it, but the fold itself is written once here.
///
/// Total matters more than it looks. The three actions have repeatedly been
/// consumed as `match raw { LOAD => .., CLEAR => .., _ => {} }`, where the
/// catch-all silently carries both DontCare and a corrupt ordinal, and where
/// `rustc` cannot say that a fourth arm is missing. Matching this enum is
/// checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadAction {
    /// The attachment's prior contents may be discarded.
    DontCare,
    /// The pass composites onto the attachment's contents.
    Load,
    /// The attachment starts at the record's clear value.
    Clear,
}

impl LoadAction {
    /// Fold a decoded ordinal, treating anything outside the closed set as
    /// DontCare.
    #[must_use]
    pub fn from_declared(raw: u16) -> Self {
        match raw {
            MTL_LOAD_ACTION_LOAD => Self::Load,
            MTL_LOAD_ACTION_CLEAR => Self::Clear,
            _ => Self::DontCare,
        }
    }

    /// Whether a pass with this action composites onto the attachment's prior
    /// contents, so this device has to resolve those contents rather than let
    /// the attachment begin at a value it invented.
    ///
    /// True for `Load`, which says so, and for `DontCare`, which does not
    /// forbid it. DontCare declares the prior contents **undefined**, and
    /// undefined permits *any* contents — including the ones already there. So
    /// preserving is a legal realization of DontCare and clearing is the only
    /// reading that destroys, which is what a two-valued `LOAD vs CLEAR`
    /// spelling silently picks. The guest relies on the preserving reading: it
    /// declares DontCare and then redraws only its damage rect, because on
    /// Apple hardware the texture memory persists, and
    /// `backend::metal::render` hands the same wire word to Metal, which
    /// preserves. A Vulkan arm that clears therefore disagrees with the Metal
    /// arm about one attachment prefix.
    ///
    /// A `match` rather than a `matches!` so a fourth action fails the build
    /// here, once, instead of being folded into whichever side the caller's
    /// catch-all happened to take.
    #[must_use]
    pub fn preserves_prior_contents(self) -> bool {
        match self {
            Self::Load | Self::DontCare => true,
            Self::Clear => false,
        }
    }

    /// The census pair for this action: how many records declared it, and how
    /// many attachment pixels those records covered.
    ///
    /// Two counters and not one because the cost of getting an action wrong is
    /// proportional to **area**, while the population is dominated by small
    /// attachments — a boot's thousands of 64x64 passes would otherwise outrank
    /// its dozens of full-screen ones. Named together here so the pair cannot
    /// drift apart, and returned rather than emitted so this module keeps no
    /// dependency on the census.
    #[must_use]
    pub fn census_routes(self) -> (&'static str, &'static str) {
        match self {
            Self::DontCare => ("color0_declared_dontcare", "color0_declared_dontcare_px"),
            Self::Load => ("color0_declared_load", "color0_declared_load_px"),
            Self::Clear => ("color0_declared_clear", "color0_declared_clear_px"),
        }
    }
}

/// Whether `raw` is one of the four store actions this device decodes by name.
///
/// The remaining SDK values require additional state not represented by this
/// wire form. Backend capability is a separate question: recognizing an action
/// here does not authorize a backend to approximate it.
#[must_use]
pub fn is_declared_store_action(raw: u16) -> bool {
    raw <= MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE
}

/// Whether the action publishes a single-sample destination the guest may
/// subsequently read.
#[must_use]
pub fn store_action_publishes_single_sample(raw: u16) -> bool {
    matches!(
        raw,
        MTL_STORE_ACTION_STORE
            | MTL_STORE_ACTION_MULTISAMPLE_RESOLVE
            | MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The accepted load set is exactly the three declared ordinals.
    ///
    /// Swept past the top constant in both directions, because the predicate's
    /// job is to be closed: every value it rejects is substituted with DontCare
    /// by its callers, so an accidentally-accepted fourth ordinal would reach a
    /// Metal enum conversion that has no arm for it.
    #[test]
    fn only_the_three_declared_load_actions_are_accepted() {
        assert!(is_declared_load_action(MTL_LOAD_ACTION_DONT_CARE));
        assert!(is_declared_load_action(MTL_LOAD_ACTION_LOAD));
        assert!(is_declared_load_action(MTL_LOAD_ACTION_CLEAR));
        for raw in (MTL_LOAD_ACTION_CLEAR + 1)..=64u16 {
            assert!(
                !is_declared_load_action(raw),
                "{raw} is not a declared MTLLoadAction"
            );
        }
        assert!(!is_declared_load_action(u16::MAX));
    }

    /// The recognized store set is exactly the four actions represented here.
    #[test]
    fn only_the_four_named_store_actions_are_accepted() {
        assert!(is_declared_store_action(MTL_STORE_ACTION_DONT_CARE));
        assert!(is_declared_store_action(MTL_STORE_ACTION_STORE));
        assert!(is_declared_store_action(
            MTL_STORE_ACTION_MULTISAMPLE_RESOLVE
        ));
        assert!(is_declared_store_action(
            MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE
        ));
        for raw in (MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE + 1)..=64u16 {
            assert!(
                !is_declared_store_action(raw),
                "{raw} is not a named MTLStoreAction"
            );
        }
        assert!(!is_declared_store_action(u16::MAX));
    }

    #[test]
    fn only_actions_with_a_single_sample_result_publish_one() {
        assert!(!store_action_publishes_single_sample(
            MTL_STORE_ACTION_DONT_CARE
        ));
        for action in [
            MTL_STORE_ACTION_STORE,
            MTL_STORE_ACTION_MULTISAMPLE_RESOLVE,
            MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE,
        ] {
            assert!(store_action_publishes_single_sample(action));
        }
        assert!(!store_action_publishes_single_sample(u16::MAX));
    }

    /// The fold is total, agrees with the predicate on the closed set, and
    /// sends everything else to DontCare.
    ///
    /// Swept past the top ordinal the same way the predicate's own test is,
    /// because the two have to answer the same question: a value the predicate
    /// rejects is exactly a value this must call DontCare, and a fourth arm
    /// appearing in one and not the other is how the catch-all this enum exists
    /// to remove would come back.
    #[test]
    fn the_load_action_fold_is_total_and_agrees_with_the_predicate() {
        assert_eq!(
            LoadAction::from_declared(MTL_LOAD_ACTION_DONT_CARE),
            LoadAction::DontCare
        );
        assert_eq!(
            LoadAction::from_declared(MTL_LOAD_ACTION_LOAD),
            LoadAction::Load
        );
        assert_eq!(
            LoadAction::from_declared(MTL_LOAD_ACTION_CLEAR),
            LoadAction::Clear
        );
        for raw in 0..=u16::MAX {
            let folded = LoadAction::from_declared(raw);
            if !is_declared_load_action(raw) {
                assert_eq!(
                    folded,
                    LoadAction::DontCare,
                    "{raw} is out of contract and must fold to DontCare"
                );
            }
        }
    }

    /// Clear is the only action that refuses the attachment's prior contents.
    ///
    /// The pin that stops the Vulkan arm going back to a two-valued
    /// `LOAD vs CLEAR` spelling. Folding DontCare onto Clear is what wrote a
    /// definite colour over live guest content on every pass the guest declared
    /// undefined — measured on a driven macos-15 boot as a `passbegin_clear`
    /// that ran exactly `color0_declared_dontcare` above the clears the guest
    /// actually asked for, on all five boots it was checked on.
    ///
    /// The out-of-contract case rides along deliberately: `from_declared` folds
    /// an unknown ordinal to DontCare, so an unknown action now preserves
    /// rather than clears. Preserving cannot destroy what the guest did not ask
    /// to have destroyed, and the value is still reported by name where it is
    /// decoded.
    #[test]
    fn only_a_clear_refuses_the_attachments_prior_contents() {
        assert!(
            LoadAction::Load.preserves_prior_contents(),
            "Load composites onto what is already there"
        );
        assert!(
            LoadAction::DontCare.preserves_prior_contents(),
            "DontCare declares the prior contents undefined, and undefined \
             permits the prior contents — so preserving is legal, and it is \
             what the Metal arm does with the same wire word"
        );
        assert!(
            !LoadAction::Clear.preserves_prior_contents(),
            "a Clear must still clear: the guest asked for its clear value"
        );
        assert!(
            LoadAction::from_declared(u16::MAX).preserves_prior_contents(),
            "an out-of-contract ordinal folds to DontCare, and must therefore \
             preserve rather than destroy"
        );
    }

    /// The six census route names are distinct, and each count is paired with
    /// its own area counter.
    ///
    /// A copied line here reads as a working census and silently adds one
    /// action's records to another's — the failure that makes a boot's ranking
    /// wrong in the direction that looks like a finding.
    #[test]
    fn every_load_action_has_its_own_pair_of_census_routes() {
        let mut seen = std::collections::BTreeSet::new();
        for action in [LoadAction::DontCare, LoadAction::Load, LoadAction::Clear] {
            let (n, px) = action.census_routes();
            assert_eq!(px, format!("{n}_px"), "the area route names its own count");
            assert!(seen.insert(n), "{n} is already another action's count");
            assert!(seen.insert(px), "{px} is already another action's area");
        }
        assert_eq!(seen.len(), 6);
    }

    /// Neither predicate can see the two adjacent words swapped.
    ///
    /// The load set is a strict subset of the store set — `{0, 1, 2}` inside
    /// `{0, 1, 2, 3}` — and the attachment prefix carries the two in adjacent
    /// words. So a decode that swaps the words can still produce values both
    /// predicates accept. Pinned
    /// because it bounds what these two can be asked to prove: they narrow a
    /// value to its own contract, and no arrangement of them detects a field
    /// offset that is two bytes out.
    #[test]
    fn the_load_set_is_a_strict_subset_of_the_store_set() {
        let load: Vec<u16> = (0..=u16::MAX)
            .filter(|&r| is_declared_load_action(r))
            .collect();
        let store: Vec<u16> = (0..=u16::MAX)
            .filter(|&r| is_declared_store_action(r))
            .collect();
        assert!(
            load.iter().all(|r| store.contains(r)),
            "a load ordinal outside the store set would make a swap detectable"
        );
        assert_eq!(
            store
                .iter()
                .filter(|r| !load.contains(r))
                .collect::<Vec<_>>(),
            vec![&MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE]
        );
    }
}
