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
//! thing that touched both was an identity `match` in [`crate::runtime::draw`]
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
//! widening — see [`crate::runtime::draw`]'s `map_load_action`, which returns
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
/// The set is *closed* in the same sense as [`crate::contract::dispatch`]'s:
/// `MTLLoadAction` has exactly these three, so a fourth ordinal is a corrupt
/// record or a wrong wire offset rather than a guest feature this device has no
/// contract for yet.
#[must_use]
pub fn is_declared_load_action(raw: u16) -> bool {
    raw <= MTL_LOAD_ACTION_CLEAR
}

/// Which band of colour attachments a census reading is about.
///
/// The band and not the slot. Eight slots times three actions times two
/// counters is forty-eight `&'static str`s to answer a question nobody asks per
/// slot, and the split that carries information is this one: slot 0 is the
/// attachment that aliases guest memory, carries the LOAD seed, and reaches the
/// present path, while slots 1..N are ordinary residents. A reading that
/// separates those two answers "did a *secondary* declare this", which is the
/// question a divergence between the two producers makes worth asking; a
/// reading per slot answers nothing further.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachmentBand {
    /// The primary colour attachment, Metal slot 0.
    Color0,
    /// Any MRT colour attachment beyond the primary, Metal slots 1 and up.
    Color1Plus,
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
/// # `Default` is `Clear`, which is not the ordinal-zero member
///
/// Nothing on the wire is ever defaulted — [`Self::from_declared`] is the only
/// boundary parse and it is total, so this impl is reached exclusively by a
/// `..Default::default()` on a backend request struct, i.e. by a producer that
/// said nothing. An attachment whose action nobody stated must be given
/// *defined* contents: `DontCare` is a licence to leave whatever was in the tile
/// memory, and handing that out to a caller who simply forgot the field is a
/// silent loss of the kind this crate refuses to ship. `Clear` costs a write and
/// cannot lose anything.
/// What a writer that lands a record's `clearColor` must do for a given
/// [`LoadAction`], from [`LoadAction::clear_seed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClearSeed {
    /// The action names the clear value; land it.
    Land,
    /// The action does not name the clear value. The payload is the census
    /// route for this refusal, so the population that stopped being painted
    /// stays countable rather than merely disappearing.
    Decline(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum LoadAction {
    /// The attachment's prior contents may be discarded.
    DontCare,
    /// The pass composites onto the attachment's contents.
    Load,
    /// The attachment starts at the record's clear value.
    #[default]
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

    /// Whether landing this record's `clearColor` into the attachment is what
    /// the action asks for, and the census route naming the refusal when it is
    /// not.
    ///
    /// `MTLRenderPass.h` states the rule as a conditional on exactly one member:
    /// *"The clear color to be used **if** the loadAction property is
    /// MTLLoadActionClear."* So one of the three consults the clear value and
    /// the other two must not, and a seed is not a cheap approximation of
    /// either of them — `Load` needs the surface's prior contents, and
    /// `DontCare` is a licence to leave them alone, which a seed spends.
    ///
    /// Named here rather than spelled at the call site because the seed path is
    /// where getting it wrong is *invisible*: that writer lands solid pixels
    /// straight into the guest's own pages, so a wrongly-admitted action is not
    /// a refused command but a surface the guest keeps and composites. The
    /// spelling this replaced admitted `DontCare` beside `Clear`, which landed
    /// Metal's own default `MTLClearColorMake(0, 0, 0, 1)` — opaque black —
    /// across every attachment whose guest never set a clear value.
    ///
    /// The decision and the refusal's name are one answer rather than two
    /// methods, so a caller cannot decline on one rule and report the other.
    /// Returned rather than emitted, following [`Self::census_routes`], so this
    /// module keeps no dependency on the census.
    #[must_use]
    pub fn clear_seed(self) -> ClearSeed {
        match self {
            Self::Clear => ClearSeed::Land,
            Self::Load => ClearSeed::Decline("clear_seed_declined_load"),
            Self::DontCare => ClearSeed::Decline("clear_seed_declined_dontcare"),
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
    ///
    /// Banded by [`AttachmentBand`] because the two producers of this device's
    /// colour attachments are separate code, and for a while only one of them
    /// had a census at all — so a boot could say how many *primary* attachments
    /// declared each action and nothing whatever about the secondaries. A
    /// question that can only be asked of half the population is how a
    /// divergence between two arms over one wire form stays theoretical.
    #[must_use]
    pub fn census_routes(self, band: AttachmentBand) -> (&'static str, &'static str) {
        match (band, self) {
            (AttachmentBand::Color0, Self::DontCare) => {
                ("color0_declared_dontcare", "color0_declared_dontcare_px")
            }
            (AttachmentBand::Color0, Self::Load) => {
                ("color0_declared_load", "color0_declared_load_px")
            }
            (AttachmentBand::Color0, Self::Clear) => {
                ("color0_declared_clear", "color0_declared_clear_px")
            }
            (AttachmentBand::Color1Plus, Self::DontCare) => (
                "color1plus_declared_dontcare",
                "color1plus_declared_dontcare_px",
            ),
            (AttachmentBand::Color1Plus, Self::Load) => {
                ("color1plus_declared_load", "color1plus_declared_load_px")
            }
            (AttachmentBand::Color1Plus, Self::Clear) => {
                ("color1plus_declared_clear", "color1plus_declared_clear_px")
            }
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

    /// Only the action that names the clear value may have it landed.
    ///
    /// Swept from the decoded ordinal rather than from the enum, because that is
    /// the hop the seed writer actually makes and the one that was wrong: it
    /// admitted ordinal 0 (`DontCare`) beside ordinal 2 (`Clear`) and painted
    /// Metal's default opaque black across the guest's own surface. An
    /// out-of-contract ordinal folds to `DontCare`, so it must decline too.
    #[test]
    fn only_clear_lands_the_clear_value_into_the_guest_surface() {
        assert_eq!(
            LoadAction::from_declared(MTL_LOAD_ACTION_CLEAR).clear_seed(),
            ClearSeed::Land
        );
        for raw in [MTL_LOAD_ACTION_DONT_CARE, MTL_LOAD_ACTION_LOAD] {
            assert!(
                matches!(
                    LoadAction::from_declared(raw).clear_seed(),
                    ClearSeed::Decline(_)
                ),
                "ordinal {raw} does not name the clear value, so it must not be seeded with it"
            );
        }
        for raw in (MTL_LOAD_ACTION_CLEAR + 1)..=64u16 {
            assert!(
                matches!(
                    LoadAction::from_declared(raw).clear_seed(),
                    ClearSeed::Decline(_)
                ),
                "out-of-contract ordinal {raw} folds to DontCare and must not be seeded"
            );
        }
    }

    /// Each declining action names a distinct route, so the census can tell the
    /// two refusals apart. A shared name would report the `DontCare` population
    /// and the `Load` population as one number.
    #[test]
    fn the_two_declining_actions_do_not_share_a_census_route() {
        let names: Vec<&'static str> = [LoadAction::DontCare, LoadAction::Load]
            .into_iter()
            .map(|a| match a.clear_seed() {
                ClearSeed::Decline(route) => route,
                ClearSeed::Land => panic!("{a:?} must decline the clear seed"),
            })
            .collect();
        assert_eq!(names.len(), 2);
        assert_ne!(names[0], names[1], "the two refusals collapsed onto one route");
    }

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
        assert!(is_declared_store_action(MTL_STORE_ACTION_MULTISAMPLE_RESOLVE));
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

    /// The twelve census route names are distinct, and each count is paired
    /// with its own area counter.
    ///
    /// A copied line here reads as a working census and silently adds one
    /// action's records to another's — the failure that makes a boot's ranking
    /// wrong in the direction that looks like a finding. With a band the copy is
    /// easier to make and harder to see: `color1plus`'s three arms are the
    /// obvious paste of `color0`'s, and a paste that forgot to change the prefix
    /// would file every secondary's declaration under the primary's counter, so
    /// the two producers would be indistinguishable in exactly the reading that
    /// exists to tell them apart.
    #[test]
    fn every_load_action_has_its_own_pair_of_census_routes_in_each_band() {
        let mut seen = std::collections::BTreeSet::new();
        for band in [AttachmentBand::Color0, AttachmentBand::Color1Plus] {
            let prefix = match band {
                AttachmentBand::Color0 => "color0_",
                AttachmentBand::Color1Plus => "color1plus_",
            };
            for action in [LoadAction::DontCare, LoadAction::Load, LoadAction::Clear] {
                let (n, px) = action.census_routes(band);
                assert_eq!(px, format!("{n}_px"), "the area route names its own count");
                assert!(
                    n.starts_with(prefix),
                    "{n} is filed under a band it did not come from"
                );
                assert!(seen.insert(n), "{n} is already another route's count");
                assert!(seen.insert(px), "{px} is already another route's area");
            }
        }
        assert_eq!(seen.len(), 12);
    }

    /// An unstated load action is `Clear`, not the ordinal-zero `DontCare`.
    ///
    /// The two differ by whether a producer that forgot the field gets defined
    /// contents or undefined ones, and the derive would have picked the first
    /// variant. Pinned so that reordering the enum to put `DontCare` first --
    /// which is the order the Metal ordinals are in, and therefore the tempting
    /// order -- cannot silently hand every defaulted request a licence to leave
    /// the attachment full of whatever was there.
    #[test]
    fn an_unstated_load_action_is_the_one_that_cannot_lose_content() {
        assert_eq!(LoadAction::default(), LoadAction::Clear);
        assert_ne!(
            LoadAction::default(),
            LoadAction::from_declared(MTL_LOAD_ACTION_DONT_CARE),
            "defaulting must not be a way to spell DontCare without declaring it"
        );
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
            store.iter()
                .filter(|r| !load.contains(r))
                .collect::<Vec<_>>(),
            vec![&MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE]
        );
    }
}
