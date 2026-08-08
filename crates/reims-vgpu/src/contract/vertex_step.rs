//! `MTLVertexStepFunction` ordinals, and the step rate that pairs with each.
//!
//! A type-7 pipeline descriptor's buffer layout carries a step function and a
//! step rate, decoded by [`crate::runtime::decode::resource`] into
//! [`VertexAttribute::step_function_ordinal`] and [`VertexAttribute::step_rate`].
//! The two are one rule and not two fields: `MTLVertexBufferLayoutDescriptor`
//! requires `stepRate == 0` for `MTLVertexStepFunctionConstant` and rejects it
//! for every other step function, because a constant-rate attribute is fetched
//! once for the whole draw and a rate of zero is how that is spelled.
//!
//! # Why the pair lives here
//!
//! Both backends narrow it and they disagreed. The Metal arm asked
//! `rate == 0 && step != Constant`; the Vulkan arm's draw validation asked
//! `rate == 0` alone, so it declined the canonical Constant spelling — with
//! `vk_draw_validate_zero_vertex_step_rate`, a decline that loses the whole
//! draw — while ignoring the rate for exactly that step function everywhere
//! downstream (its divisor is 0 whatever the rate says). The decoder's own doc
//! settles which arm was right: "a layout that declared **zero** means zero —
//! that is what `MTLVertexStepFunctionConstant` pairs with — so nothing here
//! clamps it up".
//!
//! The ordinals are here for a second reason. `backend::metal` reached them
//! through `metal` 0.33's `MTLVertexStepFunction` discriminants, and that crate
//! is measured to number the *sibling* `MTLStepFunction` wrongly in six of nine
//! places — see `backend::metal::mtl_enum`, which carries the table and now also
//! carries `const` assertions pinning this enum's five discriminants to the
//! ordinals below. (Named in prose, not linked: that module is
//! `backend-metal`-gated, so a link from here is unresolved on every Vulkan-arm
//! doc build.)
//!
//! [`VertexAttribute::step_function_ordinal`]: crate::runtime::decode::resource::VertexAttribute::step_function_ordinal
//! [`VertexAttribute::step_rate`]: crate::runtime::decode::resource::VertexAttribute::step_rate

/// `MTLVertexStepFunctionConstant` — one fetch for the whole draw.
pub const MTL_VERTEX_STEP_FUNCTION_CONSTANT: u32 = 0;
/// `MTLVertexStepFunctionPerVertex`.
pub const MTL_VERTEX_STEP_FUNCTION_PER_VERTEX: u32 = 1;
/// `MTLVertexStepFunctionPerInstance` — the rate is the instance divisor.
pub const MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE: u32 = 2;
/// `MTLVertexStepFunctionPerPatch` — tessellation only.
pub const MTL_VERTEX_STEP_FUNCTION_PER_PATCH: u32 = 3;
/// `MTLVertexStepFunctionPerPatchControlPoint` — tessellation only.
pub const MTL_VERTEX_STEP_FUNCTION_PER_PATCH_CONTROL_POINT: u32 = 4;

/// Whether the `(step function, step rate)` pair is one Metal accepts.
///
/// Zero is legal for exactly one step function and required by it. Under any
/// other, a zero rate advances nothing and `MTLVertexDescriptor` validation
/// rejects the descriptor — so refusing it by name is a report, not a policy.
///
/// The step function is taken as its raw ordinal rather than a narrowed type on
/// purpose: an undeclared ordinal has its own refusal at each backend, and this
/// predicate must not double as that one. It answers only "is the rate right for
/// this step", and for an ordinal neither backend accepts the answer is the same
/// as for `PerVertex` — which is what a caller checking the pair before the
/// ordinal would want anyway.
#[must_use]
pub fn step_rate_in_contract(step_function_ordinal: u32, step_rate: u32) -> bool {
    step_rate != 0 || step_function_ordinal == MTL_VERTEX_STEP_FUNCTION_CONSTANT
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero is in contract for Constant and for nothing else.
    ///
    /// Swept over every declared ordinal and past the top of the enum, because
    /// the rule is an equality against one ordinal and the cheap wrong version
    /// of it is a `<=` band.
    #[test]
    fn a_zero_step_rate_pairs_with_constant_and_nothing_else() {
        assert!(step_rate_in_contract(MTL_VERTEX_STEP_FUNCTION_CONSTANT, 0));
        for step in [
            MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
            MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
            MTL_VERTEX_STEP_FUNCTION_PER_PATCH,
            MTL_VERTEX_STEP_FUNCTION_PER_PATCH_CONTROL_POINT,
        ] {
            assert!(
                !step_rate_in_contract(step, 0),
                "step function {step} does not pair with a zero rate"
            );
        }
        for step in 5..=64u32 {
            assert!(!step_rate_in_contract(step, 0));
        }
        assert!(!step_rate_in_contract(u32::MAX, 0));
    }

    /// A nonzero rate is in contract under every step function, including the
    /// one that ignores it.
    ///
    /// Constant with a rate of 1 is what the tree's own Vulkan validation test
    /// was built from, and it is not what a guest sends — but nothing rejects
    /// it: `MTLVertexBufferLayoutDescriptor` only constrains the zero, and this
    /// predicate is not the place to invent a second constraint.
    #[test]
    fn a_nonzero_rate_is_in_contract_under_every_step_function() {
        for step in 0..=8u32 {
            for rate in [1u32, 2, 7, u32::MAX] {
                assert!(step_rate_in_contract(step, rate));
            }
        }
    }
}
