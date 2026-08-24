//! `MTLVertexStepFunction` ordinals, and the step rate that pairs with each.
//!
//! A render-pipeline descriptor's buffer layout carries a step function and a
//! step rate, decoded from render-pipeline vertex attributes.
//! The two are one rule and not two fields: `MTLVertexBufferLayoutDescriptor`
//! requires `stepRate == 0` for `MTLVertexStepFunctionConstant` and rejects it
//! for every other step function, because a constant-rate attribute is fetched
//! once for the whole draw and a rate of zero is how that is spelled.
//!
//! # Why the pair lives here
//!
//! Draw validation once asked `rate == 0` alone, so it declined the canonical Constant spelling — with
//! `vk_draw_validate_zero_vertex_step_rate`, a decline that loses the whole
//! draw — while ignoring the rate for exactly that step function everywhere
//! downstream (its divisor is 0 whatever the rate says). The decoder's own doc
//! settles the rule: "a layout that declared **zero** means zero —
//! that is what `MTLVertexStepFunctionConstant` pairs with — so nothing here
//! clamps it up".
//!
//! The ordinals live here because they are guest contract values consumed by
//! both decode validation and Vulkan translation.
//!
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum VertexStepFunction {
    Constant,
    #[default]
    PerVertex,
    PerInstance,
}

impl VertexStepFunction {
    pub const fn mtl_ordinal(self) -> u32 {
        match self {
            Self::Constant => MTL_VERTEX_STEP_FUNCTION_CONSTANT,
            Self::PerVertex => MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
            Self::PerInstance => MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VertexStepDecodeError {
    TessellationUnsupported(u32),
    Unknown(u32),
}

impl VertexStepDecodeError {
    pub const fn raw(self) -> u32 {
        match self {
            Self::TessellationUnsupported(raw) | Self::Unknown(raw) => raw,
        }
    }

    pub const fn slug(self) -> &'static str {
        match self {
            Self::TessellationUnsupported(_) => "vertex_step_function_per_patch",
            Self::Unknown(_) => "unknown_vertex_step_function",
        }
    }
}

pub const fn decode_vertex_step_function(
    declared: Option<u32>,
) -> Result<VertexStepFunction, VertexStepDecodeError> {
    let Some(raw) = declared else {
        return Ok(VertexStepFunction::PerVertex);
    };
    match raw {
        MTL_VERTEX_STEP_FUNCTION_CONSTANT => Ok(VertexStepFunction::Constant),
        MTL_VERTEX_STEP_FUNCTION_PER_VERTEX => Ok(VertexStepFunction::PerVertex),
        MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE => Ok(VertexStepFunction::PerInstance),
        MTL_VERTEX_STEP_FUNCTION_PER_PATCH | MTL_VERTEX_STEP_FUNCTION_PER_PATCH_CONTROL_POINT => {
            Err(VertexStepDecodeError::TessellationUnsupported(raw))
        }
        other => Err(VertexStepDecodeError::Unknown(other)),
    }
}

/// Whether the `(step function, step rate)` pair is one Metal accepts.
///
/// Zero is legal for exactly one step function and required by it. Under any
/// other, a zero rate advances nothing and `MTLVertexDescriptor` validation
/// rejects the descriptor — so refusing it by name is a report, not a policy.
///
/// The step function is taken as its raw ordinal rather than a narrowed type on
/// purpose: an undeclared ordinal has its own refusal at translation, and this
/// predicate must not double as that one. It answers only "is the rate right for
/// this step", and for an ordinal the backend cannot accept the answer is the same
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

    #[test]
    fn step_function_decode_distinguishes_tessellation_from_unknown_values() {
        assert_eq!(
            decode_vertex_step_function(None),
            Ok(VertexStepFunction::PerVertex)
        );
        assert_eq!(
            decode_vertex_step_function(Some(MTL_VERTEX_STEP_FUNCTION_CONSTANT)),
            Ok(VertexStepFunction::Constant)
        );
        assert_eq!(
            decode_vertex_step_function(Some(MTL_VERTEX_STEP_FUNCTION_PER_PATCH)),
            Err(VertexStepDecodeError::TessellationUnsupported(
                MTL_VERTEX_STEP_FUNCTION_PER_PATCH
            ))
        );
        assert_eq!(
            decode_vertex_step_function(Some(5)),
            Err(VertexStepDecodeError::Unknown(5))
        );
    }
}
