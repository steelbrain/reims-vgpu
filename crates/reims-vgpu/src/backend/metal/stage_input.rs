//! Compute stage-input descriptor construction.

use crate::backend::metal::abi::{
    ReimsVgpuComputeStageInputDescriptor, ReimsVgpuComputeStageInputLayout,
    REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES, REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS,
    REIMS_VGPU_COMPUTE_STAGE_INPUT_STRIDE_DYNAMIC,
};
use crate::backend::metal::constants::{
    MTL_BUFFER_LAYOUT_STRIDE_DYNAMIC, REIMS_VGPU_METAL_MAX_ATTRS, REIMS_VGPU_METAL_MAX_BUFFERS,
};
use crate::backend::metal::mtl_enum;
use crate::backend::metal::util::{set_err, valid_buffer_binding, ErrOut, Status};
use metal::{MTLAttributeFormat, StageInputOutputDescriptor};

// Apple Metal.framework MTLStepFunction raw values (MTLStageInputOutputDescriptor.h).
// Do not use metal-0.33's MTLStepFunction names — that crate assigns six of the
// enum's nine names to the wrong numbers, not just the X/Y Indexed pair this
// comment used to name. [`mtl_enum::step_function`] carries the full table and
// the test that pins it.
const MTL_STEP_THREAD_POS_IN_GRID_X: u32 = 5;
const MTL_STEP_THREAD_POS_IN_GRID_Y: u32 = 6;
const MTL_STEP_THREAD_POS_IN_GRID_X_INDEXED: u32 = 7;
const MTL_STEP_THREAD_POS_IN_GRID_Y_INDEXED: u32 = 8;

pub const fn step_supported(step_function: u32) -> bool {
    matches!(
        step_function,
        MTL_STEP_THREAD_POS_IN_GRID_X
            | MTL_STEP_THREAD_POS_IN_GRID_Y
            | MTL_STEP_THREAD_POS_IN_GRID_X_INDEXED
            | MTL_STEP_THREAD_POS_IN_GRID_Y_INDEXED
    )
}

pub const fn step_indexed(step_function: u32) -> bool {
    matches!(
        step_function,
        MTL_STEP_THREAD_POS_IN_GRID_X_INDEXED | MTL_STEP_THREAD_POS_IN_GRID_Y_INDEXED
    )
}

// The two lists of `MTLStepFunction` in this crate, related.
//
// `make_compute_stage_input_descriptor` runs `step_supported` and then converts
// through [`mtl_enum::step_function`], and it carries a distinct refusal
// (`metal_stage_input_step_function_unconvertible`) for the case where the first
// admits an ordinal the second cannot convert — a divergence between two lists
// of one enum rather than a guest sending something unsupported. That refusal is
// a healthy-zero alarm and stays, because the conversion returns an `Option` and
// something must handle `None`. But the property it watches for does not need a
// boot to observe: both functions are `const`, so the implication is decidable
// here, on every arm that compiles the file.
//
// The four ordinals above are Apple's, read off `MTLStageInputOutputDescriptor.h`
// because `metal` 0.33's names for them are wrong. Nothing else in the tree
// relates them to `STEP_FUNCTION_BY_ORDINAL`, which is indexed by those same
// Apple numbers — so an edit to that table's length or contents could move the
// convertible set out from under this one silently.
//
// Swept past the top of the enum rather than over the four: the implication is
// what is being pinned, and a fifth ordinal added to `step_supported` without a
// table entry is exactly the mistake it exists to catch.
const _: () = {
    let mut ordinal = 0u32;
    while ordinal <= 64 {
        assert!(
            !step_supported(ordinal) || mtl_enum::step_function(ordinal).is_some(),
            "step_supported admits a step function mtl_enum cannot convert",
        );
        assert!(
            !step_indexed(ordinal) || step_supported(ordinal),
            "an indexed step function must be a supported one",
        );
        ordinal += 1;
    }
    assert!(!step_supported(u32::MAX));
};

pub fn has_indexed_layout(stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>) -> bool {
    let Some(stage_input) = stage_input else {
        return false;
    };
    let layout_count =
        (stage_input.layout_count as usize).min(REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS);
    for i in 0..layout_count {
        if step_indexed(stage_input.layouts[i].step_function) {
            return true;
        }
    }
    false
}

pub fn layout_for_buffer(
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    binding: u32,
) -> (Option<ReimsVgpuComputeStageInputLayout>, bool) {
    let Some(stage_input) = stage_input else {
        return (None, false);
    };
    let layout_count =
        (stage_input.layout_count as usize).min(REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS);
    let mut layout = None;
    for i in 0..layout_count {
        if stage_input.layouts[i].buffer_index == binding {
            layout = Some(stage_input.layouts[i]);
            break;
        }
    }
    let attr_count =
        (stage_input.attribute_count as usize).min(REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES);
    let mut has_attribute = false;
    for i in 0..attr_count {
        if stage_input.attributes[i].buffer_index == binding {
            has_attribute = true;
            break;
        }
    }
    (layout, has_attribute)
}

pub fn make_compute_stage_input_descriptor(
    stage_input: &ReimsVgpuComputeStageInputDescriptor,
    err: ErrOut<'_>,
) -> Result<StageInputOutputDescriptor, Status> {
    if stage_input.attribute_count as usize > REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES {
        set_err(err, "invalid compute stageInputDescriptor entry count");
        return Err(Status::args("metal_stage_input_attribute_count_exceeded")
            .field("attributes", stage_input.attribute_count)
            .field("limit", REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES));
    }
    if stage_input.layout_count as usize > REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS {
        set_err(err, "invalid compute stageInputDescriptor entry count");
        return Err(Status::args("metal_stage_input_layout_count_exceeded")
            .field("layouts", stage_input.layout_count)
            .field("limit", REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS));
    }

    let descriptor = StageInputOutputDescriptor::new().to_owned();
    let mut layout_seen = [false; REIMS_VGPU_METAL_MAX_BUFFERS];
    let mut has_indexed = false;

    for i in 0..stage_input.layout_count as usize {
        let layout = &stage_input.layouts[i];
        if !valid_buffer_binding(layout.buffer_index) {
            set_err(
                err,
                format!(
                    "compute stageInputDescriptor layout buffer {} out of range",
                    layout.buffer_index
                ),
            );
            return Err(Status::args("metal_stage_input_layout_buffer_out_of_range")
                .field("buffer", layout.buffer_index)
                .field("limit", REIMS_VGPU_METAL_MAX_BUFFERS));
        }
        if layout_seen[layout.buffer_index as usize] {
            set_err(
                err,
                format!(
                    "duplicate compute stageInputDescriptor layout buffer {}",
                    layout.buffer_index
                ),
            );
            return Err(Status::args("metal_stage_input_layout_buffer_duplicate")
                .field("buffer", layout.buffer_index));
        }
        if !step_supported(layout.step_function) {
            set_err(
                err,
                format!(
                    "unsupported compute stageInputDescriptor step function {}",
                    layout.step_function
                ),
            );
            return Err(Status::args("metal_stage_input_step_function_unsupported")
                .field("step", layout.step_function));
        }
        if step_indexed(layout.step_function) {
            has_indexed = true;
        }
        let metal_layout = descriptor
            .layouts()
            .and_then(|a| a.object_at(layout.buffer_index as u64))
            .ok_or_else(|| {
                set_err(err, "compute stageInputDescriptor layouts unavailable");
                Status::execute("metal_stage_input_layouts_unavailable")
                    .field("buffer", layout.buffer_index)
            })?;
        let stride = if layout.stride == REIMS_VGPU_COMPUTE_STAGE_INPUT_STRIDE_DYNAMIC {
            MTL_BUFFER_LAYOUT_STRIDE_DYNAMIC
        } else {
            layout.stride
        };
        metal_layout.set_stride(stride);
        // `step_supported` above already narrows to the four compute forms, so
        // this cannot be `None`; it is converted rather than transmuted so the
        // two are not a cross-function soundness invariant.
        //
        // Its own slug, and not the one that guard uses: the two ask different
        // questions and this one firing means they *disagree* — an ordinal
        // `step_supported` admitted that `mtl_enum`'s table cannot convert,
        // which is a divergence between two lists of the same enum rather than
        // a guest sending something unsupported. Sharing a slug with the guard
        // also shared `fail_once`'s latch, so whichever fired first silenced the
        // other for that pipeline.
        let Some(step) = mtl_enum::step_function(layout.step_function) else {
            set_err(
                err,
                format!(
                    "unsupported compute stageInputDescriptor step function {}",
                    layout.step_function
                ),
            );
            return Err(
                Status::args("metal_stage_input_step_function_unconvertible")
                    .field("step", layout.step_function),
            );
        };
        metal_layout.set_step_function(step);
        metal_layout.set_step_rate(layout.step_rate as u64);
        layout_seen[layout.buffer_index as usize] = true;
    }

    if has_indexed {
        let Some(index_type) = mtl_enum::index_type(stage_input.index_type) else {
            set_err(
                err,
                format!(
                    "unsupported compute stageInputDescriptor index type {}",
                    stage_input.index_type
                ),
            );
            return Err(Status::args("metal_stage_input_index_type_unsupported")
                .field("index_type", stage_input.index_type));
        };
        if !valid_buffer_binding(stage_input.index_buffer_index) {
            set_err(
                err,
                format!(
                    "compute stageInputDescriptor index buffer {} out of range",
                    stage_input.index_buffer_index
                ),
            );
            return Err(Status::args("metal_stage_input_index_buffer_out_of_range")
                .field("buffer", stage_input.index_buffer_index)
                .field("limit", REIMS_VGPU_METAL_MAX_BUFFERS));
        }
        descriptor.set_index_type(index_type);
        descriptor.set_index_buffer_index(stage_input.index_buffer_index as u64);
    }

    let mut attribute_seen = [false; REIMS_VGPU_METAL_MAX_ATTRS];
    for i in 0..stage_input.attribute_count as usize {
        let attr = &stage_input.attributes[i];
        if attr.location as usize >= REIMS_VGPU_METAL_MAX_ATTRS {
            set_err(
                err,
                format!(
                    "compute stageInputDescriptor attribute {} out of range",
                    attr.location
                ),
            );
            return Err(Status::args("metal_stage_input_attribute_out_of_range")
                .field("location", attr.location)
                .field("limit", REIMS_VGPU_METAL_MAX_ATTRS));
        }
        if attribute_seen[attr.location as usize] {
            set_err(
                err,
                format!(
                    "duplicate compute stageInputDescriptor attribute {}",
                    attr.location
                ),
            );
            return Err(Status::args("metal_stage_input_attribute_duplicate")
                .field("location", attr.location));
        }
        if !valid_buffer_binding(attr.buffer_index) {
            set_err(
                err,
                format!(
                    "compute stageInputDescriptor attribute {} references missing buffer {}",
                    attr.location, attr.buffer_index
                ),
            );
            return Err(
                Status::args("metal_stage_input_attribute_buffer_out_of_range")
                    .field("location", attr.location)
                    .field("buffer", attr.buffer_index)
                    .field("limit", REIMS_VGPU_METAL_MAX_BUFFERS),
            );
        }
        if !layout_seen[attr.buffer_index as usize] {
            set_err(
                err,
                format!(
                    "compute stageInputDescriptor attribute {} references missing buffer {}",
                    attr.location, attr.buffer_index
                ),
            );
            return Err(Status::args("metal_stage_input_attribute_layout_missing")
                .field("location", attr.location)
                .field("buffer", attr.buffer_index));
        }
        // `Invalid` is a declared variant, so the conversion accepts it and the
        // explicit refusal has to stay. What the conversion replaced is the
        // upper bound beside it, which was `MTL_ATTRIBUTE_FORMAT_FLOAT_RGB9E5 =
        // 54` — wrong twice over. Apple's `FloatRGB9E5` is 55, not 54, so the
        // constant carried `FloatRG11B10`'s value under `FloatRGB9E5`'s name;
        // and a bound of any kind admits 43 and 44, which Apple leaves
        // unassigned between `UChar4Normalized_BGRA` and `UChar`.
        let Some(format) = mtl_enum::attribute_format(attr.format)
            .filter(|_| attr.format != MTLAttributeFormat::Invalid as u32)
        else {
            set_err(
                err,
                format!(
                    "unsupported compute stageInputDescriptor attribute format {}",
                    attr.format
                ),
            );
            return Err(
                Status::args("metal_stage_input_attribute_format_unsupported")
                    .field("location", attr.location)
                    .field("format", attr.format),
            );
        };
        let metal_attr = descriptor
            .attributes()
            .and_then(|a| a.object_at(attr.location as u64))
            .ok_or_else(|| {
                set_err(err, "compute stageInputDescriptor attributes unavailable");
                Status::execute("metal_stage_input_attributes_unavailable")
                    .field("location", attr.location)
            })?;
        metal_attr.set_format(format);
        metal_attr.set_offset(attr.offset as u64);
        metal_attr.set_buffer_index(attr.buffer_index as u64);
        attribute_seen[attr.location as usize] = true;
    }

    Ok(descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Emit;

    fn empty_descriptor() -> ReimsVgpuComputeStageInputDescriptor {
        // All fields are integers or fixed arrays of integer-only C structs.
        unsafe { std::mem::zeroed() }
    }

    fn refused_line(stage: &ReimsVgpuComputeStageInputDescriptor) -> String {
        let status = match make_compute_stage_input_descriptor(stage, (std::ptr::null_mut(), 0)) {
            Ok(_) => panic!("invalid stage-input descriptor unexpectedly succeeded"),
            Err(status) => status,
        };
        Emit::refusal("metal_stage_input_test", &status)
            .expect("invalid stage input must carry a refusal")
            .render()
    }

    // Both caps are derived from the decoder's own count mask, so the expected
    // lines are spelled from the constants rather than from the numbers they
    // happen to hold. Written as literals, this test asserted `limit=16` and
    // went red the day the wire's 5-bit count field widened the cap to 31 — the
    // failure was the test's own second spelling and not a refusal that had
    // stopped naming its limit.
    #[test]
    fn stage_input_attribute_and_layout_caps_have_distinct_reasons() {
        let attr_cap = REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES;
        let mut attributes = empty_descriptor();
        attributes.attribute_count = attr_cap as u32 + 1;
        assert_eq!(
            refused_line(&attributes),
            format!(
                "metal_stage_input_test reason=metal_stage_input_attribute_count_exceeded \
                 class=args attributes={} limit={attr_cap}",
                attr_cap + 1
            )
        );

        let layout_cap = REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS;
        let mut layouts = empty_descriptor();
        layouts.layout_count = layout_cap as u32 + 1;
        assert_eq!(
            refused_line(&layouts),
            format!(
                "metal_stage_input_test reason=metal_stage_input_layout_count_exceeded \
                 class=args layouts={} limit={layout_cap}",
                layout_cap + 1
            )
        );
    }
}
