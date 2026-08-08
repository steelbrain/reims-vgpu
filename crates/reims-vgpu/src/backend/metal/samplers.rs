//! Explicit + default sampler construction with content-hash cache.

use crate::backend::metal::abi::ReimsVgpuSampler;
use crate::backend::metal::cache::{sampler_insert, sampler_lookup, SamplerDescriptorKey};
use crate::backend::metal::mtl_enum;
use crate::backend::metal::runtime::cached_default_sampler;
use crate::backend::metal::util::{set_err, ErrOut, Status};
use metal::{
    Device, MTLCompareFunction, MTLSamplerAddressMode, MTLSamplerBorderColor,
    MTLSamplerMinMagFilter, MTLSamplerMipFilter, SamplerDescriptor, SamplerState,
};

pub fn make_default_sampler(device: &Device) -> SamplerState {
    cached_default_sampler(device)
}

/// One refusal shape for the eight sampler enum words, so each keeps its own
/// slug without eight copies of the message.
fn unsupported(err: ErrOut<'_>, slug: &'static str, field: &'static str, value: u32) -> Status {
    set_err(err, format!("unsupported sampler {field} {value}"));
    Status::args(slug).field(field, value)
}

pub fn validate_sampler_state(sampler: &ReimsVgpuSampler, err: ErrOut<'_>) -> Status {
    if sampler.min_filter > MTLSamplerMinMagFilter::Linear as u32 {
        set_err(err, "unsupported sampler min/mag filter");
        return Status::args("metal_sampler_min_filter_unsupported")
            .field("min_filter", sampler.min_filter);
    }
    if sampler.mag_filter > MTLSamplerMinMagFilter::Linear as u32 {
        set_err(err, "unsupported sampler min/mag filter");
        return Status::args("metal_sampler_mag_filter_unsupported")
            .field("mag_filter", sampler.mag_filter);
    }
    if sampler.mip_filter > MTLSamplerMipFilter::Linear as u32 {
        set_err(err, "unsupported sampler mip filter");
        return Status::args("metal_sampler_mip_filter_unsupported")
            .field("mip_filter", sampler.mip_filter);
    }
    if sampler.s_address_mode > MTLSamplerAddressMode::ClampToBorderColor as u32 {
        set_err(err, "unsupported sampler address mode");
        return Status::args("metal_sampler_address_s_unsupported")
            .field("mode", sampler.s_address_mode);
    }
    if sampler.t_address_mode > MTLSamplerAddressMode::ClampToBorderColor as u32 {
        set_err(err, "unsupported sampler address mode");
        return Status::args("metal_sampler_address_t_unsupported")
            .field("mode", sampler.t_address_mode);
    }
    if sampler.r_address_mode > MTLSamplerAddressMode::ClampToBorderColor as u32 {
        set_err(err, "unsupported sampler address mode");
        return Status::args("metal_sampler_address_r_unsupported")
            .field("mode", sampler.r_address_mode);
    }
    if sampler.border_color > MTLSamplerBorderColor::OpaqueWhite as u32 {
        set_err(err, "unsupported sampler border color");
        return Status::args("metal_sampler_border_color_unsupported")
            .field("border_color", sampler.border_color);
    }
    if sampler.compare_function > MTLCompareFunction::Always as u32 {
        set_err(err, "unsupported sampler compare function");
        return Status::args("metal_sampler_compare_function_unsupported")
            .field("compare", sampler.compare_function);
    }
    if sampler.max_anisotropy == 0 {
        set_err(err, "sampler maxAnisotropy must be non-zero");
        return Status::args("metal_sampler_anisotropy_zero");
    }
    Status::OK
}

pub fn make_explicit_sampler(
    device: &Device,
    sampler: &ReimsVgpuSampler,
    err: ErrOut<'_>,
) -> Result<SamplerState, Status> {
    let rc = validate_sampler_state(sampler, err);
    if !rc.is_ok() {
        return Err(rc);
    }
    // Every one of the eight enum words is converted rather than reinterpreted.
    // What this replaced was `fn enum_u64<T: Copy>(v: u32) -> T`, a *generic*
    // `transmute_copy` whose output type came from whatever setter it was
    // passed to — so one line could produce an undefined discriminant for eight
    // different Metal enums, and `transmute_copy` does not even require the
    // sizes to match the way `transmute` does.
    //
    // `validate_sampler_state` above bounds each of these, and every one of its
    // bounds happens to name its enum's real last variant, so nothing here can
    // currently be `None`. That is the point: it was true by coincidence across
    // two functions and nothing in the types said so.
    let Some(min_filter) = mtl_enum::sampler_min_mag_filter(sampler.min_filter) else {
        return Err(unsupported(
            err,
            "metal_sampler_min_filter_unsupported",
            "min_filter",
            sampler.min_filter,
        ));
    };
    let Some(mag_filter) = mtl_enum::sampler_min_mag_filter(sampler.mag_filter) else {
        return Err(unsupported(
            err,
            "metal_sampler_mag_filter_unsupported",
            "mag_filter",
            sampler.mag_filter,
        ));
    };
    let Some(mip_filter) = mtl_enum::sampler_mip_filter(sampler.mip_filter) else {
        return Err(unsupported(
            err,
            "metal_sampler_mip_filter_unsupported",
            "mip_filter",
            sampler.mip_filter,
        ));
    };
    let Some(address_s) = mtl_enum::sampler_address_mode(sampler.s_address_mode) else {
        return Err(unsupported(
            err,
            "metal_sampler_address_s_unsupported",
            "mode",
            sampler.s_address_mode,
        ));
    };
    let Some(address_t) = mtl_enum::sampler_address_mode(sampler.t_address_mode) else {
        return Err(unsupported(
            err,
            "metal_sampler_address_t_unsupported",
            "mode",
            sampler.t_address_mode,
        ));
    };
    let Some(address_r) = mtl_enum::sampler_address_mode(sampler.r_address_mode) else {
        return Err(unsupported(
            err,
            "metal_sampler_address_r_unsupported",
            "mode",
            sampler.r_address_mode,
        ));
    };
    let Some(border_color) = mtl_enum::sampler_border_color(sampler.border_color) else {
        return Err(unsupported(
            err,
            "metal_sampler_border_color_unsupported",
            "border_color",
            sampler.border_color,
        ));
    };
    let Some(compare) = mtl_enum::compare_function(sampler.compare_function) else {
        return Err(unsupported(
            err,
            "metal_sampler_compare_function_unsupported",
            "compare",
            sampler.compare_function,
        ));
    };
    let key = SamplerDescriptorKey::new(sampler);
    if let Some(hit) = sampler_lookup(&key) {
        return Ok(hit);
    }

    let descriptor = SamplerDescriptor::new();
    descriptor.set_min_filter(min_filter);
    descriptor.set_mag_filter(mag_filter);
    descriptor.set_mip_filter(mip_filter);
    descriptor.set_address_mode_s(address_s);
    descriptor.set_address_mode_t(address_t);
    descriptor.set_address_mode_r(address_r);
    descriptor.set_border_color(border_color);
    descriptor.set_compare_function(compare);
    descriptor.set_max_anisotropy(sampler.max_anisotropy as u64);
    descriptor.set_normalized_coordinates(sampler.unnormalized == 0);
    descriptor.set_lod_min_clamp(f32::from_bits(sampler.lod_min_bits));
    descriptor.set_lod_max_clamp(f32::from_bits(sampler.lod_max_bits));
    descriptor.set_lod_average(sampler.lod_average != 0);
    descriptor.set_support_argument_buffers(sampler.support_argument_buffers != 0);

    let state = device.new_sampler(&descriptor);
    Ok(sampler_insert(key, state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Emit;

    fn sampler() -> ReimsVgpuSampler {
        ReimsVgpuSampler {
            binding: 0,
            unnormalized: 0,
            min_filter: 0,
            mag_filter: 0,
            mip_filter: 0,
            s_address_mode: 0,
            t_address_mode: 0,
            r_address_mode: 0,
            border_color: 0,
            compare_function: 0,
            lod_min_bits: 0,
            lod_max_bits: 0,
            max_anisotropy: 1,
            lod_average: 0,
            support_argument_buffers: 0,
            has_lod_clamp: 0,
            clamp_lod_min_bits: 0,
            clamp_lod_max_bits: 0,
        }
    }

    fn line(status: Status) -> String {
        Emit::refusal("metal_sampler_test", &status)
            .expect("invalid sampler must carry a refusal")
            .render()
    }

    #[test]
    fn sampler_filter_axes_keep_distinct_reasons_and_values() {
        let mut min = sampler();
        min.min_filter = MTLSamplerMinMagFilter::Linear as u32 + 1;
        assert_eq!(
            line(validate_sampler_state(
                &min,
                (std::ptr::null_mut(), 0)
            )),
            "metal_sampler_test reason=metal_sampler_min_filter_unsupported class=args min_filter=2"
        );

        let mut mag = sampler();
        mag.mag_filter = MTLSamplerMinMagFilter::Linear as u32 + 1;
        assert_eq!(
            line(validate_sampler_state(
                &mag,
                (std::ptr::null_mut(), 0)
            )),
            "metal_sampler_test reason=metal_sampler_mag_filter_unsupported class=args mag_filter=2"
        );
    }
}
