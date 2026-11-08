/*
 * The layout of the libva video-processing structs vpp_ffi.rs declares,
 * from the installed headers: sizeof and offsetof of each, then the value
 * of each constant. vpp_ffi.rs's vpp_layouts_match_libva_headers test
 * compiles this and compares its output with Rust's, line for line.
 */
#include <stddef.h>
#include <stdio.h>
#include <va/va.h>
#include <va/va_vpp.h>

#define S(T) printf(#T " size %zu\n", sizeof(T))
#define F(T, f) printf(#T "." #f " %zu\n", offsetof(T, f))
#define C(c) printf(#c " %lld\n", (long long)(c))

int main(void)
{
    S(VARectangle);
    F(VARectangle, x);
    F(VARectangle, y);
    F(VARectangle, width);
    F(VARectangle, height);

    S(VAProcColorProperties);
    F(VAProcColorProperties, chroma_sample_location);
    F(VAProcColorProperties, color_range);
    F(VAProcColorProperties, colour_primaries);
    F(VAProcColorProperties, transfer_characteristics);
    F(VAProcColorProperties, matrix_coefficients);
    F(VAProcColorProperties, reserved);

    S(VAProcPipelineParameterBuffer);
    F(VAProcPipelineParameterBuffer, surface);
    F(VAProcPipelineParameterBuffer, surface_region);
    F(VAProcPipelineParameterBuffer, surface_color_standard);
    F(VAProcPipelineParameterBuffer, output_region);
    F(VAProcPipelineParameterBuffer, output_background_color);
    F(VAProcPipelineParameterBuffer, output_color_standard);
    F(VAProcPipelineParameterBuffer, pipeline_flags);
    F(VAProcPipelineParameterBuffer, filter_flags);
    F(VAProcPipelineParameterBuffer, filters);
    F(VAProcPipelineParameterBuffer, num_filters);
    F(VAProcPipelineParameterBuffer, forward_references);
    F(VAProcPipelineParameterBuffer, num_forward_references);
    F(VAProcPipelineParameterBuffer, backward_references);
    F(VAProcPipelineParameterBuffer, num_backward_references);
    F(VAProcPipelineParameterBuffer, rotation_state);
    F(VAProcPipelineParameterBuffer, blend_state);
    F(VAProcPipelineParameterBuffer, mirror_state);
    F(VAProcPipelineParameterBuffer, additional_outputs);
    F(VAProcPipelineParameterBuffer, num_additional_outputs);
    F(VAProcPipelineParameterBuffer, input_surface_flag);
    F(VAProcPipelineParameterBuffer, output_surface_flag);
    F(VAProcPipelineParameterBuffer, input_color_properties);
    F(VAProcPipelineParameterBuffer, output_color_properties);
    F(VAProcPipelineParameterBuffer, processing_mode);
    F(VAProcPipelineParameterBuffer, output_hdr_metadata);
    F(VAProcPipelineParameterBuffer, va_reserved);


    C(VAProcPipelineParameterBufferType);
    C(VAProcColorStandardNone);
    C(VAProcColorStandardBT709);
    C(VA_ROTATION_NONE);
    return 0;
}
