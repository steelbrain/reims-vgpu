// The tests' kernel of a function constant it reads only where the constant is defined
// (`reims-vgpu`'s decoder tests): made with constant values that set none, as a guest makes a
// function whose values set none of its constants, it writes 7; made with the constant set, the
// constant's value.
#include <metal_stdlib>
using namespace metal;

constant int optional_value [[function_constant(0)]];
constant bool has_optional_value = is_function_constant_defined(optional_value);

kernel void optional_constant(device int *out [[buffer(0)]]) {
    out[0] = has_optional_value ? optional_value : 7;
}
