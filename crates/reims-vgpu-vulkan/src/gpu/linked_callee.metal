// The tests' visible functions a pipeline links (`states::tests`): what `linked_caller.metal`'s
// functions call, and `linked_add_two`, which a pipeline links under the name they call by
// making it with that specialized name (`reims-vgpu`'s decoder tests).
#include <metal_stdlib>
using namespace metal;

[[visible]] int linked_add_one(int x) {
    return x + 1;
}

[[visible]] int linked_add_two(int x) {
    return x + 2;
}

// The function a pipeline links under `custom_fn` (`linked_caller.metal`'s
// `linked_effect_fragment` calls it), as RenderBox links its gradients: made with that
// specialized name and a value for a function constant of its own.
constant float linked_gradient_scale [[function_constant(1)]];

[[visible]] float4 linked_gradient(float2 p) {
    return float4(p.x * linked_gradient_scale, p.y * 0.25, 0.5, 0.0);
}
