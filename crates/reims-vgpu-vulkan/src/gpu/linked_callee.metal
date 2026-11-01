// The tests' visible functions a pipeline links (`states::tests`): what `linked_caller.metal`'s
// functions call, and `linked_add_two`, which a pipeline links under the name they call by
// making it with that specialized name (`reims-vgpu`'s decoder tests, on Metal).
#include <metal_stdlib>
using namespace metal;

[[visible]] int linked_add_one(int x) {
    return x + 1;
}

[[visible]] int linked_add_two(int x) {
    return x + 2;
}
