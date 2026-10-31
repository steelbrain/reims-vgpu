// The tests' visible function a pipeline links (`states::tests`): what `linked_caller.metal`'s
// functions call.
#include <metal_stdlib>
using namespace metal;

[[visible]] int linked_add_one(int x) {
    return x + 1;
}
