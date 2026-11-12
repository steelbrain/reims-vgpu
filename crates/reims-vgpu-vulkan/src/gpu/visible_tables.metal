// The tests' visible function tables (`reims-vgpu`'s decoder tests, `decode/function_table.rs`):
// two visible functions a pipeline links, and kernels that call them through a table, bound at a
// buffer index or held in an argument buffer. Each thread `i` writes what the table's entry
// `which[i]` answers of `i + 10`.
#include <metal_stdlib>
using namespace metal;

[[visible]] int table_add1(int x) {
    return x + 1;
}

[[visible]] int table_mul2(int x) {
    return x * 2;
}

kernel void table_kernel(visible_function_table<int(int)> table [[buffer(0)]],
                         device const uint *which [[buffer(1)]],
                         device int *out [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    out[i] = table[which[i]](int(i) + 10);
}

struct TableArguments {
    visible_function_table<int(int)> table [[id(0)]];
};

kernel void table_arg_kernel(device const TableArguments &arguments [[buffer(0)]],
                             device const uint *which [[buffer(1)]],
                             device int *out [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    out[i] = arguments.table[which[i]](int(i) + 10);
}
