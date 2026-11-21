// The rt platform's indirect command buffer tests (crates/reims-vgpu-vulkan/src/rt_platform):
// indexed triangles that a kernel encodes into an indirect command buffer and a render encoder
// executes. Kernel `icb_encode` encodes command `tid`: vertex buffer 0 the positions, fragment
// buffer 0 its own colour of `colours`, and a draw of the three 16-bit indices from
// `indices + 3 * tid`. `icb_vertex` places vertex `vid` at `positions[vid]`, and `icb_fragment`
// returns its command's colour. `icb_encode_wide` binds the positions at vertex buffers 0 to 28
// as well; `icb_vertex_N`, for N from 23 to 29, reads them from vertex buffers 0 to N - 1, so a
// command of it and `icb_fragment` binds N + 1 slots.
#include <metal_stdlib>
using namespace metal;

struct Held {
    command_buffer commands;
};

struct Placed {
    float4 position [[position]];
};

kernel void icb_encode(device const Held &held [[buffer(0)]],
                       device const float4 *positions [[buffer(1)]],
                       constant float4 *colours [[buffer(2)]],
                       device const ushort *indices [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
    render_command command(held.commands, tid);
    command.set_vertex_buffer(positions, 0);
    command.set_fragment_buffer(colours + tid, 0);
    command.draw_indexed_primitives(primitive_type::triangle, 3, indices + 3 * tid, 1, 0, 0);
}

vertex Placed icb_vertex(uint vid [[vertex_id]], device const float4 *positions [[buffer(0)]]) {
    Placed out;
    out.position = positions[vid];
    return out;
}

fragment float4 icb_fragment(constant float4 &colour [[buffer(0)]]) {
    return colour;
}

#define SET(n) command.set_vertex_buffer(positions, n);

kernel void icb_encode_wide(device const Held &held [[buffer(0)]],
                            device const float4 *positions [[buffer(1)]],
                            constant float4 *colours [[buffer(2)]],
                            device const ushort *indices [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    render_command command(held.commands, tid);
    SET(0) SET(1) SET(2) SET(3) SET(4) SET(5) SET(6) SET(7) SET(8) SET(9)
    SET(10) SET(11) SET(12) SET(13) SET(14) SET(15) SET(16) SET(17) SET(18) SET(19)
    SET(20) SET(21) SET(22) SET(23) SET(24) SET(25) SET(26) SET(27) SET(28)
    command.set_fragment_buffer(colours + tid, 0);
    command.draw_indexed_primitives(primitive_type::triangle, 3, indices + 3 * tid, 1, 0, 0);
}

#define P(n) device const float4 *p##n [[buffer(n)]]
#define S(n) + p##n[vid]

#define PARAMS_23 P(0), P(1), P(2), P(3), P(4), P(5), P(6), P(7), P(8), P(9), P(10), P(11), \
                  P(12), P(13), P(14), P(15), P(16), P(17), P(18), P(19), P(20), P(21), P(22)
#define SUM_23 (p0[vid] S(1) S(2) S(3) S(4) S(5) S(6) S(7) S(8) S(9) S(10) S(11) S(12) S(13) \
                S(14) S(15) S(16) S(17) S(18) S(19) S(20) S(21) S(22))
#define PARAMS_24 PARAMS_23, P(23)
#define SUM_24 (SUM_23 S(23))
#define PARAMS_25 PARAMS_24, P(24)
#define SUM_25 (SUM_24 S(24))
#define PARAMS_26 PARAMS_25, P(25)
#define SUM_26 (SUM_25 S(25))
#define PARAMS_27 PARAMS_26, P(26)
#define SUM_27 (SUM_26 S(26))
#define PARAMS_28 PARAMS_27, P(27)
#define SUM_28 (SUM_27 S(27))
#define PARAMS_29 PARAMS_28, P(28)
#define SUM_29 (SUM_28 S(28))

// The sum of N copies of a position: the same point once divided by its w.
#define VERTEX(n) \
    vertex Placed icb_vertex_##n(uint vid [[vertex_id]], PARAMS_##n) { \
        Placed out; \
        out.position = SUM_##n; \
        return out; \
    }

VERTEX(23)
VERTEX(24)
VERTEX(25)
VERTEX(26)
VERTEX(27)
VERTEX(28)
VERTEX(29)
