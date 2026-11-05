// The mesh render pipelines' functions of the tests (`gpu::encode_render::tests::mesh`): those of
// the graphics lane's mesh oracle (guest/bundle/tools/oracle/oracle_mesh.m), so a draw here
// compares with what it measured on Metal, a mesh function that writes a buffer, and a vertex
// function.
#include <metal_stdlib>
using namespace metal;

struct V {
    float4 p [[position]];
    float4 c;
};
using M = metal::mesh<V, void, 4, 2, topology::triangle>;
struct Payload {
    float4 c;
    uint n;
};

// Column x of n columns over the whole target, in colour c: four vertices, two triangles.
static void column(M m, uint t, uint x, uint n, float4 c) {
    float x0 = -1.0 + 2.0 * float(x) / float(n), x1 = -1.0 + 2.0 * float(x + 1) / float(n);
    if (t == 0) {
        m.set_primitive_count(2);
        m.set_index(0, 0);
        m.set_index(1, 1);
        m.set_index(2, 2);
        m.set_index(3, 2);
        m.set_index(4, 1);
        m.set_index(5, 3);
    }
    if (t < 4) {
        V v;
        v.p = float4((t & 1) ? x1 : x0, (t & 2) ? 1.0 : -1.0, 0, 1);
        v.c = c;
        m.set_vertex(t, v);
    }
}

// Column tg.x of the grid's columns, in the colour at mesh buffer 0.
[[mesh]] void m_plain(M m, uint t [[thread_index_in_threadgroup]],
                      uint3 tg [[threadgroup_position_in_grid]],
                      uint3 tgs [[threadgroups_per_grid]], constant float4 &c [[buffer(0)]]) {
    column(m, t, tg.x, tgs.x, c);
}

// The colour from mesh buffer 1, and the grid's threads in its alpha.
[[mesh]] void m_grid(M m, uint t [[thread_index_in_threadgroup]],
                     uint3 tg [[threadgroup_position_in_grid]],
                     uint3 tgs [[threadgroups_per_grid]], uint3 g [[threads_per_grid]],
                     device const float4 *c [[buffer(1)]]) {
    float4 k = c[0];
    k.a = float(g.x) / 255.0;
    column(m, t, tg.x, tgs.x, k);
}

// The colour sampled from mesh texture 0 through mesh sampler 0.
[[mesh]] void m_tex(M m, uint t [[thread_index_in_threadgroup]],
                    uint3 tg [[threadgroup_position_in_grid]], uint3 tgs [[threadgroups_per_grid]],
                    texture2d<float> tex [[texture(0)]], sampler s [[sampler(0)]]) {
    column(m, t, tg.x, tgs.x, tex.sample(s, float2(0.5), level(0)));
}

// The colour from the payload, and its threads count in its alpha.
[[mesh]] void m_payload(M m, const object_data Payload &p [[payload]],
                        uint t [[thread_index_in_threadgroup]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint3 tgs [[threadgroups_per_grid]]) {
    float4 k = p.c;
    k.a = float(p.n) / 255.0;
    column(m, t, tg.x, tgs.x, k);
}

// As m_plain, each threadgroup's thread 0 also writing its threadgroup's position, plus 1, into
// mesh buffer 2 at that position.
[[mesh]] void m_writes(M m, uint t [[thread_index_in_threadgroup]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       uint3 tgs [[threadgroups_per_grid]], constant float4 &c [[buffer(0)]],
                       device uint *out [[buffer(2)]]) {
    if (t == 0) {
        out[tg.x] = tg.x + 1;
    }
    column(m, t, tg.x, tgs.x, c);
}

// An object threadgroup writes its colour (object buffer 0 at its position, through threadgroup
// memory 0) and its threads into the payload, and launches two mesh threadgroups.
[[object]] void o_fn(object_data Payload &p [[payload]], mesh_grid_properties g,
                     uint t [[thread_index_in_threadgroup]], uint n [[threads_per_threadgroup]],
                     uint ot [[threadgroup_position_in_grid]], constant float4 *c [[buffer(0)]],
                     threadgroup float4 *s [[threadgroup(0)]]) {
    if (t == 0) {
        s[0] = c[ot];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        p.c = s[0];
        p.n = n;
        g.set_threadgroups_per_grid(uint3(2, 1, 1));
    }
}

fragment float4 f_col(V in [[stage_in]]) { return in.c; }

// A vertex function, for the refusal of a mesh pipeline whose mesh function is of another stage.
vertex V v_fn(uint i [[vertex_id]]) {
    V v;
    v.p = float4(0, 0, 0, 1);
    v.c = float4(float(i));
    return v;
}
