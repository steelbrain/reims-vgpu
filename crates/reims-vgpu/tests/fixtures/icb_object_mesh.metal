#include <metal_stdlib>
using namespace metal;

// Object stage + mesh stage dual-export metallib (fallback product path).
// Product fill selects by function type (object=8, mesh=7) when mesh SPI
// serializer-resource shape (tag 0x14) is absent. Prefer separate metallibs + mesh SPI
// tags 0x01/0x02/0x03 when available (see icb_object_stage /
// icb_mesh_with_payload).

struct VertexOut {
    float4 position [[position]];
};

struct Payload {
    float scale;
};

using MeshOut = metal::mesh<VertexOut, void, 3, 1, topology::triangle>;

[[object]]
void object_main(object_data Payload &out [[payload]], mesh_grid_properties mgp) {
    out.scale = 1.0f;
    mgp.set_threadgroups_per_grid(uint3(1, 1, 1));
}

[[mesh]]
void mesh_main(
    object_data Payload const& in [[payload]],
    uint tid [[thread_index_in_threadgroup]],
    MeshOut out)
{
    if (tid == 0) {
        float s = in.scale;
        out.set_vertex(0, VertexOut{float4(-1.0f * s, -1.0f * s, 0.0f, 1.0f)});
        out.set_vertex(1, VertexOut{float4(3.0f * s, -1.0f * s, 0.0f, 1.0f)});
        out.set_vertex(2, VertexOut{float4(-1.0f * s, 3.0f * s, 0.0f, 1.0f)});
        out.set_index(0, 0);
        out.set_index(1, 1);
        out.set_index(2, 2);
        out.set_primitive_count(1);
    }
}
