#include <metal_stdlib>
using namespace metal;

// Object-stage only metallib for the mesh serializer resource's object_func_ref (tag 0x01
// under section tag 0x14). Payload layout must match icb_mesh_with_payload.

struct Payload {
    float scale;
};

[[object]]
void object_main(object_data Payload &out [[payload]], mesh_grid_properties mgp) {
    out.scale = 1.0f;
    mgp.set_threadgroups_per_grid(uint3(1, 1, 1));
}
