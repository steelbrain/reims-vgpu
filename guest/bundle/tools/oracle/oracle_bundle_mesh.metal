// oracle_bundle's mesh and object functions (oracle_bundle.m, case functypes).
#include <metal_stdlib>
using namespace metal;

struct DivMeshVertex {
	float4 position [[position]];
};

using DivMesh = metal::mesh<DivMeshVertex, void, 3, 1, metal::topology::triangle>;

[[mesh]] void div_mesh(DivMesh output, uint tid [[thread_index_in_threadgroup]])
{
	if (tid == 0) {
		output.set_primitive_count(1);
		output.set_index(0, 0);
		output.set_index(1, 1);
		output.set_index(2, 2);
	}
	if (tid < 3) {
		DivMeshVertex v;
		v.position = float4(float(tid), 0.0f, 0.0f, 1.0f);
		output.set_vertex(tid, v);
	}
}

[[object]] void div_object(mesh_grid_properties props)
{
	props.set_threadgroups_per_grid(uint3(1, 1, 1));
}
