/*
 * RgfxClMode: see RgfxClMode.hpp. Our own C++.
 */
#include "RgfxClMode.hpp"

bool RgfxClBindingsOf(const RgfxAirEntry &entry, RgfxClBindings &out, std::string &why)
{
	out = {};
	if (entry.stage != RgfxAirEntryStage::Kernel) {
		why = "it is no kernel";
		return false;
	}
	RgfxClBindings b;
	b.parameters = static_cast<uint32_t>(entry.arguments.size());
	for (const RgfxAirNode &n : entry.arguments) {
		if (!n.hasLocation || n.indexIsGlobal || n.countIsGlobal || n.predicated) {
			why = "an argument of " + n.kind + " is bound at no fixed index";
			return false;
		}
		const uint32_t index = n.index + kRgfxClHiddenBuffers;
		if (n.kind == "air.constant") {
			if (!n.hasPosition) {
				why = "a value argument has no position";
				return false;
			}
			b.byValue.push_back(index);
			b.byValuePositions.push_back(n.position);
			continue;
		}
		if (n.kind == "air.buffer" && n.hasAddressSpace && (n.addressSpace == 1 || n.addressSpace == 2)) {
			b.buffers.push_back(index);
			continue;
		}
		if (n.kind == "air.buffer" && n.hasAddressSpace && n.addressSpace == 3)
			continue; // __local: the threadgroup memory at its own index
		if ((n.kind == "air.texture" || n.kind == "air.sampler") && n.count == 1) {
			(n.kind == "air.texture" ? b.textures : b.samplers).push_back(n.index); // at its own index
			continue;
		}
		why = "an argument of " + n.kind + " is not one OpenCL mode was measured with";
		return false;
	}
	b.hasWorkGroupSize = entry.hasWorkGroupSize;
	for (int d = 0; d < 3; d++)
		b.workGroupSize[d] = entry.workGroupSize[d];
	out = std::move(b);
	return true;
}
