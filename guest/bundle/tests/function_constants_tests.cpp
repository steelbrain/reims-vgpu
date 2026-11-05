/*
 * function_constants_tests: the names Metal's errors give MTLDataType's values
 * (RgfxFunctionConstants.hpp), part of the serialiser check (serializer_tests.cpp calls
 * rgfxFunctionConstantsTests; guest/bundle/Makefile `test`). No Metal.
 */

#include <cstdint>
#include <cstdio>
#include <string>

#include "RgfxFunctionConstants.hpp"
#include "rgfx_test_ring.hpp" // CHECK, g_checks

namespace {

// Every MTLDataType value and the name Metal's error gave it, a value of that type set for an int
// constant (measured: guest/bundle/tools/oracle/oracle_names, the same on the AMD Radeon Pro
// 5300M and the Intel UHD Graphics 630, macOS 26.7.1).
struct Measured {
	uint32_t type;
	const char *name;
};
constexpr Measured kMeasured[] = {
	{ 0, "MTLDataTypeNone" },
	{ 1, "MTLDataTypeStruct" },
	{ 2, "MTLDataTypeArray" },
	{ 3, "MTLDataTypeFloat" },
	{ 4, "MTLDataTypeFloat2" },
	{ 5, "MTLDataTypeFloat3" },
	{ 6, "MTLDataTypeFloat4" },
	{ 7, "MTLDataTypeFloat2x2" },
	{ 8, "MTLDataTypeFloat2x3" },
	{ 9, "MTLDataTypeFloat2x4" },
	{ 10, "MTLDataTypeFloat3x2" },
	{ 11, "MTLDataTypeFloat3x3" },
	{ 12, "MTLDataTypeFloat3x4" },
	{ 13, "MTLDataTypeFloat4x2" },
	{ 14, "MTLDataTypeFloat4x3" },
	{ 15, "MTLDataTypeFloat4x4" },
	{ 16, "MTLDataTypeHalf" },
	{ 17, "MTLDataTypeHalf2" },
	{ 18, "MTLDataTypeHalf3" },
	{ 19, "MTLDataTypeHalf4" },
	{ 20, "MTLDataTypeHalf2x2" },
	{ 21, "MTLDataTypeHalf2x3" },
	{ 22, "MTLDataTypeHalf2x4" },
	{ 23, "MTLDataTypeHalf3x2" },
	{ 24, "MTLDataTypeHalf3x3" },
	{ 25, "MTLDataTypeHalf3x4" },
	{ 26, "MTLDataTypeHalf4x2" },
	{ 27, "MTLDataTypeHalf4x3" },
	{ 28, "MTLDataTypeHalf4x4" },
	{ 29, "MTLDataTypeInt" },
	{ 30, "MTLDataTypeInt2" },
	{ 31, "MTLDataTypeInt3" },
	{ 32, "MTLDataTypeInt4" },
	{ 33, "MTLDataTypeUInt" },
	{ 34, "MTLDataTypeUInt2" },
	{ 35, "MTLDataTypeUInt3" },
	{ 36, "MTLDataTypeUInt4" },
	{ 37, "MTLDataTypeShort" },
	{ 38, "MTLDataTypeShort2" },
	{ 39, "MTLDataTypeShort3" },
	{ 40, "MTLDataTypeShort4" },
	{ 41, "MTLDataTypeUShort" },
	{ 42, "MTLDataTypeUShort2" },
	{ 43, "MTLDataTypeUShort3" },
	{ 44, "MTLDataTypeUShort4" },
	{ 45, "MTLDataTypeChar" },
	{ 46, "MTLDataTypeChar2" },
	{ 47, "MTLDataTypeChar3" },
	{ 48, "MTLDataTypeChar4" },
	{ 49, "MTLDataTypeUChar" },
	{ 50, "MTLDataTypeUChar2" },
	{ 51, "MTLDataTypeUChar3" },
	{ 52, "MTLDataTypeUChar4" },
	{ 53, "MTLDataTypeBool" },
	{ 54, "MTLDataTypeBool2" },
	{ 55, "MTLDataTypeBool3" },
	{ 56, "MTLDataTypeBool4" },
	{ 58, "MTLDataTypeTexture" },
	{ 59, "MTLDataTypeSampler" },
	{ 60, "MTLDataTypePointer" },
	{ 62, "MTLDataTypeR8Unorm" },
	{ 63, "MTLDataTypeR8Snorm" },
	{ 64, "MTLDataTypeR16Unorm" },
	{ 65, "MTLDataTypeR16Snorm" },
	{ 66, "MTLDataTypeRG8Unorm" },
	{ 67, "MTLDataTypeRG8Snorm" },
	{ 68, "MTLDataTypeRG16Unorm" },
	{ 69, "MTLDataTypeRG16Snorm" },
	{ 70, "MTLDataTypeRGBA8Unorm" },
	{ 71, "MTLDataTypeRGBA8Unorm_sRGB" },
	{ 72, "MTLDataTypeRGBA8Snorm" },
	{ 73, "MTLDataTypeRGBA16Unorm" },
	{ 74, "MTLDataTypeRGBA16Snorm" },
	{ 75, "MTLDataTypeRGB10A2Unorm" },
	{ 76, "MTLDataTypeRG11B10Float" },
	{ 77, "MTLDataTypeRGB9E5Float" },
	{ 78, "MTLDataTypeRenderPipeline" },
	{ 79, "MTLDataTypeComputePipeline" },
	{ 80, "MTLDataTypeIndirectCommandBuffer" },
	{ 81, "MTLDataTypeLong" },
	{ 82, "MTLDataTypeLong2" },
	{ 83, "MTLDataTypeLong3" },
	{ 84, "MTLDataTypeLong4" },
	{ 85, "MTLDataTypeULong" },
	{ 86, "MTLDataTypeULong2" },
	{ 87, "MTLDataTypeULong3" },
	{ 88, "MTLDataTypeULong4" },
	{ 115, "MTLDataTypeVisibleFunctionTable" },
	{ 116, "MTLDataTypeIntersectionFunctionTable" },
	{ 117, "MTLDataTypePrimitiveAccelerationStructure" },
	{ 118, "MTLDataTypeInstanceAccelerationStructure" },
	{ 121, "MTLDataTypeBFloat" },
	{ 122, "MTLDataTypeBFloat2" },
	{ 123, "MTLDataTypeBFloat3" },
	{ 124, "MTLDataTypeBFloat4" },
};

} // namespace

void rgfxFunctionConstantsTests()
{
	for (const Measured &m : kMeasured) {
		if (rgfxDataTypeName(m.type) != m.name)
			std::fprintf(stderr, "type %u: %s, Metal's %s\n", m.type,
			             rgfxDataTypeName(m.type).c_str(), m.name);
		CHECK(rgfxDataTypeName(m.type) == m.name);
	}
	// A value outside the enum: ours.
	CHECK(rgfxDataTypeName(57) == "MTLDataType(57)");
	CHECK(rgfxDataTypeName(1000) == "MTLDataType(1000)");
}
