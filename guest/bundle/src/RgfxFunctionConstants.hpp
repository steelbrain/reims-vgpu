/*
 * RgfxFunctionConstants: a function's constants and the values a process sets for them,
 * resolved as Metal resolves them.
 *
 * A .metallib holds each function's constants in its public metadata (the `CNST` tag): the
 * constant's name, its MTLDataType, its index and whether it is required. Metal specialises a
 * function by those constants alone (measured: macOS 27.0.1 Metal on an M3 Max, the bundle's
 * report):
 * - each constant takes the value set at its index (by index or by range); with none there, the
 *   value set under its name; a value set at another index or name is ignored, unchecked;
 * - the value's type must be the constant's own, exactly (a uint for an int fails): otherwise
 *   the function is nil, with MTLLibraryErrorCompileFailure;
 * - a required constant left without a value reads 0, and the function is made;
 * - a constant that is not required (one the function tests with
 *   `is_function_constant_defined`) left without a value is not defined.
 *
 * Our own C++; no Apple code.
 */

#ifndef RGFX_FUNCTION_CONSTANTS_HPP_
#define RGFX_FUNCTION_CONSTANTS_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "reims_graphics_wire.h" // RgfxFunctionConstant

/// One constant a function declares: its `CNST` entry.
struct RgfxConstantDecl {
	std::string name;
	uint32_t type;  // MTLDataType
	uint32_t index; // [[function_constant(index)]]
	bool required;
};

/// One value a process set on an MTLFunctionConstantValues: its MTLDataType and its bytes, as
/// many as the type packs (none for a type no function constant takes).
struct RgfxConstantValue {
	uint32_t type = 0;
	std::vector<uint8_t> bytes;
};

/// The packed size of a function constant of MTLDataType `type` (a float3 is 12 bytes); 0 for a
/// type no function constant takes.
uint32_t rgfxConstantSize(uint32_t type);

/// MTLDataType `type`'s name as Metal's errors spell it ("MTLDataTypeBool").
std::string rgfxDataTypeName(uint32_t type);

/// Reads the constants of the function whose public metadata starts `offset` bytes into the
/// .metallib's public metadata section, from the `length` bytes at `bytes`. A function with no
/// `CNST` tag declares none. Answers false, leaving `out` empty, where the bytes do not hold the
/// entry whole.
bool rgfxReadConstantTable(const uint8_t *bytes, size_t length, uint64_t offset,
                           std::vector<RgfxConstantDecl> &out);

/// A value whose type is not its constant's: the constant (an index into the declarations) and
/// the value's type.
struct RgfxConstantMismatch {
	size_t decl = 0;
	uint32_t valueType = 0;
};

/// Resolves the values set at each index (`byIndex`) and under each name (`byName`) against the
/// function's constants `decls`, as Metal does (the file's comment): `out` gets one wire record
/// for each constant given a value, required ones without a value given 0. Answers false where a
/// value's type is not its constant's, with `*mismatch` naming the first such constant.
bool rgfxResolveConstants(const std::vector<RgfxConstantDecl> &decls,
                          const std::map<uint32_t, RgfxConstantValue> &byIndex,
                          const std::map<std::string, RgfxConstantValue> &byName,
                          std::vector<RgfxFunctionConstant> &out, RgfxConstantMismatch *mismatch);

#endif // RGFX_FUNCTION_CONSTANTS_HPP_
