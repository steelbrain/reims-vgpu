/*
 * RgfxGLLowering: a GL shader's IR lowered to an AIR module, and that module to a .metallib of
 * one function (RgfxMetallibWriter.hpp), in the process that compiles the shader.
 *
 * macOS's OpenGL renderer over Metal (AppleMetalOpenGLRenderer) hands a Metal device each shader
 * as GL IR (newFunctionWithGLCoreIR:inputsDescription:functionType:) and each variant of it as
 * plugin data (newFunctionWithPluginData:bitcodeType:). Metal's own devices lower the two to AIR
 * in their compiler service, through the GL code generator the system ships beside the renderer
 * (AppleMetalOpenGLRenderer.bundle/Contents/PlugIns/libGLDPlugin<V>.dylib, V the version of
 * GPUCompiler.framework, whose libLLVM.dylib it runs on). The device has no compiler service, so
 * it calls the same generator here, in place, through its C entry points:
 *   - oglCodeGenServiceInitialize(), once per process;
 *   - oglCodeGenServiceGenerateIRWithPluginDataAndPluginReturnData(context, ir, irLength,
 *     plugin, pluginLength, &returnData, &returnLength), which answers the llvm::Module of the
 *     variant in `context` (an LLVMContextRef), and its return data: a binary property list,
 *     malloc'ed, that Metal answers as the pipeline reflection's vertexPluginReturnData,
 *     fragmentPluginReturnData or pluginReturnData (measured: macOS 26.7.1, the same bytes).
 * The module is standard AIR (measured: macOS 26.7.1, AIR 2.3, entry point `__main`, the
 * language "OpenGL GLSL"), which the system's LLVM writes as bitcode through its C API. The
 * module and its context are disposed of before the call returns.
 *
 * Measured on an x86 Mac (macOS 26.7.1, Intel UHD 630 and AMD 5300M): the .metallib of each of
 * a vertex and a fragment shader's variants loads with newLibraryWithData:, makes a render
 * pipeline, and draws what the renderer's own pipeline draws (tools/gld_lower.m).
 *
 * Our own C++. It calls the system's libraries in place; nothing of them is copied.
 */
#ifndef RGFX_GL_LOWERING_HPP_
#define RGFX_GL_LOWERING_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "RgfxMetallibWriter.hpp"

/// A GL shader variant, lowered.
struct RgfxGLLowered {
	std::vector<uint8_t> metallib;   // a .metallib of the one function (RgfxWriteMetallib)
	std::string entry;               // its entry point
	RgfxMetallibStage stage = RgfxMetallibStage::Vertex;
	std::vector<uint8_t> returnData; // the generator's return data (empty where it gave none)
};

/// Lowers the `irLength` bytes of GL IR at `ir` (the renderer's IR after its u32 length) with
/// the `pluginLength` bytes of plugin data at `plugin` to `out`. False, with `why`, where the
/// system's libraries are not there, the generator makes no module, or the module names no entry
/// point. Calls are serialised: the generator is one per process.
bool RgfxLowerGLIR(const uint8_t *ir, size_t irLength, const uint8_t *plugin, size_t pluginLength,
                   RgfxGLLowered &out, std::string &why);

/// The GL IR a renderer's pointer holds: the u32 length at `ir`, then that many bytes. Writes the
/// bytes' start and their length; false where `ir` is null.
bool RgfxGLIRPayload(const void *ir, const uint8_t **payload, size_t *length);

#endif // RGFX_GL_LOWERING_HPP_
