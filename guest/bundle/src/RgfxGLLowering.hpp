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
 * Two things in the generator's stage metadata make Metal refuse a pipeline of its functions, so
 * a vertex or fragment variant's metadata is first made what `metal` writes
 * (RgfxGLNormalizeStageText, over the module's text, which the system's LLVM parses again):
 *   - an integer varying's type name is the signed one in a fragment input and the unsigned one
 *     in the vertex output for the same GLSL `flat int` (measured: macOS 26.7.1, "int" and
 *     "uint" for user(slot8)), and Metal fails the pipeline's link on the difference ("The type
 *     of fragment input user(slot8) does not match the type of the corresponding vertex shader
 *     output"); the renderer then draws the shader in software. An integer varying is named
 *     unsigned on both sides: AIR holds no signedness, and the bits pass as they are;
 *   - an output without air.arg_type_name (measured: a fragment output of air.depth, from
 *     gl_FragDepth) makes Metal's reflection reader dereference a null string and end the
 *     process. Such a node gets the type name of its value's LLVM type and, where it has none, an
 *     air.arg_name "#<kind>", as the generator names air.position ("#air_position").
 * A node that has both, or only lacks air.arg_name (each fragment's air.render_target, measured
 * to make pipelines), is left as it is.
 *
 * More in the generator's module is not in the AIR `metal` writes. Metal's devices take it, but
 * the Vulkan host's runtime reads AIR's metadata as a closed grammar and refuses a pipeline of a
 * function with any of it (metal2vulkan's AIR decoder): every GL pipeline was refused on a 5080
 * host ("UnknownKey" at gld.plugin_info, macOS 26.6.1), and the rest was found by compiling the
 * pipelines of tools/gl_cases.m, dumped through tools/gl_inject.mm on the Intel 630, with the
 * decoder's compile API (macOS 26.7.1: 6 of 29 still refused after the first rule). The
 * generator links its builtins (GLDBuiltins.metal and the like, compiled as AIR 2.8 and Metal
 * 4.0) into a shader that uses them, so their lists' entries follow the module's own. Every
 * variant's module, a kernel's too, is made what one compile of `metal` writes:
 *   - the generator's own named lists (gld.plugin_info, which holds the plugin return data the
 *     generator also answers apart) are dropped;
 *   - air.version and air.language_version hold the module's own entry alone, the first, and the
 *     language, "OpenGL GLSL" 1.0.0, is "Metal" at the AIR version's major and minor (2.3, the
 *     language `metal -mmacosx-version-min=11.0` writes with AIR 2.3);
 *   - air.compile_options holds the module's own option of each kind (fast math, framebuffer
 *     fetch, denormals), the first, and no air.compile.native_double_disable, which `metal` does
 *     not write; llvm.module.flags holds the first flag of each name (the module's own limits,
 *     4096 buffers, not the builtins' 31);
 *   - a vertex input's attribute name, a bare string after its air.location_index's two
 *     operands, is dropped (the input keeps its air.arg_name, which holds the same name);
 *   - a flat fragment input has no sampling key: `metal` writes [[flat]] as air.flat alone, and
 *     the generator writes air.center before it;
 *   - a buffer input has the air.address_space `metal` writes, its parameter's pointer's, and,
 *     where it lacks them, the air.arg_type_size and air.arg_type_align_size of its
 *     air.arg_type_name. The generator writes no address space for a uniform buffer, and no
 *     pointee size for one with air.buffer_size (measured: macOS 26.7.1). The device's reflection
 *     reads both: without them a pipeline had no reflection, and gl_cases' uniforms and block
 *     cases read black on the RTX 5080 host.
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
/// system's libraries are not there, the generator makes no module, the module names no entry
/// point, or the system's LLVM refuses the module's normalized text. Calls are serialised: the generator is one per process.
bool RgfxLowerGLIR(const uint8_t *ir, size_t irLength, const uint8_t *plugin, size_t pluginLength,
                   RgfxGLLowered &out, std::string &why);

/// The Metal type names (as `metal` writes air.arg_type_name: "float", "half4", "uint2", "bool";
/// integers unsigned; "" for a type with none) of an AIR entry point's outputs, its result's
/// members in order (or its one result), and of its parameters.
struct RgfxGLSignature {
	std::vector<std::string> outputs, parameters;
};

/// Makes `text`, a module as LLVM prints it, follow the rules above: the module's (no gld.*
/// lists; one version, its language Metal's; its own compile options and module flags, once
/// each), and those of the stage metadata of its vertex or fragment entry point where it has one:
/// an integer user() varying's air.arg_type_name is the unsigned name, a vertex input has no
/// bare attribute name, a flat fragment input no sampling key, and an output or input node
/// without air.arg_type_name gets the name `signature` gives its value (an input's by its
/// parameter index), and an air.arg_name where it has none. True where it changed `text`; `text`
/// is unchanged otherwise.
bool RgfxGLNormalizeStageText(std::string &text, const RgfxGLSignature &signature);

/// The GL IR a renderer's pointer holds: the u32 length at `ir`, then that many bytes. Writes the
/// bytes' start and their length; false where `ir` is null.
bool RgfxGLIRPayload(const void *ir, const uint8_t **payload, size_t *length);

#endif // RGFX_GL_LOWERING_HPP_
