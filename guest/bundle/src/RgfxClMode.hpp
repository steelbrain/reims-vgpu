/*
 * RgfxClMode: what a kernel's pipeline of OpenCL mode binds, from its AIR's nodes as Metal's
 * compiler writes them for OpenCL C.
 *
 * macOS's OpenGL renderer over Metal (AppleMetalOpenGLRenderer) runs a GPU OpenCL program on a
 * Metal device. It compiles the program's source with newLibraryWithSource (sourceLanguage 1),
 * whose library holds each kernel's AIR module under `__entry_<name>` (RgfxMetallibOfClLibrary),
 * makes each kernel's function with newFunctionWithPluginData:bitcodeType:2, and then a compute
 * pipeline of that function with openCLModeEnabled. Metal links the kernel for OpenCL mode as it
 * makes the pipeline; on our device the host does (RGFX_FEATURE_OPENCL_MODE), from the module as
 * the library holds it. What a pipeline of OpenCL mode answers (measured: macOS 26.7.1, the Intel
 * UHD 630 and the AMD Radeon Pro 5300M alike, through the renderer, for kernels of global,
 * constant and local pointers, values and a struct passed by value, and reqd_work_group_size):
 *   - three buffers ahead of the kernel's own, which the renderer binds at every dispatch:
 *     `__global_offset_and_num_dims` (buffer 0, uint, read only), `__printf_buffer` (1, char, read
 *     and write) and `__image_order_data_type_table` (2, uint, read only);
 *   - each __global and __constant pointer of the kernel a buffer at its own index plus 3, and
 *     each argument passed by value one too, read only, which the renderer fills with the value;
 *     each __local pointer the threadgroup memory at its own index;
 *   - the function's plugin return data a binary property list of `LinkerScript_buffer_bindings`
 *     (the pointers' buffer indices, in their order), `LinkerScript_byVal_bindings` (the values',
 *     only where there are any), `LinkerScript_function_name` (the entry's name) and, for a kernel
 *     of reqd_work_group_size, `air.work_group_size` (its three sizes); the renderer answers
 *     CL_KERNEL_COMPILE_WORK_GROUP_SIZE from the last.
 * In the kernel's module (read: macOS 26.7.1 and macOS 27) a __global pointer is an `air.buffer`
 * of address space 1, a __constant one of address space 2 and a __local one of address space 3;
 * an argument passed by value is an `air.constant`, a struct's with its layout; and
 * reqd_work_group_size is the entry node's `air.work_group_size` (RgfxAirEntry). A texture's and
 * a sampler's indices in OpenCL mode were not measured, and a kernel of either is not made.
 *
 * Our own C++.
 */
#ifndef RGFX_CL_MODE_HPP_
#define RGFX_CL_MODE_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "RgfxAirArguments.hpp" // RgfxAirEntry

/// What a kernel's pipeline of OpenCL mode binds, as its function's plugin return data and its
/// pipeline's reflection answer it.
struct RgfxClBindings {
	std::vector<uint32_t> buffers;          // the __global and __constant pointers' buffer indices
	std::vector<uint32_t> byValue;          // the values' buffer indices
	std::vector<uint32_t> byValuePositions; // the values' parameter positions
	uint32_t parameters = 0;                // the kernel's own parameter count
	bool hasWorkGroupSize = false;          // reqd_work_group_size
	uint32_t workGroupSize[3] = {1, 1, 1};
};

/// The buffers ahead of a kernel's own in OpenCL mode.
constexpr uint32_t kRgfxClHiddenBuffers = 3;

/// What `entry`, a kernel of Metal's compile of OpenCL C, binds in OpenCL mode, in `out`. False,
/// with `why`, where it is no kernel, or an argument is of a kind or address space the rule does
/// not cover (a texture, a sampler, or any other); `out` then holds nothing.
bool RgfxClBindingsOf(const RgfxAirEntry &entry, RgfxClBindings &out, std::string &why);

#endif // RGFX_CL_MODE_HPP_
