// The tests' Core Image kernel library (`device::tests`): one colour kernel, compiled as Core
// Image compiles its kernels (`metal -fcikernel`, `metallib -cikernel`), which marks the
// metallib's header with file type 1, a Core Image kernel library.
#include <CoreImage/CoreImage.h>

extern "C" {
namespace coreimage {

// The sample's colour, its red and blue channels swapped.
float4 swap_red_blue(sample_t s) {
    return s.bgra;
}

}
}
