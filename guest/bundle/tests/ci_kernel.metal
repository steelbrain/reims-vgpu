// The shell tests' Core Image kernel library (Makefile: `metal -fcikernel`, `metallib -cikernel`):
// two color kernels, which Metal lists as extern functions of functionType 4, never as entry
// points.
#include <CoreImage/CoreImage.h>

extern "C" {
namespace coreimage {

float4 swap_red_blue(sample_t s) { return s.bgra; }

float4 scale_red_green(sample_t s, float k) { return float4(s.r * k, s.g * k, s.b, s.a); }

}
}
