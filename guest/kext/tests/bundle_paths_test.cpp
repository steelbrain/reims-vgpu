// bundle_paths_test: the paths the accelerator looks up for its bundle (RgfxBundlePaths.hpp).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "RgfxBundlePaths.hpp"

static int checks;
#define CHECK(c)                                                                                   \
	do {                                                                                       \
		checks++;                                                                          \
		if (!(c)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c); \
			std::abort();                                                              \
		}                                                                                  \
	} while (0)

int main()
{
	RgfxBundlePaths paths;
	// The executable Metal loads, and the marker the package writes for the wire's version.
	CHECK(rgfxBundlePaths("reims-graphics", 2, &paths));
	CHECK(!std::strcmp(paths.executable,
	                   "/Library/GPUBundles/reims-graphics.bundle/Contents/MacOS/reims-graphics"));
	CHECK(!std::strcmp(paths.marker,
	                   "/Library/GPUBundles/reims-graphics.bundle/Contents/Resources/wire-version-2"));
	// Another version names another marker, and the same executable.
	RgfxBundlePaths other;
	CHECK(rgfxBundlePaths("reims-graphics", 13, &other));
	CHECK(!std::strcmp(other.executable, paths.executable));
	CHECK(!std::strcmp(other.marker,
	                   "/Library/GPUBundles/reims-graphics.bundle/Contents/Resources/wire-version-13"));
	// The executable's path is the longer, 43 bytes and the name twice: a 106-byte name fills its
	// 255 bytes, and a 107-byte one does not fit, so it is refused.
	std::string edge(106, 'e');
	CHECK(rgfxBundlePaths(edge.c_str(), 2, &paths));
	CHECK(std::strlen(paths.executable) == 255);
	std::string over(107, 'o');
	CHECK(!rgfxBundlePaths(over.c_str(), 2, &paths));
	// The widest version still fits the marker.
	std::string fits(80, 'f');
	CHECK(rgfxBundlePaths(fits.c_str(), 4294967295u, &paths));
	CHECK(std::strstr(paths.marker, "wire-version-4294967295") != nullptr);
	// What counts on disk, given RgfxGate's pass: the bundle, once no pass is under way.
	CHECK(rgfxBundleRegisters(kRgfxGateAbsent, true, true));
	CHECK(rgfxBundleRegisters(kRgfxGateDone, true, true));
	CHECK(!rgfxBundleRegisters(kRgfxGateInstalling, true, true));
	CHECK(!rgfxBundleRegisters(kRgfxGateAbsent, false, true));
	CHECK(!rgfxBundleRegisters(kRgfxGateDone, false, true));
	CHECK(!rgfxBundleRegisters(kRgfxGateInstalling, false, true));
	// No framebuffer started: no registration, whatever the disk holds.
	CHECK(!rgfxBundleRegisters(kRgfxGateAbsent, true, false));
	CHECK(!rgfxBundleRegisters(kRgfxGateDone, true, false));
	CHECK(!rgfxBundleRegisters(kRgfxGateInstalling, true, false));
	CHECK(!rgfxBundleRegisters(kRgfxGateDone, false, false));
	CHECK(!std::strcmp(kRgfxGateInstallDoneKey, "RgfxGateInstallDone"));
	// A read of MetalPluginName waits only while RgfxGate's watch runs and its last pass ended
	// under another root than the reader's: the full system's root, just after FileVault's unlock
	// left the BaseSystem whose pass registered the accelerator.
	const unsigned long long baseRoot = 0x1000000041ull, fullRoot = 0x1000000013ull;
	CHECK(rgfxMetalNameWaits(true, baseRoot, true, fullRoot));
	CHECK(!rgfxMetalNameWaits(true, fullRoot, true, fullRoot));
	CHECK(!rgfxMetalNameWaits(true, baseRoot, true, baseRoot));
	// No watch (a boot that starts in its full system, the watch over, no RgfxGate): no wait.
	CHECK(!rgfxMetalNameWaits(false, baseRoot, true, fullRoot));
	CHECK(!rgfxMetalNameWaits(false, 0, true, fullRoot));
	// No root to compare: no wait.
	CHECK(!rgfxMetalNameWaits(true, baseRoot, false, 0));
	CHECK(!std::strcmp(kRgfxGatePassRootKey, "RgfxGatePassRoot"));
	std::printf("bundle_paths_test: OK (%d checks)\n", checks);
	return 0;
}
