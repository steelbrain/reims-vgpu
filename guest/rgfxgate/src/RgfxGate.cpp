// RgfxGate plugin entry. At patcher load it installs the kernel hooks and
// kicks off the boot-time bundle install. Running only in the normal boot
// environment (not the installer/Recovery) keeps it inert where there is no
// GPU to drive; the install also guards that itself.
#include <Headers/plugin_start.hpp>
#include <Headers/kern_api.hpp>
#include <Headers/kern_util.hpp>

#include <mach/kmod.h>

#include "RgfxGate.hpp"

static void rgfxOnPatcherLoad(void *, KernelPatcher &patcher) {
	rgfxgate::installHooks(patcher);
	rgfxgate::startInstaller();
}

static void rgfxPluginStart() {
	lilu.onPatcherLoadForce(rgfxOnPatcherLoad);
}

static const char *bootargOff[] { "-rgfxgateoff" };

PluginConfiguration ADDPR(config) {
	xStringify(PRODUCT_NAME),
	parseModuleVersion(xStringify(MODULE_VERSION)),
	LiluAPI::AllowNormal,
	bootargOff,
	arrsize(bootargOff),
	nullptr,
	0,
	nullptr,
	0,
	KernelVersion::BigSur,
	KernelVersion::Tahoe,
	rgfxPluginStart
};

// The kmod_info OpenCore and the kernel need, pointing at the kern_start /
// kern_stop that Lilu's plugin_start.cpp defines for us (ADDPR(kern_start)).
extern "C" {
kern_return_t RgfxGate_kern_start(kmod_info_t *, void *);
kern_return_t RgfxGate_kern_stop(kmod_info_t *, void *);
__attribute__((visibility("default")))
KMOD_EXPLICIT_DECL(com.steelbrain.RgfxGate, xStringify(MODULE_VERSION),
                   RgfxGate_kern_start, RgfxGate_kern_stop)
}
