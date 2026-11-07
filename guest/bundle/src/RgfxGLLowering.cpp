/*
 * RgfxGLLowering: see RgfxGLLowering.hpp. Our own C++.
 */
#include "RgfxGLLowering.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

constexpr char kCompilerVersions[] = "/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/";
constexpr char kPlugins[] = "/System/Library/Extensions/AppleMetalOpenGLRenderer.bundle/Contents/PlugIns/";
// The version where GPUCompiler.framework's Current names none (measured: macOS 26.6.1, 26.7.1 and
// 27.0.1 all ship 32023).
constexpr char kDefaultVersion[] = "32023";

typedef void *Ref;
typedef Ref (*GenerateFn)(Ref context, const void *ir, size_t length, const void *plugin, size_t pluginLength,
                          void **returnData, size_t *returnLength);

struct Libraries {
	bool loaded = false;
	std::string why;
	Ref (*contextCreate)(void) = nullptr;
	void (*contextDispose)(Ref) = nullptr;
	void (*disposeModule)(Ref) = nullptr;
	Ref (*writeBitcode)(Ref module) = nullptr;
	const char *(*bufferStart)(Ref) = nullptr;
	size_t (*bufferSize)(Ref) = nullptr;
	void (*disposeBuffer)(Ref) = nullptr;
	unsigned (*namedMetadataCount)(Ref module, const char *name) = nullptr;
	void (*namedMetadata)(Ref module, const char *name, Ref *dest) = nullptr;
	unsigned (*nodeCount)(Ref node) = nullptr;
	void (*nodeOperands)(Ref node, Ref *dest) = nullptr;
	const char *(*valueName)(Ref value, size_t *length) = nullptr;
	unsigned long long (*constIntValue)(Ref value) = nullptr;
	Ref (*isConstantInt)(Ref value) = nullptr;
	GenerateFn generate = nullptr;
};

std::string compilerVersion()
{
	char link[PATH_MAX];
	const std::string current = std::string(kCompilerVersions) + "Current";
	const ssize_t n = readlink(current.c_str(), link, sizeof link - 1);
	if (n <= 0)
		return kDefaultVersion;
	link[n] = 0;
	return std::strchr(link, '/') != nullptr ? kDefaultVersion : link;
}

template <typename F>
bool symbol(void *handle, const char *name, F &out, std::string &why)
{
	out = reinterpret_cast<F>(dlsym(handle, name));
	if (out == nullptr)
		why = std::string("the system's LLVM has no ") + name;
	return out != nullptr;
}

Libraries &libraries()
{
	static Libraries libs;
	static std::once_flag once;
	std::call_once(once, [] {
		const std::string version = compilerVersion();
		const std::string llvmPath = std::string(kCompilerVersions) + version + "/Libraries/libLLVM.dylib";
		const std::string pluginPath = std::string(kPlugins) + "libGLDPlugin" + version + ".dylib";
		void *llvm = dlopen(llvmPath.c_str(), RTLD_NOW | RTLD_LOCAL);
		if (llvm == nullptr) {
			libs.why = "no " + llvmPath;
			return;
		}
		std::string &why = libs.why;
		if (!symbol(llvm, "LLVMContextCreate", libs.contextCreate, why) ||
		    !symbol(llvm, "LLVMContextDispose", libs.contextDispose, why) ||
		    !symbol(llvm, "LLVMDisposeModule", libs.disposeModule, why) ||
		    !symbol(llvm, "LLVMWriteBitcodeToMemoryBuffer", libs.writeBitcode, why) ||
		    !symbol(llvm, "LLVMGetBufferStart", libs.bufferStart, why) ||
		    !symbol(llvm, "LLVMGetBufferSize", libs.bufferSize, why) ||
		    !symbol(llvm, "LLVMDisposeMemoryBuffer", libs.disposeBuffer, why) ||
		    !symbol(llvm, "LLVMGetNamedMetadataNumOperands", libs.namedMetadataCount, why) ||
		    !symbol(llvm, "LLVMGetNamedMetadataOperands", libs.namedMetadata, why) ||
		    !symbol(llvm, "LLVMGetMDNodeNumOperands", libs.nodeCount, why) ||
		    !symbol(llvm, "LLVMGetMDNodeOperands", libs.nodeOperands, why) ||
		    !symbol(llvm, "LLVMGetValueName2", libs.valueName, why) ||
		    !symbol(llvm, "LLVMConstIntGetZExtValue", libs.constIntValue, why) ||
		    !symbol(llvm, "LLVMIsAConstantInt", libs.isConstantInt, why))
			return;
		void *plugin = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL);
		if (plugin == nullptr) {
			why = "no " + pluginPath;
			return;
		}
		void (*initialize)(void) = nullptr;
		if (!symbol(plugin, "oglCodeGenServiceInitialize", initialize, why) ||
		    !symbol(plugin, "oglCodeGenServiceGenerateIRWithPluginDataAndPluginReturnData", libs.generate, why))
			return;
		initialize();
		libs.loaded = true;
	});
	return libs;
}

// The operands of the first node of the named metadata `name` of `module`; empty where it has none.
std::vector<Ref> firstNode(const Libraries &l, Ref module, const char *name)
{
	const unsigned n = l.namedMetadataCount(module, name);
	if (n == 0)
		return {};
	std::vector<Ref> nodes(n);
	l.namedMetadata(module, name, nodes.data());
	const unsigned k = l.nodeCount(nodes[0]);
	std::vector<Ref> ops(k);
	if (k != 0)
		l.nodeOperands(nodes[0], ops.data());
	return ops;
}

} // namespace

bool RgfxGLIRPayload(const void *ir, const uint8_t **payload, size_t *length)
{
	if (ir == nullptr)
		return false;
	uint32_t n = 0;
	std::memcpy(&n, ir, sizeof n);
	*payload = static_cast<const uint8_t *>(ir) + sizeof n;
	*length = n;
	return true;
}

bool RgfxLowerGLIR(const uint8_t *ir, size_t irLength, const uint8_t *plugin, size_t pluginLength,
                   RgfxGLLowered &out, std::string &why)
{
	Libraries &l = libraries();
	if (!l.loaded) {
		why = l.why;
		return false;
	}
	static std::mutex serial;
	std::lock_guard<std::mutex> hold(serial);
	Ref context = l.contextCreate();
	void *returnData = nullptr;
	size_t returnLength = 0;
	Ref module = l.generate(context, ir, irLength, plugin, pluginLength, &returnData, &returnLength);
	if (returnData != nullptr) {
		const uint8_t *r = static_cast<const uint8_t *>(returnData);
		out.returnData.assign(r, r + returnLength);
		free(returnData);
	}
	bool ok = module != nullptr;
	if (!ok)
		why = "the GL code generator made no module";
	const struct {
		const char *metadata;
		RgfxMetallibStage stage;
	} kStages[] = {{"air.vertex", RgfxMetallibStage::Vertex},
	               {"air.fragment", RgfxMetallibStage::Fragment},
	               {"air.kernel", RgfxMetallibStage::Kernel}};
	RgfxMetallibFunction f;
	for (const auto &s : kStages) {
		if (!ok || !f.name.empty())
			break;
		std::vector<Ref> ops = firstNode(l, module, s.metadata);
		size_t length = 0;
		const char *name = !ops.empty() && ops[0] != nullptr ? l.valueName(ops[0], &length) : nullptr;
		if (name != nullptr && length != 0) {
			f.name.assign(name, length);
			f.stage = s.stage;
		}
	}
	if (ok && f.name.empty()) {
		ok = false;
		why = "the GL code generator's module names no entry point";
	}
	if (ok) {
		// !air.version = !{i32 major, i32 minor, i32 patch}; the language version VERS names is the
		// AIR version's, as `metal -mmacosx-version-min=11.0` writes it for AIR 2.3.
		std::vector<Ref> v = firstNode(l, module, "air.version");
		if (v.size() >= 2 && v[0] != nullptr && v[1] != nullptr && l.isConstantInt(v[0]) != nullptr &&
		    l.isConstantInt(v[1]) != nullptr) {
			f.airMajor = f.languageMajor = static_cast<uint16_t>(l.constIntValue(v[0]));
			f.airMinor = f.languageMinor = static_cast<uint16_t>(l.constIntValue(v[1]));
		}
		Ref buffer = l.writeBitcode(module);
		size_t length = l.bufferSize(buffer);
		f.bitcode = RgfxUnwrapBitcode(reinterpret_cast<const uint8_t *>(l.bufferStart(buffer)), &length);
		f.bitcodeLength = length;
		out.metallib = RgfxWriteMetallib(f);
		out.entry = f.name;
		out.stage = f.stage;
		l.disposeBuffer(buffer);
	}
	if (module != nullptr)
		l.disposeModule(module);
	l.contextDispose(context);
	return ok;
}
