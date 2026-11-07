/*
 * RgfxGLLowering: see RgfxGLLowering.hpp. Our own C++.
 */
#include "RgfxGLLowering.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
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
	Ref (*namedFunction)(Ref module, const char *name) = nullptr;
	Ref (*globalValueType)(Ref global) = nullptr;
	Ref (*returnType)(Ref function) = nullptr;
	unsigned (*paramCount)(Ref function) = nullptr;
	void (*paramTypes)(Ref function, Ref *dest) = nullptr;
	unsigned (*structCount)(Ref type) = nullptr;
	void (*structTypes)(Ref type, Ref *dest) = nullptr;
	int (*typeKind)(Ref type) = nullptr;
	unsigned (*vectorSize)(Ref type) = nullptr;
	Ref (*elementType)(Ref type) = nullptr;
	unsigned (*intWidth)(Ref type) = nullptr;
	char *(*printModule)(Ref module) = nullptr;
	void (*disposeMessage)(char *message) = nullptr;
	Ref (*bufferCopy)(const char *data, size_t length, const char *name) = nullptr;
	int (*parseIR)(Ref context, Ref buffer, Ref *module, char **message) = nullptr;
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
		    !symbol(llvm, "LLVMIsAConstantInt", libs.isConstantInt, why) ||
		    !symbol(llvm, "LLVMGetNamedFunction", libs.namedFunction, why) ||
		    !symbol(llvm, "LLVMGlobalGetValueType", libs.globalValueType, why) ||
		    !symbol(llvm, "LLVMGetReturnType", libs.returnType, why) ||
		    !symbol(llvm, "LLVMCountParamTypes", libs.paramCount, why) ||
		    !symbol(llvm, "LLVMGetParamTypes", libs.paramTypes, why) ||
		    !symbol(llvm, "LLVMCountStructElementTypes", libs.structCount, why) ||
		    !symbol(llvm, "LLVMGetStructElementTypes", libs.structTypes, why) ||
		    !symbol(llvm, "LLVMGetTypeKind", libs.typeKind, why) ||
		    !symbol(llvm, "LLVMGetVectorSize", libs.vectorSize, why) ||
		    !symbol(llvm, "LLVMGetElementType", libs.elementType, why) ||
		    !symbol(llvm, "LLVMGetIntTypeWidth", libs.intWidth, why) ||
		    !symbol(llvm, "LLVMPrintModuleToString", libs.printModule, why) ||
		    !symbol(llvm, "LLVMDisposeMessage", libs.disposeMessage, why) ||
		    !symbol(llvm, "LLVMCreateMemoryBufferWithMemoryRangeCopy", libs.bufferCopy, why) ||
		    !symbol(llvm, "LLVMParseIRInContext", libs.parseIR, why))
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


// LLVM-C's LLVMTypeKind values the signature names (llvm-c/Core.h; the enum only grows).
enum : int { kHalfKind = 1, kFloatKind = 2, kIntegerKind = 8, kStructKind = 10, kVectorKind = 13 };

// The Metal name of an LLVM type, as `metal` writes it in air.arg_type_name: "float", "half4",
// "uint2", "bool"; integers unsigned, the signedness LLVM's types do not hold. "" for any other.
std::string metalTypeName(const Libraries &l, Ref type)
{
	unsigned lanes = 1;
	if (l.typeKind(type) == kVectorKind) {
		lanes = l.vectorSize(type);
		type = l.elementType(type);
	}
	std::string name;
	switch (l.typeKind(type)) {
	case kHalfKind:
		name = "half";
		break;
	case kFloatKind:
		name = "float";
		break;
	case kIntegerKind:
		switch (l.intWidth(type)) {
		case 1:
			name = "bool";
			break;
		case 8:
			name = "uchar";
			break;
		case 16:
			name = "ushort";
			break;
		case 32:
			name = "uint";
			break;
		case 64:
			name = "ulong";
			break;
		default:
			return "";
		}
		break;
	default:
		return "";
	}
	if (lanes > 1)
		name += std::to_string(lanes);
	return name;
}

RgfxGLSignature signatureOf(const Libraries &l, Ref module, const std::string &entry)
{
	RgfxGLSignature s;
	Ref function = l.namedFunction(module, entry.c_str());
	if (function == nullptr)
		return s;
	Ref type = l.globalValueType(function);
	Ref result = l.returnType(type);
	if (l.typeKind(result) == kStructKind) {
		std::vector<Ref> members(l.structCount(result));
		if (!members.empty())
			l.structTypes(result, members.data());
		for (Ref m : members)
			s.outputs.push_back(metalTypeName(l, m));
	} else {
		s.outputs.push_back(metalTypeName(l, result));
	}
	std::vector<Ref> params(l.paramCount(type));
	if (!params.empty())
		l.paramTypes(type, params.data());
	for (Ref p : params)
		s.parameters.push_back(metalTypeName(l, p));
	return s;
}

// A metadata node's line of a module's text, "!N = [distinct ]!{op, op, ...}", split into what
// comes before its "!{" and its operands at the top level. A string operand is !"..." with no
// raw quote inside (LLVM writes one as \22).
struct NodeText {
	std::string prefix;
	std::vector<std::string> ops;
};

bool splitNode(const std::string &line, NodeText &out)
{
	const size_t eq = line.find(" = ");
	const size_t open = eq == std::string::npos ? eq : line.find("!{", eq);
	if (open == std::string::npos || line.empty() || line.back() != '}')
		return false;
	out.prefix = line.substr(0, open);
	out.ops.clear();
	const std::string body = line.substr(open + 2, line.size() - open - 3);
	int depth = 0;
	bool quoted = false;
	std::string op;
	for (char c : body) {
		if (c == '"')
			quoted = !quoted;
		else if (!quoted && (c == '{' || c == '(' || c == '<' || c == '['))
			depth++;
		else if (!quoted && (c == '}' || c == ')' || c == '>' || c == ']'))
			depth--;
		if (!quoted && depth == 0 && c == ',') {
			out.ops.push_back(op);
			op.clear();
			continue;
		}
		if (op.empty() && c == ' ')
			continue;
		op += c;
	}
	if (!op.empty())
		out.ops.push_back(op);
	return true;
}

std::string joinNode(const NodeText &n)
{
	std::string line = n.prefix + "!{";
	for (size_t i = 0; i < n.ops.size(); i++)
		line += (i != 0 ? ", " : "") + n.ops[i];
	return line + "}";
}

// The string of an operand !"s"; false for any other operand.
bool stringOp(const std::string &op, std::string &s)
{
	if (op.size() < 3 || op[0] != '!' || op[1] != '"' || op.back() != '"')
		return false;
	s = op.substr(2, op.size() - 3);
	return true;
}

// The node an operand !N names; false for any other operand.
bool refOp(const std::string &op, unsigned &n)
{
	if (op.size() < 2 || op[0] != '!' || op[1] < '0' || op[1] > '9')
		return false;
	n = static_cast<unsigned>(std::strtoul(op.c_str() + 1, nullptr, 10));
	return true;
}

// The value of an operand "i32 N"; false for any other operand.
bool intOp(const std::string &op, unsigned &n)
{
	if (op.compare(0, 4, "i32 ") != 0)
		return false;
	n = static_cast<unsigned>(std::strtoul(op.c_str() + 4, nullptr, 10));
	return true;
}

// Whether `t` is a signed integer type's Metal name: char, short, int or long, or a vector of one.
bool signedIntegerName(const std::string &t)
{
	std::string base = t;
	if (!base.empty() && base.back() >= '2' && base.back() <= '4')
		base.pop_back();
	return base == "char" || base == "short" || base == "int" || base == "long";
}

// Makes one output or input node's text what the rules in RgfxGLLowering.hpp say; true where it
// changed. `type` is the Metal name of the value's type from the entry point's signature.
bool normalizeNode(std::string &line, bool input, const std::string &type)
{
	NodeText n;
	if (!splitNode(line, n))
		return false;
	// A vertex input's attribute name, which the generator writes as a bare string after the
	// location's two operands, is not in the node `metal` writes.
	bool edited = false;
	for (size_t i = 0; input && i + 3 < n.ops.size(); i++) {
		std::string key, bare;
		unsigned a = 0, b = 0;
		if (stringOp(n.ops[i], key) && key == "air.location_index" && intOp(n.ops[i + 1], a) &&
		    intOp(n.ops[i + 2], b) && stringOp(n.ops[i + 3], bare) && bare.compare(0, 4, "air.") != 0) {
			n.ops.erase(n.ops.begin() + static_cast<std::ptrdiff_t>(i + 3));
			line = joinNode(n);
			edited = true;
			break;
		}
	}
	// A flat fragment input's sampling key: `metal` writes [[flat]] as air.flat alone.
	bool flat = false, fragmentInput = false;
	for (const std::string &op : n.ops) {
		std::string key;
		if (stringOp(op, key)) {
			flat |= key == "air.flat";
			fragmentInput |= key == "air.fragment_input";
		}
	}
	if (input && flat && fragmentInput) {
		for (size_t i = 0; i < n.ops.size();) {
			std::string key;
			if (stringOp(n.ops[i], key) && (key == "air.center" || key == "air.centroid" || key == "air.sample")) {
				n.ops.erase(n.ops.begin() + static_cast<std::ptrdiff_t>(i));
				line = joinNode(n);
				edited = true;
			} else {
				i++;
			}
		}
	}
	size_t kindAt = n.ops.size(), typeAt = n.ops.size();
	bool named = false;
	std::string kind, s;
	for (size_t i = 0; i < n.ops.size(); i++) {
		if (!stringOp(n.ops[i], s))
			continue;
		if (kindAt == n.ops.size()) {
			kindAt = i;
			kind = s;
		} else if (s == "air.arg_type_name" && i + 1 < n.ops.size()) {
			typeAt = i + 1;
		} else if (s == "air.arg_name") {
			named = true;
		}
	}
	if (kindAt == n.ops.size() || kind.compare(0, 4, "air.") != 0 || (input && kindAt == 0))
		return edited;
	std::string current;
	if (typeAt != n.ops.size()) {
		std::string user;
		const bool interpolant = (kind == "air.vertex_output" || kind == "air.fragment_input") &&
		                         kindAt + 1 < n.ops.size() && stringOp(n.ops[kindAt + 1], user) &&
		                         user.compare(0, 5, "user(") == 0;
		if (!interpolant || !stringOp(n.ops[typeAt], current) || !signedIntegerName(current))
			return edited;
		n.ops[typeAt] = "!\"u" + current + "\"";
		line = joinNode(n);
		return true;
	}
	if (type.empty())
		return edited;
	n.ops.push_back("!\"air.arg_type_name\"");
	n.ops.push_back("!\"" + type + "\"");
	if (!named) {
		std::string name = "#" + kind;
		for (char &c : name)
			c = c == '.' ? '_' : c;
		unsigned index = 0;
		if (kindAt + 1 < n.ops.size() && intOp(n.ops[kindAt + 1], index))
			name += "_" + std::to_string(index);
		n.ops.push_back("!\"air.arg_name\"");
		n.ops.push_back("!\"" + name + "\"");
	}
	line = joinNode(n);
	return true;
}

} // namespace

bool RgfxGLNormalizeStageText(std::string &text, const RgfxGLSignature &signature)
{
	std::vector<std::string> lines;
	for (size_t at = 0; at <= text.size();) {
		size_t end = text.find('\n', at);
		if (end == std::string::npos)
			end = text.size();
		lines.push_back(text.substr(at, end - at));
		at = end + 1;
	}
	// Each numbered node's line, and the lines of the named lists the rules read.
	std::vector<size_t> nodeLine;
	size_t entryList = lines.size(), versionList = lines.size(), languageList = lines.size(),
	       optionsList = lines.size(), flagsList = lines.size();
	std::vector<bool> dropped(lines.size(), false);
	bool changed = false;
	auto starts = [](const std::string &line, const char *name) {
		return line.compare(0, std::strlen(name), name) == 0;
	};
	for (size_t i = 0; i < lines.size(); i++) {
		const std::string &line = lines[i];
		unsigned n = 0;
		if (refOp(line, n) && line.find(" = ") != std::string::npos) {
			if (nodeLine.size() <= n)
				nodeLine.resize(n + 1, lines.size());
			nodeLine[n] = i;
		} else if (entryList == lines.size() &&
		           (starts(line, "!air.vertex = ") || starts(line, "!air.fragment = "))) {
			entryList = i;
		} else if (starts(line, "!air.version = ")) {
			versionList = i;
		} else if (starts(line, "!air.language_version = ")) {
			languageList = i;
		} else if (starts(line, "!air.compile_options = ")) {
			optionsList = i;
		} else if (starts(line, "!llvm.module.flags = ")) {
			flagsList = i;
		} else if (starts(line, "!gld.") && line.find(" = ") != std::string::npos) {
			dropped[i] = changed = true; // the generator's own list
		}
	}
	auto node = [&](unsigned n, NodeText &out) {
		return n < nodeLine.size() && nodeLine[n] < lines.size() && splitNode(lines[nodeLine[n]], out);
	};
	// A named list's entries, each kept where `slot` names a slot no entry before it took ("" keeps
	// every entry; nullptr drops it), so the list holds what one module's compile writes.
	auto keepFirst = [&](size_t list, const std::function<const char *(const NodeText &, std::string &)> &slot) {
		NodeText named;
		if (list >= lines.size() || !splitNode(lines[list], named))
			return;
		std::vector<std::string> kept, taken;
		for (const std::string &op : named.ops) {
			unsigned n = 0;
			NodeText entry;
			std::string scratch;
			const char *s = refOp(op, n) && node(n, entry) ? slot(entry, scratch) : "";
			if (s == nullptr)
				continue;
			if (s[0] != '\0') {
				if (std::find(taken.begin(), taken.end(), s) != taken.end())
					continue;
				taken.emplace_back(s);
			}
			kept.push_back(op);
		}
		if (kept.size() != named.ops.size()) {
			named.ops = kept;
			lines[list] = joinNode(named);
			changed = true;
		}
	};
	// The versions: the module's own, the list's first (the generator links its builtins,
	// compiled as another AIR and language version, in after it).
	auto one = [](const NodeText &, std::string &) -> const char * { return "one"; };
	keepFirst(versionList, one);
	keepFirst(languageList, one);
	// The compile options: the module's own of each (fast math, framebuffer fetch, denormals), and
	// no air.compile.native_double_disable.
	keepFirst(optionsList, [](const NodeText &option, std::string &s) -> const char * {
		if (option.ops.size() != 1 || !stringOp(option.ops[0], s))
			return "";
		if (s == "air.compile.native_double_disable")
			return nullptr;
		if (s.rfind("air.compile.fast_math_", 0) == 0)
			return "fast_math";
		if (s.rfind("air.compile.framebuffer_fetch_", 0) == 0)
			return "framebuffer_fetch";
		return s.c_str();
	});
	// The module flags: the module's own of each.
	keepFirst(flagsList, [](const NodeText &flag, std::string &s) -> const char * {
		return flag.ops.size() >= 2 && stringOp(flag.ops[1], s) ? s.c_str() : "";
	});
	// The language: Metal's, at the AIR version's major and minor.
	auto firstOf = [&](size_t list, unsigned &n) {
		NodeText named;
		return list < lines.size() && splitNode(lines[list], named) && !named.ops.empty() &&
		       refOp(named.ops[0], n);
	};
	unsigned languageAt = 0;
	NodeText language;
	std::string name;
	if (firstOf(languageList, languageAt) && node(languageAt, language) && !language.ops.empty() &&
	    stringOp(language.ops[0], name) && name != "Metal") {
		unsigned versionAt = 0, major = 2, minor = 3;
		NodeText version;
		if (firstOf(versionList, versionAt) && node(versionAt, version) && version.ops.size() >= 2) {
			intOp(version.ops[0], major);
			intOp(version.ops[1], minor);
		}
		language.ops = {"!\"Metal\"", "i32 " + std::to_string(major), "i32 " + std::to_string(minor),
		                "i32 0"};
		lines[nodeLine[languageAt]] = joinNode(language);
		changed = true;
	}
	// The entry point's outputs and inputs.
	unsigned entry = 0, outputsAt = 0, inputsAt = 0;
	NodeText e, outputs, inputs;
	NodeText named;
	if (entryList < lines.size() && splitNode(lines[entryList], named) && !named.ops.empty() &&
	    refOp(named.ops[0], entry) && node(entry, e) && e.ops.size() >= 3 && refOp(e.ops[1], outputsAt) &&
	    refOp(e.ops[2], inputsAt) && node(outputsAt, outputs) && node(inputsAt, inputs)) {
		for (size_t i = 0; i < outputs.ops.size(); i++) {
			unsigned n = 0;
			if (refOp(outputs.ops[i], n) && n < nodeLine.size() && nodeLine[n] < lines.size())
				changed |= normalizeNode(lines[nodeLine[n]], false,
				                         i < signature.outputs.size() ? signature.outputs[i] : std::string());
		}
		for (const std::string &op : inputs.ops) {
			unsigned n = 0, param = 0;
			NodeText in;
			if (!refOp(op, n) || !node(n, in) || in.ops.empty() || !intOp(in.ops[0], param))
				continue;
			changed |= normalizeNode(lines[nodeLine[n]], true,
			                         param < signature.parameters.size() ? signature.parameters[param]
			                                                             : std::string());
		}
	}
	if (!changed)
		return false;
	std::string joined;
	bool first = true;
	for (size_t i = 0; i < lines.size(); i++) {
		if (dropped[i])
			continue;
		joined += (first ? "" : "\n") + lines[i];
		first = false;
	}
	text = joined;
	return true;
}

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
		char *printed = l.printModule(module);
		std::string text = printed != nullptr ? printed : "";
		if (printed != nullptr)
			l.disposeMessage(printed);
		if (RgfxGLNormalizeStageText(text, signatureOf(l, module, f.name))) {
			Ref parsed = nullptr;
			char *message = nullptr;
			Ref buffer = l.bufferCopy(text.data(), text.size(), "rgfx-gl");
			if (l.parseIR(context, buffer, &parsed, &message) != 0 || parsed == nullptr) {
				ok = false;
				why = std::string("the system's LLVM refused the module's text: ") +
				      (message != nullptr ? message : "no message");
			} else {
				l.disposeModule(module);
				module = parsed;
			}
			if (message != nullptr)
				l.disposeMessage(message);
		}
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
