/*
 * RgfxGLLowering: see RgfxGLLowering.hpp. Our own C++.
 */
#include "RgfxGLLowering.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
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
	Ref (*globalValueType)(Ref global) = nullptr; // null where the system's LLVM lacks it
	Ref (*typeOf)(Ref value) = nullptr;
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
		    !symbol(llvm, "LLVMTypeOf", libs.typeOf, why) ||
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
		// Optional: GPUCompiler 31001's LLVM does not export it (functionType).
		libs.globalValueType = reinterpret_cast<Ref (*)(Ref)>(dlsym(llvm, "LLVMGlobalGetValueType"));
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
enum : int {
	kHalfKind = 1,
	kFloatKind = 2,
	kIntegerKind = 8,
	kFunctionKind = 9,
	kStructKind = 10,
	kPointerKind = 12,
	kVectorKind = 13
};

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

// A function's type. The system's LLVM answers it where it exports LLVMGlobalGetValueType, as macOS
// 26.7.2's GPUCompiler 32023 does. macOS 11.7.11's, 12.7.6's and 13.7.8's GPUCompiler 31001 does
// not (measured on x86 guests), and without the call every GL shader variant was refused there: the
// renderer fell back to software processing and a triangle left the clear color behind (measured: a
// GL 4.1 core draw into an IOSurface). Where it lacks it, the function's type is the pointee of its
// value's type: the generator's LLVM types its pointers, as its modules show (`<4 x float>
// addrspace(2)*`, above). Null where neither answers a function type.
Ref functionType(const Libraries &l, Ref function)
{
	if (l.globalValueType != nullptr)
		return l.globalValueType(function);
	Ref pointer = l.typeOf(function);
	if (pointer == nullptr || l.typeKind(pointer) != kPointerKind)
		return nullptr;
	Ref type = l.elementType(pointer);
	return type != nullptr && l.typeKind(type) == kFunctionKind ? type : nullptr;
}

RgfxGLSignature signatureOf(const Libraries &l, Ref module, const std::string &entry)
{
	RgfxGLSignature s;
	Ref function = l.namedFunction(module, entry.c_str());
	if (function == nullptr)
		return s;
	Ref type = functionType(l, function);
	if (type == nullptr)
		return s;
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

// A parameter of an entry point's function type: the address space of its pointer (N for a pointer
// into addrspace(N), 0 for one into none) and the pointer's pointee type, or -1 and the type of a
// value that is not a pointer.
struct ParameterType {
	int space = -1;
	std::string pointee;
};

// The parameters of the function type an entry node's first operand holds
// ("{ <4 x float> } (i8 addrspace(2)*, i32)* @__main").
std::vector<ParameterType> parameterTypes(const std::string &op)
{
	std::vector<ParameterType> params;
	// The parameter list: the outermost parentheses after the return type.
	size_t open = std::string::npos;
	int depth = 0;
	for (size_t i = 0; i < op.size(); i++) {
		const char c = op[i];
		if (c == '{' || c == '<' || c == '[') {
			depth++;
		} else if (c == '}' || c == '>' || c == ']') {
			depth--;
		} else if (c == '(' && depth == 0) {
			open = i;
			break;
		}
	}
	if (open == std::string::npos)
		return params;
	std::string param;
	depth = 0;
	for (size_t i = open + 1; i < op.size(); i++) {
		const char c = op[i];
		if (c == '(' || c == '{' || c == '<' || c == '[')
			depth++;
		else if (c == ')' || c == '}' || c == '>' || c == ']')
			depth--;
		if (depth < 0 || (depth == 0 && c == ',')) {
			if (!param.empty()) {
				ParameterType t;
				t.pointee = param;
				if (param.back() == '*') {
					t.pointee.pop_back();
					t.space = 0;
					const size_t at = t.pointee.rfind(" addrspace(");
					if (at != std::string::npos && t.pointee.back() == ')') {
						t.space = std::atoi(t.pointee.c_str() + at + 11);
						t.pointee.erase(at);
					}
					while (!t.pointee.empty() && t.pointee.back() == ' ')
						t.pointee.pop_back();
				}
				params.push_back(t);
			}
			param.clear();
			if (depth < 0)
				break;
			continue;
		}
		if (param.empty() && c == ' ')
			continue;
		param += c;
	}
	return params;
}

// The allocation size and alignment of LLVM type `t` in an AIR module's data layout, which
// metal2vulkan checks a buffer's air.arg_type_size against and `metal` writes: an integer's bytes
// rounded up to a power of two, half 2, float 4, double 8; a vector its elements' size rounded up
// to a power of two, aligned to that; an array its count of its element's. False for any other
// type (a struct, a pointer).
bool llvmTypeSize(std::string t, uint64_t &size, uint64_t &align)
{
	while (!t.empty() && t.front() == ' ')
		t.erase(0, 1);
	while (!t.empty() && t.back() == ' ')
		t.pop_back();
	auto powerOfTwo = [](uint64_t n) {
		uint64_t p = 1;
		while (p < n)
			p <<= 1;
		return p;
	};
	if (t == "half" || t == "float" || t == "double") {
		size = align = t == "half" ? 2 : t == "float" ? 4 : 8;
		return true;
	}
	if (t.size() > 1 && t[0] == 'i' && t.find_first_not_of("0123456789", 1) == std::string::npos) {
		const uint64_t bits = std::strtoull(t.c_str() + 1, nullptr, 10);
		if (bits == 0 || bits > 64)
			return false;
		size = align = powerOfTwo((bits + 7) / 8);
		return true;
	}
	const bool vector = t.size() > 2 && t.front() == '<' && t.back() == '>';
	const bool array = t.size() > 2 && t.front() == '[' && t.back() == ']';
	if (!vector && !array)
		return false;
	const std::string inner = t.substr(1, t.size() - 2);
	const size_t x = inner.find(" x ");
	if (x == std::string::npos)
		return false;
	char *end = nullptr;
	const uint64_t count = std::strtoull(inner.c_str(), &end, 10);
	if (end != inner.c_str() + x || count == 0)
		return false;
	uint64_t elementSize = 0, elementAlign = 0;
	if (!llvmTypeSize(inner.substr(x + 3), elementSize, elementAlign))
		return false;
	if (array) {
		size = count * elementSize;
		align = elementAlign;
	} else {
		size = align = powerOfTwo(count * elementSize);
	}
	return true;
}

// The Metal name of LLVM type `t` as a module's text writes it, as metalTypeName names a type:
// "float2" for <2 x float>, "uint" for i32; integers unsigned. "" for any other type.
std::string metalTextTypeName(std::string t)
{
	while (!t.empty() && t.front() == ' ')
		t.erase(0, 1);
	while (!t.empty() && t.back() == ' ')
		t.pop_back();
	unsigned lanes = 1;
	if (t.size() > 2 && t.front() == '<' && t.back() == '>') {
		const std::string inner = t.substr(1, t.size() - 2);
		const size_t x = inner.find(" x ");
		char *end = nullptr;
		lanes = static_cast<unsigned>(std::strtoul(inner.c_str(), &end, 10));
		if (x == std::string::npos || end != inner.c_str() + x || lanes < 2)
			return "";
		t = inner.substr(x + 3);
	}
	static const struct {
		const char *llvm, *metal;
	} kNames[] = {{"half", "half"},   {"float", "float"}, {"i1", "bool"},  {"i8", "uchar"},
	              {"i16", "ushort"}, {"i32", "uint"},    {"i64", "ulong"}};
	for (const auto &n : kNames)
		if (t == n.llvm)
			return lanes > 1 ? n.metal + std::to_string(lanes) : n.metal;
	return "";
}

// The member types of the result of the function type an operand holds
// ("{ <4 x float>, <2 x float> } (i32, i8*)* @fn"): a struct's members, or the one result.
std::vector<std::string> resultMembers(const std::string &op)
{
	std::vector<std::string> members;
	if (op.empty() || op.front() != '{') {
		const size_t open = op.find(" (");
		if (open != std::string::npos)
			members.push_back(op.substr(0, open));
		return members;
	}
	std::string member;
	int depth = 0;
	for (size_t i = 1; i < op.size(); i++) {
		const char c = op[i];
		if (c == '(' || c == '{' || c == '<' || c == '[')
			depth++;
		else if (c == ')' || c == '}' || c == '>' || c == ']')
			depth--;
		if (depth < 0 || (depth == 0 && c == ',')) {
			members.push_back(member);
			member.clear();
			if (depth < 0)
				break;
			continue;
		}
		member += c;
	}
	return members;
}

// Drops an input's attribute name, which the generator writes as a bare string after its
// location's two operands and `metal` does not; true where it did.
bool dropAttributeName(NodeText &n)
{
	for (size_t i = 0; i + 3 < n.ops.size(); i++) {
		std::string key, bare;
		unsigned a = 0, b = 0;
		if (stringOp(n.ops[i], key) && key == "air.location_index" && intOp(n.ops[i + 1], a) &&
		    intOp(n.ops[i + 2], b) && stringOp(n.ops[i + 3], bare) && bare.compare(0, 4, "air.") != 0) {
			n.ops.erase(n.ops.begin() + static_cast<std::ptrdiff_t>(i + 3));
			return true;
		}
	}
	return false;
}

// Makes a field node of a patch control point input what `metal` writes: nothing between its
// location's two operands and its air.arg_type_name (the generator writes the attribute name there,
// and, for a field that carries gl_Position, the key air.position), and the Metal name `type` of
// the field's member of the control point function's result where the generator wrote the key
// air.arg_type_name in place of a type name. True where it changed.
bool normalizeControlPointField(std::string &line, const std::string &type)
{
	NodeText n;
	if (!splitNode(line, n))
		return false;
	bool edited = false;
	for (size_t i = 0; i + 2 < n.ops.size(); i++) {
		std::string key;
		unsigned a = 0, b = 0;
		if (!stringOp(n.ops[i], key) || key != "air.location_index" || !intOp(n.ops[i + 1], a) ||
		    !intOp(n.ops[i + 2], b))
			continue;
		size_t end = i + 3;
		while (end < n.ops.size() && !(stringOp(n.ops[end], key) && key == "air.arg_type_name"))
			end++;
		if (end < n.ops.size() && end > i + 3) {
			n.ops.erase(n.ops.begin() + static_cast<std::ptrdiff_t>(i + 3), n.ops.begin() + static_cast<std::ptrdiff_t>(end));
			edited = true;
		}
		break;
	}
	for (size_t i = 0; i + 1 < n.ops.size(); i++) {
		std::string key, value;
		if (stringOp(n.ops[i], key) && key == "air.arg_type_name" && stringOp(n.ops[i + 1], value) &&
		    value.compare(0, 4, "air.") == 0 && !type.empty()) {
			n.ops[i + 1] = "!\"" + type + "\"";
			edited = true;
			break;
		}
	}
	if (edited)
		line = joinNode(n);
	return edited;
}

// Makes one output or input node's text what the rules in RgfxGLLowering.hpp say; true where it
// changed. `type` is the Metal name of the value's type from the entry point's signature, and
// `param` an input's parameter in the entry point's function type (parameterTypes).
bool normalizeNode(std::string &line, bool input, const std::string &type, const ParameterType &param = {})
{
	NodeText n;
	if (!splitNode(line, n))
		return false;
	// An input's attribute name (a vertex function's vertex input, a kernel's input) is not in the
	// node `metal` writes.
	bool edited = input && dropAttributeName(n);
	if (edited)
		line = joinNode(n);
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
	// A buffer's address space and pointee size, which the generator does not write where `metal`
	// does: air.address_space after the access (or the location), and air.arg_type_size and
	// air.arg_type_align_size before air.arg_type_name, from the parameter's pointer. The size is
	// the pointee's, not the type name's: the generator names a uniform block's pointer
	// <4 x float> addrspace(2)* "uchar" (measured: macOS 26.7.1, gl_cases' block case), and
	// metal2vulkan refuses a size other than the pointee's allocation size.
	std::string first;
	if (input && n.ops.size() > 1 && stringOp(n.ops[1], first) && first == "air.buffer") {
		size_t accessAt = n.ops.size(), locationAt = n.ops.size(), nameAt = n.ops.size();
		bool hasSpace = false, hasSize = false;
		for (size_t i = 0; i < n.ops.size(); i++) {
			std::string key;
			if (!stringOp(n.ops[i], key))
				continue;
			if (key == "air.read" || key == "air.read_write" || key == "air.write")
				accessAt = i;
			else if (key == "air.location_index")
				locationAt = i;
			else if (key == "air.address_space")
				hasSpace = true;
			else if (key == "air.arg_type_size")
				hasSize = true;
			else if (key == "air.arg_type_name")
				nameAt = i;
		}
		uint64_t size = 0, align = 0;
		if (!hasSize && nameAt != n.ops.size() && param.space >= 0 &&
		    llvmTypeSize(param.pointee, size, align)) {
			n.ops.insert(n.ops.begin() + static_cast<std::ptrdiff_t>(nameAt),
			             {"!\"air.arg_type_size\"", "i32 " + std::to_string(size),
			              "!\"air.arg_type_align_size\"", "i32 " + std::to_string(align)});
			edited = true;
		}
		const size_t after = accessAt != n.ops.size() ? accessAt + 1
		                     : locationAt + 2 < n.ops.size() ? locationAt + 3
		                                                     : n.ops.size();
		if (!hasSpace && param.space > 0 && after <= n.ops.size()) {
			n.ops.insert(n.ops.begin() + static_cast<std::ptrdiff_t>(after),
			             {"!\"air.address_space\"", "i32 " + std::to_string(param.space)});
			edited = true;
		}
		if (edited)
			line = joinNode(n);
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

bool RgfxGLSortDeclarations(std::string &text)
{
	struct Line {
		size_t begin, end; // end at its '\n' or the text's end
	};
	std::vector<Line> lines;
	std::vector<size_t> slots; // the declarations' lines
	for (size_t a = 0; a < text.size();) {
		size_t b = text.find('\n', a);
		if (b == std::string::npos)
			b = text.size();
		if (text.compare(a, 8, "declare ") == 0)
			slots.push_back(lines.size());
		lines.push_back({a, b});
		a = b + 1;
	}
	if (slots.size() < 2)
		return false;
	// A declaration's name: from its '@' to its '(' (past a quoted name's closing quote).
	auto nameOf = [&](size_t i) {
		const std::string line = text.substr(lines[i].begin, lines[i].end - lines[i].begin);
		const size_t at = line.find(" @");
		if (at == std::string::npos)
			return line;
		size_t from = at + 2;
		if (from < line.size() && line[from] == '"') {
			const size_t quote = line.find('"', from + 1);
			from = quote == std::string::npos ? line.size() : quote;
		}
		const size_t paren = line.find('(', from);
		return line.substr(at + 1, paren == std::string::npos ? std::string::npos : paren - at - 1);
	};
	std::vector<std::pair<std::string, size_t>> order;
	for (size_t i : slots)
		order.emplace_back(nameOf(i), i);
	// By name, then by the whole line, so that the order is the text's own whatever came first.
	std::sort(order.begin(), order.end(), [&](const auto &x, const auto &y) {
		if (x.first != y.first)
			return x.first < y.first;
		return text.compare(lines[x.second].begin, lines[x.second].end - lines[x.second].begin, text,
		                    lines[y.second].begin, lines[y.second].end - lines[y.second].begin) < 0;
	});
	bool moved = false;
	for (size_t k = 0; k < slots.size(); k++)
		if (order[k].second != slots[k] &&
		    text.compare(lines[slots[k]].begin, lines[slots[k]].end - lines[slots[k]].begin, text,
		                 lines[order[k].second].begin, lines[order[k].second].end - lines[order[k].second].begin) != 0)
			moved = true;
	if (!moved)
		return false;
	std::string out;
	out.reserve(text.size());
	size_t next = 0;
	for (size_t i = 0; i < lines.size(); i++) {
		const Line &from = next < slots.size() && slots[next] == i ? lines[order[next++].second] : lines[i];
		out.append(text, from.begin, from.end - from.begin);
		if (lines[i].end < text.size())
			out.push_back('\n');
	}
	text = std::move(out);
	return true;
}

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
		           (starts(line, "!air.vertex = ") || starts(line, "!air.fragment = ") ||
		            starts(line, "!air.kernel = "))) {
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
		const std::vector<ParameterType> params = parameterTypes(e.ops[0]);
		for (const std::string &op : inputs.ops) {
			unsigned n = 0, param = 0;
			NodeText in;
			if (!refOp(op, n) || !node(n, in) || in.ops.empty() || !intOp(in.ops[0], param))
				continue;
			changed |= normalizeNode(lines[nodeLine[n]], true,
			                         param < signature.parameters.size() ? signature.parameters[param]
			                                                             : std::string(),
			                         param < params.size() ? params[param] : ParameterType());
			// A patch control point input's fields, one per member of its control point function's
			// result, in order.
			std::string kind;
			unsigned function = 0;
			NodeText f;
			if (in.ops.size() < 3 || !stringOp(in.ops[1], kind) || kind != "air.patch_control_point_input" ||
			    !refOp(in.ops[2], function) || !node(function, f) || f.ops.size() < 2)
				continue;
			const std::vector<std::string> members = resultMembers(f.ops[1]);
			for (size_t k = 0; k + 3 < in.ops.size(); k++) {
				unsigned field = 0;
				if (refOp(in.ops[k + 3], field) && field < nodeLine.size() && nodeLine[field] < lines.size())
					changed |= normalizeControlPointField(
					    lines[nodeLine[field]], k < members.size() ? metalTextTypeName(members[k]) : std::string());
			}
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

std::string RgfxGLStageIn::key() const
{
	std::string k;
	auto put = [&k](uint32_t v) { k.append(reinterpret_cast<const char *>(&v), sizeof v); };
	for (const Attribute &a : attributes) {
		put(a.format);
		put(a.offset);
		put(a.bufferIndex);
	}
	for (const Layout &l : layouts) {
		put(l.stride);
		put(l.stepFunction);
		put(l.stepRate);
	}
	put(indexType);
	put(indexBufferIndex);
	return k;
}

namespace {

// MTLStepFunction's values the fetch reads (Metal/MTLStageInputOutputDescriptor.h).
enum : uint32_t {
	kStepConstant = 0,
	kStepGridX = 5,
	kStepGridY = 6,
	kStepGridXIndexed = 7,
};

// How an MTLAttributeFormat's bytes hold its components: `count` of them, of `bits` each,
// unsigned or signed integers ('u', 's'), halves or floats ('h', 'f'), normalized or not; or one of
// the packed formats.
struct AttributeLayout {
	unsigned count = 0, bits = 0;
	char kind = 0;
	bool normalized = false;
	enum Packed { None, Snorm1010102, Unorm1010102, Bgra8, Rg11b10f, Rgb9e5 } packed = None;
};

// Metal/MTLStageInputOutputDescriptor.h's MTLAttributeFormat values.
bool attributeLayout(uint32_t format, AttributeLayout &f)
{
	struct Row {
		uint32_t first;
		unsigned bits;
		char kind;
		bool normalized;
	};
	// The formats of 2, 3 and 4 components in runs of three, from their first value.
	static const Row kRuns[] = {{1, 8, 'u', false},   {4, 8, 's', false},   {7, 8, 'u', true},
	                            {10, 8, 's', true},   {13, 16, 'u', false}, {16, 16, 's', false},
	                            {19, 16, 'u', true},  {22, 16, 's', true},  {25, 16, 'h', false}};
	for (const Row &r : kRuns)
		if (format >= r.first && format < r.first + 3) {
			f = {format - r.first + 2, r.bits, r.kind, r.normalized, AttributeLayout::None};
			return true;
		}
	if (format >= 28 && format <= 31) {
		f = {format - 27, 32, 'f', false, AttributeLayout::None};
		return true;
	}
	if (format >= 32 && format <= 39) {
		f = {(format - 32) % 4 + 1, 32, format < 36 ? 's' : 'u', false, AttributeLayout::None};
		return true;
	}
	static const struct {
		uint32_t format;
		AttributeLayout layout;
	} kOthers[] = {
	    {40, {4, 32, 's', true, AttributeLayout::Snorm1010102}},
	    {41, {4, 32, 'u', true, AttributeLayout::Unorm1010102}},
	    {42, {4, 8, 'u', true, AttributeLayout::Bgra8}},
	    {45, {1, 8, 'u', false, AttributeLayout::None}},
	    {46, {1, 8, 's', false, AttributeLayout::None}},
	    {47, {1, 8, 'u', true, AttributeLayout::None}},
	    {48, {1, 8, 's', true, AttributeLayout::None}},
	    {49, {1, 16, 'u', false, AttributeLayout::None}},
	    {50, {1, 16, 's', false, AttributeLayout::None}},
	    {51, {1, 16, 'u', true, AttributeLayout::None}},
	    {52, {1, 16, 's', true, AttributeLayout::None}},
	    {53, {1, 16, 'h', false, AttributeLayout::None}},
	    {54, {3, 32, 'f', false, AttributeLayout::Rg11b10f}},
	    {55, {3, 32, 'f', false, AttributeLayout::Rgb9e5}},
	};
	for (const auto &o : kOthers)
		if (o.format == format) {
			f = o.layout;
			return true;
		}
	return false;
}

// An LLVM scalar or vector type's element type text and lanes ("<4 x float>": "float", 4).
bool splitVector(std::string t, std::string &element, unsigned &lanes)
{
	while (!t.empty() && t.front() == ' ')
		t.erase(0, 1);
	while (!t.empty() && t.back() == ' ')
		t.pop_back();
	lanes = 1;
	element = t;
	if (t.size() > 2 && t.front() == '<' && t.back() == '>') {
		const std::string inner = t.substr(1, t.size() - 2);
		const size_t x = inner.find(" x ");
		if (x == std::string::npos)
			return false;
		lanes = static_cast<unsigned>(std::strtoul(inner.c_str(), nullptr, 10));
		element = inner.substr(x + 3);
	}
	return element == "float" || element == "half" || element == "i32" || element == "i16" || element == "i8" ||
	       element == "i1" || element == "i64";
}

// The IR a fetch is written in: numbered values, one instruction a line.
struct Emitter {
	std::string body;
	unsigned next = 0;
	std::string value() { return "%rgfx." + std::to_string(next++); }
	std::string emit(const std::string &instruction)
	{
		const std::string v = value();
		body += "  " + v + " = " + instruction + "\n";
		return v;
	}
	void line(const std::string &instruction) { body += "  " + instruction + "\n"; }
};

std::string floatConstant(double d)
{
	// LLVM's text writes a float constant as the double of its value, in hex.
	const float f = static_cast<float>(d);
	const double exact = f;
	uint64_t bits = 0;
	std::memcpy(&bits, &exact, sizeof bits);
	char text[24];
	std::snprintf(text, sizeof text, "0x%016llX", static_cast<unsigned long long>(bits));
	return text;
}

// One component as float of an unsigned or signed integer `v` of `bits` bits, normalized as Metal's
// stage input fetch converts it: unsigned c / (2^bits - 1); signed max(c / (2^(bits - 1) - 1), -1),
// save that `byReciprocal` multiplies c by the float nearest 1 / (2^(bits - 1) - 1) instead
// (measured: the M3 Max under macOS 27.0.1 fetches Int1010102Normalized's 10-bit components so,
// and every other normalized format's by the quotient; tests/gl_tests.mm's stage input oracle).
std::string normalizedComponent(Emitter &e, const std::string &v, unsigned bits, bool isSigned,
                                bool byReciprocal = false)
{
	const std::string i = "i" + std::to_string(bits);
	if (!isSigned) {
		const std::string f = e.emit("uitofp " + i + " " + v + " to float");
		return e.emit("fdiv float " + f + ", " + floatConstant(static_cast<double>((1ull << bits) - 1)));
	}
	const std::string f = e.emit("sitofp " + i + " " + v + " to float");
	const float largest = static_cast<float>((1ull << (bits - 1)) - 1);
	const std::string q = byReciprocal ? e.emit("fmul float " + f + ", " + floatConstant(1.0f / largest))
	                                   : e.emit("fdiv float " + f + ", " + floatConstant(largest));
	const std::string low = e.emit("fcmp olt float " + q + ", " + floatConstant(-1.0));
	return e.emit("select i1 " + low + ", float " + floatConstant(-1.0) + ", float " + q);
}

// The float of an unsigned small float of `mantissa` mantissa bits and 5 exponent bits, `v` an i32
// holding it in its low bits (RG11B10's: 6 or 5 mantissa bits, bias 15).
std::string smallFloat(Emitter &e, const std::string &v, unsigned mantissa)
{
	const std::string m = e.emit("and i32 " + v + ", " + std::to_string((1u << mantissa) - 1));
	const std::string ex = e.emit("lshr i32 " + v + ", " + std::to_string(mantissa));
	const std::string x = e.emit("and i32 " + ex + ", 31");
	// Normal: the float's bits (x - 15 + 127) << 23 | m << (23 - mantissa); 31: infinity where m is
	// 0, else the quiet NaN 0x7fc00000, whatever m (measured: the M3 Max under macOS 27.0.1,
	// tests/gl_tests.mm's stage input oracle); 0: the denormal m * 2^(-14 - mantissa).
	const std::string biased = e.emit("add i32 " + x + ", 112");
	const std::string hi = e.emit("shl i32 " + biased + ", 23");
	const std::string lo = e.emit("shl i32 " + m + ", " + std::to_string(23 - mantissa));
	const std::string normal = e.emit("or i32 " + hi + ", " + lo);
	const std::string infinite = e.emit("icmp eq i32 " + m + ", 0");
	const std::string special = e.emit("select i1 " + infinite + ", i32 2139095040, i32 2143289344");
	const std::string isMax = e.emit("icmp eq i32 " + x + ", 31");
	const std::string bitsValue = e.emit("select i1 " + isMax + ", i32 " + special + ", i32 " + normal);
	const std::string asFloat = e.emit("bitcast i32 " + bitsValue + " to float");
	const std::string mf = e.emit("uitofp i32 " + m + " to float");
	const std::string denormal =
	    e.emit("fmul float " + mf + ", " + floatConstant(std::ldexp(1.0, -14 - static_cast<int>(mantissa))));
	const std::string isZero = e.emit("icmp eq i32 " + x + ", 0");
	return e.emit("select i1 " + isZero + ", float " + denormal + ", float " + asFloat);
}

// A component's natural value: an integer of the format's width and signedness, or a float.
struct Component {
	std::string value;
	std::string type; // "float", "half", "iN"
	bool isSigned = false;
};

// The components the format at `pointer` (an i8 addrspace(1)*) holds, as Metal's vertex fetch
// reads them. Each is loaded at its own size's alignment, which it has: Metal takes an attribute's
// offset and a layout's stride only as multiples of 4.
std::vector<Component> readComponents(Emitter &e, const std::string &pointer, const AttributeLayout &f)
{
	std::vector<Component> out;
	auto load = [&](const std::string &type, unsigned byteOffset) {
		std::string p = pointer;
		if (byteOffset != 0)
			p = e.emit("getelementptr inbounds i8, i8 addrspace(1)* " + pointer + ", i64 " +
			           std::to_string(byteOffset));
		const std::string typed =
		    type == "i8" ? p : e.emit("bitcast i8 addrspace(1)* " + p + " to " + type + " addrspace(1)*");
		const char *align = type == "i8" ? "1" : type == "i16" ? "2" : "4";
		return e.emit("load " + type + ", " + type + " addrspace(1)* " + typed + ", align " + align);
	};
	if (f.packed == AttributeLayout::None) {
		const std::string type = f.kind == 'f' ? "float" : f.kind == 'h' ? "half" : "i" + std::to_string(f.bits);
		for (unsigned c = 0; c < f.count; c++) {
			std::string v = load(f.kind == 'h' ? "i16" : type, c * f.bits / 8);
			if (f.kind == 'h')
				v = e.emit("bitcast i16 " + v + " to half");
			if (f.normalized)
				out.push_back({normalizedComponent(e, v, f.bits, f.kind == 's'), "float", false});
			else
				out.push_back({v, type, f.kind == 's'});
		}
		return out;
	}
	if (f.packed == AttributeLayout::Bgra8) {
		std::string bytes[4];
		for (unsigned c = 0; c < 4; c++)
			bytes[c] = load("i8", c);
		const unsigned order[4] = {2, 1, 0, 3};
		for (unsigned c = 0; c < 4; c++)
			out.push_back({normalizedComponent(e, bytes[order[c]], 8, false), "float", false});
		return out;
	}
	const std::string word = load("i32", 0);
	auto field = [&](unsigned shift, unsigned width) {
		const std::string s = shift != 0 ? e.emit("lshr i32 " + word + ", " + std::to_string(shift)) : word;
		return e.emit("trunc i32 " + s + " to i" + std::to_string(width));
	};
	switch (f.packed) {
	case AttributeLayout::Snorm1010102:
	case AttributeLayout::Unorm1010102: {
		const bool isSigned = f.packed == AttributeLayout::Snorm1010102;
		for (unsigned c = 0; c < 3; c++)
			out.push_back({normalizedComponent(e, field(10 * c, 10), 10, isSigned, isSigned), "float", false});
		out.push_back({normalizedComponent(e, field(30, 2), 2, isSigned), "float", false});
		break;
	}
	case AttributeLayout::Rg11b10f: {
		const unsigned shifts[3] = {0, 11, 22}, widths[3] = {11, 11, 10};
		for (unsigned c = 0; c < 3; c++) {
			const std::string s = shifts[c] != 0 ? e.emit("lshr i32 " + word + ", " + std::to_string(shifts[c])) : word;
			const std::string bits = e.emit("and i32 " + s + ", " + std::to_string((1u << widths[c]) - 1));
			out.push_back({smallFloat(e, bits, widths[c] - 5), "float", false});
		}
		break;
	}
	case AttributeLayout::Rgb9e5: {
		// Each of 9 mantissa bits times 2^(e - 15 - 9), e the shared exponent (bits 27 to 31).
		const std::string ex = e.emit("lshr i32 " + word + ", 27");
		const std::string biased = e.emit("add i32 " + ex + ", 103");
		const std::string scaleBits = e.emit("shl i32 " + biased + ", 23");
		const std::string scale = e.emit("bitcast i32 " + scaleBits + " to float");
		for (unsigned c = 0; c < 3; c++) {
			const std::string s = c != 0 ? e.emit("lshr i32 " + word + ", " + std::to_string(9 * c)) : word;
			const std::string m = e.emit("and i32 " + s + ", 511");
			const std::string mf = e.emit("uitofp i32 " + m + " to float");
			out.push_back({e.emit("fmul float " + mf + ", " + scale), "float", false});
		}
		break;
	}
	default:
		break;
	}
	return out;
}

// `c` converted to `element` ("float", "half", "iN"): floats widened or narrowed, integers
// extended by their signedness or truncated, integers to floats by value, floats to integers
// toward zero.
std::string convert(Emitter &e, const Component &c, const std::string &element)
{
	if (c.type == element)
		return c.value;
	const bool fromFloat = c.type == "float" || c.type == "half";
	const bool toFloat = element == "float" || element == "half";
	if (fromFloat && toFloat)
		return e.emit((element == "float" ? "fpext " : "fptrunc ") + c.type + " " + c.value + " to " + element);
	if (fromFloat)
		return e.emit("fptosi " + c.type + " " + c.value + " to " + element);
	if (toFloat)
		return e.emit((c.isSigned ? "sitofp " : "uitofp ") + c.type + " " + c.value + " to " + element);
	const unsigned from = static_cast<unsigned>(std::strtoul(c.type.c_str() + 1, nullptr, 10));
	const unsigned to = static_cast<unsigned>(std::strtoul(element.c_str() + 1, nullptr, 10));
	if (from > to)
		return e.emit("trunc " + c.type + " " + c.value + " to " + element);
	return e.emit((c.isSigned ? "sext " : "zext ") + c.type + " " + c.value + " to " + element);
}

// The constant `v` (0 or 1) of `element`.
std::string constantOf(const std::string &element, int v)
{
	if (element == "float" || element == "half")
		return v == 0 ? "0.0" : "1.0";
	return std::to_string(v);
}

// The parameters' types of the function type an operand holds ("void (i8*, <2 x i32>)* @k").
std::vector<std::string> parameterTexts(const std::string &op)
{
	std::vector<std::string> params;
	size_t open = std::string::npos;
	int depth = 0;
	for (size_t i = 0; i < op.size(); i++) {
		const char c = op[i];
		if (c == '{' || c == '<' || c == '[')
			depth++;
		else if (c == '}' || c == '>' || c == ']')
			depth--;
		else if (c == '(' && depth == 0) {
			open = i;
			break;
		}
	}
	if (open == std::string::npos)
		return params;
	std::string param;
	depth = 0;
	for (size_t i = open + 1; i < op.size(); i++) {
		const char c = op[i];
		if (c == '(' || c == '{' || c == '<' || c == '[')
			depth++;
		else if (c == ')' || c == '}' || c == '>' || c == ']')
			depth--;
		if (depth < 0 || (depth == 0 && c == ',')) {
			while (!param.empty() && param.back() == ' ')
				param.pop_back();
			if (!param.empty())
				params.push_back(param);
			param.clear();
			if (depth < 0)
				break;
			continue;
		}
		if (param.empty() && c == ' ')
			continue;
		param += c;
	}
	return params;
}

// The x (`lane` 0) or y (1) of a value of LLVM type `type` (a vector of i32 or one i32; 0 for y
// of a scalar).
std::string laneOf(Emitter &e, const std::string &type, const std::string &v, unsigned lane)
{
	if (type == "i32")
		return lane == 0 ? v : "0";
	return e.emit("extractelement " + type + " " + v + ", i32 " + std::to_string(lane));
}

} // namespace

bool RgfxGLSpecializeStageInText(std::string &text, const RgfxGLStageIn &in, std::string &why)
{
	std::vector<std::string> lines;
	for (size_t at = 0; at <= text.size();) {
		size_t end = text.find('\n', at);
		if (end == std::string::npos)
			end = text.size();
		lines.push_back(text.substr(at, end - at));
		at = end + 1;
	}
	std::vector<size_t> nodeLine;
	size_t kernelList = lines.size();
	unsigned nextNode = 0;
	for (size_t i = 0; i < lines.size(); i++) {
		unsigned n = 0;
		if (refOp(lines[i], n) && lines[i].find(" = ") != std::string::npos) {
			if (nodeLine.size() <= n)
				nodeLine.resize(n + 1, lines.size());
			nodeLine[n] = i;
			nextNode = std::max(nextNode, n + 1);
		} else if (kernelList == lines.size() && lines[i].compare(0, 14, "!air.kernel = ") == 0) {
			kernelList = i;
		}
	}
	auto node = [&](unsigned n, NodeText &out) {
		return n < nodeLine.size() && nodeLine[n] < lines.size() && splitNode(lines[nodeLine[n]], out);
	};
	NodeText named, entry, inputs;
	unsigned entryNode = 0, inputsNode = 0;
	if (kernelList == lines.size() || !splitNode(lines[kernelList], named) || named.ops.empty() ||
	    !refOp(named.ops[0], entryNode) || !node(entryNode, entry) || entry.ops.size() < 3 ||
	    !refOp(entry.ops[2], inputsNode) || !node(inputsNode, inputs)) {
		why = "the module has no kernel entry point";
		return false;
	}
	const size_t at = entry.ops[0].rfind(" @");
	if (at == std::string::npos) {
		why = "the kernel's entry node names no function";
		return false;
	}
	const std::string name = entry.ops[0].substr(at + 2);
	const std::vector<std::string> params = parameterTexts(entry.ops[0]);

	// The kernel's inputs by argument: the control point input, the region's origin, the thread
	// position, and its buffers by index.
	struct Input {
		unsigned node = 0, arg = 0;
		NodeText text;
		std::string kind;
	};
	std::vector<Input> args;
	const Input *controlPoints = nullptr;
	int origin = -1, position = -1;
	std::vector<int> bufferAt(32, -1);
	for (const std::string &op : inputs.ops) {
		Input input;
		if (!refOp(op, input.node) || !node(input.node, input.text) || input.text.ops.size() < 2 ||
		    !intOp(input.text.ops[0], input.arg) || !stringOp(input.text.ops[1], input.kind) ||
		    input.arg >= params.size()) {
			why = "the kernel's input " + op + " is not an argument's";
			return false;
		}
		args.push_back(input);
	}
	for (const Input &input : args) {
		if (input.kind == "air.patch_control_point_input")
			controlPoints = &input;
		else if (input.kind == "air.stage_in_grid_origin")
			origin = static_cast<int>(input.arg);
		else if (input.kind == "air.thread_position_in_grid")
			position = static_cast<int>(input.arg);
		else if (input.kind == "air.buffer")
			for (size_t i = 2; i + 1 < input.text.ops.size(); i++) {
				std::string key;
				unsigned index = 0;
				if (stringOp(input.text.ops[i], key) && key == "air.location_index" &&
				    intOp(input.text.ops[i + 1], index) && index < bufferAt.size())
					bufferAt[index] = static_cast<int>(input.arg);
			}
	}
	unsigned functionNode = 0;
	NodeText function;
	if (controlPoints == nullptr || controlPoints->text.ops.size() < 3 ||
	    !refOp(controlPoints->text.ops[2], functionNode) || !node(functionNode, function) ||
	    function.ops.size() < 2) {
		why = "the kernel takes no patch control point input";
		return false;
	}
	const size_t fnAt = function.ops[1].rfind(" @");
	const size_t paren = function.ops[1].find(" (");
	if (fnAt == std::string::npos || paren == std::string::npos) {
		why = "the control point input names no function";
		return false;
	}
	const std::string fetchName = function.ops[1].substr(fnAt + 2);
	const std::string resultType = function.ops[1].substr(0, paren);
	const std::vector<std::string> members = resultMembers(function.ops[1]);

	// Each field: its attribute, layout, member type; the buffers the fields read, in index order.
	struct Field {
		unsigned location = 0;
		RgfxGLStageIn::Attribute attribute;
		RgfxGLStageIn::Layout layout;
		AttributeLayout format;
		std::string element;
		unsigned lanes = 1;
	};
	std::vector<Field> fields;
	std::vector<unsigned> slots; // buffer indices, sorted
	auto slotOf = [&slots](unsigned index) {
		const auto it = std::lower_bound(slots.begin(), slots.end(), index);
		if (it == slots.end() || *it != index)
			return slots.insert(it, index) - slots.begin();
		return it - slots.begin();
	};
	bool indexed = false, byY = false;
	for (size_t k = 3; k < controlPoints->text.ops.size(); k++) {
		unsigned fieldNode = 0;
		NodeText f;
		Field field;
		if (!refOp(controlPoints->text.ops[k], fieldNode) || !node(fieldNode, f) || k - 3 >= members.size() ||
		    !splitVector(members[k - 3], field.element, field.lanes)) {
			why = "the control point input's field " + std::to_string(k - 3) + " has no member type";
			return false;
		}
		bool located = false;
		for (size_t i = 0; i + 1 < f.ops.size() && !located; i++) {
			std::string key;
			located = stringOp(f.ops[i], key) && key == "air.location_index" && intOp(f.ops[i + 1], field.location);
		}
		if (!located || field.location >= 31) {
			why = "the control point input's field " + std::to_string(k - 3) + " has no attribute index";
			return false;
		}
		field.attribute = in.attributes[field.location];
		if (!attributeLayout(field.attribute.format, field.format) || field.attribute.bufferIndex >= 31) {
			why = "the stage input descriptor's attribute " + std::to_string(field.location) +
			      " has no format the fetch reads (" + std::to_string(field.attribute.format) + ")";
			return false;
		}
		field.layout = in.layouts[field.attribute.bufferIndex];
		const uint32_t step = field.layout.stepFunction;
		if (step != kStepConstant && step != kStepGridX && step != kStepGridY && step != kStepGridXIndexed) {
			why = "the stage input descriptor's layout " + std::to_string(field.attribute.bufferIndex) +
			      " steps by function " + std::to_string(step) + ", which no compute stage input takes";
			return false;
		}
		indexed |= step == kStepGridXIndexed;
		byY |= step == kStepGridY;
		fields.push_back(field);
	}
	for (const Field &f : fields)
		slotOf(f.attribute.bufferIndex);
	if (indexed)
		slotOf(in.indexBufferIndex);
	if (indexed && in.indexBufferIndex >= 31) {
		why = "the stage input descriptor's index buffer index is out of range";
		return false;
	}

	// The kernel of the same name: the arguments but the input, then the buffers it lacks, then the
	// origin and thread position where it lacks them.
	const std::string inner = name + ".rgfx_gl";
	const unsigned p = controlPoints->arg;
	std::vector<std::string> wrapperTypes;
	std::vector<int> wrapperOf(params.size(), -1);
	for (unsigned k = 0; k < params.size(); k++)
		if (k != p) {
			wrapperOf[k] = static_cast<int>(wrapperTypes.size());
			wrapperTypes.push_back(params[k]);
		}
	std::vector<std::string> added; // the new input nodes' text, before their numbers
	std::vector<std::string> slotValue(slots.size());
	for (size_t j = 0; j < slots.size(); j++) {
		const unsigned index = slots[j];
		if (bufferAt[index] >= 0 && static_cast<unsigned>(bufferAt[index]) != p) {
			const unsigned k = static_cast<unsigned>(bufferAt[index]);
			slotValue[j] = "%rgfx.a" + std::to_string(k);
			continue;
		}
		const unsigned position = static_cast<unsigned>(wrapperTypes.size());
		wrapperTypes.push_back("i8 addrspace(1)*");
		slotValue[j] = "%rgfx.b" + std::to_string(index);
		added.push_back("!{i32 " + std::to_string(position) +
		                ", !\"air.buffer\", !\"air.location_index\", i32 " + std::to_string(index) +
		                ", i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, "
		                "!\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", "
		                "!\"air.arg_name\", !\"rgfx_stage_in_buffer_" +
		                std::to_string(index) + "\"}");
	}
	std::string originValue, originType = "<2 x i32>", positionValue, positionType = "<2 x i32>";
	if (origin >= 0) {
		originValue = "%rgfx.a" + std::to_string(origin);
		originType = params[static_cast<unsigned>(origin)];
	} else {
		added.push_back("!{i32 " + std::to_string(wrapperTypes.size()) +
		                ", !\"air.stage_in_grid_origin\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", "
		                "!\"rgfx_grid_origin\"}");
		wrapperTypes.push_back(originType);
		originValue = "%rgfx.origin";
	}
	if (position >= 0) {
		positionValue = "%rgfx.a" + std::to_string(position);
		positionType = params[static_cast<unsigned>(position)];
	} else if (byY) {
		added.push_back("!{i32 " + std::to_string(wrapperTypes.size()) +
		                ", !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", "
		                "!\"rgfx_thread_position\"}");
		wrapperTypes.push_back(positionType);
		positionValue = "%rgfx.position";
	}

	// The record the fetch reads: the buffers, the origin's x and the thread position's y.
	const std::string record = "{ [" + std::to_string(slots.size()) + " x i8 addrspace(1)*], i32, i32 }";
	Emitter w;
	w.body = "define void @" + name + "(";
	{
		std::vector<std::string> names;
		for (unsigned k = 0; k < params.size(); k++)
			if (k != p)
				names.push_back("%rgfx.a" + std::to_string(k));
		for (size_t j = 0; j < slots.size(); j++)
			if (slotValue[j].compare(0, 7, "%rgfx.b") == 0)
				names.push_back(slotValue[j]);
		if (origin < 0)
			names.push_back(originValue);
		if (position < 0 && byY)
			names.push_back(positionValue);
		for (size_t i = 0; i < names.size(); i++)
			w.body += (i != 0 ? ", " : "") + wrapperTypes[i] + " " + names[i];
	}
	w.body += ") {\nentry:\n";
	const std::string r = w.emit("alloca " + record + ", align 8");
	for (size_t j = 0; j < slots.size(); j++) {
		const std::string slot = w.emit("getelementptr inbounds " + record + ", " + record + "* " + r +
		                                ", i32 0, i32 0, i32 " + std::to_string(j));
		std::string v = slotValue[j];
		const std::string type = slotValue[j].compare(0, 7, "%rgfx.a") == 0
		                             ? params[std::strtoul(slotValue[j].c_str() + 7, nullptr, 10)]
		                             : std::string("i8 addrspace(1)*");
		if (type != "i8 addrspace(1)*")
			v = w.emit("bitcast " + type + " " + v + " to i8 addrspace(1)*");
		w.line("store i8 addrspace(1)* " + v + ", i8 addrspace(1)** " + slot + ", align 8");
	}
	const std::string ox = laneOf(w, originType, originValue, 0);
	const std::string originField =
	    w.emit("getelementptr inbounds " + record + ", " + record + "* " + r + ", i32 0, i32 1");
	w.line("store i32 " + ox + ", i32* " + originField + ", align 4");
	const std::string ty = positionValue.empty() ? std::string("0") : laneOf(w, positionType, positionValue, 1);
	const std::string positionField =
	    w.emit("getelementptr inbounds " + record + ", " + record + "* " + r + ", i32 0, i32 2");
	w.line("store i32 " + ty + ", i32* " + positionField + ", align 4");
	const std::string handle = w.emit("bitcast " + record + "* " + r + " to " + params[p]);
	{
		std::string call = "call void @" + inner + "(";
		for (unsigned k = 0; k < params.size(); k++)
			call += (k != 0 ? ", " : "") + params[k] + " " + (k == p ? handle : "%rgfx.a" + std::to_string(k));
		w.line(call + ")");
	}
	w.line("ret void");
	w.body += "}\n";

	// The fetch.
	Emitter f;
	f.body = "define internal " + resultType + " @" + fetchName + "(i32 %rgfx.i, i8* %rgfx.h) alwaysinline {\nentry:\n";
	const std::string fr = f.emit("bitcast i8* %rgfx.h to " + record + "*");
	auto loadSlot = [&](unsigned index) {
		const unsigned j = static_cast<unsigned>(std::lower_bound(slots.begin(), slots.end(), index) - slots.begin());
		const std::string slot = f.emit("getelementptr inbounds " + record + ", " + record + "* " + fr +
		                                ", i32 0, i32 0, i32 " + std::to_string(j));
		return f.emit("load i8 addrspace(1)*, i8 addrspace(1)** " + slot + ", align 8");
	};
	const std::string originAt = f.emit("getelementptr inbounds " + record + ", " + record + "* " + fr + ", i32 0, i32 1");
	const std::string fox = f.emit("load i32, i32* " + originAt + ", align 4");
	const std::string positionAt =
	    f.emit("getelementptr inbounds " + record + ", " + record + "* " + fr + ", i32 0, i32 2");
	const std::string fty = f.emit("load i32, i32* " + positionAt + ", align 4");
	const std::string byPosition = f.emit("add i32 %rgfx.i, " + fox);
	std::string byIndex;
	if (indexed) {
		const std::string ib = loadSlot(in.indexBufferIndex);
		const std::string it = in.indexType == 1 ? "i32" : "i16";
		const std::string typed = f.emit("bitcast i8 addrspace(1)* " + ib + " to " + it + " addrspace(1)*");
		const std::string i64 = f.emit("zext i32 %rgfx.i to i64");
		const std::string at =
		    f.emit("getelementptr inbounds " + it + ", " + it + " addrspace(1)* " + typed + ", i64 " + i64);
		std::string index =
		    f.emit("load " + it + ", " + it + " addrspace(1)* " + at + (it == "i32" ? ", align 4" : ", align 2"));
		if (it != "i32")
			index = f.emit("zext i16 " + index + " to i32");
		byIndex = f.emit("add i32 " + index + ", " + fox);
	}
	std::string result = "undef";
	for (size_t k = 0; k < fields.size(); k++) {
		const Field &field = fields[k];
		std::string element;
		switch (field.layout.stepFunction) {
		case kStepGridX:
			element = byPosition;
			break;
		case kStepGridXIndexed:
			element = byIndex;
			break;
		case kStepGridY:
			element = field.layout.stepRate == 0
			              ? std::string("0")
			              : f.emit("udiv i32 " + fty + ", " + std::to_string(field.layout.stepRate));
			break;
		default:
			element = "0";
			break;
		}
		const std::string scaled = f.emit("mul i32 " + element + ", " + std::to_string(field.layout.stride));
		const std::string offset = f.emit("add i32 " + scaled + ", " + std::to_string(field.attribute.offset));
		const std::string wide = f.emit("zext i32 " + offset + " to i64");
		const std::string base = loadSlot(field.attribute.bufferIndex);
		const std::string pointer = f.emit("getelementptr inbounds i8, i8 addrspace(1)* " + base + ", i64 " + wide);
		const std::vector<Component> components = readComponents(f, pointer, field.format);
		std::string v;
		if (field.lanes == 1) {
			v = components.empty() ? constantOf(field.element, 0) : convert(f, components[0], field.element);
		} else {
			const std::string vector = "<" + std::to_string(field.lanes) + " x " + field.element + ">";
			v = "undef";
			for (unsigned c = 0; c < field.lanes; c++) {
				const std::string lane =
				    c < components.size() ? convert(f, components[c], field.element) : constantOf(field.element, c == 3 ? 1 : 0);
				v = f.emit("insertelement " + vector + " " + v + ", " + field.element + " " + lane + ", i32 " +
				           std::to_string(c));
			}
		}
		std::string memberType = members[k];
		while (!memberType.empty() && memberType.front() == ' ')
			memberType.erase(0, 1);
		while (!memberType.empty() && memberType.back() == ' ')
			memberType.pop_back();
		if (members.size() == 1 && resultType.front() != '{') {
			result = v;
		} else {
			result = f.emit("insertvalue " + resultType + " " + result + ", " + memberType + " " + v + ", " +
			                std::to_string(k));
		}
	}
	f.line("ret " + resultType + " " + result);
	f.body += "}";

	// The text: the kernel renamed and internal, the fetch in place of its declaration, the
	// kernel's entry node naming the wrapper and its inputs, the wrapper after the module.
	bool renamed = false, defined = false;
	const std::string defineAs = "define void @" + name + "(";
	for (std::string &line : lines) {
		if (!renamed && line.compare(0, defineAs.size(), defineAs) == 0) {
			line = "define internal void @" + inner + "(" + line.substr(defineAs.size());
			renamed = true;
		} else if (!defined && line.compare(0, 8, "declare ") == 0 &&
		           line.find(" @" + fetchName + "(") != std::string::npos) {
			line = f.body;
			defined = true;
		}
	}
	if (!renamed || !defined) {
		why = !renamed ? "the module defines no kernel @" + name + " to wrap"
		               : "the module declares no control point function @" + fetchName;
		return false;
	}
	std::vector<std::string> inputRefs;
	for (const Input &input : args) {
		if (input.arg == p)
			continue;
		NodeText t = input.text;
		t.ops[0] = "i32 " + std::to_string(wrapperOf[input.arg]);
		lines[nodeLine[input.node]] = joinNode(t);
		inputRefs.push_back("!" + std::to_string(input.node));
	}
	std::string appended;
	for (const std::string &a : added) {
		appended += "\n!" + std::to_string(nextNode) + " = " + a;
		inputRefs.push_back("!" + std::to_string(nextNode++));
	}
	std::string list = "!" + std::to_string(nextNode) + " = !{";
	for (size_t i = 0; i < inputRefs.size(); i++)
		list += (i != 0 ? ", " : "") + inputRefs[i];
	appended += "\n" + list + "}";
	entry.ops[2] = "!" + std::to_string(nextNode++);
	std::string fnType = "void (";
	for (size_t i = 0; i < wrapperTypes.size(); i++)
		fnType += (i != 0 ? ", " : "") + wrapperTypes[i];
	entry.ops[0] = fnType + ")* @" + name;
	lines[nodeLine[entryNode]] = joinNode(entry);
	std::string joined;
	for (size_t i = 0; i < lines.size(); i++)
		joined += (i != 0 ? "\n" : "") + lines[i];
	while (!joined.empty() && joined.back() == '\n')
		joined.pop_back();
	text = joined + "\n\n" + w.body + appended + "\n";
	return true;
}

bool RgfxGLSpecializeStageIn(const std::string &stageInText, const RgfxGLStageIn &in, RgfxGLLowered &out,
                             std::string &why)
{
	std::string text = stageInText;
	if (!RgfxGLSpecializeStageInText(text, in, why))
		return false;
	Libraries &l = libraries();
	if (!l.loaded) {
		why = l.why;
		return false;
	}
	Ref context = l.contextCreate();
	Ref module = nullptr;
	char *message = nullptr;
	Ref buffer = l.bufferCopy(text.data(), text.size(), "rgfx-gl-stage-in");
	bool ok = l.parseIR(context, buffer, &module, &message) == 0 && module != nullptr;
	if (!ok)
		why = std::string("the system's LLVM refused the kernel's specialized text: ") +
		      (message != nullptr ? message : "no message");
	if (message != nullptr)
		l.disposeMessage(message);
	if (ok) {
		RgfxMetallibFunction f;
		std::vector<Ref> named = firstNode(l, module, "air.kernel");
		size_t length = 0;
		const char *name = !named.empty() && named[0] != nullptr ? l.valueName(named[0], &length) : nullptr;
		f.name = name != nullptr ? std::string(name, length) : std::string();
		f.stage = RgfxMetallibStage::Kernel;
		std::vector<Ref> v = firstNode(l, module, "air.version");
		if (v.size() >= 2 && v[0] != nullptr && v[1] != nullptr && l.isConstantInt(v[0]) != nullptr &&
		    l.isConstantInt(v[1]) != nullptr) {
			f.airMajor = f.languageMajor = static_cast<uint16_t>(l.constIntValue(v[0]));
			f.airMinor = f.languageMinor = static_cast<uint16_t>(l.constIntValue(v[1]));
		}
		Ref bitcode = l.writeBitcode(module);
		length = l.bufferSize(bitcode);
		f.bitcode = RgfxUnwrapBitcode(reinterpret_cast<const uint8_t *>(l.bufferStart(bitcode)), &length);
		f.bitcodeLength = length;
		out.metallib = RgfxWriteMetallib(f);
		out.entry = f.name;
		out.stage = f.stage;
		l.disposeBuffer(bitcode);
	}
	if (module != nullptr)
		l.disposeModule(module);
	l.contextDispose(context);
	return ok;
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
		const bool sorted = RgfxGLSortDeclarations(text);
		const bool normalized = RgfxGLNormalizeStageText(text, signatureOf(l, module, f.name)) || sorted;
		// A kernel of a patch control point input is made per stage input descriptor, from its text.
		if (f.stage == RgfxMetallibStage::Kernel && text.find("!\"air.patch_control_point_input\"") != std::string::npos)
			out.stageInText = text;
		if (normalized) {
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
