// The Metal protocol members the bundle's classes answer, against the register of how each is
// answered (tests/protocol_members.txt), so a member that does nothing, or answers a constant,
// cannot pass for one that is carried out. A member of a protocol a class of ours adopts that
// the register does not name fails here until someone classifies it, and so does a new member a
// macOS adds to a protocol.
//   protocol_members_tests BUNDLE REGISTER RELEASES SRCDIR [--propose]
//     loads BUNDLE, the device's x86-64 bundle (run under Rosetta, as the guest runs it), walks
//     through the Objective-C runtime every MTL* protocol each class whose name starts with Rgfx
//     adopts (with the protocols those adopt), and each one's SPI counterpart, MTLFooSPI for
//     MTLFoo, which Metal's own objects adopt and the guest's frameworks send without asking
//     respondsToSelector: (macOS 13's Maps sends MTLLibrarySPI's
//     newFunctionWithName:constantValues:pipelineLibrary:error: to every library), every instance
//     member, required and optional, adds the members RELEASES lists (those the guest's macOS
//     releases carry in a protocol the Mac's Metal does not), and reads the members' bodies from
//     SRCDIR's .mm files. --propose prints a register line for each member the register lacks,
//     with the kind the code suggests, and fails as without it.
//
// The register's kinds, one line "selector kind text" per member under its class's [Class]
// heading (a member a subclass inherits from a class of ours is listed under the class that
// answers it):
//   implemented  the member carries out what Metal's does. Its body must not be empty or only
//                answer a constant (read from its source, and from its machine code), and must
//                not be marked RGFX_UNSUPPORTED.
//   partial      carried out, with a case marked RGFX_UNSUPPORTED in its body; the text names it.
//   unsupported  marked RGFX_UNSUPPORTED in its body (src/RgfxUnsupported.h); the text names the
//                feature.
//   metal        answers what Metal's own answers for this device, with nothing to carry out (a
//                device without the feature, a value the object never changes); the text says
//                why.
//   gap          answers without carrying out what Metal's does, and is not marked
//                RGFX_UNSUPPORTED: a known gap, which the text names. The run prints each.
//   base         answered by a base class that is not ours (the device's IOGPU base); the text
//                says why.
//   absent       no class answers it, so a call raises an unrecognized selector (or goes where
//                the class forwards it); the text says why.
//   pending      no class answers it yet, and what it should answer is not yet decided: an SPI
//                member whose answer is still to be taken from Metal's own devices (the text
//                gives their verdict). Checked as absent.
// A line "%name text" names a reason; an entry's text "%name" gives it.
//
// RELEASES (tests/protocol_members_releases.txt): one line per member that some guest macOS from
// 11 to 26 carries in an MTL* protocol and the Mac's Metal does not, "Protocol selector
// required|optional releases"; its header says how the list is made. Each class of ours that
// adopts the protocol (or its public counterpart) has the member, which the register classifies
// as any other.
//
// It fails on: a member the register does not list; a register line no class has; a kind the
// code contradicts (above); a kind other than implemented with no text; a RELEASES line whose
// member the Mac's protocol has (it is walked already), or whose protocol the Mac's Metal lacks.
#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <objc/runtime.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures;

void fail(const std::string &message)
{
	printf("FAIL: %s\n", message.c_str());
	g_failures++;
}

// --- the members' bodies, read from the sources ---------------------------------------------

struct Body {
	std::string file;
	int line = 0;
	std::string text; // the body between its braces, comments and string literals blanked
	std::string raw;  // the same, as written
	bool unsupported = false;
};

// `src` with its comments and string literals' contents blanked to spaces (newlines kept), so
// braces and tokens inside them are not read as code.
std::string blankCommentsAndStrings(const std::string &src)
{
	std::string out = src;
	size_t i = 0;
	const size_t n = src.size();
	while (i < n) {
		if (src.compare(i, 2, "//") == 0) {
			while (i < n && src[i] != '\n')
				out[i++] = ' ';
		} else if (src.compare(i, 2, "/*") == 0) {
			out[i] = out[i + 1] = ' ';
			i += 2;
			while (i < n && src.compare(i, 2, "*/") != 0) {
				if (src[i] != '\n')
					out[i] = ' ';
				i++;
			}
			if (i < n) {
				out[i] = out[i + 1] = ' ';
				i += 2;
			}
		} else if (src[i] == '"' || src[i] == '\'') {
			const char quote = src[i++];
			while (i < n && src[i] != quote) {
				if (src[i] == '\\' && i + 1 < n) {
					out[i] = out[i + 1] = ' ';
					i += 2;
					continue;
				}
				if (src[i] != '\n')
					out[i] = ' ';
				i++;
			}
			i++;
		} else {
			i++;
		}
	}
	return out;
}

bool isIdent(char c)
{
	return isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// The index past the bracket that closes the one at `i` (`open` at `i`), or npos.
size_t skipBalanced(const std::string &s, size_t i, char open, char close)
{
	int depth = 0;
	for (; i < s.size(); i++) {
		if (s[i] == open)
			depth++;
		else if (s[i] == close && --depth == 0)
			return i + 1;
	}
	return std::string::npos;
}

size_t skipSpace(const std::string &s, size_t i)
{
	while (i < s.size() && isspace(static_cast<unsigned char>(s[i])))
		i++;
	return i;
}

// The selector a method declaration `head` names ("- (T)a:(U)x b:(V)y" -> "a:b:"), or "".
std::string selectorOf(const std::string &head)
{
	size_t i = skipSpace(head, 1); // past the - or +
	if (i < head.size() && head[i] == '(')
		i = skipBalanced(head, i, '(', ')');
	std::string sel;
	while (i != std::string::npos && i < head.size()) {
		i = skipSpace(head, i);
		size_t start = i;
		while (i < head.size() && isIdent(head[i]))
			i++;
		if (i == start)
			break;
		const std::string word = head.substr(start, i - start);
		i = skipSpace(head, i);
		if (i < head.size() && head[i] == ':') {
			sel += word + ":";
			i = skipSpace(head, i + 1);
			if (i < head.size() && head[i] == '(')
				i = skipBalanced(head, i, '(', ')');
			if (i == std::string::npos)
				break;
			i = skipSpace(head, i);
			while (i < head.size() && isIdent(head[i])) // the parameter's name
				i++;
		} else {
			if (sel.empty())
				sel = word;
			break; // a unary selector, or attributes after the last parameter
		}
	}
	return sel;
}

// Every method body in the @implementation blocks of `path`, by "Class selector" (instance
// members only; a category's under its class).
void readBodies(const std::string &path, std::map<std::string, Body> &out)
{
	NSString *raw = [NSString stringWithContentsOfFile:@(path.c_str())
	                                          encoding:NSUTF8StringEncoding
	                                             error:nil];
	if (raw == nil) {
		fail("cannot read " + path);
		return;
	}
	const std::string written = raw.UTF8String;
	const std::string src = blankCommentsAndStrings(written);
	size_t at = 0;
	while ((at = src.find("@implementation", at)) != std::string::npos) {
		size_t i = skipSpace(src, at + strlen("@implementation"));
		size_t start = i;
		while (i < src.size() && isIdent(src[i]))
			i++;
		const std::string cls = src.substr(start, i - start);
		const size_t end = src.find("@end", i);
		if (end == std::string::npos)
			break;
		// Skip the ivar block, at the line's end of the @implementation.
		size_t j = skipSpace(src, i);
		if (j < end && src[j] == '(')
			j = skipBalanced(src, j, '(', ')');
		j = skipSpace(src, j);
		if (j < end && src[j] == '{')
			j = skipBalanced(src, j, '{', '}');
		// Methods: a - or + first on its line, its head up to the first { outside parentheses.
		// Anything else in the block with braces (a function, a static's initialiser) is skipped.
		bool lineStart = true;
		while (j < end) {
			const char c = src[j];
			if (c == '\n') {
				lineStart = true;
				j++;
				continue;
			}
			if (isspace(static_cast<unsigned char>(c))) {
				j++;
				continue;
			}
			if (c == '{') {
				j = skipBalanced(src, j, '{', '}');
				if (j == std::string::npos || j > end)
					break;
				lineStart = false;
				continue;
			}
			if ((c == '-' || c == '+') && lineStart) {
				size_t k = j;
				int paren = 0;
				while (k < end && !((src[k] == '{' || src[k] == ';') && paren == 0)) {
					if (src[k] == '(')
						paren++;
					else if (src[k] == ')')
						paren--;
					k++;
				}
				if (k >= end || src[k] == ';') {
					j = k + 1;
					lineStart = false;
					continue;
				}
				const std::string head = src.substr(j, k - j);
				const size_t close = skipBalanced(src, k, '{', '}');
				if (close == std::string::npos || close > end)
					break;
				Body body;
				body.file = path.substr(path.find_last_of('/') + 1);
				body.line = 1 + static_cast<int>(std::count(src.begin(), src.begin() + j, '\n'));
				body.text = src.substr(k + 1, close - k - 2);
				body.raw = written.substr(k + 1, close - k - 2);
				body.unsupported = body.text.find("RGFX_UNSUPPORTED") != std::string::npos;
				const std::string sel = selectorOf(head);
				if (!sel.empty())
					out[cls + " " + (c == '+' ? "+" : "") + sel] = body;
				j = close;
				lineStart = false;
				continue;
			}
			lineStart = false;
			j++;
		}
		at = end;
	}
}

// --- what a body does: nothing, or a constant ------------------------------------------------

std::string trim(const std::string &s)
{
	size_t a = 0, b = s.size();
	while (a < b && isspace(static_cast<unsigned char>(s[a])))
		a++;
	while (b > a && isspace(static_cast<unsigned char>(s[b - 1])))
		b--;
	return s.substr(a, b - a);
}

// Whether `expr` is a constant: literals, nil/NO/YES, enum constants (MTL..., k...), all-capitals
// macros, casts, braces, operators and MTL...Make of constants; no ivar, self, message or other
// call.
bool isConstantExpression(const std::string &expr)
{
	size_t i = 0;
	bool any = false;
	while (i < expr.size()) {
		const char c = expr[i];
		if (isspace(static_cast<unsigned char>(c)) || strchr("(){},|+-*<>@", c) != nullptr) {
			if (c == '@' && i + 1 < expr.size() && (expr[i + 1] == '[' || expr[i + 1] == '"'))
				return false; // an array or string literal: treat as work (it allocates or names)
			i++;
			continue;
		}
		if (isdigit(static_cast<unsigned char>(c)) || c == '.') {
			while (i < expr.size() && (isIdent(expr[i]) || expr[i] == '.'))
				i++;
			any = true;
			continue;
		}
		if (isIdent(c)) {
			size_t start = i;
			while (i < expr.size() && isIdent(expr[i]))
				i++;
			const std::string word = expr.substr(start, i - start);
			static const std::set<std::string> constants = {
			    "nil",        "Nil",        "NULL",     "nullptr",     "NO",
			    "YES",        "true",       "false",    "NSNotFound",  "NSUInteger",
			    "NSInteger",  "BOOL",       "uint64_t", "uint32_t",    "int64_t",
			    "double",     "float",      "id",       "const",       "static_cast",
			    "void",       "char",       "unsigned", "int",         "NSUIntegerMax",
			    "CFTimeInterval"};
			bool macro = word.size() > 1; // an all-capitals name: KERN_SUCCESS, UINT64_MAX
			for (char ch : word)
				macro = macro && (isupper(static_cast<unsigned char>(ch)) ||
				                  isdigit(static_cast<unsigned char>(ch)) || ch == '_');
			const bool enumConstant =
			    macro || (word.size() > 3 && word.compare(0, 3, "MTL") == 0) ||
			    (word.size() > 1 && word[0] == 'k' && isupper(static_cast<unsigned char>(word[1])));
			if (!enumConstant && constants.count(word) == 0)
				return false;
			any = true;
			continue;
		}
		return false;
	}
	return any;
}

// "empty" where `body` does nothing, "constant" where it does nothing but answer a constant, ""
// where it does work. Statements of only (void)x are nothing.
std::string sourceTriviality(const std::string &body)
{
	std::vector<std::string> statements;
	std::string current;
	int depth = 0;
	for (char c : body) {
		if (c == '(' || c == '{' || c == '[')
			depth++;
		else if (c == ')' || c == '}' || c == ']')
			depth--;
		if (c == ';' && depth == 0) {
			statements.push_back(trim(current));
			current.clear();
		} else {
			current += c;
		}
	}
	if (!trim(current).empty())
		return ""; // a trailing statement with no ';': an if/for block, say
	bool returnsConstant = false;
	for (const std::string &s : statements) {
		if (s.empty())
			continue;
		if (s.compare(0, 6, "(void)") == 0) {
			const std::string rest = trim(s.substr(6));
			bool plain = !rest.empty();
			for (char c : rest)
				plain = plain && (isIdent(c) || c == '.');
			if (plain)
				continue;
			return "";
		}
		if (s.compare(0, 6, "return") == 0 && (s.size() == 6 || !isIdent(s[6]))) {
			const std::string expr = trim(s.substr(6));
			if (expr.empty() || isConstantExpression(expr)) {
				returnsConstant = true;
				continue;
			}
		}
		return "";
	}
	return returnsConstant ? "constant" : "empty";
}

// The same, read from the x86-64 machine code at `imp`: "empty" where it only returns, "constant"
// where it only sets a return register to a constant and returns (with or without a frame), ""
// where it does anything else or the code is not one of these forms. Only exact forms count, so
// a body that does work is never called empty.
std::string codeTriviality(IMP imp)
{
#if defined(__x86_64__)
	const uint8_t *p = reinterpret_cast<const uint8_t *>(imp);
	bool frame = false, constant = false;
	if (p[0] == 0x55 && p[1] == 0x48 && p[2] == 0x89 && p[3] == 0xe5) { // push rbp; mov rbp, rsp
		frame = true;
		p += 4;
	}
	for (int step = 0; step < 4; step++) {
		if (p[0] == 0x31 && (p[1] == 0xc0 || p[1] == 0xd2)) { // xor eax/edx, eax/edx
			p += 2;
		} else if (p[0] == 0x0f && p[1] == 0x57 && p[2] == 0xc0) { // xorps xmm0, xmm0
			p += 3;
		} else if (p[0] == 0x66 && p[1] == 0x0f && p[2] == 0x57 && p[3] == 0xc0) { // xorpd
			p += 4;
		} else if (p[0] == 0xb8 || p[0] == 0xba) { // mov eax/edx, imm32
			p += 5;
		} else if (p[0] == 0xb0) { // mov al, imm8
			p += 2;
		} else if (p[0] == 0x48 && p[1] == 0xc7 && p[2] == 0xc0) { // mov rax, simm32
			p += 7;
		} else if (p[0] == 0x48 && p[1] == 0xb8) { // movabs rax, imm64
			p += 10;
		} else if ((p[0] == 0xf2 || p[0] == 0xf3) && p[1] == 0x0f && p[2] == 0x10 &&
		           p[3] == 0x05) { // movsd/movss xmm0, [rip + d32]: a literal
			p += 8;
		} else {
			break;
		}
		constant = true;
	}
	if (frame) {
		if (p[0] != 0x5d) // pop rbp
			return "";
		p++;
	}
	if (p[0] != 0xc3) // ret
		return "";
	return constant ? "constant" : "empty";
#else
	(void)imp;
	return "";
#endif
}

// --- the classes and their protocols ----------------------------------------------------------

struct Member {
	std::string key;     // "Class selector": the class that answers it, or the adopter's
	std::string state;   // "ours", "base" or "absent"
	std::string definer; // the class whose method list holds it (empty when absent)
	std::string protocol;
	bool required = false;
	IMP imp = nullptr;
};

Class definingClass(Class cls, SEL sel)
{
	for (Class c = cls; c != Nil; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *ms = class_copyMethodList(c, &n);
		bool found = false;
		for (unsigned i = 0; i < n && !found; i++)
			found = method_getName(ms[i]) == sel;
		free(ms);
		if (found)
			return c;
	}
	return Nil;
}

bool isOurs(const char *name)
{
	return strncmp(name, "Rgfx", 4) == 0;
}

Member memberOf(Class cls, SEL sel, const std::string &protocol, bool required);

void walkProtocol(Class cls, Protocol *p, std::set<std::string> &seen,
                  std::map<std::string, Member> &out)
{
	const std::string pname = protocol_getName(p);
	if (!seen.insert(pname).second)
		return;
	if (pname.compare(0, 3, "MTL") == 0) {
		for (int required = 1; required >= 0; required--) {
			unsigned n = 0;
			objc_method_description *ds = protocol_copyMethodDescriptionList(p, required, YES, &n);
			for (unsigned i = 0; i < n; i++) {
				const Member m = memberOf(cls, ds[i].name, pname, required);
				auto it = out.find(m.key);
				if (it == out.end() || (m.required && !it->second.required))
					out[m.key] = m;
			}
			free(ds);
		}
	}
	unsigned n = 0;
	Protocol *__unsafe_unretained *adopted = protocol_copyProtocolList(p, &n);
	for (unsigned i = 0; i < n; i++)
		walkProtocol(cls, adopted[i], seen, out);
	free(adopted);
	// Its SPI counterpart, which Metal's own objects of the protocol adopt.
	const std::string spi = pname + "SPI";
	if (pname.compare(0, 3, "MTL") == 0 && pname.size() > 3 &&
	    pname.compare(pname.size() - 3, 3, "SPI") != 0) {
		if (Protocol *counterpart = objc_getProtocol(spi.c_str()))
			walkProtocol(cls, counterpart, seen, out);
	}
}

// The member `sel` of `cls`, from `protocol`, as walkProtocol makes it.
Member memberOf(Class cls, SEL sel, const std::string &protocol, bool required)
{
	const Class def = definingClass(cls, sel);
	Member m;
	m.protocol = protocol;
	m.required = required;
	if (def != Nil && isOurs(class_getName(def))) {
		m.state = "ours";
		m.definer = class_getName(def);
		m.key = m.definer + " " + sel_getName(sel);
		m.imp = method_getImplementation(class_getInstanceMethod(def, sel));
	} else {
		m.state = def != Nil ? "base" : "absent";
		m.definer = def != Nil ? class_getName(def) : "";
		m.key = std::string(class_getName(cls)) + " " + sel_getName(sel);
	}
	return m;
}

// The members RELEASES lists, added to each class of ours whose walked protocols (`walked`, by
// class) hold the line's protocol or its public counterpart.
void addReleaseMembers(const std::string &path,
                       const std::map<std::string, std::set<std::string>> &walked,
                       std::map<std::string, Member> &out)
{
	NSString *raw = [NSString stringWithContentsOfFile:@(path.c_str())
	                                          encoding:NSUTF8StringEncoding
	                                             error:nil];
	if (raw == nil) {
		fail("cannot read the releases' members " + path);
		return;
	}
	int lineNo = 0;
	std::set<std::string> lines;
	for (NSString *ns in [raw componentsSeparatedByString:@"\n"]) {
		lineNo++;
		const std::string line = trim(ns.UTF8String);
		const std::string where = path + ":" + std::to_string(lineNo);
		if (line.empty() || line[0] == '#')
			continue;
		std::istringstream in(line);
		std::string protocol, sel, required, releases;
		in >> protocol >> sel >> required;
		std::getline(in, releases);
		if (sel.empty() || (required != "required" && required != "optional") ||
		    trim(releases).empty()) {
			fail(where + ": not 'Protocol selector required|optional releases'");
			continue;
		}
		if (!lines.insert(protocol + " " + sel).second)
			fail(where + ": " + protocol + " " + sel + " is listed twice");
		Protocol *p = objc_getProtocol(protocol.c_str());
		if (p == nullptr) {
			fail(where + ": the Mac's Metal has no protocol " + protocol);
			continue;
		}
		const SEL s = sel_registerName(sel.c_str());
		const bool onMac = protocol_getMethodDescription(p, s, YES, YES).name != nullptr ||
		                   protocol_getMethodDescription(p, s, NO, YES).name != nullptr;
		if (onMac) {
			fail(where + ": the Mac's " + protocol + " has " + sel + ": drop the line");
			continue;
		}
		const std::string base =
		    protocol.size() > 3 && protocol.compare(protocol.size() - 3, 3, "SPI") == 0 ?
		        protocol.substr(0, protocol.size() - 3) :
		        protocol;
		int adopters = 0;
		for (const auto &[cls, names] : walked) {
			if (names.count(protocol) == 0 && names.count(base) == 0)
				continue;
			adopters++;
			const Member m = memberOf(objc_getClass(cls.c_str()), s, protocol, required == "required");
			auto it = out.find(m.key);
			if (it == out.end() || (m.required && !it->second.required))
				out[m.key] = m;
		}
		if (adopters == 0)
			fail(where + ": no class of ours adopts " + protocol + " or " + base);
	}
}

// --- the register -------------------------------------------------------------------------------

struct Entry {
	std::string kind, text;
	int line = 0;
};

const std::set<std::string> kKinds = {"implemented", "partial", "unsupported", "metal",
                                      "gap",         "base",    "absent",      "pending"};

// The register's members by "Class selector". A line "%name text" names a reason that an entry's
// text can give as "%name" (with more text after it, or none).
std::map<std::string, Entry> readRegister(const std::string &path,
                                          std::map<std::string, std::string> &reasons)
{
	std::map<std::string, Entry> out;
	NSString *raw = [NSString stringWithContentsOfFile:@(path.c_str())
	                                          encoding:NSUTF8StringEncoding
	                                             error:nil];
	if (raw == nil) {
		fail("cannot read the register " + path);
		return out;
	}
	std::string cls;
	int lineNo = 0;
	std::vector<std::pair<std::string, int>> referenced;
	for (NSString *ns in [raw componentsSeparatedByString:@"\n"]) {
		lineNo++;
		const std::string line = trim(ns.UTF8String);
		const std::string where = path + ":" + std::to_string(lineNo);
		if (line.empty() || line[0] == '#')
			continue;
		if (line[0] == '%') {
			const size_t sp = line.find_first_of(" \t");
			if (sp == std::string::npos)
				fail(where + ": a reason with no text");
			else if (!reasons.emplace(line.substr(0, sp), trim(line.substr(sp))).second)
				fail(where + ": reason " + line.substr(0, sp) + " is named twice");
			continue;
		}
		if (line[0] == '[' && line.back() == ']') {
			cls = line.substr(1, line.size() - 2);
			continue;
		}
		const size_t s1 = line.find_first_of(" \t");
		const std::string sel = line.substr(0, s1);
		std::string rest = s1 == std::string::npos ? "" : trim(line.substr(s1));
		const size_t s2 = rest.find_first_of(" \t");
		Entry e;
		e.kind = rest.substr(0, s2);
		e.text = s2 == std::string::npos ? "" : trim(rest.substr(s2));
		e.line = lineNo;
		if (cls.empty()) {
			fail(where + ": a member before any [Class] heading");
			continue;
		}
		if (kKinds.count(e.kind) == 0) {
			fail(where + ": unknown kind '" + e.kind + "'");
			continue;
		}
		if (e.kind != "implemented" && e.text.empty())
			fail(where + ": " + cls + " " + sel + " is '" + e.kind + "' with no text saying why");
		if (!e.text.empty() && e.text[0] == '%')
			referenced.push_back({e.text.substr(0, e.text.find_first_of(" \t")), lineNo});
		if (!out.emplace(cls + " " + sel, e).second)
			fail(where + ": " + cls + " " + sel + " is listed twice");
	}
	for (const auto &[name, line] : referenced) {
		if (reasons.count(name) == 0)
			fail(path + ":" + std::to_string(line) + ": no reason is named " + name);
	}
	return out;
}

// The text of the first RGFX_UNSUPPORTED* in `body`'s source: its string argument, or the
// string a constant it names holds in `file`'s source; else "".
std::string unsupportedText(const Body &body, const std::string &fileSource)
{
	const size_t at = body.raw.find("RGFX_UNSUPPORTED");
	if (at == std::string::npos)
		return "";
	const size_t open = body.raw.find('(', at);
	const size_t close = body.raw.find(");", open);
	if (open == std::string::npos || close == std::string::npos)
		return "";
	// The feature is RGFX_UNSUPPORTED's first argument, the _ERROR forms' second.
	const bool errorForm = body.raw.compare(at, strlen("RGFX_UNSUPPORTED_ERROR"),
	                                        "RGFX_UNSUPPORTED_ERROR") == 0;
	std::vector<std::string> args(1);
	int depth = 0;
	for (size_t i = open + 1; i < close; i++) {
		const char c = body.raw[i];
		depth += c == '(' ? 1 : c == ')' ? -1 : 0;
		if (c == ',' && depth == 0)
			args.emplace_back();
		else
			args.back() += c;
	}
	const std::string arg = trim(args[errorForm && args.size() > 1 ? 1 : 0]);
	if (!arg.empty() && arg[0] == '"')
		return arg.substr(1, arg.find('"', 1) - 1);
	std::string name;
	for (char c : arg) {
		if (!isIdent(c))
			break;
		name += c;
	}
	size_t def = 0;
	while ((def = fileSource.find(name + " = \"", def)) != std::string::npos && def > 0 &&
	       isIdent(fileSource[def - 1]))
		def++;
	if (name.empty() || def == std::string::npos)
		return "";
	const size_t from = def + name.size() + 4;
	return fileSource.substr(from, fileSource.find('"', from) - from);
}

// `text` with each RGFX_UNSUPPORTED* statement taken out.
std::string withoutUnsupported(std::string text)
{
	size_t at;
	while ((at = text.find("RGFX_UNSUPPORTED")) != std::string::npos) {
		const size_t open = text.find('(', at);
		const size_t close = open == std::string::npos ? open : skipBalanced(text, open, '(', ')');
		const size_t semi = close == std::string::npos ? close : text.find(';', close);
		if (semi == std::string::npos)
			break;
		text.erase(at, semi + 1 - at);
	}
	return text;
}

bool isCreatorName(const std::string &sel)
{
	for (const char *prefix : {"new", "compile", "load", "alloc", "functionHandle", "resolve"}) {
		if (sel.compare(0, strlen(prefix), prefix) == 0)
			return true;
	}
	return false;
}

} // namespace

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 5) {
			fprintf(stderr,
			        "usage: protocol_members_tests BUNDLE REGISTER RELEASES SRCDIR [--propose]\n");
			return 2;
		}
		const bool propose = argc > 5 && strcmp(argv[5], "--propose") == 0;
		if (dlopen(argv[1], RTLD_NOW) == nullptr) {
			printf("FAIL: cannot load %s: %s\n", argv[1], dlerror());
			return 1;
		}

		std::map<std::string, Body> bodies;
		std::map<std::string, std::string> sources; // each file's source, by name
		NSArray<NSString *> *files =
		    [[NSFileManager defaultManager] contentsOfDirectoryAtPath:@(argv[4]) error:nil];
		for (NSString *f in [files sortedArrayUsingSelector:@selector(compare:)]) {
			if (![f hasSuffix:@".mm"])
				continue;
			const std::string path = std::string(argv[4]) + "/" + f.UTF8String;
			readBodies(path, bodies);
			NSString *text = [NSString stringWithContentsOfFile:@(path.c_str())
			                                           encoding:NSUTF8StringEncoding
			                                              error:nil];
			sources[f.UTF8String] = text != nil ? text.UTF8String : "";
		}

		std::map<std::string, Member> members;
		std::map<std::string, std::set<std::string>> walkedBy; // the protocols, by class
		unsigned count = 0;
		Class *classes = objc_copyClassList(&count);
		int walked = 0;
		for (unsigned i = 0; i < count; i++) {
			if (!isOurs(class_getName(classes[i])))
				continue;
			std::set<std::string> seen;
			for (Class c = classes[i]; c != Nil; c = class_getSuperclass(c)) {
				unsigned pn = 0;
				Protocol *__unsafe_unretained *ps = class_copyProtocolList(c, &pn);
				for (unsigned j = 0; j < pn; j++)
					walkProtocol(classes[i], ps[j], seen, members);
				free(ps);
			}
			walkedBy[class_getName(classes[i])] = seen;
			walked++;
		}
		free(classes);
		addReleaseMembers(argv[3], walkedBy, members);

		std::map<std::string, std::string> reasons;
		const std::map<std::string, Entry> reg = readRegister(argv[2], reasons);
		std::map<std::string, std::vector<std::string>> proposals; // by class
		std::map<std::string, int> kinds;
		std::vector<std::string> gaps;
		for (const auto &[key, m] : members) {
			const auto bodyIt = bodies.find(key);
			const Body *body = bodyIt == bodies.end() ? nullptr : &bodyIt->second;
			const std::string fromSource = body != nullptr ? sourceTriviality(body->text) : "";
			const std::string fromCode = m.imp != nullptr ? codeTriviality(m.imp) : "";
			const std::string trivial = !fromSource.empty() ? fromSource : fromCode;
			const std::string where =
			    body != nullptr ? " (" + body->file + ":" + std::to_string(body->line) + ")" : "";
			const std::string what = key + where + ", " + (m.required ? "required" : "optional") +
			                         " in " + m.protocol;
			const std::string sel = key.substr(key.find(' ') + 1);
			const auto it = reg.find(key);
			if (it == reg.end()) {
				fail("unclassified: " + what + ": list it in the register");
				std::string line = sel + "  ";
				if (m.state == "base")
					line += std::string("base  ") +
					        (isCreatorName(sel) ? "%base-creator" : "%base-spi");
				else if (m.state == "absent")
					line += "absent  TODO: why";
				else if (body != nullptr && body->unsupported)
					line += std::string(sourceTriviality(withoutUnsupported(body->text)).empty() ?
					                        "partial  " :
					                        "unsupported  ") +
					        unsupportedText(*body, sources[body->file]);
				else if (!trivial.empty())
					line += "metal  TODO (" + trivial + ", from the " +
					        (!fromSource.empty() ? "source" : "machine code") + ")";
				else
					line += "implemented";
				proposals[key.substr(0, key.find(' '))].push_back(line);
				continue;
			}
			const Entry &e = it->second;
			kinds[e.kind]++;
			const std::string as = what + " is listed '" + e.kind + "'";
			if (e.kind == "base" || e.kind == "absent" || e.kind == "pending") {
				if (m.state != (e.kind == "pending" ? "absent" : e.kind))
					fail(as + ", but " +
					     (m.state == "ours" ? "a class of ours answers it" :
					      m.state == "base" ? "the base " + m.definer + " answers it" :
					                          "no class answers it"));
				continue;
			}
			if (m.state != "ours") {
				fail(as + ", but " +
				     (m.state == "base" ? "only the base " + m.definer + " answers it" :
				                          "no class answers it"));
				continue;
			}
			if (e.kind == "unsupported" || e.kind == "partial") {
				if (body == nullptr)
					fail(as + ", but no body of it is in the sources");
				else if (!body->unsupported)
					fail(as + ", but its body has no RGFX_UNSUPPORTED");
				continue;
			}
			if (body != nullptr && body->unsupported)
				fail(as + ", but its body is marked RGFX_UNSUPPORTED: list it 'unsupported' or "
				          "'partial'");
			if (e.kind == "implemented" && !trivial.empty())
				fail(as + ", but its body " +
				     (trivial == "empty" ? "does nothing" : "only answers a constant") +
				     " (read from its " + (!fromSource.empty() ? "source" : "machine code") +
				     "): carry it out, or list it 'metal' with why, or mark it RGFX_UNSUPPORTED");
			if (e.kind == "gap")
				gaps.push_back(key + where + ": " + e.text);
		}
		for (const auto &[key, e] : reg) {
			if (members.count(key) == 0)
				fail("stale: " + key + " (register line " + std::to_string(e.line) +
				     ") is listed, but no protocol member of a class of ours is it");
		}
		for (const std::string &g : gaps)
			printf("gap: %s\n", g.c_str());
		if (propose) {
			for (const auto &[cls, lines] : proposals) {
				printf("[%s]\n", cls.c_str());
				for (const std::string &l : lines)
					printf("%s\n", l.c_str());
			}
		}
		std::string summary;
		for (const auto &[k, n] : kinds)
			summary += " " + k + " " + std::to_string(n) + ",";
		if (!summary.empty())
			summary.pop_back();
		printf("protocol_members_tests: %zu members of %d classes' Metal protocols:%s; %d "
		       "failures\n",
		       members.size(), walked, summary.c_str(), g_failures);
		return g_failures == 0 ? 0 : 1;
	}
}
