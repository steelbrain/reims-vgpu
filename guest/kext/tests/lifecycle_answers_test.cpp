// The native test of RgfxLifecycleAnswers.hpp (lifecycle_answers_test, and lifecycle_answers_arm64_test
// with RGFX_GUEST_ARM64=1): every answer type a kext thread waits for in postLifecycle is one the
// event handler hands to its waiter, POOL_TAKE's POOL_BLOCK among them (a boot whose handler
// dropped it left WindowServer's first take waiting for good), and the events the handler takes
// itself are not.
//
// The types a thread waits for are read from the kext's sources (RGFX_KEXT_SRC, the Makefile's):
// the reply type of every postLifecycle call in src/*.cpp, named as a constant or through a
// `replyType` variable set from constants in the same file. They must be the types kWaited lists,
// no more and no fewer, so a call that waits for a new type fails here until it is listed both
// here and in rgfxLifecycleAnswer. postLifecycle refuses at once a type the list lacks, before it
// posts anything: PRIVATE_TAKE's PRIVATE_BLOCK was missing, and every Private take failed in the
// kext with only an IOLog, the host never seeing one.
#include "RgfxLifecycleAnswers.hpp"

#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef RGFX_KEXT_SRC
#error "RGFX_KEXT_SRC, the kext's src directory, comes from the Makefile"
#endif

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "lifecycle_answers_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// The answer types the kext's postLifecycle callers wait for.
struct Waited {
	const char *name;
	uint32_t type;
};
const Waited kWaited[] = {
	{"RGFX_EVT_COMPLETE", RGFX_EVT_COMPLETE},
	{"RGFX_EVT_PIPELINE_REFLECTION", RGFX_EVT_PIPELINE_REFLECTION},
	{"RGFX_EVT_SIZE_AND_ALIGN", RGFX_EVT_SIZE_AND_ALIGN},
	{"RGFX_EVT_SAMPLER_STATE", RGFX_EVT_SAMPLER_STATE},
	{"RGFX_EVT_ICB_STATE", RGFX_EVT_ICB_STATE},
	{"RGFX_EVT_SHARED_EVENT_VALUE", RGFX_EVT_SHARED_EVENT_VALUE},
	{"RGFX_EVT_RATE_MAP_STATE", RGFX_EVT_RATE_MAP_STATE},
	{"RGFX_EVT_RATE_MAP_ANSWER", RGFX_EVT_RATE_MAP_ANSWER},
	{"RGFX_EVT_FUNCTION_POINTER_STATE", RGFX_EVT_FUNCTION_POINTER_STATE},
	{"RGFX_EVT_POOL_BLOCK", RGFX_EVT_POOL_BLOCK},
	{"RGFX_EVT_PRIVATE_BLOCK", RGFX_EVT_PRIVATE_BLOCK},
	// VT_MESSAGE's answer: the arm64e kext's alone.
	{"RGFX_EVT_VT_STATUS", RGFX_EVT_VT_STATUS},
};

std::string trimmed(const std::string &s)
{
	const size_t first = s.find_first_not_of(" \t\r\n");
	if (first == std::string::npos)
		return "";
	return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

// The text of argument `index` (from 0) of the call whose '(' is at `open`, trimmed; empty where
// the call has fewer arguments.
std::string argument(const std::string &text, size_t open, unsigned index)
{
	int depth = 0;
	unsigned arg = 0;
	size_t start = open + 1;
	for (size_t i = open; i < text.size(); i++) {
		const char c = text[i];
		if (c == '(' || c == '[' || c == '{') {
			depth++;
		} else if (c == ')' || c == ']' || c == '}') {
			if (--depth == 0)
				return arg == index ? trimmed(text.substr(start, i - start)) : "";
		} else if (c == ',' && depth == 1) {
			if (arg == index)
				return trimmed(text.substr(start, i - start));
			arg++;
			start = i + 1;
		}
	}
	return "";
}

bool endsWith(const std::string &s, const char *tail)
{
	const size_t n = std::strlen(tail);
	return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
}

// The reply types the postLifecycle calls in the kext's sources wait for, by name; the calls
// read in `*calls`.
std::set<std::string> waitedInSources(int *calls)
{
	*calls = 0;
	std::set<std::string> names;
	DIR *dir = opendir(RGFX_KEXT_SRC);
	check(dir != nullptr, "the kext's src directory opens");
	const std::regex constant("^RGFX_EVT_[A-Z0-9_]+$");
	const std::regex assigned("\\breplyType\\s*=\\s*(RGFX_EVT_[A-Z0-9_]+)");
	while (const dirent *entry = readdir(dir)) {
		const std::string file = entry->d_name;
		if (!endsWith(file, ".cpp"))
			continue;
		std::ifstream in(std::string(RGFX_KEXT_SRC) + "/" + file);
		std::stringstream bytes;
		bytes << in.rdbuf();
		const std::string text = bytes.str();
		static const char kCall[] = "postLifecycle(";
		for (size_t at = text.find(kCall); at != std::string::npos;
		     at = text.find(kCall, at + 1)) {
			const size_t lineStart = text.rfind('\n', at) == std::string::npos
			                             ? 0
			                             : text.rfind('\n', at) + 1;
			const std::string before = trimmed(text.substr(lineStart, at - lineStart));
			// Comments, and the declarations and definitions (`IOReturn postLifecycle(`,
			// `IOReturn ReimsGraphics::postLifecycle(`).
			if (before.find("//") != std::string::npos || before.rfind("*", 0) == 0 ||
			    endsWith(before, "IOReturn") || endsWith(before, "::"))
				continue;
			(*calls)++;
			const std::string reply = argument(text, at + std::strlen(kCall) - 1, 3);
			if (reply == "0")
				continue;
			if (std::regex_match(reply, constant)) {
				names.insert(reply);
				continue;
			}
			if (reply == "replyType") {
				// Set from constants in the same file, or a pass-through's parameter, whose
				// callers are calls of their own here.
				for (std::sregex_iterator it(text.begin(), text.end(), assigned), end; it != end;
				     ++it)
					names.insert((*it)[1].str());
				continue;
			}
			std::fprintf(stderr, "lifecycle_answers_test: %s: a postLifecycle reply type of `%s`\n",
			             file.c_str(), reply.c_str());
			check(false, "every postLifecycle reply type is 0, a constant or `replyType`");
		}
	}
	closedir(dir);
	return names;
}
} // namespace

int main()
{
	// The answer types the kext's postLifecycle callers name.
	for (const Waited &waited : kWaited) {
		if (waited.type == RGFX_EVT_VT_STATUS)
			check(rgfxLifecycleAnswer(waited.type) == (RGFX_GUEST_ARM64 != 0), waited.name);
		else
			check(rgfxLifecycleAnswer(waited.type), waited.name);
	}

	// They are the types the calls in the kext's sources wait for.
	int calls = 0;
	const std::set<std::string> found = waitedInSources(&calls);
	check(calls >= 10, "the sources' postLifecycle calls are read");
	std::set<std::string> listed;
	for (const Waited &waited : kWaited)
		listed.insert(waited.name);
	for (const std::string &name : found) {
		if (!listed.count(name))
			std::fprintf(stderr,
			             "lifecycle_answers_test: a postLifecycle call waits for %s: list it in "
			             "kWaited and in rgfxLifecycleAnswer\n",
			             name.c_str());
		check(listed.count(name) != 0, "every type a call waits for is listed");
	}
	for (const std::string &name : listed) {
		if (!found.count(name))
			std::fprintf(stderr, "lifecycle_answers_test: no postLifecycle call waits for %s\n",
			             name.c_str());
		check(found.count(name) != 0, "every listed type is one a call waits for");
	}

	// The events the handler takes itself, which no thread waits for.
	check(!rgfxLifecycleAnswer(RGFX_MSG_PAD), "padding");
	check(!rgfxLifecycleAnswer(RGFX_EVT_DISPLAYS), "DISPLAYS");
	check(!rgfxLifecycleAnswer(RGFX_EVT_DISPLAY_CHANGED), "DISPLAY_CHANGED");
	check(!rgfxLifecycleAnswer(RGFX_EVT_VBLANK), "VBLANK");
	check(!rgfxLifecycleAnswer(RGFX_EVT_POINTER_OVER), "POINTER_OVER");
	check(!rgfxLifecycleAnswer(RGFX_EVT_POINTER_MOTION), "POINTER_MOTION");
	check(!rgfxLifecycleAnswer(RGFX_EVT_SUBMIT_ROOM), "SUBMIT_ROOM");
	check(!rgfxLifecycleAnswer(RGFX_EVT_VT_MESSAGE), "VT_MESSAGE");
	check(!rgfxLifecycleAnswer(RGFX_EVT_VT_RING), "VT_RING");

	std::printf("lifecycle_answers_test: OK (%d checks, %d postLifecycle calls read)\n", checks,
	            calls);
	return 0;
}
