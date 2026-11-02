// reflection_check: the reflection the bundle builds for a library's stitchable functions
// (src/RgfxReflection.*, from the signature src/RgfxAirArguments.* reads) against the reflection
// Metal's own compiler gives the same functions on the device it runs on, getter by getter.
//   reflection_check LIB.metallib [-v]
// For each visible function of LIB (a library of our own, tests/air_arguments.metal), on the
// default device: its -returnType and -arguments (and that -bindings is the same object), and
// ours. Both are walked alike: every getter of no argument each object's class and superclasses
// list (up to NSObject), its value, and the objects and arrays it gives, recursively. A getter
// is first called in a forked child, as some end the process where they do not apply (a buffer's
// alignment on a texture's binding): such a getter is "asserts" on both, never called in the
// tool's own process. Each object's class is compared too, and -isEqual: both ways.
// Where ours is refused (a signature that holds an acceleration structure), Metal's own is
// probed in a process of its own (the tool, run as `reflection_check LIB --probe FUNCTION`), and
// agrees where it ends the process or answers nil.
// Prints one line per difference (with -v, every line of both walks), and
//   reflection_check: F functions, G values, D differ
// Exits 1 where any differs. It runs on a Mac and in a guest (stock: the device's own Metal):
// Metal's own answers are the contract the bundle's must equal.
// Build: make reflection-check (build/reflection_check-x86_64); ARCH=arm64 for an Apple-silicon
// Mac. The library: make build/air_arguments.metallib.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

extern char **environ;

#include "RgfxAirArguments.hpp"
#import "RgfxReflection.h"

namespace {

// The text of `o`'s getter `s` of return type `rt`, and the object it gives.
std::string text(id o, SEL s, const char *rt, id __strong *obj)
{
	*obj = nil;
	char buf[64];
	switch (rt[0]) {
	case '@': {
		id r = ((id(*)(id, SEL))objc_msgSend)(o, s);
		*obj = r;
		if (r == nil)
			return "nil";
		if ([r isKindOfClass:NSString.class])
			return std::string("\"") + [(NSString *)r UTF8String] + "\"";
		if ([r isKindOfClass:NSNumber.class])
			return [[(NSNumber *)r stringValue] UTF8String];
		if ([r isKindOfClass:NSArray.class]) {
			snprintf(buf, sizeof buf, "array of %lu", (unsigned long)[(NSArray *)r count]);
			return buf;
		}
		return std::string("<") + class_getName([r class]) + ">";
	}
	case 'B':
	case 'c':
	case 'C':
		snprintf(buf, sizeof buf, "%d", (int)((unsigned char (*)(id, SEL))objc_msgSend)(o, s));
		return buf;
	case 'f':
		snprintf(buf, sizeof buf, "%g", (double)((float (*)(id, SEL))objc_msgSend)(o, s));
		return buf;
	case 'd':
		snprintf(buf, sizeof buf, "%g", ((double (*)(id, SEL))objc_msgSend)(o, s));
		return buf;
	default:
		snprintf(buf, sizeof buf, "%llu", ((unsigned long long (*)(id, SEL))objc_msgSend)(o, s));
		return buf;
	}
}

// The getter's text, from a forked child; "asserts" where the child does not exit cleanly.
std::string safely(id o, SEL s, const char *rt)
{
	fflush(stdout);
	fflush(stderr);
	int fd[2];
	if (pipe(fd) != 0)
		return "(no pipe)";
	const pid_t p = fork();
	if (p == 0) {
		close(fd[0]);
		// The child's asserts are expected: its stderr goes nowhere.
		const int null = open("/dev/null", O_WRONLY);
		if (null >= 0)
			dup2(null, 2);
		id obj;
		const std::string t = text(o, s, rt, &obj);
		ssize_t done = write(fd[1], t.data(), t.size());
		(void)done;
		_exit(0);
	}
	close(fd[1]);
	std::string out;
	char buf[4096];
	ssize_t n;
	while ((n = read(fd[0], buf, sizeof buf)) > 0)
		out.append(buf, static_cast<size_t>(n));
	close(fd[0]);
	int status = 0;
	waitpid(p, &status, 0);
	return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? out : "asserts";
}

// Metal's own reflection of `function`, asked in a new process of this tool: "nil", "made", or
// "ends the process".
std::string probe(const char *tool, const char *lib, const char *function)
{
	fflush(stdout);
	int fd[2];
	if (pipe(fd) != 0)
		return "(no pipe)";
	posix_spawn_file_actions_t actions;
	posix_spawn_file_actions_init(&actions);
	posix_spawn_file_actions_adddup2(&actions, fd[1], 1);
	posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", O_WRONLY, 0);
	posix_spawn_file_actions_addclose(&actions, fd[0]);
	char *const args[] = {const_cast<char *>(tool), const_cast<char *>(lib), const_cast<char *>("--probe"),
	                      const_cast<char *>(function), nullptr};
	pid_t p = 0;
	const int spawned = posix_spawnp(&p, tool, &actions, nullptr, args, environ);
	posix_spawn_file_actions_destroy(&actions);
	close(fd[1]);
	std::string out;
	char buf[256];
	ssize_t n;
	while ((n = read(fd[0], buf, sizeof buf)) > 0)
		out.append(buf, static_cast<size_t>(n));
	close(fd[0]);
	if (spawned != 0)
		return "(not spawned)";
	int status = 0;
	waitpid(p, &status, 0);
	return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? out : "ends the process";
}

bool skipped(NSString *s)
{
	static NSSet *never = [NSSet setWithArray:@[
		@"dealloc", @"release", @"retain", @"autorelease", @"retainCount", @"description",
		@"debugDescription", @"hash", @"self", @"class", @"zone", @"superclass"
	]];
	return [never containsObject:s] || [s hasPrefix:@"new"] || [s hasPrefix:@"init"] || [s hasPrefix:@"set"] ||
	       [s hasPrefix:@"copy"] || [s hasPrefix:@"mutableCopy"] || [s hasPrefix:@"."] || [s hasPrefix:@"_"];
}

// Every getter's line of `o`, under `path`.
void walk(id o, const std::string &path, int depth, std::vector<std::string> &lines)
{
	if (o == nil)
		return;
	if (depth > 12) {
		lines.push_back(path + " (deeper than the walk goes)");
		return;
	}
	if ([o isKindOfClass:NSArray.class]) {
		NSUInteger i = 0;
		for (id e in (NSArray *)o) {
			const std::string at = path + "[" + std::to_string(i++) + "]";
			lines.push_back(at + " class " + class_getName([e class]));
			walk(e, at, depth + 1, lines);
		}
		return;
	}
	NSMutableArray<NSString *> *names = [NSMutableArray array];
	NSMutableDictionary<NSString *, NSString *> *types = [NSMutableDictionary dictionary];
	for (Class c = [o class]; c != Nil && c != NSObject.class; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *ms = class_copyMethodList(c, &n);
		for (unsigned i = 0; i < n; i++) {
			NSString *s = @(sel_getName(method_getName(ms[i])));
			char rt[64];
			method_getReturnType(ms[i], rt, sizeof rt);
			if (types[s] != nil || method_getNumberOfArguments(ms[i]) != 2 || skipped(s) ||
			    strchr("@QqIiBcCLlSsfd", rt[0]) == nullptr)
				continue;
			types[s] = @(rt);
			[names addObject:s];
		}
		free(ms);
	}
	[names sortUsingSelector:@selector(compare:)];
	for (NSString *s in names) {
		const char *rt = types[s].UTF8String;
		const SEL sel = NSSelectorFromString(s);
		const std::string v = safely(o, sel, rt);
		const std::string at = path + "." + s.UTF8String;
		lines.push_back(at + " = " + v);
		if (rt[0] != '@' || v == "asserts")
			continue;
		id obj;
		text(o, sel, rt, &obj);
		if (obj != nil && ([obj isKindOfClass:NSArray.class] || [NSStringFromClass([obj class]) hasPrefix:@"MTL"]))
			walk(obj, at, depth + 1, lines);
	}
}

std::vector<std::string> walked(id o, const std::string &path)
{
	std::vector<std::string> lines;
	lines.push_back(path + " class " + (o != nil ? class_getName([o class]) : "nil"));
	walk(o, path, 0, lines);
	return lines;
}

} // namespace

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: reflection_check LIB.metallib [-v]\n");
			return 2;
		}
		if (argc == 4 && strcmp(argv[2], "--probe") == 0) {
			id<MTLDevice> d = MTLCreateSystemDefaultDevice();
			id<MTLLibrary> l = [d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:nil];
			id<MTLFunction> f = [l newFunctionWithName:@(argv[3])];
			id r = ((id(*)(id, SEL))objc_msgSend)(f, sel_registerName("returnType"));
			id a = ((id(*)(id, SEL))objc_msgSend)(f, sel_registerName("arguments"));
			printf("%s", r == nil && a == nil ? "nil" : "made");
			return 0;
		}
		const bool verbose = argc > 2 && strcmp(argv[2], "-v") == 0;
		NSData *bytes = [NSData dataWithContentsOfFile:@(argv[1])];
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		NSError *error = nil;
		id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
		if (bytes == nil || lib == nil) {
			fprintf(stderr, "reflection_check: %s: no library (%s)\n", argv[1], error.description.UTF8String);
			return 2;
		}
		printf("reflection_check: %s on %s\n", argv[1], device.name.UTF8String);
		const uint8_t *p = static_cast<const uint8_t *>(bytes.bytes);
		size_t n = bytes.length;
		rgfxMetallibAirSlice(p, n);
		unsigned functions = 0, values = 0, differ = 0;
		for (NSString *name in [lib.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
			id<MTLFunction> f = [lib newFunctionWithName:name];
			if (f.functionType != MTLFunctionTypeVisible)
				continue;
			functions++;
			const std::string fn = name.UTF8String;
			// Ours, from the function's own module.
			const uint8_t *bitcode = nullptr;
			size_t bitcodeSize = 0;
			std::vector<RgfxAirVisible> read;
			const RgfxAirVisible *visible = nullptr;
			if (rgfxMetallibBitcode(p, n, fn, &bitcode, &bitcodeSize) &&
			    rgfxReadAirVisible(bitcode, bitcodeSize, read))
				for (const RgfxAirVisible &v : read)
					if (v.name == fn)
						visible = &v;
			id ourReturn = nil;
			NSArray *ourArguments = nil;
			if (visible == nullptr) {
				printf("%s: its signature does not read\n", fn.c_str());
				differ++;
				continue;
			}
			NSString *why = nil;
			if (visible->signature == RgfxAirVisible::Signature::None) {
				ourArguments = @[];
			} else if (!rgfxStitchableReflection(*visible, &ourReturn, &ourArguments, &why)) {
				// Refused: Metal's own must end the process or answer nil.
				const std::string theirs = probe(argv[0], argv[1], fn.c_str());
				const bool agrees =
				    visible->signature == RgfxAirVisible::Signature::Read && (theirs == "ends the process" || theirs == "nil");
				printf("%s: no reflection built (%s); Metal's own: %s\n", fn.c_str(), why.UTF8String, theirs.c_str());
				values++;
				differ += agrees ? 0 : 1;
				continue;
			}
			id theirReturn = ((id(*)(id, SEL))objc_msgSend)(f, sel_registerName("returnType"));
			id theirArguments = ((id(*)(id, SEL))objc_msgSend)(f, sel_registerName("arguments"));
			id theirBindings = ((id(*)(id, SEL))objc_msgSend)(f, sel_registerName("bindings"));
			if (theirBindings != theirArguments) {
				printf("%s: Metal's bindings are not its arguments' object\n", fn.c_str());
				differ++;
			}
			const std::vector<std::string> theirs[] = {walked(theirReturn, fn + ".returnType"),
			                                           walked(theirArguments, fn + ".arguments")};
			const std::vector<std::string> ours[] = {walked(ourReturn, fn + ".returnType"),
			                                         walked(ourArguments, fn + ".arguments")};
			for (size_t k = 0; k < 2; k++) {
				const std::vector<std::string> &a = theirs[k], &b = ours[k];
				for (size_t i = 0; i < std::max(a.size(), b.size()); i++) {
					const std::string ta = i < a.size() ? a[i] : "(none)", tb = i < b.size() ? b[i] : "(none)";
					values++;
					if (ta != tb) {
						differ++;
						printf("DIFF Metal: %s\n     ours:  %s\n", ta.c_str(), tb.c_str());
					} else if (verbose) {
						printf("     %s\n", ta.c_str());
					}
				}
			}
			// Metal's own equality, both ways.
			const bool equalReturn =
			    theirReturn == nil ? ourReturn == nil : [theirReturn isEqual:ourReturn] && [ourReturn isEqual:theirReturn];
			bool equalArguments = [theirArguments count] == [ourArguments count] &&
			                      (theirArguments == nil) == (ourArguments == nil);
			for (NSUInteger i = 0; equalArguments && i < [ourArguments count]; i++)
				equalArguments = [theirArguments[i] isEqual:ourArguments[i]] && [ourArguments[i] isEqual:theirArguments[i]];
			if (!equalReturn || !equalArguments) {
				differ++;
				printf("%s: isEqual: says the %s differ\n", fn.c_str(), !equalReturn ? "return types" : "arguments");
			}
		}
		printf("reflection_check: %u functions, %u values, %u differ\n", functions, values, differ);
		return differ == 0 && functions > 0 ? 0 : 1;
	}
}
