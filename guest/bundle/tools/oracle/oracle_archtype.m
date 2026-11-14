// oracle_archtype: what a device answers to `targetDeviceArchitecture` (MTLDeviceSPI), the GPU
// compiler target Metal's loader keys a binary archive's slices on, and what Metal's archive and
// dynamic library readers answer about a file for the device:
// `+[_MTLBinaryArchive archiveTypeAtURL:device:error:]` and
// `+[_MTLDynamicLibrary dynamicLibraryTypeAtURL:device:error:]`, which MTLAssetUpgraderD asks of
// every .metallib of an app it is told of, for every device. Lines:
//   arch nil                         the device answers nil
//   arch class NAME super SUPER      the class of its answer
//   arch method SEL TYPES            each instance method of that class, with its type encoding
//   arch value SEL VALUE             each of them that takes no argument and returns an integer
//                                    or a BOOL, called
//   archtype FILE -> N               the archive reader's answer (MTLAssetUpgraderD names them
//                                    0 Invalid, 1 Legacy, 2 Compatible, 3 Rebuild, 4 NoVendorSlice),
//                                    and its error where it sets one
//   dyltype FILE -> N                the dynamic library reader's answer (0 Invalid, 1 AIROnly,
//                                    2 Compatible, 3 Rebuild, 4 NoVendorSlice) and its error
//   ... -> ended by signal S: TEXT   the call ended its process; TEXT is what it wrote to stderr
// The files are those named, then a binary archive and a dynamic library the device serializes
// itself (one compute kernel and one function, compiled from source here). Each reader call runs
// in a child process of its own.
//   oracle_archtype DEVICE [FILE...]
// Build: see README.md.
#import "oracle_common.h"
#include <limits.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

static void printArchitecture(id<MTLDevice> d)
{
	if (![(id)d respondsToSelector:@selector(targetDeviceArchitecture)]) {
		printf("arch absent (the device does not respond)\n");
		return;
	}
	id a = ((id (*)(id, SEL))objc_msgSend)((id)d, @selector(targetDeviceArchitecture));
	if (a == nil) {
		printf("arch nil\n");
		return;
	}
	Class c = object_getClass(a);
	printf("arch class %s super %s\n", class_getName(c), class_getName(class_getSuperclass(c)));
	unsigned n = 0;
	Method *ms = class_copyMethodList(c, &n);
	for (unsigned i = 0; i < n; i++) {
		SEL s = method_getName(ms[i]);
		const char *types = method_getTypeEncoding(ms[i]);
		printf("arch method %s %s\n", sel_getName(s), types ? types : "?");
	}
	for (unsigned i = 0; i < n; i++) {
		SEL s = method_getName(ms[i]);
		if (method_getNumberOfArguments(ms[i]) != 2)
			continue;
		char ret[32] = "";
		method_getReturnType(ms[i], ret, sizeof ret);
		const char *r = ret;
		while (*r == 'r' || *r == 'n' || *r == 'N' || *r == 'o' || *r == 'O' || *r == 'R' ||
		       *r == 'V')
			r++;
		if (strchr("cCsSiIlLqQB", *r) == NULL || r[1] != '\0')
			continue;
		const char *name = sel_getName(s);
		if (strncmp(name, "init", 4) == 0 || strncmp(name, "new", 3) == 0 ||
		    strcmp(name, "dealloc") == 0 || strncmp(name, "copy", 4) == 0 ||
		    strcmp(name, ".cxx_destruct") == 0)
			continue;
		unsigned long long v = ((unsigned long long (*)(id, SEL))objc_msgSend)(a, s);
		switch (*r) {
		case 'c': case 'C': case 'B': v &= 0xff; break;
		case 's': case 'S': v &= 0xffff; break;
		case 'i': case 'I': v &= 0xffffffffull; break;
		default: break;
		}
		printf("arch value %s 0x%llx (%llu)\n", name, v, v);
	}
	free(ms);
	printf("arch description %s\n", [[a description] UTF8String]);
}

// The child: one reader call, its answer and error on stdout.
static int childCall(id<MTLDevice> d, const char *kind, const char *file)
{
	NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:file]];
	NSError *err = nil;
	Class c = objc_getClass(strcmp(kind, "archtype") == 0 ? "_MTLBinaryArchive" : "_MTLDynamicLibrary");
	SEL s = strcmp(kind, "archtype") == 0 ? @selector(archiveTypeAtURL:device:error:)
	                                      : @selector(dynamicLibraryTypeAtURL:device:error:);
	if (c == Nil || ![(id)c respondsToSelector:s]) {
		printf("absent\n");
		return 0;
	}
	unsigned long long v =
	    ((unsigned long long (*)(id, SEL, NSURL *, id, NSError **))objc_msgSend)(c, s, url, d, &err);
	printf("%u", (unsigned)(v & 0xffffffffu));
	if (err)
		printf(" error %s", err.description.UTF8String);
	printf("\n");
	return 0;
}

static void child(const char *device, const char *kind, const char *file)
{
	int out[2], errp[2];
	if (pipe(out) != 0 || pipe(errp) != 0)
		exit(2);
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, out[1], 1);
	posix_spawn_file_actions_adddup2(&fa, errp[1], 2);
	posix_spawn_file_actions_addclose(&fa, out[0]);
	posix_spawn_file_actions_addclose(&fa, errp[0]);
	char exe[PATH_MAX];
	uint32_t size = sizeof exe;
	extern int _NSGetExecutablePath(char *, uint32_t *);
	if (_NSGetExecutablePath(exe, &size) != 0)
		exit(2);
	char *args[] = {exe, (char *)device, (char *)"--child", (char *)kind, (char *)file, NULL};
	pid_t pid = 0;
	if (posix_spawn(&pid, exe, &fa, NULL, args, environ) != 0)
		exit(2);
	posix_spawn_file_actions_destroy(&fa);
	close(out[1]);
	close(errp[1]);
	NSMutableData *o = [NSMutableData data], *r = [NSMutableData data];
	char chunk[4096];
	ssize_t got;
	while ((got = read(out[0], chunk, sizeof chunk)) > 0)
		[o appendBytes:chunk length:(NSUInteger)got];
	while ((got = read(errp[0], chunk, sizeof chunk)) > 0)
		[r appendBytes:chunk length:(NSUInteger)got];
	close(out[0]);
	close(errp[0]);
	int status = 0;
	waitpid(pid, &status, 0);
	NSString *line = [[NSString alloc] initWithData:o encoding:NSUTF8StringEncoding];
	NSString *text = [[NSString alloc] initWithData:r encoding:NSUTF8StringEncoding];
	printf("%s %s -> ", kind, file);
	if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
		printf("%s", line.UTF8String);
	} else if (WIFSIGNALED(status)) {
		NSString *flat = [text stringByReplacingOccurrencesOfString:@"\n" withString:@" | "];
		printf("ended by signal %d: %s\n", WTERMSIG(status), flat.UTF8String);
	} else {
		printf("exited %d\n", WEXITSTATUS(status));
	}
}

static NSString *serializeOwn(id<MTLDevice> d, NSString *dir, BOOL dynamic)
{
	NSError *err = nil;
	MTLCompileOptions *opts = [MTLCompileOptions new];
	NSString *src;
	if (dynamic) {
		opts.libraryType = MTLLibraryTypeDynamic;
		opts.installName = @"@executable_path/liboracle_archtype_dyn.metallib";
		src = @"#include <metal_stdlib>\nusing namespace metal;\n"
		      @"[[visible]] uint oracle_archtype_f(uint x) { return x * 3u + 1u; }\n";
	} else {
		src = @"#include <metal_stdlib>\nusing namespace metal;\n"
		      @"kernel void oracle_archtype_k(device uint *o [[buffer(0)]], "
		      @"uint i [[thread_position_in_grid]]) { o[i] = i * 2u; }\n";
	}
	id<MTLLibrary> lib = [d newLibraryWithSource:src options:opts error:&err];
	if (lib == nil) {
		printf("own %s: library: %s\n", dynamic ? "dylib" : "archive",
		       err ? err.description.UTF8String : "nil");
		return nil;
	}
	NSString *path;
	if (dynamic) {
		path = [dir stringByAppendingPathComponent:@"own-dylib.metallib"];
		id<MTLDynamicLibrary> dl = [d newDynamicLibrary:lib error:&err];
		if (dl == nil || ![dl serializeToURL:[NSURL fileURLWithPath:path] error:&err]) {
			printf("own dylib: %s\n", err ? err.description.UTF8String : "nil");
			return nil;
		}
	} else {
		path = [dir stringByAppendingPathComponent:@"own-archive.metallib"];
		MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
		id<MTLBinaryArchive> ar = [d newBinaryArchiveWithDescriptor:ad error:&err];
		MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
		cd.computeFunction = [lib newFunctionWithName:@"oracle_archtype_k"];
		if (ar == nil || ![ar addComputePipelineFunctionsWithDescriptor:cd error:&err] ||
		    ![ar serializeToURL:[NSURL fileURLWithPath:path] error:&err]) {
			printf("own archive: %s\n", err ? err.description.UTF8String : "nil");
			return nil;
		}
	}
	NSData *bytes = [NSData dataWithContentsOfFile:path];
	const uint8_t *b = bytes.bytes;
	printf("own %s: %s, %lu bytes, head %02x%02x%02x%02x %02x%02x%02x%02x\n",
	       dynamic ? "dylib" : "archive", path.UTF8String, (unsigned long)bytes.length,
	       bytes.length > 0 ? b[0] : 0, bytes.length > 1 ? b[1] : 0, bytes.length > 2 ? b[2] : 0,
	       bytes.length > 3 ? b[3] : 0, bytes.length > 4 ? b[4] : 0, bytes.length > 5 ? b[5] : 0,
	       bytes.length > 6 ? b[6] : 0, bytes.length > 7 ? b[7] : 0);
	return path;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: %s DEVICE [FILE...]\n", argv[0]);
			return 2;
		}
		id<MTLDevice> d = oracle_device(argv[1]);
		if (argc == 5 && strcmp(argv[2], "--child") == 0)
			return childCall(d, argv[3], argv[4]);
		oracle_header("oracle_archtype", d);
		printArchitecture(d);
		NSMutableArray<NSString *> *files = [NSMutableArray array];
		for (int i = 2; i < argc; i++)
			[files addObject:[NSString stringWithUTF8String:argv[i]]];
		char tmpl[] = "/tmp/oracle_archtype.XXXXXX";
		NSString *dir = mkdtemp(tmpl) ? [NSString stringWithUTF8String:tmpl] : @"/tmp";
		NSString *own = serializeOwn(d, dir, NO);
		if (own)
			[files addObject:own];
		NSString *ownDyl = serializeOwn(d, dir, YES);
		if (ownDyl)
			[files addObject:ownDyl];
		fflush(stdout);
		for (NSString *f in files) {
			child(argv[1], "archtype", f.UTF8String);
			fflush(stdout);
			child(argv[1], "dyltype", f.UTF8String);
			fflush(stdout);
		}
	}
	return 0;
}
