// oracle_urls: what Metal answers at each call of a device, a library and a binary archive that
// takes a URL, for each form of URL: nil, a path of no scheme, a file URL, a URL of another scheme
// (`rgfxprobe:`, `http:`) of a path, and a URL of no path (`data:`), where the file at the path is
// there and where it is not; and the device's library of a path (`newLibraryWithFile:error:`), of
// nil, the file's path and a missing one's (`no such form` for the others). Each case runs in a child process of its own, so a call that ends the
// process (a failed assertion) shows how it ended.
//   oracle_urls DEVICE METALLIB [CALL]   every call (or the one named), each form in a child
// METALLIB is a library of one kernel or more (tests/shaders/compute_add.metallib); its first
// kernel is the function an archive is given. A read's forms name its fixture: the library, a
// binary archive the parent serialized from it, or a raw file of 4 KiB; a write's forms name a
// file of its own in the scratch directory (`missing`: in a directory that is not there).
// Columns: the call, the form, then its answer: the object's class or nil, YES or NO; the error
// (domain, code, description and userInfo's keys, an underlying error's domain and code) or
// `error nil`; for a write, `wrote N` (the bytes at the path after it) or `no file`; or `ends the
// process` with the signal and the assertion's words on stderr. `no such member` where the object
// does not answer the call.
// Build: see README.md.
#import "oracle_common.h"
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#import <objc/message.h>

extern char **environ;

static id<MTLDevice> D;
static NSString *Dir;     // the scratch directory, made by the parent
static NSString *LibPath; // the library's copy there

typedef enum { kRead, kWrite } Kind;

// A call's fixture, which its read forms name.
typedef enum { kLibrary, kArchive, kRaw } Fixture;

static NSString *fixturePath(Fixture f)
{
	switch (f) {
	case kLibrary:
		return LibPath;
	case kArchive:
		return [Dir stringByAppendingPathComponent:@"archive.bin"];
	case kRaw:
		return [Dir stringByAppendingPathComponent:@"raw.bin"];
	}
	return nil;
}

static const char *const kForms[] = {"nil",   "path",  "path-missing", "file", "file-missing", "other",
                                     "other-missing", "http", "http-missing", "data"};

// The URL of `form`: `there` is the path where the file is (a read's fixture, a write's target),
// `missing` one where it is not.
static NSURL *urlOf(const char *form, NSString *there, NSString *missing)
{
	if (!strcmp(form, "nil"))
		return nil;
	if (!strcmp(form, "path"))
		return [NSURL URLWithString:there];
	if (!strcmp(form, "path-missing"))
		return [NSURL URLWithString:missing];
	if (!strcmp(form, "file"))
		return [NSURL fileURLWithPath:there];
	if (!strcmp(form, "file-missing"))
		return [NSURL fileURLWithPath:missing];
	if (!strcmp(form, "other"))
		return [NSURL URLWithString:[@"rgfxprobe://" stringByAppendingString:there]];
	if (!strcmp(form, "other-missing"))
		return [NSURL URLWithString:[@"rgfxprobe://" stringByAppendingString:missing]];
	if (!strcmp(form, "http"))
		return [NSURL URLWithString:[@"http://127.0.0.1:9" stringByAppendingString:there]];
	if (!strcmp(form, "http-missing"))
		return [NSURL URLWithString:[@"http://127.0.0.1:9" stringByAppendingString:missing]];
	if (!strcmp(form, "data"))
		return [NSURL URLWithString:@"data:application/octet-stream;base64,AAAA"];
	return nil;
}

static void printError(NSError *e)
{
	if (e == nil) {
		printf("error nil");
		return;
	}
	printf("error %s %ld \"%s\" keys [", e.domain.UTF8String, (long)e.code, e.localizedDescription.UTF8String);
	NSArray *keys = [e.userInfo.allKeys sortedArrayUsingSelector:@selector(compare:)];
	for (NSUInteger i = 0; i < keys.count; i++)
		printf("%s%s", i ? " " : "", [keys[i] UTF8String]);
	printf("]");
	NSError *under = e.userInfo[NSUnderlyingErrorKey];
	if ([under isKindOfClass:[NSError class]])
		printf(" under %s %ld", under.domain.UTF8String, (long)under.code);
}

static void printObject(id o)
{
	printf("%s ", o != nil ? NSStringFromClass([o class]).UTF8String : "nil");
}

static void printWritten(NSString *path)
{
	struct stat st;
	if (path != nil && stat(path.fileSystemRepresentation, &st) == 0)
		printf(" wrote %lld", (long long)st.st_size);
	else
		printf(" no file");
}

static id<MTLLibrary> libraryOfData(void)
{
	NSData *bytes = [NSData dataWithContentsOfFile:LibPath];
	dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
	NSError *e = nil;
	id<MTLLibrary> lib = [D newLibraryWithData:data error:&e];
	if (lib == nil)
		printf("(no library of data: %s) ", e.description.UTF8String);
	return lib;
}

static id<MTLLibrary> libraryOfURL(void)
{
	NSError *e = nil;
	id<MTLLibrary> lib = [D newLibraryWithURL:[NSURL fileURLWithPath:LibPath] error:&e];
	if (lib == nil)
		printf("(no library of a URL: %s) ", e.description.UTF8String);
	return lib;
}

// An archive with the library's first kernel added, or none added where `empty`.
static id<MTLBinaryArchive> archiveOf(BOOL empty)
{
	NSError *e = nil;
	id<MTLBinaryArchive> a = [D newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
	if (a == nil) {
		printf("(no archive: %s) ", e.description.UTF8String);
		return nil;
	}
	if (empty)
		return a;
	id<MTLLibrary> lib = libraryOfData();
	MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
	cd.computeFunction = [lib newFunctionWithName:lib.functionNames.firstObject];
	if (![a addComputePipelineFunctionsWithDescriptor:cd error:&e])
		printf("(no function added: %s) ", e.description.UTF8String);
	return a;
}

typedef struct {
	const char *name;
	Kind kind;
	Fixture fixture;     // a read's
	const char *selector; // the member, for `no such member`
} Call;

static const Call kCalls[] = {
    {"device.newLibraryWithURL", kRead, kLibrary, "newLibraryWithURL:error:"},
    {"device.newLibraryWithFile", kRead, kLibrary, "newLibraryWithFile:error:"},
    {"library.data.serializeToURL", kWrite, kLibrary, "serializeToURL:error:"},
    {"library.url.serializeToURL", kWrite, kLibrary, "serializeToURL:error:"},
    {"device.newBinaryArchive.url", kRead, kArchive, "newBinaryArchiveWithDescriptor:error:"},
    {"device.newBinaryArchive.url.library", kRead, kLibrary, "newBinaryArchiveWithDescriptor:error:"},
    {"archive.serializeToURL", kWrite, kArchive, "serializeToURL:error:"},
    {"archive.empty.serializeToURL", kWrite, kArchive, "serializeToURL:error:"},
    {"archive.serializeToURL.options", kWrite, kArchive, "serializeToURL:options:error:"},
    {"archive.recompileToArchiveWithURL", kWrite, kArchive, "recompileToArchiveWithURL:error:"},
    {"device.newDynamicLibraryWithURL", kRead, kLibrary, "newDynamicLibraryWithURL:error:"},
    {"device.newDynamicLibraryWithURL.options", kRead, kLibrary, "newDynamicLibraryWithURL:options:error:"},
    {"device.newDynamicLibraryFromURL", kRead, kLibrary, "newDynamicLibraryFromURL:error:"},
    {"device.newBinaryLibraryWithOptions.url", kRead, kLibrary, "newBinaryLibraryWithOptions:url:error:"},
    {"device.newArchiveWithURL", kRead, kArchive, "newArchiveWithURL:error:"},
    {"device.newIOFileHandleWithURL", kRead, kRaw, "newIOFileHandleWithURL:error:"},
    {"device.newIOFileHandleWithURL.method", kRead, kRaw, "newIOFileHandleWithURL:compressionMethod:error:"},
    {"device.newIOHandleWithURL", kRead, kRaw, "newIOHandleWithURL:error:"},
    {"device.newIOHandleWithURL.method", kRead, kRaw, "newIOHandleWithURL:compressionMethod:error:"},
    {"device.newIOHandleWithURL.type", kRead, kRaw, "newIOHandleWithURL:compressionType:error:"},
    {"device.newUncachedIOHandleWithURL", kRead, kRaw, "newUncachedIOHandleWithURL:error:"},
    {"device.newUncachedIOFileHandleWithURL", kRead, kRaw, "newUncachedIOFileHandleWithURL:error:"},
    {"device.newUncachedIOHandleWithURL.type", kRead, kRaw, "newUncachedIOHandleWithURL:compressionType:error:"},
    {"device.newUncachedIOFileHandleWithURL.method", kRead, kRaw,
     "newUncachedIOFileHandleWithURL:compressionMethod:error:"},
};

typedef id (*UrlErr)(id, SEL, NSURL *, NSError **);
typedef id (*UrlIntErr)(id, SEL, NSURL *, NSInteger, NSError **);
typedef id (*IntUrlErr)(id, SEL, NSInteger, NSURL *, NSError **);
typedef BOOL (*BoolUrlErr)(id, SEL, NSURL *, NSError **);
typedef BOOL (*BoolUrlIntErr)(id, SEL, NSURL *, NSUInteger, NSError **);

// One case: `call` at the URL of `form`, its answer on one line.
static int runCase(const Call *call, const char *form)
{
	NSString *there = nil, *missing = nil;
	if (call->kind == kRead) {
		there = fixturePath(call->fixture);
		missing = [Dir stringByAppendingPathComponent:@"no-such-file.bin"];
	} else {
		NSString *own = [NSString stringWithFormat:@"w-%s-%s.bin", call->name, form];
		there = [Dir stringByAppendingPathComponent:own];
		missing = [[Dir stringByAppendingPathComponent:@"no-such-dir"] stringByAppendingPathComponent:own];
	}
	NSURL *url = urlOf(form, there, missing);
	SEL sel = sel_registerName(call->selector);
	const char *n = call->name;
	id target = D;
	if (!strcmp(n, "library.data.serializeToURL"))
		target = libraryOfData();
	else if (!strcmp(n, "library.url.serializeToURL"))
		target = libraryOfURL();
	else if (!strncmp(n, "archive.", 8))
		target = archiveOf(!strcmp(n, "archive.empty.serializeToURL"));
	if (target == nil || ![target respondsToSelector:sel]) {
		printf("no such member\n");
		return 0;
	}
	NSError *e = nil;
	if (!strcmp(n, "device.newLibraryWithFile")) {
		// A path, not a URL: nil, the file's or a missing one's; the other forms are no path.
		NSString *path = !strcmp(form, "nil")                                   ? nil
		                 : !strcmp(form, "path") || !strcmp(form, "file")       ? there
		                 : !strcmp(form, "path-missing") || !strcmp(form, "file-missing") ? missing
		                                                                         : @"";
		if (path != nil && path.length == 0) {
			printf("no such form\n");
			return 0;
		}
		printObject([D newLibraryWithFile:path error:&e]);
		printError(e);
	} else if (!strcmp(n, "device.newBinaryArchive.url") || !strcmp(n, "device.newBinaryArchive.url.library")) {
		MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
		ad.url = url;
		printObject([D newBinaryArchiveWithDescriptor:ad error:&e]);
		printError(e);
	} else if (call->kind == kWrite && strstr(call->selector, "options:") != NULL) {
		BOOL ok = ((BoolUrlIntErr)objc_msgSend)(target, sel, url, 0, &e);
		printf("%s ", ok ? "YES" : "NO");
		printError(e);
	} else if (call->kind == kWrite) {
		BOOL ok = ((BoolUrlErr)objc_msgSend)(target, sel, url, &e);
		printf("%s ", ok ? "YES" : "NO");
		printError(e);
	} else if (!strcmp(call->selector, "newBinaryLibraryWithOptions:url:error:")) {
		printObject(((IntUrlErr)objc_msgSend)(target, sel, 0, url, &e));
		printError(e);
	} else if (strstr(call->selector, "compression") != NULL || strstr(call->selector, "options:") != NULL) {
		// LZ4 (MTLIOCompressionMethodLZ4) for a compression's method or type, 0 for options.
		const NSInteger v = strstr(call->selector, "compression") != NULL ? MTLIOCompressionMethodLZ4 : 0;
		printObject(((UrlIntErr)objc_msgSend)(target, sel, url, v, &e));
		printError(e);
	} else {
		printObject(((UrlErr)objc_msgSend)(target, sel, url, &e));
		printError(e);
	}
	if (call->kind == kWrite) {
		printWritten(there);
		if (strcmp(form, "nil") != 0)
			printWritten(missing);
	}
	printf("\n");
	return 0;
}

// The parent's fixtures in `Dir`: the library's copy, a raw file, and an archive of the library's
// first kernel serialized to a file URL.
static void makeFixtures(NSString *metallib)
{
	NSFileManager *fm = [NSFileManager defaultManager];
	[fm createDirectoryAtPath:Dir withIntermediateDirectories:YES attributes:nil error:nil];
	NSData *bytes = [NSData dataWithContentsOfFile:metallib];
	if (bytes == nil || ![bytes writeToFile:LibPath atomically:NO]) {
		fprintf(stderr, "oracle_urls: cannot copy %s to %s\n", metallib.UTF8String, LibPath.UTF8String);
		exit(2);
	}
	[[NSMutableData dataWithLength:4096] writeToFile:fixturePath(kRaw) atomically:NO];
	NSError *e = nil;
	id<MTLBinaryArchive> a = archiveOf(NO);
	if (![a serializeToURL:[NSURL fileURLWithPath:fixturePath(kArchive)] error:&e]) {
		printf("  ! archive fixture: %s\n", e.description.UTF8String);
	}
}

// Runs this program again for `call` at `form`; prints its line, or how it ended.
static void child(const Call *call, const char *form, const char *device)
{
	int out[2], err[2];
	if (pipe(out) != 0 || pipe(err) != 0)
		exit(2);
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, out[1], 1);
	posix_spawn_file_actions_adddup2(&fa, err[1], 2);
	posix_spawn_file_actions_addclose(&fa, out[0]);
	posix_spawn_file_actions_addclose(&fa, err[0]);
	char exe[PATH_MAX];
	uint32_t size = sizeof exe;
	extern int _NSGetExecutablePath(char *, uint32_t *);
	if (_NSGetExecutablePath(exe, &size) != 0)
		exit(2);
	char *args[] = {exe, (char *)device, (char *)"--case", (char *)call->name, (char *)form,
	                (char *)Dir.UTF8String, NULL};
	pid_t pid = 0;
	if (posix_spawn(&pid, exe, &fa, NULL, args, environ) != 0)
		exit(2);
	posix_spawn_file_actions_destroy(&fa);
	close(out[1]);
	close(err[1]);
	NSMutableData *o = [NSMutableData data], *r = [NSMutableData data];
	char chunk[4096];
	ssize_t got;
	while ((got = read(out[0], chunk, sizeof chunk)) > 0)
		[o appendBytes:chunk length:(NSUInteger)got];
	while ((got = read(err[0], chunk, sizeof chunk)) > 0)
		[r appendBytes:chunk length:(NSUInteger)got];
	close(out[0]);
	close(err[0]);
	int status = 0;
	waitpid(pid, &status, 0);
	NSString *line = [[NSString alloc] initWithData:o encoding:NSUTF8StringEncoding];
	NSString *errText = [[NSString alloc] initWithData:r encoding:NSUTF8StringEncoding];
	printf("  %-46s %-14s -> ", call->name, form);
	if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
		printf("%s", line.UTF8String);
		if (![line hasSuffix:@"\n"])
			printf("\n");
		return;
	}
	NSString *words = @"";
	for (NSString *l in [errText componentsSeparatedByString:@"\n"])
		if ([l containsString:@"failed assertion"] || [l containsString:@"*** "])
			words = l;
	if (WIFSIGNALED(status))
		printf("%sends the process: signal %d: %s\n", line.UTF8String, WTERMSIG(status), words.UTF8String);
	else
		printf("%sexits %d: %s\n", line.UTF8String, WEXITSTATUS(status), words.UTF8String);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		if (argc >= 6 && !strcmp(argv[2], "--case")) {
			D = oracle_device(argv[1]);
			Dir = @(argv[5]);
			LibPath = [Dir stringByAppendingPathComponent:@"lib.metallib"];
			for (size_t i = 0; i < sizeof kCalls / sizeof kCalls[0]; i++)
				if (!strcmp(kCalls[i].name, argv[3]))
					return runCase(&kCalls[i], argv[4]);
			return 2;
		}
		if (argc < 3) {
			fprintf(stderr, "usage: %s DEVICE METALLIB [CALL]\n", argv[0]);
			return 2;
		}
		D = oracle_device(argv[1]);
		oracle_header("oracle_urls", D);
		Dir = [NSTemporaryDirectory() stringByAppendingPathComponent:
		                                  [NSString stringWithFormat:@"oracle_urls-%d", getpid()]];
		LibPath = [Dir stringByAppendingPathComponent:@"lib.metallib"];
		makeFixtures(@(argv[2]));
		for (size_t i = 0; i < sizeof kCalls / sizeof kCalls[0]; i++) {
			if (argc > 3 && strcmp(argv[3], kCalls[i].name) != 0)
				continue;
			for (size_t f = 0; f < sizeof kForms / sizeof kForms[0]; f++)
				child(&kCalls[i], kForms[f], argv[1]);
		}
		[[NSFileManager defaultManager] removeItemAtPath:Dir error:nil];
	}
	return 0;
}
