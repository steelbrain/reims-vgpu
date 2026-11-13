// oracle_profile: what Metal's own base device (_MTLDevice) answers under a feature profile, the
// SPI `featureProfile` a GPU's device answers and Metal reads at device creation. A subclass of
// _MTLDevice made here answers featureProfile with PROFILE and nothing else; the probe initialises
// it as the base does (init, then initLimits, initFeatureQueries and initGPUFamilySupport where
// init did not run them) and prints, one line each, "PROFILE KIND MEMBER VALUE":
//   getter   each getter of MTLDevice and MTLDeviceSPI of no argument the base answers, that
//            returns a BOOL, an integer or an MTLSize (not one whose name acts: new..., set...)
//   family   supportsFamily: of each MTLGPUFamily raw value
//   featset  supportsFeatureSet: of each MTLFeatureSet raw value
//   query    each method of the device's _MTLDeviceFeatureQueries of no argument that returns a
//            BOOL or an integer
// What the base asks of the device that it does not answer itself (a GPU driver's members) is
// answered 0, printed "PROFILE asked MEMBER". A member that ends the process is printed
// "PROFILE skip MEMBER ends the process" and asked no more: each profile runs in a child
// process, run again past what ended it. No GPU is used, no object of Metal's GPU driver is made.
//   oracle_profile PROFILE...    e.g. oracle_profile 10 10002
// Build: see README.md.
#import "oracle_common.h"
#include <ctype.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

static NSUInteger g_profile;

static NSUInteger profileAnswer(id self, SEL _cmd)
{
	(void)self;
	(void)_cmd;
	return g_profile;
}

// A member Metal's base asks of a GPU's device and does not answer itself (maxBufferLength, ...):
// answered 0 here, its name printed once ("PROFILE asked MEMBER"), whatever its arguments.
static NSMethodSignature *signatureFor(id self, SEL _cmd, SEL sel)
{
	(void)_cmd;
	struct objc_super sup = {self, class_getSuperclass(object_getClass(self))};
	NSMethodSignature *s =
	    ((NSMethodSignature * (*)(struct objc_super *, SEL, SEL)) objc_msgSendSuper)(
	        &sup, @selector(methodSignatureForSelector:), sel);
	if (s != nil)
		return s;
	NSMutableString *types = [NSMutableString stringWithString:@"Q@:"];
	for (const char *c = sel_getName(sel); *c; c++)
		if (*c == ':')
			[types appendString:@"Q"];
	return [NSMethodSignature signatureWithObjCTypes:types.UTF8String];
}

static void forwardTo0(id self, SEL _cmd, NSInvocation *inv)
{
	(void)self;
	(void)_cmd;
	static NSMutableSet *said;
	if (said == nil)
		said = [NSMutableSet set];
	NSString *name = NSStringFromSelector(inv.selector);
	if (![said containsObject:name]) {
		[said addObject:name];
		printf("%lu asked %s\n", (unsigned long)g_profile, name.UTF8String);
	}
	uint8_t zero[64] = {0};
	if (inv.methodSignature.methodReturnLength <= sizeof zero && inv.methodSignature.methodReturnLength)
		[inv setReturnValue:zero];
}

static BOOL skipped(const char *name, int argc, char **argv)
{
	for (int i = 0; i < argc; i++)
		if (!strcmp(argv[i], name))
			return YES;
	return NO;
}

static BOOL acts(const char *name)
{
	static const char *const verbs[] = {"new", "commit", "wait", "set", "use", "make", "copy", "start",
	                                    "end", "reset", "map", "unmap", "release", "retain", "dealloc",
	                                    "init", "compile", "serialize", "load", "unload", "purge",
	                                    "reserve", "allocate", "deallocate", "free", "dispose", "get",
	                                    "write", "read", "register", "unregister", "add", "remove",
	                                    "_", "."};
	for (size_t i = 0; i < sizeof verbs / sizeof verbs[0]; i++) {
		const size_t n = strlen(verbs[i]);
		if (!strncmp(name, verbs[i], n) && (name[n] == '\0' || isupper((unsigned char)name[n]) ||
		                                    verbs[i][0] == '_' || verbs[i][0] == '.'))
			return YES;
	}
	return NO;
}

// Calls `sel` of no argument on `obj` by its method's return type, printing its answer.
static void ask(const char *kind, id obj, Class cls, SEL sel, int skipc, char **skipv)
{
	const char *name = sel_getName(sel);
	if (strchr(name, ':') || acts(name))
		return;
	Method m = class_getInstanceMethod(cls, sel);
	if (m == NULL)
		return;
	const char *types = method_getTypeEncoding(m);
	if (types == NULL)
		return;
	char key[256];
	snprintf(key, sizeof key, "%s.%s", kind, name);
	if (skipped(key, skipc, skipv)) {
		printf("%lu skip %s ends the process\n", (unsigned long)g_profile, key);
		return;
	}
	char rt = types[0];
	while (rt == 'r' || rt == 'n' || rt == 'N' || rt == 'o' || rt == 'O' || rt == 'R' || rt == 'V')
		rt = *(++types);
	IMP imp = method_getImplementation(m);
	switch (rt) {
	case 'c': case 'B': case 'C': case 'i': case 's': case 'l': case 'q': case 'I': case 'S':
	case 'L': case 'Q':
		break;
	case '{':
		if (strncmp(types, "{?=QQQ}", 7) && strncmp(types, "{MTLSize=QQQ}", 13))
			return;
		break;
	default:
		return;
	}
	printf("C %s\n", key);
	fflush(stdout);
	switch (rt) {
	case 'c': case 'B': case 'C':
		printf("%lu %s %s %d\n", (unsigned long)g_profile, kind, name,
		       (int)((BOOL(*)(id, SEL))imp)(obj, sel));
		break;
	case 'i': case 's': case 'I': case 'S':
		printf("%lu %s %s %ld\n", (unsigned long)g_profile, kind, name,
		       (long)((int (*)(id, SEL))imp)(obj, sel));
		break;
	case 'l': case 'q': case 'L': case 'Q':
		printf("%lu %s %s %llu\n", (unsigned long)g_profile, kind, name,
		       (unsigned long long)((uint64_t(*)(id, SEL))imp)(obj, sel));
		break;
	case '{': {
		MTLSize s = ((MTLSize(*)(id, SEL))imp)(obj, sel);
		printf("%lu %s %s %lu,%lu,%lu\n", (unsigned long)g_profile, kind, name,
		       (unsigned long)s.width, (unsigned long)s.height, (unsigned long)s.depth);
		break;
	}
	}
	fflush(stdout);
}

static void walk(const char *kind, id obj, Protocol *p, NSMutableSet<NSString *> *seen, int skipc,
                 char **skipv)
{
	if (p == NULL || [seen containsObject:@(protocol_getName(p))])
		return;
	[seen addObject:@(protocol_getName(p))];
	if (!strncmp(protocol_getName(p), "MTL", 3)) {
		for (int req = 1; req >= 0; req--) {
			unsigned n = 0;
			struct objc_method_description *d = protocol_copyMethodDescriptionList(p, req, YES, &n);
			for (unsigned i = 0; i < n; i++)
				ask(kind, obj, object_getClass(obj), d[i].name, skipc, skipv);
			free(d);
		}
	}
	unsigned n = 0;
	Protocol *__unsafe_unretained *a = protocol_copyProtocolList(p, &n);
	for (unsigned i = 0; i < n; i++)
		walk(kind, obj, a[i], seen, skipc, skipv);
	free(a);
}

static int child(NSUInteger profile, int skipc, char **skipv)
{
	g_profile = profile;
	Class base = objc_getClass("_MTLDevice");
	if (base == Nil) {
		printf("%lu error no _MTLDevice\n", (unsigned long)profile);
		return 1;
	}
	Class sub = objc_allocateClassPair(base, "OracleProfileDevice", 0);
	class_addMethod(sub, sel_registerName("featureProfile"), (IMP)profileAnswer, "Q16@0:8");
	class_addMethod(sub, @selector(methodSignatureForSelector:), (IMP)signatureFor, "@24@0:8:16");
	class_addMethod(sub, @selector(forwardInvocation:), (IMP)forwardTo0, "v24@0:8@16");
	objc_registerClassPair(sub);
	printf("C init\n");
	fflush(stdout);
	id dev = ((id(*)(id, SEL))objc_msgSend)(class_createInstance(sub, 0), sel_registerName("init"));
	printf("%lu init %s\n", (unsigned long)profile, dev ? "made" : "nil");
	if (dev == nil)
		return 1;
	const char *inits[] = {"initLimits", "initFeatureQueries", "initGPUFamilySupport"};
	for (int i = 0; i < 3; i++) {
		char key[64];
		snprintf(key, sizeof key, "init.%s", inits[i]);
		if (skipped(key, skipc, skipv))
			continue;
		printf("C %s\n", key);
		fflush(stdout);
		((void (*)(id, SEL))objc_msgSend)(dev, sel_registerName(inits[i]));
		printf("%lu init %s done\n", (unsigned long)profile, inits[i]);
	}
	NSMutableSet *seen = [NSMutableSet set];
	walk("getter", dev, objc_getProtocol("MTLDevice"), seen, skipc, skipv);
	walk("getter", dev, objc_getProtocol("MTLDeviceSPI"), seen, skipc, skipv);
	const long families[] = {1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 2001, 2002,
	                         3001, 3002, 3003, 4001, 4002, 5001, 5002, -1009};
	for (size_t i = 0; i < sizeof families / sizeof families[0]; i++) {
		char key[64];
		snprintf(key, sizeof key, "family.%ld", families[i]);
		if (skipped(key, skipc, skipv))
			continue;
		printf("C %s\n", key);
		fflush(stdout);
		printf("%lu family %ld %d\n", (unsigned long)profile, families[i],
		       (int)((BOOL(*)(id, SEL, long))objc_msgSend)(dev, sel_registerName("supportsFamily:"),
		                                                  families[i]));
	}
	const long sets[] = {0,     1,     2,     3,     4,     5,     6,     7,     8,     9,
	                     10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
	                     20,    10000, 10001, 10002, 10003, 10004, 10005, 30000, 30001, 30002};
	for (size_t i = 0; i < sizeof sets / sizeof sets[0]; i++) {
		char key[64];
		snprintf(key, sizeof key, "featset.%ld", sets[i]);
		if (skipped(key, skipc, skipv))
			continue;
		printf("C %s\n", key);
		fflush(stdout);
		printf("%lu featset %ld %d\n", (unsigned long)profile, sets[i],
		       (int)((BOOL(*)(id, SEL, long))objc_msgSend)(dev, sel_registerName("supportsFeatureSet:"),
		                                                  sets[i]));
	}
	Ivar fq = class_getInstanceVariable(base, "_featureQueries");
	id queries = fq ? object_getIvar(dev, fq) : nil;
	printf("%lu queries %s\n", (unsigned long)profile, queries ? class_getName(object_getClass(queries)) : "none");
	if (queries != nil) {
		unsigned n = 0;
		Method *ms = class_copyMethodList(object_getClass(queries), &n);
		for (unsigned i = 0; i < n; i++)
			ask("query", queries, object_getClass(queries), method_getName(ms[i]), skipc, skipv);
		free(ms);
	}
	fflush(stdout);
	return 0;
}

// Runs the child for `profile` again past each member that ends it.
static void run(const char *argv0, const char *profile)
{
	NSMutableArray<NSString *> *skips = [NSMutableArray array];
	for (int round = 0; round < 64; round++) {
		NSPipe *pipe = [NSPipe pipe];
		posix_spawn_file_actions_t fa;
		posix_spawn_file_actions_init(&fa);
		posix_spawn_file_actions_adddup2(&fa, pipe.fileHandleForWriting.fileDescriptor, 1);
		NSMutableArray *args = [NSMutableArray arrayWithObjects:@(argv0), @"--one", @(profile), nil];
		[args addObjectsFromArray:skips];
		char **av = calloc(args.count + 1, sizeof(char *));
		for (NSUInteger i = 0; i < args.count; i++)
			av[i] = (char *)[args[i] UTF8String];
		pid_t pid;
		int rc = posix_spawn(&pid, argv0, &fa, NULL, av, environ);
		free(av);
		posix_spawn_file_actions_destroy(&fa);
		[pipe.fileHandleForWriting closeFile];
		if (rc != 0) {
			printf("%s error spawn\n", profile);
			return;
		}
		NSData *out = [pipe.fileHandleForReading readDataToEndOfFile];
		int st = 0;
		waitpid(pid, &st, 0);
		NSString *text = [[NSString alloc] initWithData:out encoding:NSUTF8StringEncoding];
		NSString *last = nil;
		for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
			if ([line hasPrefix:@"C "])
				last = [line substringFromIndex:2];
			else if (line.length != 0 && (WIFEXITED(st) && WEXITSTATUS(st) == 0))
				printf("%s\n", line.UTF8String);
		}
		if (WIFEXITED(st) && WEXITSTATUS(st) == 0)
			return;
		if (last == nil) {
			printf("%s error the child ended before any member\n", profile);
			return;
		}
		[skips addObject:last];
	}
	printf("%s error too many members end the process\n", profile);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		if (argc >= 3 && !strcmp(argv[1], "--one"))
			return child((NSUInteger)strtoull(argv[2], NULL, 10), argc - 3, argv + 3);
		if (argc < 2) {
			fprintf(stderr, "usage: %s PROFILE...\n", argv[0]);
			return 2;
		}
		id<MTLDevice> d = MTLCopyAllDevices().firstObject;
		if (d != nil)
			oracle_header("oracle_profile", d);
		for (int i = 1; i < argc; i++)
			run(argv[0], argv[i]);
	}
	return 0;
}
