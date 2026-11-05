// oracle_walk: the walk oracle_reflection and the bundle's pipeline_reflection_check print a
// reflection by: every getter of no argument each object's class and its superclasses list (up to
// NSObject), its value, and the objects and arrays it gives, recursively, as `PATH.GETTER = VALUE`
// lines; with Ivars set, each object's instance variables too. A getter that ends the process
// where it does not apply (Metal asserts) is "asserts", caught in this process.
// Our own code. Include it once, from one .m or .mm.
#ifndef ORACLE_WALK_H_
#define ORACLE_WALK_H_

#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <fcntl.h>
#include <setjmp.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

static NSMutableOrderedSet<NSString *> *Seen; // the classes the walks met

// The text of `o`'s getter `s` of return type `rt`, and the object it gives.
static NSString *text(id o, SEL s, const char *rt, id __strong *obj)
{
	*obj = nil;
	switch (rt[0]) {
	case '@': {
		id r = ((id(*)(id, SEL))objc_msgSend)(o, s);
		*obj = r;
		if (r == nil)
			return @"nil";
		if ([r isKindOfClass:NSString.class])
			return [NSString stringWithFormat:@"\"%@\"", r];
		if ([r isKindOfClass:NSNumber.class])
			return [(NSNumber *)r stringValue];
		if ([r isKindOfClass:NSArray.class])
			return [NSString stringWithFormat:@"array of %lu", (unsigned long)[(NSArray *)r count]];
		return [NSString stringWithFormat:@"<%s>", class_getName([r class])];
	}
	case 'B':
	case 'c':
	case 'C':
		return [NSString stringWithFormat:@"%d", (int)((unsigned char (*)(id, SEL))objc_msgSend)(o, s)];
	case 'f':
		return [NSString stringWithFormat:@"%g", (double)((float (*)(id, SEL))objc_msgSend)(o, s)];
	case 'd':
		return [NSString stringWithFormat:@"%g", ((double (*)(id, SEL))objc_msgSend)(o, s)];
	default:
		return [NSString stringWithFormat:@"%llu", ((unsigned long long (*)(id, SEL))objc_msgSend)(o, s)];
	}
}

// The getter's text, called in this process with the signals an assertion raises caught: a
// getter that ends the process where it does not apply (Metal asserts) is "asserts". Calling it in
// a forked child, as tools/reflection_check.mm does, takes seconds a getter beside Metal's driver
// on the oracle.
static sigjmp_buf Jump;
static volatile sig_atomic_t Catching;

static void caught(int sig)
{
	if (Catching)
		siglongjmp(Jump, sig);
	signal(sig, SIG_DFL);
	raise(sig);
}

static NSString *safely(id o, SEL s, const char *rt)
{
	static BOOL installed;
	if (!installed) {
		installed = YES;
		const int sigs[] = {SIGABRT, SIGTRAP, SIGILL, SIGSEGV, SIGBUS};
		for (unsigned i = 0; i < sizeof sigs / sizeof sigs[0]; i++) {
			struct sigaction a;
			memset(&a, 0, sizeof a);
			a.sa_handler = caught;
			a.sa_flags = SA_NODEFER;
			sigemptyset(&a.sa_mask);
			sigaction(sigs[i], &a, NULL);
		}
	}
	fflush(stdout);
	const int saved = dup(2);
	const int null = open("/dev/null", O_WRONLY);
	if (null >= 0)
		dup2(null, 2);
	NSString *out = nil;
	if (sigsetjmp(Jump, 1) == 0) {
		Catching = 1;
		id obj;
		out = text(o, s, rt, &obj);
		Catching = 0;
	} else {
		Catching = 0;
		out = @"asserts";
	}
	dup2(saved, 2);
	close(saved);
	if (null >= 0)
		close(null);
	return out;
}

static BOOL skipped(NSString *s)
{
	static NSSet *never;
	if (never == nil)
		never = [NSSet setWithArray:@[
			@"dealloc", @"release", @"retain", @"autorelease", @"retainCount", @"description",
			@"debugDescription", @"hash", @"self", @"class", @"zone", @"superclass", @"formattedDescription:"
		]];
	return [never containsObject:s] || [s hasPrefix:@"new"] || [s hasPrefix:@"init"] || [s hasPrefix:@"set"] ||
	       [s hasPrefix:@"copy"] || [s hasPrefix:@"mutableCopy"] || [s hasPrefix:@"."] || [s hasPrefix:@"_"];
}

// With ORACLE_IVARS set in the environment: each object's instance variables too, as
// `PATH ivar NAME TYPE = VALUE`; an object's value is its class and the first path the walk met it
// at (`@ PATH`), scalars their value, bitfields and structs the raw bytes at their offset.
static BOOL Ivars;
static NSMapTable *PathOf;                        // object -> the first path it was met at
static NSMutableArray<NSArray *> *PendingIvars;   // (line index, prefix, object) to resolve

static void meet(id o, NSString *path)
{
	if (Ivars && o != nil && [PathOf objectForKey:o] == nil)
		[PathOf setObject:path forKey:o];
}

static void ivars(id o, NSString *path, NSMutableArray<NSString *> *lines)
{
	for (Class c = [o class]; c != Nil && c != NSObject.class; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Ivar *vs = class_copyIvarList(c, &n);
		for (unsigned i = 0; i < n; i++) {
			const char *t = ivar_getTypeEncoding(vs[i]) ?: "?";
			const ptrdiff_t off = ivar_getOffset(vs[i]);
			const uint8_t *at = (const uint8_t *)(__bridge void *)o + off;
			NSString *prefix = [NSString stringWithFormat:@"%@ ivar %s %s = ", path, ivar_getName(vs[i]), t];
			NSString *v = nil;
			switch (t[0]) {
			case '@': {
				id x = object_getIvar(o, vs[i]);
				if (x == nil) {
					v = @"nil";
				} else {
					[PendingIvars addObject:@[ @(lines.count), prefix, x ]];
					v = @"?";
				}
				break;
			}
			case 'c': case 'C': case 'B': v = [NSString stringWithFormat:@"%u", *at]; break;
			case 's': case 'S': v = [NSString stringWithFormat:@"%u", *(const uint16_t *)at]; break;
			case 'i': case 'I': v = [NSString stringWithFormat:@"%u", *(const uint32_t *)at]; break;
			case 'q': case 'Q': v = [NSString stringWithFormat:@"%llu", *(const unsigned long long *)at]; break;
			default: {
				NSUInteger size = 8, align = 0;
				if (t[0] == '{') {
					@try {
						NSGetSizeAndAlignment(t, &size, &align);
					} @catch (NSException *e) {
						size = 8; // a struct of bitfields, which it does not size
					}
				}
				else if (t[0] == 'b')
					size = 2;
				NSMutableString *hex = [NSMutableString stringWithString:@"bytes"];
				for (NSUInteger k = 0; k < size && k < 16; k++)
					[hex appendFormat:@" %02x", at[k]];
				v = hex;
				break;
			}
			}
			[lines addObject:[prefix stringByAppendingString:v]];
		}
		free(vs);
	}
}

static void resolve_ivars(NSMutableArray<NSString *> *lines)
{
	for (NSArray *p in PendingIvars) {
		id x = p[2];
		NSString *where = [PathOf objectForKey:x];
		NSString *v = [x isKindOfClass:NSString.class]   ? [NSString stringWithFormat:@"\"%@\"", x]
		              : [x isKindOfClass:NSArray.class] ? [NSString stringWithFormat:@"array of %lu @ %@",
		                                                                (unsigned long)[(NSArray *)x count], where ?: @"(unmet)"]
		                                                : [NSString stringWithFormat:@"<%s> @ %@", class_getName([x class]),
		                                                                where ?: @"(unmet)"];
		lines[[p[0] unsignedIntegerValue]] = [p[1] stringByAppendingString:v];
	}
	[PendingIvars removeAllObjects];
	[PathOf removeAllObjects];
}

static void walk(id o, NSString *path, int depth, NSMutableArray<NSString *> *lines)
{
	meet(o, path);
	if (o == nil)
		return;
	if (depth > 12) {
		[lines addObject:[path stringByAppendingString:@" (deeper than the walk goes)"]];
		return;
	}
	if ([o isKindOfClass:NSArray.class]) {
		NSUInteger i = 0;
		for (id e in (NSArray *)o) {
			NSString *at = [NSString stringWithFormat:@"%@[%lu]", path, (unsigned long)i++];
			[Seen addObject:NSStringFromClass([e class])];
			[lines addObject:[NSString stringWithFormat:@"%@ class %s", at, class_getName([e class])]];
			walk(e, at, depth + 1, lines);
		}
		return;
	}
	[Seen addObject:NSStringFromClass([o class])];
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
			    strchr("@QqIiBcCLlSsfd", rt[0]) == NULL)
				continue;
			types[s] = @(rt);
			[names addObject:s];
		}
		free(ms);
	}
	[names sortUsingSelector:@selector(compare:)];
	if (Ivars)
		ivars(o, path, lines);
	for (NSString *s in names) {
		const char *rt = types[s].UTF8String;
		const SEL sel = NSSelectorFromString(s);
		NSString *v = safely(o, sel, rt);
		NSString *at = [NSString stringWithFormat:@"%@.%@", path, s];
		[lines addObject:[NSString stringWithFormat:@"%@ = %@", at, v]];
		if (rt[0] != '@' || [v isEqualToString:@"asserts"])
			continue;
		id obj;
		text(o, sel, rt, &obj);
		if (obj != nil && ([obj isKindOfClass:NSArray.class] || [NSStringFromClass([obj class]) hasPrefix:@"MTL"]))
			walk(obj, at, depth + 1, lines);
	}
}

// Starts the walk's state: Ivars from ORACLE_IVARS in the environment.
static void walk_init(void)
{
	Seen = [NSMutableOrderedSet orderedSet];
	Ivars = getenv("ORACLE_IVARS") != NULL;
	PathOf = [NSMapTable mapTableWithKeyOptions:NSPointerFunctionsObjectPointerPersonality
	                               valueOptions:NSPointerFunctionsStrongMemory];
	PendingIvars = [NSMutableArray array];
}

#endif // ORACLE_WALK_H_
