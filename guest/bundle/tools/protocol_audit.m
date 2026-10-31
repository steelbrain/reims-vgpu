// protocol_audit: which members of Metal's public protocols each of the bundle's classes answers.
//   protocol_audit [BUNDLE]  loads BUNDLE (a Mach-O, default: none, for a process where Metal
//                            loaded ours) and, for every class whose name starts with "Rgfx", walks
//                            the MTL* protocols it adopts (with the protocols they adopt) and checks
//                            each instance member, required and optional, against the class and its
//                            superclasses.
// It prints one line per member the class does not answer:
//   MISSING  required|optional  Class  Protocol  selector
// one line per member a class answers only through a superclass that is not ours (the device's
// IOGPU base):
//   SUPER    required|optional  Class  Protocol  selector  DefiningClass
// and a summary: "AUDIT classes=N required-missing=R optional-missing=O super=S". It changes
// nothing and calls no member. Exits 0.
// Build (guest): xcrun clang -arch x86_64 -fobjc-arc -framework Foundation -framework Metal
//                protocol_audit.m -o protocol_audit
// On the guest, run it with no argument after a Metal call has loaded the bundle (it makes the
// default device first), or with the bundle's path.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <string.h>

static int g_reqMissing, g_optMissing, g_super;

// The class in `cls`'s chain whose own method list holds `sel` (nil when none does).
static Class definingClass(Class cls, SEL sel)
{
	for (Class c = cls; c != Nil; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *ms = class_copyMethodList(c, &n);
		BOOL found = NO;
		for (unsigned i = 0; i < n && !found; i++) {
			found = method_getName(ms[i]) == sel;
		}
		free(ms);
		if (found) {
			return c;
		}
	}
	return Nil;
}

static void auditProtocol(Class cls, Protocol *p, NSMutableSet<NSString *> *seen)
{
	NSString *pname = [NSString stringWithUTF8String:protocol_getName(p)];
	if ([seen containsObject:pname]) {
		return;
	}
	[seen addObject:pname];
	if ([pname hasPrefix:@"MTL"]) {
		for (int required = 1; required >= 0; required--) {
			unsigned n = 0;
			struct objc_method_description *ds =
			    protocol_copyMethodDescriptionList(p, (BOOL)required, YES, &n);
			for (unsigned i = 0; i < n; i++) {
				SEL sel = ds[i].name;
				Class def = definingClass(cls, sel);
				const char *kind = required ? "required" : "optional";
				if (def == Nil) {
					printf("MISSING  %s  %s  %s  %s\n", kind, class_getName(cls),
					       pname.UTF8String, sel_getName(sel));
					if (required) {
						g_reqMissing++;
					} else {
						g_optMissing++;
					}
				} else if (strncmp(class_getName(def), "Rgfx", 4) != 0 &&
				           strcmp(class_getName(def), "NSObject") != 0) {
					printf("SUPER    %s  %s  %s  %s  %s\n", kind, class_getName(cls),
					       pname.UTF8String, sel_getName(sel), class_getName(def));
					g_super++;
				}
			}
			free(ds);
		}
	}
	unsigned n = 0;
	Protocol *__unsafe_unretained *inherited = protocol_copyProtocolList(p, &n);
	for (unsigned i = 0; i < n; i++) {
		auditProtocol(cls, inherited[i], seen);
	}
	free(inherited);
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc > 1) {
			if (dlopen(argv[1], RTLD_NOW) == NULL) {
				printf("AUDIT FAIL: dlopen %s: %s\n", argv[1], dlerror());
				return 1;
			}
		} else {
			id<MTLDevice> d = MTLCreateSystemDefaultDevice();
			printf("device: %s (%s)\n", d ? d.name.UTF8String : "(nil)",
			       d ? class_getName([(id)d class]) : "-");
		}
		unsigned count = 0;
		Class *classes = objc_copyClassList(&count);
		int audited = 0;
		for (unsigned i = 0; i < count; i++) {
			Class cls = classes[i];
			if (strncmp(class_getName(cls), "Rgfx", 4) != 0) {
				continue;
			}
			NSMutableSet<NSString *> *seen = [NSMutableSet set];
			for (Class c = cls; c != Nil; c = class_getSuperclass(c)) {
				unsigned pn = 0;
				Protocol *__unsafe_unretained *ps = class_copyProtocolList(c, &pn);
				for (unsigned j = 0; j < pn; j++) {
					auditProtocol(cls, ps[j], seen);
				}
				free(ps);
			}
			audited++;
		}
		free(classes);
		printf("AUDIT classes=%d required-missing=%d optional-missing=%d super=%d\n", audited,
		       g_reqMissing, g_optMissing, g_super);
	}
	return 0;
}
