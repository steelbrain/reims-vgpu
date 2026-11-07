// gl_redirect: a library inserted into one process (DYLD_INSERT_LIBRARIES) that makes that
// process's OpenGL see an IOGLBundleName on the reims-graphics accelerator, as the kext's
// personality would carry it: IORegistryEntryCreateCFProperty and
// IORegistryEntryCreateCFProperties answer GL_REDIRECT_BUNDLE (default AppleMetalOpenGLRenderer)
// for the key on an accelerator of class GL_REDIRECT_CLASS (default RgfxAccelerator) that has
// none, or, with GL_REDIRECT_FORCE=1, in place of the one it has (so another Mac's accelerator
// can be pointed at the same renderer). Nothing outside the process changes, so a
// GL renderer can be tried on the device before the kext names one. Each answer it changes is
// written to stderr once.
// Our own code.
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DYLD_INTERPOSE(_replacement, _replacee)                                              \
	__attribute__((used)) static struct {                                                    \
		const void *replacement;                                                             \
		const void *replacee;                                                                \
	} _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = {              \
	    (const void *)(unsigned long)&_replacement, (const void *)(unsigned long)&_replacee};

static const char *bundle_name(void) {
	const char *n = getenv("GL_REDIRECT_BUNDLE");
	return n && n[0] ? n : "AppleMetalOpenGLRenderer";
}

static int is_ours(io_registry_entry_t entry) {
	const char *want = getenv("GL_REDIRECT_CLASS");
	io_name_t cls;
	return IOObjectGetClass(entry, cls) == KERN_SUCCESS &&
	       strcmp(cls, want && want[0] ? want : "RgfxAccelerator") == 0;
}

static int forced(void) {
	const char *f = getenv("GL_REDIRECT_FORCE");
	return f && strcmp(f, "1") == 0;
}

static int is_key(CFStringRef key) {
	return key && CFGetTypeID(key) == CFStringGetTypeID() &&
	       CFStringCompare(key, CFSTR("IOGLBundleName"), 0) == kCFCompareEqualTo;
}

static void note(const char *what) {
	static int said[2];
	int i = what[0] == 's' ? 1 : 0;
	if (!said[i]) {
		said[i] = 1;
		const char *want = getenv("GL_REDIRECT_CLASS");
		fprintf(stderr, "gl_redirect: %s answers IOGLBundleName %s for %s\n", what, bundle_name(),
		        want && want[0] ? want : "RgfxAccelerator");
	}
}

static CFTypeRef my_property(io_registry_entry_t entry, CFStringRef key, CFAllocatorRef allocator,
                             IOOptionBits options) {
	CFTypeRef v = IORegistryEntryCreateCFProperty(entry, key, allocator, options);
	if ((v == NULL || forced()) && is_key(key) && is_ours(entry)) {
		if (v) CFRelease(v);
		note("single");
		return CFStringCreateWithCString(allocator, bundle_name(), kCFStringEncodingUTF8);
	}
	return v;
}

static kern_return_t my_properties(io_registry_entry_t entry, CFMutableDictionaryRef *properties,
                                   CFAllocatorRef allocator, IOOptionBits options) {
	kern_return_t kr = IORegistryEntryCreateCFProperties(entry, properties, allocator, options);
	if (kr == KERN_SUCCESS && properties && *properties && is_ours(entry) &&
	    (forced() || !CFDictionaryContainsKey(*properties, CFSTR("IOGLBundleName")))) {
		note("set");
		CFStringRef name = CFStringCreateWithCString(allocator, bundle_name(), kCFStringEncodingUTF8);
		CFDictionarySetValue(*properties, CFSTR("IOGLBundleName"), name);
		CFRelease(name);
	}
	return kr;
}

DYLD_INTERPOSE(my_property, IORegistryEntryCreateCFProperty)
DYLD_INTERPOSE(my_properties, IORegistryEntryCreateCFProperties)
