// protocol_dump: every instance member of every MTL* protocol this process's Metal defines, through
// the Objective-C runtime: one line per member, "PROTOCOL SELECTOR required|optional", and one per
// protocol it adopts, "PROTOCOL adopts OTHER". It makes no device and calls no member. Run in each
// guest macOS and on the Mac that runs the bundle's tests: a member a guest prints for a protocol
// the bundle's classes adopt (or its SPI counterpart) that the Mac's protocol, with the protocols
// it adopts, lacks is a line of tests/protocol_members_releases.txt.
// Build (on a Mac): xcrun clang -arch x86_64 -mmacosx-version-min=11.0 -fobjc-arc
//                   -framework Foundation -framework Metal protocol_dump.m -o protocol_dump
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <objc/runtime.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
	@autoreleasepool {
		// Metal registers its protocols as it loads; naming one makes the linker keep it.
		(void)@protocol(MTLDevice);
		unsigned count = 0;
		Protocol *__unsafe_unretained *protocols = objc_copyProtocolList(&count);
		for (unsigned i = 0; i < count; i++) {
			const char *name = protocol_getName(protocols[i]);
			if (strncmp(name, "MTL", 3) != 0)
				continue;
			for (int required = 1; required >= 0; required--) {
				unsigned n = 0;
				struct objc_method_description *members =
				    protocol_copyMethodDescriptionList(protocols[i], required, YES, &n);
				for (unsigned j = 0; j < n; j++)
					printf("%s %s %s\n", name, sel_getName(members[j].name),
					       required ? "required" : "optional");
				free(members);
			}
			unsigned m = 0;
			Protocol *__unsafe_unretained *adopted = protocol_copyProtocolList(protocols[i], &m);
			for (unsigned j = 0; j < m; j++)
				printf("%s adopts %s\n", name, protocol_getName(adopted[j]));
			free(adopted);
		}
		free(protocols);
	}
	return 0;
}
