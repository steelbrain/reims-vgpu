// base_answers_probe: what a Metal device answers to members that take no argument, one line per
// member, so the answers the reims-graphics device leaves to its IOGPU base (the "base" lines of
// tests/protocol_members.txt) can be set beside what Metal's own devices answer.
//   base_answers_probe [INDEX] < SELECTORS
//     reads one selector per line and asks device INDEX of MTLCopyAllDevices (default: the system
//     default device) each, printing "selector<TAB>answer": the value of a BOOL, integer or
//     floating-point answer; an NSString's or NSNumber's text, or the class of another object;
//     the fields of a pointed-to struct of 32-bit, 64-bit and float fields (the limit table);
//     "-" where the device does not answer it; "args" where it takes arguments; "type T" for
//     an answer of another type; "died S" where the call ended the process with signal S.
// Each member is asked in a process of its own (the probe runs itself with --one), so a member
// that asserts or faults ends only that process. The probe reads; it sends no setter.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Foundation -framework Metal \
//        tools/base_answers_probe.m -o base_answers_probe   (ARCH=arm64 for an Apple-silicon Mac)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <objc/message.h>
#include <objc/runtime.h>
#include <spawn.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>

extern char **environ;

static id<MTLDevice> deviceAt(long index)
{
	if (index < 0)
		return MTLCreateSystemDefaultDevice();
	NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
	return index < (long)all.count ? all[(NSUInteger)index] : nil;
}

// Asks `device` `name` and prints the answer; the child's whole work.
static int askOne(id<MTLDevice> device, const char *name)
{
	SEL sel = sel_registerName(name);
	if (device == nil || ![(id)device respondsToSelector:sel]) {
		printf("%s\t-\n", name);
		return 0;
	}
	NSMethodSignature *sig = [(id)device methodSignatureForSelector:sel];
	if (sig.numberOfArguments != 2) {
		printf("%s\targs\n", name);
		return 0;
	}
	const char *type = sig.methodReturnType;
	while (*type == 'r' || *type == 'n' || *type == 'N' || *type == 'o' || *type == 'O' ||
	       *type == 'R' || *type == 'V')
		type++;
	id d = (id)device;
	switch (*type) {
	case 'c':
	case 'B':
		printf("%s\t%d\n", name, (int)((BOOL (*)(id, SEL))objc_msgSend)(d, sel));
		break;
	case 'C':
		printf("%s\t%u\n", name, (unsigned)((unsigned char (*)(id, SEL))objc_msgSend)(d, sel));
		break;
	case 's':
	case 'i':
		printf("%s\t%d\n", name, ((int (*)(id, SEL))objc_msgSend)(d, sel));
		break;
	case 'S':
	case 'I':
		printf("%s\t%u\n", name, ((unsigned (*)(id, SEL))objc_msgSend)(d, sel));
		break;
	case 'l':
	case 'q':
		printf("%s\t%lld\n", name, ((long long (*)(id, SEL))objc_msgSend)(d, sel));
		break;
	case 'L':
	case 'Q':
		printf("%s\t%llu\n", name,
		       ((unsigned long long (*)(id, SEL))objc_msgSend)(d, sel));
		break;
	case 'f':
		printf("%s\t%g\n", name, (double)((float (*)(id, SEL))objc_msgSend)(d, sel));
		break;
	case 'd':
		printf("%s\t%g\n", name, ((double (*)(id, SEL))objc_msgSend)(d, sel));
		break;
	case '@': {
		id v = ((id (*)(id, SEL))objc_msgSend)(d, sel);
		if (v == nil)
			printf("%s\tnil\n", name);
		else if ([v isKindOfClass:[NSString class]] || [v isKindOfClass:[NSNumber class]])
			printf("%s\t%s\n", name, [[v description] UTF8String]);
		else
			printf("%s\tobject %s\n", name, class_getName([v class]));
		break;
	}
	case '^': {
		// A pointer to a struct of only 32-bit, 64-bit and float fields (the device's limit
		// table): its fields, in order.
		const char *body = strchr(type, '=');
		bool plain = body != NULL && type[1] == '{';
		for (const char *c = body ? body + 1 : ""; plain && *c && *c != '}'; c++)
			plain = strchr("IiQqf", *c) != NULL;
		if (!plain) {
			printf("%s\ttype %s\n", name, sig.methodReturnType);
			break;
		}
		const uint8_t *at = ((const uint8_t *(*)(id, SEL))objc_msgSend)(d, sel);
		printf("%s\t", name);
		size_t off = 0;
		for (const char *c = body + 1; at != NULL && *c != '}'; c++) {
			if (*c == 'Q' || *c == 'q') {
				off = (off + 7) & ~(size_t)7;
				uint64_t v;
				memcpy(&v, at + off, 8);
				printf("%s%llu", off ? "," : "", (unsigned long long)v);
				off += 8;
			} else if (*c == 'f') {
				float v;
				memcpy(&v, at + off, 4);
				printf("%s%g", off ? "," : "", (double)v);
				off += 4;
			} else {
				uint32_t v;
				memcpy(&v, at + off, 4);
				printf("%s%u", off ? "," : "", v);
				off += 4;
			}
		}
		printf("%s\n", at == NULL ? "NULL" : "");
		break;
	}
	default:
		printf("%s\ttype %s\n", name, sig.methodReturnType);
	}
	fflush(stdout);
	return 0;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc == 4 && strcmp(argv[1], "--one") == 0)
			return askOne(deviceAt(atol(argv[2])), argv[3]);
		const char *index = argc > 1 ? argv[1] : "-1";
		id<MTLDevice> device = deviceAt(atol(index));
		printf("# device %s (%s)\n", device ? device.name.UTF8String : "(none)",
		       device ? class_getName([(id)device class]) : "-");
		fflush(stdout);
		char line[512];
		while (fgets(line, sizeof line, stdin) != NULL) {
			line[strcspn(line, "\r\n")] = 0;
			if (line[0] == 0 || line[0] == '#')
				continue;
			char *args[] = {argv[0], (char *)"--one", (char *)index, line, NULL};
			pid_t pid;
			if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ) != 0) {
				printf("%s\tspawn failed\n", line);
				continue;
			}
			int status = 0;
			waitpid(pid, &status, 0);
			if (WIFSIGNALED(status)) {
				printf("%s\tdied %d\n", line, WTERMSIG(status));
				fflush(stdout);
			}
		}
	}
	return 0;
}
