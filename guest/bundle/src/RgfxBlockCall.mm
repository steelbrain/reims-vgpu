/*
 * RgfxBlockCall: see the header. Our own ObjC++.
 */

#import "RgfxBlockCall.h"

#include <cstdint>
#include <mutex>
#include <set>

namespace {

// clang's block ABI (its "Block Implementation Specification"): the layout of a block and of its
// descriptor, and the flags that say which of the descriptor's optional parts it has.
constexpr int32_t kBlockHasCopyDispose = 1 << 25;
constexpr int32_t kBlockHasSignature = 1 << 30;

struct BlockDescriptor {
	unsigned long reserved;
	unsigned long size;
	// Then the copy and dispose helpers (kBlockHasCopyDispose), then the signature
	// (kBlockHasSignature).
};

struct BlockLayout {
	void *isa;
	int32_t flags;
	int32_t reserved;
	void (*invoke)(void *, ...);
	BlockDescriptor *descriptor;
};

const BlockLayout *layoutOf(id block)
{
	return (__bridge const BlockLayout *)block;
}

const char *signatureOf(id block)
{
	if (block == nil)
		return nullptr;
	const BlockLayout *layout = layoutOf(block);
	if ((layout->flags & kBlockHasSignature) == 0 || layout->descriptor == nullptr)
		return nullptr;
	const uint8_t *at =
	    reinterpret_cast<const uint8_t *>(layout->descriptor) + sizeof(BlockDescriptor);
	if ((layout->flags & kBlockHasCopyDispose) != 0)
		at += 2 * sizeof(void *);
	return *reinterpret_cast<const char *const *>(at);
}

// An argument's encoding without the class name an object's carries (`@"NSError"` is `@`).
NSString *withoutClassName(const char *type)
{
	NSString *s = @(type);
	if ([s hasPrefix:@"@\""])
		return @"@";
	return s;
}

void logUncalled(SEL cmd, id handler)
{
	static std::mutex lock;
	static std::set<SEL> logged;
	{
		std::lock_guard<std::mutex> hold(lock);
		if (!logged.insert(cmd).second)
			return;
	}
	const char *signature = signatureOf(handler);
	NSLog(@"RGFX: %s: a completion handler of signature %s, neither (object, error) nor (object, "
	      @"reflection, error): not called",
	      sel_getName(cmd), signature != nullptr ? signature : "(none)");
}

} // namespace

NSArray<NSString *> *RgfxBlockArgumentTypes(id block)
{
	const char *signature = signatureOf(block);
	if (signature == nullptr)
		return nil;
	NSMethodSignature *types = nil;
	@try {
		types = [NSMethodSignature signatureWithObjCTypes:signature];
	} @catch (NSException *) {
		return nil;
	}
	// Argument 0 is the block itself.
	if (types.numberOfArguments < 1)
		return nil;
	NSMutableArray<NSString *> *args = [NSMutableArray array];
	for (NSUInteger i = 1; i < types.numberOfArguments; i++)
		[args addObject:withoutClassName([types getArgumentTypeAtIndex:i])];
	return args;
}

BOOL RgfxCallCreationHandler(id handler, id object, NSError *error, SEL cmd)
{
	return RgfxCallCreationHandlerWithReflection(handler, object, nil, error, cmd);
}

BOOL RgfxCallCreationHandlerWithReflection(id handler, id object, id reflection, NSError *error, SEL cmd)
{
	if (handler == nil)
		return YES;
	NSArray<NSString *> *types = RgfxBlockArgumentTypes(handler);
	const BOOL allObjects = types != nil && [types indexOfObjectPassingTest:^BOOL(
	                                                    NSString *t, NSUInteger, BOOL *) {
		                                        return ![t isEqualToString:@"@"];
	                                        }] == NSNotFound;
	void *invoke = reinterpret_cast<void *>(layoutOf(handler)->invoke);
	if (allObjects && types.count == 2) {
		reinterpret_cast<void (*)(id, id, id)>(invoke)(handler, object, error);
		return YES;
	}
	if (allObjects && types.count == 3) {
		reinterpret_cast<void (*)(id, id, id, id)>(invoke)(handler, object, reflection, error);
		return YES;
	}
	logUncalled(cmd, handler);
	return NO;
}

namespace {

// The creations RgfxRunCreation started and that have not ended.
dispatch_group_t creations()
{
	static dispatch_group_t group;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		group = dispatch_group_create();
	});
	return group;
}

} // namespace

void RgfxRunCreation(dispatch_block_t creation)
{
	dispatch_group_async(creations(), dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
		@autoreleasepool {
			creation();
		}
	});
}

BOOL RgfxWaitForCreations(dispatch_time_t timeout)
{
	return dispatch_group_wait(creations(), timeout) == 0;
}

id RgfxCopyOf(id object)
{
	return [object respondsToSelector:@selector(copyWithZone:)] ? [object copy] : object;
}
