/*
 * RgfxUnsupported: see RgfxUnsupported.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxUnsupported.h"

#import <objc/runtime.h>

#include <mutex>
#include <set>
#include <utility>

namespace {

std::mutex g_lock;
std::set<std::pair<const void *, const void *>> *g_logged; // (class, selector) pairs asked

/// Records (cls, cmd); answers whether it is the first time.
bool firstAsk(Class cls, SEL cmd)
{
	std::lock_guard<std::mutex> hold(g_lock);
	if (g_logged == nullptr) {
		g_logged = new std::set<std::pair<const void *, const void *>>();
	}
	return g_logged->insert({(__bridge const void *)cls, (const void *)cmd}).second;
}

} // namespace

void RgfxLogUnsupported(Class cls, SEL cmd, const char *what)
{
	if (firstAsk(cls, cmd)) {
		NSLog(@"RGFX: unsupported: -[%s %s]: %s",
		      class_getName(cls), sel_getName(cmd), what);
	}
}

BOOL RgfxUnsupportedWasLogged(Class cls, SEL cmd)
{
	std::lock_guard<std::mutex> hold(g_lock);
	return g_logged != nullptr &&
	       g_logged->count({(__bridge const void *)cls, (const void *)cmd}) != 0;
}

NSError *RgfxUnsupportedError(Class cls, SEL cmd, const char *what, NSErrorDomain domain,
                              NSInteger code)
{
	NSString *message = [NSString stringWithFormat:@"-[%s %s]: %s is not supported by this device",
	                                               class_getName(cls), sel_getName(cmd), what];
	return [NSError errorWithDomain:domain
	                           code:code
	                       userInfo:@{NSLocalizedDescriptionKey : message}];
}
