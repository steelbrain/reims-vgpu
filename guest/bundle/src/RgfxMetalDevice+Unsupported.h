/*
 * RgfxMetalDevice (Unsupported): the MTLDevice creators of what the device does not make,
 * answered as Metal answers a creator that fails: nil, with an NSError where the creator has
 * an error out (in Metal's own domain where it names the failure, else RgfxMetalErrorDomain's
 * RgfxMetalErrorUnsupported), and the completion handler called with nil and that error. Each
 * logs its first use (RgfxUnsupported.h).
 *
 * What each waits on is listed in the effort's protocol-gaps-wire.md. Ray tracing, function
 * pointers, dynamic libraries, Metal 4, placement sparse, counters and rasterization rate maps
 * are what the device answers no for (RgfxCaps); indirect command buffers, argument encoders,
 * binary archives, tile pipelines, residency sets, shared textures, log states and
 * tensors are what the wire does not carry yet.
 *
 * Without these the IOGPU base's own creators answer, which build on IOGPU objects the device
 * does not have. The same goes for the base classes' creators no header declares, refused here
 * too where the device does not make what they make: tests/inherited_tests.mm keeps every
 * base-class creator answered by the device or listed, with why, as left to the base. Like the
 * rest of the device it is compiled into the bundle but not linked into the unit tests
 * (RgfxMetalDevice.h). Our own ObjC++.
 */

#ifndef RGFX_METAL_DEVICE_UNSUPPORTED_H_
#define RGFX_METAL_DEVICE_UNSUPPORTED_H_

#import "RgfxMetalDevice.h"

@interface RgfxMetalDevice (Unsupported)
@end

#endif // RGFX_METAL_DEVICE_UNSUPPORTED_H_
