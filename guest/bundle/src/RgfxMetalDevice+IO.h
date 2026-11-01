/*
 * RgfxMetalDevice (IO): fast resource loading, MTLIOCommandQueue and its file handles. The device
 * makes Metal's own IO queue and raw and compressed file handles over itself (RgfxMetalPrivate.h):
 * the queue reads and decompresses files on its own threads and loads buffers, process memory and
 * textures through the device's command queues, shared events and blits, so nothing of it is on
 * the wire. Where the guest's Metal lacks one of the classes, its creator is refused as Metal
 * refuses a creator that fails (RgfxUnsupported.h).
 *
 * Like the rest of the device it is compiled into the bundle but not linked into the unit tests
 * (RgfxMetalDevice.h). Our own ObjC++.
 */

#ifndef RGFX_METAL_DEVICE_IO_H_
#define RGFX_METAL_DEVICE_IO_H_

#import "RgfxMetalDevice.h"

@interface RgfxMetalDevice (IO)
@end

#endif // RGFX_METAL_DEVICE_IO_H_
