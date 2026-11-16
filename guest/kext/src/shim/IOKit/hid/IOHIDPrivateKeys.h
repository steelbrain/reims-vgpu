/*
 * Ours, empty: the macOS SDK's Kernel.framework IOHIDDevice.h includes this header, which the SDK
 * does not ship (Xcode 27.2's MacOSX27.2.sdk). IOHIDDevice.h's declarations use none of its keys,
 * and RgfxPointerDevice uses none either.
 */
#ifndef RGFX_SHIM_IOHIDPRIVATEKEYS_H
#define RGFX_SHIM_IOHIDPRIVATEKEYS_H
#endif
