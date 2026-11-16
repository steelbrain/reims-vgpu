/*
 * Ours: the macOS SDK's Kernel.framework IOHIDDevice.h includes HIDDriverKit's IOHIDDevice.h for
 * the two method lists its class declaration expands (OSDeclareDefaultStructorsWithDispatch:
 * IOHIDDevice_Methods and IOHIDDevice_KernelMethods), and the SDK does not ship that header. Of
 * those lists only Dispatch is virtual, an override of OSObject's that takes no new slot; the rest
 * are plain members that RgfxPointerDevice never calls. So the class keeps its layout with
 * Dispatch alone declared here.
 */
#ifndef RGFX_SHIM_HIDDRIVERKIT_IOHIDDEVICE_H
#define RGFX_SHIM_HIDDRIVERKIT_IOHIDDEVICE_H

#define IOHIDDevice_Methods                                                                        \
public:                                                                                            \
	virtual kern_return_t Dispatch(const IORPC rpc) APPLE_KEXT_OVERRIDE;

#define IOHIDDevice_KernelMethods

#endif
