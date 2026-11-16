/*
 * RgfxPointerDevice's two IOReporting forwards (RgfxPointerDevice.hpp). configureReport and
 * updateReport are IOService's slots, which IOHIDDevice overrides from macOS 13 on and x86-64 11
 * and 12 do not: each forward is a qualified call, which under -fapple-kext loads the method from
 * IOHIDDevice's live vtable (__ZTV11IOHIDDevice) at that slot, so a release runs its IOHIDDevice's
 * override where it has one and IOService's where it does not, and the kext imports neither by
 * name. IOHIDDevice declares its overrides private, so the call compiles only with C++ access
 * control off, which the Makefile turns off for this file alone (-fno-access-control). Nothing
 * else belongs here.
 */

#include "RgfxPointerDevice.hpp"

IOReturn RgfxPointerDevice::configureReport(IOReportChannelList *channels,
                                            IOReportConfigureAction action, void *result,
                                            void *destination)
{
	return IOHIDDevice::configureReport(channels, action, result, destination);
}

IOReturn RgfxPointerDevice::updateReport(IOReportChannelList *channels, IOReportUpdateAction action,
                                         void *result, void *destination)
{
	return IOHIDDevice::updateReport(channels, action, result, destination);
}
