#!/usr/bin/env python3
"""Report what the guest's Metal device claims, via the Objective-C runtime.

Runs inside the macOS guest. The guest has no compiler and adding one would
change the snapshot under measurement, so this drives `objc_msgSend` through
`ctypes` instead.

This exists because a browser that refuses to use Metal does not say which
capability query it lost on — ANGLE reports only "Initialization of all (1) EGL
display types failed". The device's own answers are the other half of that
comparison.

Every value printed is read from `MTLDevice`; nothing here is inferred.
"""

import ctypes
import ctypes.util
import sys

objc = ctypes.CDLL(ctypes.util.find_library("objc"))
ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
# CoreGraphics must be in the process BEFORE the first Metal device call, or
# `MTLCreateSystemDefaultDevice()` reports nil on a machine whose device is
# perfectly healthy.
#
# Metal resolves the system default through `CGSCreateDefaultMetalDevice`, which
# it looks up with `dlsym` after `dlopen`ing CoreGraphics, and it returns nil
# with no fallback when that lookup fails. The work is behind a `dispatch_once`,
# so the answer is latched for the life of the process: loading CoreGraphics
# afterwards and asking again still returns nil.
#
# A normal application links CoreGraphics and never sees this. A bare `ctypes`
# process does not, which made this probe report a nil default device and no
# browser could be blamed for it — the nil was ours. Keep this above the first
# Metal call.
ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")

objc.objc_getClass.restype = ctypes.c_void_p
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.sel_registerName.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]


def msg(restype, obj, sel, *args, argtypes=None):
    """One `objc_msgSend` per call signature.

    `objc_msgSend` is variadic in C but not in the ABI: ctypes must be told the
    exact argument and return types for each call or the wrong registers are
    read. Rebinding restype/argtypes per call is the price of not writing a
    compiler-backed probe.

    `argtypes` is explicit rather than derived from the values, because a
    derived type cannot describe an out-parameter: `ctypes.byref(x)` produces a
    CArgObject, which has no `from_param` and raises at bind time.
    """
    fn = objc.objc_msgSend
    fn.restype = restype
    if argtypes is None:
        argtypes = [ctypes.c_ulonglong if isinstance(a, int) else type(a) for a in args]
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + list(argtypes)
    return fn(ctypes.c_void_p(obj), ctypes.c_void_p(objc.sel_registerName(sel)), *args)


def nsstring(ptr):
    if not ptr:
        return None
    utf8 = msg(ctypes.c_char_p, ptr, b"UTF8String")
    return utf8.decode() if utf8 else None


# MTLGPUFamily, from MTLDevice.h. ANGLE's Metal backend and Dawn both gate on
# these rather than on the deprecated feature sets.
FAMILIES = [
    ("Apple1", 1001), ("Apple2", 1002), ("Apple3", 1003), ("Apple4", 1004),
    ("Apple5", 1005), ("Apple6", 1006), ("Apple7", 1007), ("Apple8", 1008),
    ("Mac1", 2001), ("Mac2", 2002),
    ("Common1", 3001), ("Common2", 3002), ("Common3", 3003),
    ("MacCatalyst1", 4001), ("MacCatalyst2", 4002),
]

# MTLFeatureSet_macOS_GPUFamily{1,2}_v{1..4}. Deprecated, but this is what
# "Metal Support: Metal 2" in System Information is derived from, and some
# clients still ask.
FEATURE_SETS = [
    ("macOS_GPUFamily1_v1", 10000), ("macOS_GPUFamily1_v2", 10001),
    ("macOS_GPUFamily1_v3", 10003), ("macOS_GPUFamily1_v4", 10004),
    ("macOS_GPUFamily2_v1", 10005),
]


def main():
    dev = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    dev.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    dev.MTLCopyAllDevices.restype = ctypes.c_void_p
    d = dev.MTLCreateSystemDefaultDevice()
    print("MTLCreateSystemDefaultDevice:", "nil" if not d else "ok")
    # ANGLE-Metal and Dawn both ask for the system default and give up when it
    # is nil, so this line decides whether a browser can reach the GPU at all.
    # It is expected to read "ok" here: see the CoreGraphics note at the top of
    # this file for the one way it can lie, and `guest_default_device.py` for a
    # probe that distinguishes every reason it could legitimately be nil.
    arr = dev.MTLCopyAllDevices()
    n = msg(ctypes.c_ulonglong, arr, b"count") if arr else 0
    print("MTLCopyAllDevices count     :", n)
    if not d:
        if not n:
            return 1
        d = msg(ctypes.c_void_p, arr, b"objectAtIndex:", ctypes.c_ulonglong(0))
        print("(reporting MTLCopyAllDevices[0] instead)")
    print("name                        :", nsstring(msg(ctypes.c_void_p, d, b"name")))
    supports_opengl_sel = objc.sel_registerName(b"supportsOpenGL")
    supports_opengl_available = msg(
        ctypes.c_bool,
        d,
        b"respondsToSelector:",
        ctypes.c_void_p(supports_opengl_sel),
    )
    if supports_opengl_available:
        supports_opengl = msg(ctypes.c_bool, d, b"supportsOpenGL")
        print("supportsOpenGL              :", supports_opengl)
    else:
        print("supportsOpenGL              :", "selector unavailable")
    print("registryID                  :", msg(ctypes.c_ulonglong, d, b"registryID"))
    print("location                    :", msg(ctypes.c_ulonglong, d, b"location"))
    print("lowPower                    :", msg(ctypes.c_bool, d, b"isLowPower"))
    print("headless                    :", msg(ctypes.c_bool, d, b"isHeadless"))
    print("removable                   :", msg(ctypes.c_bool, d, b"isRemovable"))
    print("hasUnifiedMemory            :", msg(ctypes.c_bool, d, b"hasUnifiedMemory"))
    print("recommendedMaxWorkingSetSize:", msg(ctypes.c_ulonglong, d, b"recommendedMaxWorkingSetSize"))
    print("maxBufferLength             :", msg(ctypes.c_ulonglong, d, b"maxBufferLength"))
    print("argumentBuffersSupport      :", msg(ctypes.c_ulonglong, d, b"argumentBuffersSupport"))
    print("readWriteTextureSupport     :", msg(ctypes.c_ulonglong, d, b"readWriteTextureSupport"))
    print("depth24Stencil8Supported    :", msg(ctypes.c_bool, d, b"isDepth24Stencil8PixelFormatSupported"))
    print("programmableSamplePositions :", msg(ctypes.c_bool, d, b"areProgrammableSamplePositionsSupported"))
    print("rasterOrderGroupsSupported  :", msg(ctypes.c_bool, d, b"areRasterOrderGroupsSupported"))
    print("barycentricCoordsSupported  :", msg(ctypes.c_bool, d, b"areBarycentricCoordsSupported"))
    print("supportsShaderBarycentric   :", msg(ctypes.c_bool, d, b"supportsShaderBarycentricCoordinates"))
    print("supports32BitFloatFiltering :", msg(ctypes.c_bool, d, b"supports32BitFloatFiltering"))
    print("supportsBCTextureCompression:", msg(ctypes.c_bool, d, b"supportsBCTextureCompression"))
    print("supportsPullModelInterp     :", msg(ctypes.c_bool, d, b"supportsPullModelInterpolation"))
    print("supportsRaytracing          :", msg(ctypes.c_bool, d, b"supportsRaytracing"))
    print("supportsFunctionPointers    :", msg(ctypes.c_bool, d, b"supportsFunctionPointers"))
    print("supportsDynamicLibraries    :", msg(ctypes.c_bool, d, b"supportsDynamicLibraries"))
    print("maxThreadgroupMemoryLength  :", msg(ctypes.c_ulonglong, d, b"maxThreadgroupMemoryLength"))
    print("maxArgumentBufferSamplers   :", msg(ctypes.c_ulonglong, d, b"maxArgumentBufferSamplerCount"))

    fams = [n for n, v in FAMILIES if msg(ctypes.c_bool, d, b"supportsFamily:", ctypes.c_ulonglong(v))]
    print("supportsFamily              :", " ".join(fams) or "(none)")
    fss = [n for n, v in FEATURE_SETS if msg(ctypes.c_bool, d, b"supportsFeatureSet:", ctypes.c_ulonglong(v))]
    print("supportsFeatureSet          :", " ".join(fss) or "(none)")

    # ANGLE's Metal backend compiles its shaders at runtime; a device that
    # cannot build a library from source cannot back a WebGL context no matter
    # what the capability bits say.
    src_cls = objc.objc_getClass(b"NSString")
    src = msg(
        ctypes.c_void_p,
        src_cls,
        b"stringWithUTF8String:",
        ctypes.c_char_p(b"#include <metal_stdlib>\nusing namespace metal;\n"
                        b"vertex float4 v(uint i [[vertex_id]]) { return float4(0,0,0,1); }\n"),
    )
    err = ctypes.c_void_p(0)
    lib = msg(
        ctypes.c_void_p, d, b"newLibraryWithSource:options:error:",
        ctypes.c_void_p(src), ctypes.c_void_p(0), ctypes.byref(err),
        argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
    )
    print("runtime MSL compile         :", "ok" if lib else "FAILED")
    if not lib and err:
        print("  error                     :", nsstring(msg(ctypes.c_void_p, err.value, b"localizedDescription")))

    # A device that cannot make a command queue is not merely "not the default";
    # it is unusable, and that would be a different bug from a selection one.
    q = msg(ctypes.c_void_p, d, b"newCommandQueue")
    print("newCommandQueue             :", "ok" if q else "FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
