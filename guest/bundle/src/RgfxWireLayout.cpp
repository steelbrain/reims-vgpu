/*
 * RgfxWireLayout: the wire structs' sizes, asserted at the guest's own compile.
 *
 * reims_graphics_wire.h is generated from crates/reims-graphics/src/wire.rs and
 * drift-checked against it by `cargo xtask header --check`. wire.rs asserts every
 * struct's size in a `const _` block; this is the same wall on the C side, so a C
 * header that ever disagrees with the sizes the guest and host both rely on fails
 * the bundle's build here, not silently at run time on the wire.
 */

#include "reims_graphics_wire.h"

// Transport (stage 2).
static_assert(sizeof(RgfxRingHeader) == 256, "RgfxRingHeader");
static_assert(__builtin_offsetof(RgfxRingHeader, consumer) == 0x40, "ring consumer line");
static_assert(__builtin_offsetof(RgfxRingHeader, consumer_idle) == 0x80, "ring consumer_idle line");
static_assert(__builtin_offsetof(RgfxRingHeader, armed) == 0xc0, "ring armed line");
static_assert(sizeof(RgfxMsgHeader) == 16, "RgfxMsgHeader");

// Stage-3 object lifecycle and completion records.
static_assert(sizeof(RgfxObjectDestroy) == 24, "RgfxObjectDestroy");
static_assert(sizeof(RgfxObjectCreate) == 24, "RgfxObjectCreate");
static_assert(sizeof(RgfxSharedEventCreate) == 32, "RgfxSharedEventCreate");
static_assert(sizeof(RgfxQueueCreate) == 56, "RgfxQueueCreate");
static_assert(sizeof(RgfxHeapCreate) == 48, "RgfxHeapCreate");
static_assert(sizeof(RgfxHeapOverHostMemory) == 56, "RgfxHeapOverHostMemory");
static_assert(sizeof(RgfxBufferCreate) == 48, "RgfxBufferCreate");
static_assert(sizeof(RgfxBufferOverHostMemory) == 56, "RgfxBufferOverHostMemory");
static_assert(sizeof(RgfxBufferAtOffset) == 56, "RgfxBufferAtOffset");
static_assert(sizeof(RgfxCbComplete) == 48, "RgfxCbComplete");
static_assert(sizeof(RgfxEventSignalled) == 32, "RgfxEventSignalled");
static_assert(sizeof(RgfxTextureDescriptor) == 48, "RgfxTextureDescriptor");
static_assert(sizeof(RgfxTextureCreate) == 104, "RgfxTextureCreate");
static_assert(sizeof(RgfxTextureFromBuffer) == 104, "RgfxTextureFromBuffer");
static_assert(sizeof(RgfxSamplerCreate) == 72, "RgfxSamplerCreate");
static_assert(sizeof(RgfxStencilDescriptor) == 24, "RgfxStencilDescriptor");
static_assert(sizeof(RgfxDepthStencilCreate) == 80, "RgfxDepthStencilCreate");
static_assert(sizeof(RgfxLibraryCreate) == 32, "RgfxLibraryCreate");
static_assert(RGFX_LIBRARY_BYTES_OFFSET == sizeof(RgfxLibraryCreate), "library bytes offset");
static_assert(sizeof(RgfxDeviceInfo) == 376, "RgfxDeviceInfo");

// Every message size is a multiple of 8, as a ring message must be.
static_assert(sizeof(RgfxObjectDestroy) % 8 == 0, "RgfxObjectDestroy aligned");
static_assert(sizeof(RgfxQueueCreate) % 8 == 0, "RgfxQueueCreate aligned");
static_assert(sizeof(RgfxTextureCreate) % 8 == 0, "RgfxTextureCreate aligned");
static_assert(sizeof(RgfxDepthStencilCreate) % 8 == 0, "RgfxDepthStencilCreate aligned");
