/*
 * RgfxObjectModel: the guest bundle's object model for the reims-graphics lane
 * (docs/graphics-kext/design.md, "Stage 3 wire protocol: the Metal-call stream",
 * Decided 6 and 7).
 *
 * Every guest Metal object the bundle wraps carries a 64-bit ObjId the guest
 * picks, unique per device; the host keeps the ObjId -> Platform-handle table.
 * The guest also picks each buffer/texture/heap GPU address and each texture
 * resource id, and the host pins them in the platform's own address space, so a
 * create needs no round trip. Host device addresses never reach the guest.
 *
 * RgfxIdSpace hands out those three number lines for one device (one guest
 * process = one context, Decided 7). The lines are monotonic and never reuse a
 * value while the device lives: host handles are refcounted, and reuse would
 * alias a live handle. The precise per-context partitioning of the address and
 * id spaces, and sub-allocation of GPU addresses inside host memory blocks,
 * land with the region/block work; this hands out a flat, aligned bump range
 * that the host pins as given.
 *
 * This is our own code: it knows nothing of Apple's Metal SPI. The Metal device
 * shell (built later) allocates an ObjId per object it creates and stores it on
 * the wrapper; here we only mint the numbers.
 */

#ifndef RGFX_OBJECT_MODEL_HPP_
#define RGFX_OBJECT_MODEL_HPP_

#include <atomic>
#include <cstdint>

/// A guest-picked object id, unique per device. 0 names no object.
typedef uint64_t RgfxObjId;

/// The id that names no object, used where a message's field is optional (a
/// device-placed texture's `heap_obj_id`, for one).
static const RgfxObjId RGFX_OBJ_NONE = 0;

/// The three number lines one device's context hands out: object ids, GPU
/// addresses and texture resource ids. Every method is safe to call from any
/// thread (Metal creates objects on the app's threads).
class RgfxIdSpace {
public:
	/// A space whose GPU addresses start at `address_base`. Object ids and
	/// resource ids start at 1 (0 is [`RGFX_OBJ_NONE`]).
	explicit RgfxIdSpace(uint64_t address_base = kDefaultAddressBase);

	/// The next object id, never 0 and never reused.
	RgfxObjId allocObjId();

	/// The next texture resource id, never 0 and never reused.
	uint64_t allocResourceId();

	/// A GPU address range of `size` bytes aligned up to `align` (a power of
	/// two, at least 1). The guest owns the range; the host pins it.
	uint64_t allocAddress(uint64_t size, uint64_t align);

	/// Where this space's GPU addresses begin by default: above the low 4 GiB,
	/// clear of any identity-mapped low range and of zero.
	static const uint64_t kDefaultAddressBase = 0x1'0000'0000ull;

private:
	std::atomic<uint64_t> m_nextObjId;
	std::atomic<uint64_t> m_nextResourceId;
	std::atomic<uint64_t> m_nextAddress;
};

#endif // RGFX_OBJECT_MODEL_HPP_
