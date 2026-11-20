// RgfxSurfaceRefill: when a cached surface's device cache is filled from its pages again at the
// CPU generation it holds (RGFX_SURFACE_USE_REFILL, where the host took
// RGFX_FEATURE_SURFACE_REFILL).
//
// A read-write CPU lock of a cached surface moves its CPU generation (RgfxSurfaceCaches.hpp), and
// the host fills the cache from the pages before the first use of a newer generation. The fill
// can come while the CPU still writes under that lock: a commit that names the generation while
// the lock is held, or a fill the host makes for its own reasons at that generation (the first
// fill a block owes, a block's pages named again). The pixels the CPU writes after it would then
// never reach the cache, which holds the generation, until the next read-write lock. So each
// generation carries a refill record in the block's header (kRgfxSurfaceRefillOffset), one
// 64-bit word every holder of the surface reads and writes:
//   - the kext writes it as it caches the block (generation 0) and at each read-write lock, as it
//     moves the generation: owed, at the surface's unlock seed then, which the lock's unlock
//     moves (IOSurface moves the lock seed at a read-write lock and the unlock seed at its
//     unlock: RgfxUserClientABI.hpp, kRgfxAccelMethodSurfaceSeeds). The kext moves the
//     generation inside the lock, before IOSurface moves the lock seed, so the unlock seed it
//     records is the one before the lock's.
//   - the first use that names the generation once the unlock seed has moved past the record's
//     asks the host to fill the cache again, and settles the record (rgfxRefillStep). The host
//     fills it only where it holds that generation: one that holds an older one fills it anyway,
//     once, so a generation the cache was never filled at mid-write costs nothing more.
// A use that finds the unlock seed where the record has it, the lock held still or being taken,
// asks nothing and leaves the record owed. A lock held again as the record is settled (a lock
// seed past the unlock seed) leaves it owed at the unlock seed then. A GPU write moves both seeds
// too, so the first use after one asks again: the host keeps a cache a command buffer wrote
// since its last copy into the pages (its Refill::Kept), which a fill would lose.
//
// A record of an older generation than the use's (the kext's lost to another holder's write, or
// none written) is made as a commit names the generation first, as before the kext wrote one:
// owed where a lock is held then, else unsure, and the next use asks once the unlock seed moves,
// or settles it. A record of a newer generation (a lock moved it after the use read the
// generation) is left as it is. Without the kext's seeds a record settles at once. Each write is
// a compare-and-swap of the word as it was read, so a holder never writes over a newer record.
//
// Header-only and free of the kernel's headers: the kext and the bundle build it, and the kext's
// native test (tests/surface_caches_test.cpp) builds it as they do.
//
// Our own code.
#ifndef RGFX_SURFACE_REFILL_HPP_
#define RGFX_SURFACE_REFILL_HPP_

#include <stdint.h>

// Where a cached block's header holds its refill record: the bundle's RgfxSurfaceHeader's
// unlockSeed and writeHeld, which a copy's header uses for its synced write instead.
enum {
	kRgfxSurfaceRefillOffset = 24,
};

// A refill record's state: settled, nothing is asked again at its generation; owed, the first
// use once the unlock seed moves past the record's asks a refill; unsure, the first commit that
// named the generation saw no lock held, which may yet be one being taken, so the next asks
// once more.
enum RgfxRefillState : uint32_t {
	kRgfxRefillSettled = 0,
	kRgfxRefillOwed = 1,
	kRgfxRefillUnsure = 2,
};

// A refill record: the low 30 bits of its generation, its state and the unlock seed it was
// recorded at. In the header's word, the unlock seed is the low 32 bits and the generation and
// state the high 32, the generation above the state's two bits.
struct RgfxRefillRecord {
	uint32_t gen;
	RgfxRefillState state;
	uint32_t unlockSeed;
};

constexpr uint32_t kRgfxRefillGenMask = (1u << 30) - 1;

inline uint64_t rgfxRefillWord(uint64_t gen, RgfxRefillState state, uint32_t unlockSeed)
{
	const uint32_t high = ((static_cast<uint32_t>(gen) & kRgfxRefillGenMask) << 2) | state;
	return static_cast<uint64_t>(high) << 32 | unlockSeed;
}

inline RgfxRefillRecord rgfxRefillRead(uint64_t word)
{
	const uint32_t high = static_cast<uint32_t>(word >> 32);
	return {high >> 2, static_cast<RgfxRefillState>(high & 3), static_cast<uint32_t>(word)};
}

// What a use of the surface at generation `gen` does with the record `word` read from the
// header: whether it asks the host to fill the cache again, and the word it writes in its place
// where `write`. `seeds` where the kext answered the surface's lock and unlock seeds now;
// `midWrite` where the use names the generation while a CPU may hold the surface locked for
// writing since a fill could be made, the cache's fill then possibly mid-write.
struct RgfxRefillStep {
	bool refill = false;
	bool write = false;
	bool midWrite = false;
	uint64_t word = 0;
};

// Whether a use at `gen` asks nothing of the kext: the record is its generation's, settled, or a
// newer generation's.
inline bool rgfxRefillSettled(uint64_t word, uint64_t gen)
{
	const RgfxRefillRecord r = rgfxRefillRead(word);
	const uint32_t g = static_cast<uint32_t>(gen) & kRgfxRefillGenMask;
	if (r.gen == g)
		return r.state == kRgfxRefillSettled;
	// Newer where it is less than half the generations' range ahead.
	return ((r.gen - g) & kRgfxRefillGenMask) < (1u << 29);
}

inline RgfxRefillStep rgfxRefillStep(uint64_t word, uint64_t gen, bool seeds, uint32_t lockSeed,
                                     uint32_t unlockSeed)
{
	RgfxRefillStep s;
	if (rgfxRefillSettled(word, gen))
		return s;
	const RgfxRefillRecord r = rgfxRefillRead(word);
	if (!seeds) {
		s.write = true;
		s.word = rgfxRefillWord(gen, kRgfxRefillSettled, 0);
		return s;
	}
	const bool held = lockSeed != unlockSeed;
	if (r.gen != (static_cast<uint32_t>(gen) & kRgfxRefillGenMask)) {
		// No record of this generation: named first now.
		s.write = true;
		s.midWrite = held;
		s.word = rgfxRefillWord(gen, held ? kRgfxRefillOwed : kRgfxRefillUnsure, unlockSeed);
		return s;
	}
	const bool released = unlockSeed != r.unlockSeed;
	if (!released && r.state == kRgfxRefillOwed) {
		s.midWrite = true;
		return s;
	}
	s.refill = released;
	s.write = true;
	s.word = rgfxRefillWord(gen, held ? kRgfxRefillOwed : kRgfxRefillSettled, unlockSeed);
	return s;
}

#endif // RGFX_SURFACE_REFILL_HPP_
