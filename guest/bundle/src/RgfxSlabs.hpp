/*
 * RgfxSlabs: where the memory of the device's Shared and Managed buffers comes from
 * (docs/graphics-kext/design.md, "Shared and Managed buffers lie in slabs").
 *
 * A buffer takes no block of the kext's and no host object of its own memory: it takes a range
 * of a slab, a larger piece of memory this allocator holds, and lies there at the range's offset.
 * What a slab is and how the host sees it is its source's (RgfxSlabSource): a block of the device
 * region (RgfxRegionSlabSource), whose Shared slabs are each one placement heap over host memory
 * (HEAP_OVER_HOST_MEMORY) with every buffer a BUFFER_AT_OFFSET in it, and whose Managed slabs are
 * plain blocks with each buffer a BUFFER_OVER_HOST_MEMORY over its own range of the block, its CPU
 * copy; or a superpage of the process's own (RgfxPagesSlabSource), one heap over its guest RAM
 * (HEAP_OVER_PAGES) with its buffers placed in it the same way, or for one large Shared buffer
 * several superpages, the buffer a BUFFER_OVER_PAGES over them. So a create is a take from a
 * free list and one post on ring 0, and a free a give back: neither calls the kext, and the
 * region's block table holds one entry per slab.
 *
 * What the buffers answer, as both x86-64 Macs' GPUs (AMD 5300M, Intel 630) agree (measured with
 * tools/shmem's align and zero cases): a buffer takes whole 4 KiB pages (kGranule; allocatedSize
 * 4096 for a 256-byte buffer), its contents and GPU address start a page, and a new buffer reads
 * zeros, whatever its memory held before.
 *
 * THE CLASSES. Buffers share a slab only where their options are the same once the hazard
 * tracking is resolved (the caller resolves it): the host's placed buffer takes its heap's
 * storage, CPU cache and hazard tracking modes, and refuses another storage or cache mode.
 *
 * MANAGED OVER A COPY. Where the device took RGFX_FEATURE_MANAGED_BUFFER_SLABS (the sources'
 * `managedOverCopy`), a Managed buffer lies in the Shared class of its cache and hazard tracking
 * modes: its range is a Shared buffer's, made as any (BUFFER_AT_OFFSET in the slab's heap, or
 * BUFFER_OVER_PAGES over a larger slab of superpages), and the Managed buffer a
 * MANAGED_BUFFER_CREATE over it, its CPU copy. So a Managed buffer of any size lies where a Shared
 * one of its size does, the process's own superpages first, on every host. Its range goes back
 * once the copy's destroy follows the buffer's. Without the feature a Managed buffer has a class
 * of its own (below, the region's source).
 *
 * THE SIZES. A buffer of up to kSmallMax bytes takes a range of one of its class's slabs, first
 * fit over the slabs in the order they were made, so older slabs fill first and newer ones empty.
 * A class's slabs grow from kSlabFirst, doubling, to kSlabMax, and are never smaller than twice
 * the buffer that makes one. A larger buffer takes a slab of its own (dedicated), its length
 * rounded up to the source's unit; once freed, that slab is cached for a later buffer of the same
 * class that needs at least half of it, oldest let go first past a sixteenth of the source's
 * capacity, and let go after kCacheIdleMs unused.
 *
 * ZEROING. A freed range, and a slab the source did not zero, are dirty. A create zeroes only
 * the dirty part of its range, past the bytes its caller writes itself; the background thread
 * (start) zeroes dirty free bytes in pieces of kZeroPiece, so in a steady state a create finds
 * its range clean. A piece being zeroed is in no free run, so no create takes it meanwhile.
 *
 * LETTING GO. A class keeps up to two empty slabs, each for kCacheIdleMs at most; a third empty
 * one goes at once, and one left empty that long goes too, as does a cached slab past the cap or
 * idle, so a process at rest holds only the slabs its buffers lie in. A slab goes by its source
 * (its heap's OBJECT_DESTROY, then its block's release, whose BLOCK_RELEASE the kext waits for),
 * on the background thread, so a buffer's free never waits for the host; without the thread, at
 * once. Every buffer in a slab had its destroy posted before its range came back, so the heap's
 * destroy follows theirs.
 *
 * THE SOURCES. A new slab comes from the first source where one is set (setFirst), at its unit,
 * and from the region's source only where the first has no memory for it; a slab keeps the source
 * it came from, which makes its buffers and takes it back, and its cache's cap is a sixteenth of
 * that source's capacity.
 *
 * THE FALLBACKS, in order, where a buffer finds no room in its class's slabs: a new slab; then,
 * once the sources had no memory for one, the process's own empty and cached slabs let go and a
 * new slab asked again; then, for a small buffer, region slabs half as large down to the buffer's
 * own rounded size. Only where no source has memory even for that is the answer nil: the region
 * is full. (Where the kext's block source places blocks, WindowServer's asks may take its
 * reserve.) A slab whose heap the device never got (the kext's submit refused it) is nil at once.
 *
 * Thread-safe: Metal creates and frees buffers on the app's threads. Our own C++, free of
 * Metal's headers, so the serialiser tests build it as the bundle does.
 */

#ifndef RGFX_SLABS_HPP_
#define RGFX_SLABS_HPP_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

class RgfxRegion;
class RgfxSerializer;

/// Where slabs come from, and how the host sees them and the buffers in them.
class RgfxSlabSource {
public:
	/// A slab's memory: `size` bytes this process maps at `bytes`. `regionOffset` is where they
	/// lie in the device region, for a source of region blocks. `zeroed` is whether every byte
	/// reads zero as it comes.
	struct Memory {
		uint8_t *bytes = nullptr;
		uint64_t size = 0;
		uint64_t regionOffset = 0;
		bool zeroed = false;
		uint64_t blockSize = 0; // a pages block's size, for a source of the process's own pages
	};

	/// The host's object over a slab of a class whose buffers are placed in it (heaped): its
	/// object id and GPU address. Both 0 for a slab of a class with none.
	struct Heap {
		uint64_t objId = 0;
		uint64_t gpuAddress = 0;
	};

	/// A buffer made in a slab: its object id, GPU address, and whether its create picked that
	/// address, which then goes back after its destroy (a heaped buffer's is its heap's).
	struct Made {
		uint64_t objId = 0;
		uint64_t gpuAddress = 0;
		bool ownsAddress = false;
	};

	virtual ~RgfxSlabSource() = default;

	/// What a slab's size is a whole multiple of (a power of two).
	virtual uint64_t unit() const = 0;

	/// Whether this source makes a slab of `size` bytes (a multiple of unit()) for buffers of
	/// `options`: a first source that does not leaves it to the region's (RgfxSlabs::setFirst).
	virtual bool serves(uint32_t options, uint64_t size) const { return true; }

	/// The most bytes this source's slabs could ever hold, which sizes the cache of slabs.
	virtual uint64_t capacity() const = 0;

	/// Whether buffers of `options` lie in a heap over their slab (makeHeap), else each over
	/// its own range.
	virtual bool heaped(uint32_t options) const = 0;

	/// Whether the slab over `memory`, of a class of `options`, has a heap its buffers lie in
	/// (makeHeap), else each buffer over its own range: its class's answer (heaped) unless the
	/// source says otherwise of a slab.
	virtual bool heapedSlab(const Memory &memory, uint32_t options) const
	{
		return heaped(options);
	}

	/// The memory of a slab of `size` bytes (a multiple of unit()), or false where it has none.
	virtual bool take(uint64_t size, Memory *out) = 0;

	/// Gives a slab's memory back, once no buffer lies in it and its heap is dropped.
	virtual void give(const Memory &memory) = 0;

	/// The heap of a heaped class's slab over `memory`, of `options`; false where none was made.
	virtual bool makeHeap(const Memory &memory, uint32_t options, Heap *out) = 0;

	/// Drops a heap makeHeap made, once every buffer placed in it is destroyed.
	virtual void dropHeap(const Heap &heap) = 0;

	/// A buffer of `length` bytes and `options` at `offset` in the slab of `memory`, whose heap
	/// is `heap` (0s for a class with none); false where the device never got it.
	virtual bool makeBuffer(const Memory &memory, const Heap &heap, uint64_t offset,
	                        uint64_t length, uint32_t options, Made *out) = 0;

	/// The options of the class a buffer of `options` lies in (MANAGED OVER A COPY): the Shared
	/// class's for a Managed buffer of a source over copies, else `options`.
	virtual uint32_t classOptions(uint32_t options) const { return options; }

	/// Whether a buffer of `options` is a Managed buffer over a Shared copy (makeOverCopy).
	virtual bool overCopy(uint32_t options) const { return false; }

	/// The Managed buffer of `length` bytes and `options` whose CPU copy is the Shared buffer
	/// `copy` made in a slab of this source; false where the device never got it.
	virtual bool makeOverCopy(const Made &copy, uint64_t length, uint32_t options, Made *out)
	{
		return false;
	}

	/// Destroys the Shared buffer `copy` a Managed buffer lay over, once that buffer's destroy is
	/// posted, its address given back where its create picked it: `then` once the destroy is.
	virtual void dropCopy(const Made &copy, std::function<void()> then) { then(); }
};

/// Slabs of the device region: each a block of the kext's (RgfxRegion::allocate), in Metal's
/// 16 KiB pages, its bytes whatever the block held before. A Shared slab is a Placement heap over
/// host memory and its buffers BUFFER_AT_OFFSET there. A Managed buffer is, where the device took
/// RGFX_FEATURE_MANAGED_BUFFER_SLABS (`managedOverCopy`), a MANAGED_BUFFER_CREATE over a Shared
/// buffer of the Shared class (MANAGED OVER A COPY). Else Managed buffers have slabs of their own:
/// where the device took RGFX_FEATURE_MANAGED_AT_OFFSET (`managedAtOffset`), a Shared Placement
/// heap over host memory too, each buffer a BUFFER_MANAGED_AT_OFFSET at an address of its own whose
/// CPU copy the host places in the heap at its range; else a block whose buffers are each a
/// BUFFER_OVER_HOST_MEMORY over their own range, their CPU copy.
class RgfxRegionSlabSource : public RgfxSlabSource {
public:
	RgfxRegionSlabSource(RgfxRegion &region, RgfxSerializer &serializer,
	                     bool managedAtOffset = false, bool managedOverCopy = false);

	uint64_t unit() const override { return kUnit; }
	uint64_t capacity() const override;
	bool heaped(uint32_t options) const override;
	bool take(uint64_t size, Memory *out) override;
	void give(const Memory &memory) override;
	bool makeHeap(const Memory &memory, uint32_t options, Heap *out) override;
	void dropHeap(const Heap &heap) override;
	bool makeBuffer(const Memory &memory, const Heap &heap, uint64_t offset, uint64_t length,
	                uint32_t options, Made *out) override;
	uint32_t classOptions(uint32_t options) const override;
	bool overCopy(uint32_t options) const override;
	bool makeOverCopy(const Made &copy, uint64_t length, uint32_t options, Made *out) override;
	void dropCopy(const Made &copy, std::function<void()> then) override;

	/// Metal's page, which a heap over host memory is made of.
	static constexpr uint64_t kUnit = 0x4000;

protected:
	RgfxRegion &m_region;
	RgfxSerializer &m_serializer;
	bool m_managedAtOffset;
	bool m_managedOverCopy;
};

/// The process's own memory a source of pages slabs takes from the kernel: calls the device wires
/// (RgfxMetalDevice: mach_vm_allocate) and the tests fake.
struct RgfxPagesMemory {
	void *context = nullptr;
	/// `size` bytes of 2 MiB superpages, zeroed and wired: 0 (KERN_SUCCESS), or the kernel's
	/// refusal, KERN_NO_SPACE where no contiguous 2 MiB is free and KERN_RESOURCE_SHORTAGE past the
	/// user wire budget.
	int (*superpages)(void *context, uint64_t size, uint8_t **bytes) = nullptr;
	/// Gives back what superpages made.
	void (*unmap)(void *context, uint8_t *bytes, uint64_t size) = nullptr;
};

/// Slabs of the process's own pages: 2 MiB superpages wired into a pages block of the kext's
/// (RgfxRegion::acquirePages; RGFX_FEATURE_PAGES_BUFFERS), so their buffers' memory is the guest's
/// RAM, not the device region's, for the classes whose buffers lie in heaps. A slab of one
/// superpage, one run of guest RAM, is a heap the host makes over that RAM (HEAP_OVER_PAGES;
/// RGFX_FEATURE_HEAP_OVER_PAGES), its buffers placed there as in a region slab's heap. A larger
/// slab is more than one run, and holds one dedicated Shared buffer, a BUFFER_OVER_PAGES over the
/// block's view, or a Managed buffer over such a Shared copy (MANAGED OVER A COPY); a larger
/// Managed one of a class of its own, and a slab where the kernel has no superpages to give or the
/// kext no pages block, are the region's. A slab goes with its heap's destroy, then its block's
/// release (the kext's Free, which waits for the device's answer), then its pages.
class RgfxPagesSlabSource : public RgfxRegionSlabSource {
public:
	RgfxPagesSlabSource(RgfxRegion &region, RgfxSerializer &serializer,
	                    const RgfxPagesMemory &memory, uint64_t capacity,
	                    bool managedAtOffset = false, bool managedOverCopy = false);

	uint64_t unit() const override { return kUnit; }
	bool serves(uint32_t options, uint64_t size) const override;
	bool heapedSlab(const Memory &memory, uint32_t options) const override;
	uint64_t capacity() const override { return m_capacity; }
	bool take(uint64_t size, Memory *out) override;
	void give(const Memory &memory) override;
	bool makeHeap(const Memory &memory, uint32_t options, Heap *out) override;
	bool makeBuffer(const Memory &memory, const Heap &heap, uint64_t offset, uint64_t length,
	                uint32_t options, Made *out) override;

	/// A superpage, and the size of a slab with a heap.
	static constexpr uint64_t kUnit = 2ull << 20;

private:
	RgfxPagesMemory m_memory;
	uint64_t m_capacity;
};

class RgfxSlabs {
public:
	/// A range of a slab a buffer lies in: `size` bytes (allocatedSize) at `offset` of its slab,
	/// at `contents` here. `slab` names it for give().
	struct Range {
		const void *slab = nullptr;
		uint64_t offset = 0;
		uint64_t size = 0;
		uint8_t *contents = nullptr;
	};

	/// What the allocator holds, for the tests and diagnostics.
	struct Stats {
		uint64_t slabs = 0;        // every slab held, dedicated ones and cached ones among them
		uint64_t slabBytes = 0;    // their bytes
		uint64_t emptySlabs = 0;   // small slabs with no buffer in them
		uint64_t cachedSlabs = 0;  // dedicated slabs with no buffer, kept for reuse
		uint64_t cachedBytes = 0;
		uint64_t dirtyBytes = 0;   // free bytes not yet zeroed
		uint64_t dooming = 0;      // slabs let go whose drop has not run yet
		uint64_t usedBytes = 0;    // bytes taken by buffers
		uint64_t firstSlabs = 0;   // of the slabs, those the first source made
	};

	/// Slabs of `source`: the region's, which RgfxRegion::setReclaim asks to give room back.
	explicit RgfxSlabs(RgfxSlabSource &source);

	/// A source asked before the region's for every new slab (THE SOURCES), or null for none. Set
	/// before any take.
	void setFirst(RgfxSlabSource *first);

	/// Stops the background thread. Gives nothing back: the slabs' memory and heaps go with the
	/// process's context, as its other blocks do.
	~RgfxSlabs();

	RgfxSlabs(const RgfxSlabs &) = delete;
	RgfxSlabs &operator=(const RgfxSlabs &) = delete;

	/// Starts the background thread that zeroes freed bytes and lets slabs go. False where it
	/// could not be made: creates then zero what they take, and frees let slabs go themselves.
	bool start();

	/// A range for a buffer of `length` bytes and `options` (hazard tracking resolved), zeroed
	/// past its first `keep` bytes, which the caller writes. False where no slab can hold it.
	bool take(uint64_t length, uint32_t options, uint64_t keep, Range *out);

	/// The buffer of `length` bytes and `options` over `range` (RgfxSlabSource::makeBuffer): a
	/// Managed one of a source over copies over a Shared buffer made there first, which give()
	/// destroys (MANAGED OVER A COPY), where the Managed buffer was not made too.
	bool makeBuffer(const Range &range, uint64_t length, uint32_t options,
	                RgfxSlabSource::Made *out);

	/// The clock the idle times are read from, in ms: `now(context)`, for the tests; the steady
	/// clock where none is set. Set before any take.
	typedef uint64_t (*Clock)(void *context);
	void setClock(Clock now, void *context);

	/// Gives `range` back, once its buffer's destroy is posted: after the destroy of the Shared
	/// copy a Managed buffer over it lay over, where it has one.
	void give(const Range &range);

	/// Lets every empty and cached slab go, with the doomed ones, where the source had no room
	/// for another block of the process's (RgfxRegion::setReclaim): true where one went. Called
	/// without the lock.
	bool reclaim();

	/// Does one piece of the background's work (a drop, a piece zeroed, an idle cached slab let
	/// go): false where there was none. The thread runs it; the tests call it.
	bool step();

	Stats stats();

	/// A buffer's page: what its range is a multiple of, and its offset in its slab.
	static constexpr uint64_t kGranule = 0x1000;
	/// The largest buffer a class's slabs hold; a larger one takes a slab of its own.
	static constexpr uint64_t kSmallMax = 1ull << 20;
	/// A class's first slab, and its largest.
	static constexpr uint64_t kSlabFirst = 128ull << 10;
	static constexpr uint64_t kSlabMax = 2ull << 20;
	/// The most bytes the background zeroes at a time.
	static constexpr uint64_t kZeroPiece = 256ull << 10;
	/// How long a cached slab is kept unused.
	static constexpr uint64_t kCacheIdleMs = 2000;
	/// The empty slabs a class keeps.
	static constexpr unsigned kEmptyKept = 2;

private:
	/// Runs of bytes, by start: none overlap or touch, and `bytes` is their sum.
	struct Runs {
		std::map<uint64_t, uint64_t> runs;
		uint64_t bytes = 0;
		void add(uint64_t at, uint64_t length);
		bool takeFirstFit(uint64_t length, uint64_t *at);
		bool takeAt(uint64_t at, uint64_t length);
		/// Removes the bytes of [at, at + length) it holds, and appends them to `out`.
		void cut(uint64_t at, uint64_t length, std::vector<std::pair<uint64_t, uint64_t>> *out);
	};

	struct Slab {
		RgfxSlabSource *source = nullptr; // the source it came from, which takes it back
		RgfxSlabSource::Memory memory;
		RgfxSlabSource::Heap heap;
		uint32_t options = 0;
		bool dedicated = false;
		uint64_t serial = 0;       // the order slabs were made in
		Runs free;                 // bytes no buffer takes and the background is not zeroing
		Runs dirty;                // of the free bytes, those that may not read zero
		uint64_t used = 0;         // bytes taken by buffers
		uint32_t zeroing = 0;      // pieces the background is zeroing
		bool doomed = false;       // let go: no buffer takes from it again
		uint64_t idleSince = 0;    // the time, in ms, a cached slab was cached, an empty one emptied
		// The Shared copies Managed buffers lie over, by their ranges' offsets.
		std::map<uint64_t, RgfxSlabSource::Made> copies;
	};

	struct BySerial {
		bool operator()(const Slab *a, const Slab *b) const { return a->serial < b->serial; }
	};

	struct Class {
		std::vector<Slab *> slabs;           // its slabs of small buffers
		std::set<Slab *, BySerial> open;     // of them, those with free bytes
		uint64_t nextSize = kSlabFirst;
		unsigned empty = 0;
	};

	/// A new slab of `size` bytes for buffers of `options`, through `source`; null where it gave
	/// none, with `*refused` set where it gave the memory but the device never got its heap.
	/// Called without the lock.
	Slab *makeSlab(RgfxSlabSource &source, uint64_t size, uint32_t options, bool dedicated,
	               bool *refused);
	/// The bytes of the cached slabs of `source`. Caller holds the lock.
	uint64_t cachedBytesOf(const RgfxSlabSource *source) const;
	/// Takes `size` bytes of `slab` at its first fit, zeroing past `keep` what was dirty: the
	/// pieces to zero go in `*zero`. Caller holds the lock.
	bool takeIn(Slab *slab, uint64_t size, uint64_t keep,
	            std::vector<std::pair<uint64_t, uint64_t>> *zero, uint64_t *at);
	/// Lets `slab` go: out of its class or the cache, onto the doomed list. Caller holds the lock.
	void doom(Slab *slab);
	/// Removes `slab` from the cache. Caller holds the lock.
	void uncache(Slab *slab);
	/// Drops the doomed slabs no piece is being zeroed in, and lets every empty and cached slab
	/// go first where `evict`: the process's own room given back before a create is refused.
	/// Answers how many it dropped.
	size_t dropDoomed(bool evict);
	/// Gives a slab's heap and memory back, and frees it. Called without the lock.
	void drop(Slab *slab);
	/// Gives `range` back to its slab, its buffers' destroys all posted.
	void giveBack(const Range &range);
	/// The class of `options`. Caller holds the lock.
	Class &classOf(uint32_t options);
	/// Lets go the empty slabs of small buffers left empty kCacheIdleMs by `now`, and sets when
	/// the next one is due (m_emptyDue, 0 for none). Caller holds the lock.
	void decayEmpty(uint64_t now);
	/// How long the thread may wait before a cached or empty slab is due to go, in ms; -1 where
	/// none is. Caller holds the lock.
	int64_t idleWait(uint64_t now) const;
	void run();
	uint64_t nowMs() const;

	RgfxSlabSource &m_source;
	RgfxSlabSource *m_first = nullptr;
	std::mutex m_lock;
	std::condition_variable m_wake;
	std::map<uint32_t, Class> m_classes;
	std::set<Slab *> m_all;
	std::vector<Slab *> m_cached;          // dedicated slabs with no buffer, oldest first
	uint64_t m_cachedBytes = 0;
	uint64_t m_emptyDue = 0;               // when an empty small slab is next due to go; 0 for none
	Clock m_clock = nullptr;
	void *m_clockContext = nullptr;
	std::vector<Slab *> m_doomed;
	std::set<Slab *, BySerial> m_dirtySlabs; // slabs with dirty free bytes
	uint64_t m_nextSerial = 1;
	std::thread m_thread;
	bool m_running = false;
	bool m_stop = false;
};

#endif // RGFX_SLABS_HPP_
