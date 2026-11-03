/*
 * RgfxIOSurface: see RgfxIOSurface.h. Our own ObjC++.
 */

#import "RgfxIOSurface.h"

#import "RgfxBuffer.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <dispatch/dispatch.h>
#include <map>
#include <mutex>
#include <os/log.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <tuple>
#include <utility>

#include "RgfxEncoder.hpp"
#include "RgfxMemoryShadow.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"
#include "../../kext/src/RgfxUserClientABI.hpp"

namespace {

/// The row pitch of a copy: a multiple of every linear-texture row alignment a host
/// GPU asks for.
const uint64_t kShadowRowAlign = 256;

/// The guest's page, which the kext wires a surface's allocation in whole of for its GPU view in
/// place.
const uint64_t kSurfacePage = 0x1000;

uint64_t alignUp(uint64_t v, uint64_t a)
{
	return (v + a - 1) & ~(a - 1);
}

/// The shape of one plane of a surface, as its CPU mapping lays it out.
struct PlaneShape {
	uint64_t rowBytes;        // bytes from one row of elements to the next
	uint64_t rows;            // rows of elements
	uint64_t bytesPerElement; // bytes of one element
	uint64_t elementWidth;    // pixels across one element
	uint64_t elementHeight;   // pixels down one element
};

/// The shape of plane `plane` of `s`; a surface of no planes has the one plane 0.
/// False when the surface has no such plane.
bool planeShape(IOSurfaceRef s, size_t plane, PlaneShape *out)
{
	size_t planes = IOSurfaceGetPlaneCount(s);
	size_t height;
	if (planes == 0) {
		if (plane != 0) {
			return false;
		}
		out->rowBytes = IOSurfaceGetBytesPerRow(s);
		height = IOSurfaceGetHeight(s);
		out->bytesPerElement = IOSurfaceGetBytesPerElement(s);
		out->elementWidth = IOSurfaceGetElementWidth(s);
		out->elementHeight = IOSurfaceGetElementHeight(s);
	} else {
		if (plane >= planes) {
			return false;
		}
		out->rowBytes = IOSurfaceGetBytesPerRowOfPlane(s, plane);
		height = IOSurfaceGetHeightOfPlane(s, plane);
		out->bytesPerElement = IOSurfaceGetBytesPerElementOfPlane(s, plane);
		out->elementWidth = IOSurfaceGetElementWidthOfPlane(s, plane);
		out->elementHeight = IOSurfaceGetElementHeightOfPlane(s, plane);
	}
	uint64_t eh = std::max<uint64_t>(out->elementHeight, 1);
	out->rows = (height + eh - 1) / eh;
	return out->rowBytes != 0 && out->rows != 0;
}

/// The CPU address of plane `plane` of `s` in this process (null when unmapped).
uint8_t *planeBase(IOSurfaceRef s, size_t plane)
{
	void *p = IOSurfaceGetPlaneCount(s) == 0 ? IOSurfaceGetBaseAddress(s)
	                                         : IOSurfaceGetBaseAddressOfPlane(s, plane);
	return static_cast<uint8_t *>(p);
}

/// One plane of a surface's GPU view: its shape on the CPU, and where its rows lie in the view (in
/// a copy, both kShadowRowAlign multiples).
struct ShadowPlane {
	PlaneShape shape = {};
	uint64_t pitch = 0;  // the view's row pitch
	uint64_t offset = 0; // the plane's first byte in the view
};

/// printf into a std::string.
std::string format(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
std::string format(const char *fmt, ...)
{
	char line[512];
	va_list ap;
	va_start(ap, fmt);
	int n = vsnprintf(line, sizeof(line), fmt, ap);
	va_end(ap);
	return n > 0 ? std::string(line) : std::string();
}

/// The planes of `s` where its own memory holds them, for its GPU view in place: each at its
/// offset from the surface's base (CoreVideo's planar buffers start plane 0 past a header), with
/// its own rows. False, saying why in `*why`, where the host's textures cannot lie over them: the
/// base is not a page's, a plane's offset or rows are not a multiple of `align` (0: none known),
/// or a plane passes the allocation.
bool placeInPlace(IOSurfaceRef s, std::vector<ShadowPlane> *planes, uint64_t align,
                  std::string *why)
{
	uintptr_t base = reinterpret_cast<uintptr_t>(IOSurfaceGetBaseAddress(s));
	if (align == 0) {
		*why = "the host's linear texture alignment is not known";
		return false;
	}
	if (base == 0 || base % kSurfacePage != 0) {
		*why = format("its base 0x%lx is not a page's", static_cast<unsigned long>(base));
		return false;
	}
	uint64_t alloc = IOSurfaceGetAllocSize(s);
	for (size_t k = 0; k < planes->size(); k++) {
		ShadowPlane &p = (*planes)[k];
		uintptr_t at = reinterpret_cast<uintptr_t>(planeBase(s, k));
		if (at < base) {
			*why = format("plane %zu lies before its base", k);
			return false;
		}
		p.offset = at - base;
		p.pitch = p.shape.rowBytes;
		if (p.offset % align != 0 || p.pitch % align != 0) {
			*why = format("plane %zu's offset %llu or rows of %llu bytes are not a multiple of "
			              "the host's %llu",
			              k, (unsigned long long)p.offset, (unsigned long long)p.pitch,
			              (unsigned long long)align);
			return false;
		}
		if (p.offset > alloc || p.pitch * p.shape.rows > alloc - p.offset) {
			*why = format("plane %zu (%llu bytes at %llu) passes its %llu-byte allocation", k,
			              (unsigned long long)(p.pitch * p.shape.rows),
			              (unsigned long long)p.offset, (unsigned long long)alloc);
			return false;
		}
	}
	return true;
}

/// The surface's facts for a log line: its id, OSType, size, allocation and each plane's size
/// and rows.
std::string describe(IOSurfaceRef s)
{
	if (s == nullptr) {
		return "no IOSurface";
	}
	uint32_t f = IOSurfaceGetPixelFormat(s);
	char fourcc[5] = {char(f >> 24), char(f >> 16), char(f >> 8), char(f), 0};
	bool printable = true;
	for (int i = 0; i < 4; i++)
		printable = printable && fourcc[i] >= 0x20 && fourcc[i] < 0x7f;
	std::string out =
	    format("IOSurface %u ", IOSurfaceGetID(s)) +
	    (printable ? format("'%s'", fourcc) : format("0x%08x", f)) +
	    format(" %zux%zu, %zu bytes, %zu planes", IOSurfaceGetWidth(s), IOSurfaceGetHeight(s),
	           IOSurfaceGetAllocSize(s), IOSurfaceGetPlaneCount(s));
	size_t planes = std::max<size_t>(IOSurfaceGetPlaneCount(s), 1);
	for (size_t k = 0; k < planes && k < 4; k++) {
		PlaneShape shape = {};
		if (planeShape(s, k, &shape)) {
			out += format("%s plane %zu: %llu rows of %llu bytes", k == 0 ? ";" : ",", k,
			              (unsigned long long)shape.rows, (unsigned long long)shape.rowBytes);
		}
	}
	return out;
}

/// The textures over surfaces answered nil, by surface and cause: how many, and the line logged
/// for the first (rgfxIOSurfaceNilLog). A surface is known by its id, its object, its OSType and
/// its allocation together, since IOSurface gives a freed surface's id to the next.
struct NilLog {
	using Key = std::tuple<uint32_t, uintptr_t, uint32_t, size_t, std::string>;
	std::mutex lock;
	std::map<Key, std::pair<uint64_t, std::string>> seen;

	static Key key(IOSurfaceRef s, const char *cause)
	{
		if (s == nullptr) {
			return Key(0, 0, 0, 0, cause);
		}
		return Key(IOSurfaceGetID(s), reinterpret_cast<uintptr_t>(s), IOSurfaceGetPixelFormat(s),
		           IOSurfaceGetAllocSize(s), cause);
	}
};

NilLog &nilLog()
{
	static NilLog *l = new NilLog(); // never destroyed: textures are made at any time
	return *l;
}

/// newTextureWithDescriptor:iosurface:plane: answers nil for `surface`, as Metal does a texture
/// it cannot make, for `cause`, which `detail` says more of: logged once per surface and cause,
/// so a texture asked for every frame logs once.
void logNoTexture(IOSurfaceRef surface, const char *cause, const std::string &detail)
{
	std::string line = describe(surface) + ": no texture (" + cause + "): " + detail;
	NilLog &l = nilLog();
	{
		std::lock_guard<std::mutex> g(l.lock);
		// Bounded: a long session's surfaces come and go.
		if (l.seen.size() >= 4096) {
			l.seen.clear();
		}
		auto &entry = l.seen[NilLog::key(surface, cause)];
		if (entry.first++ != 0) {
			return;
		}
		entry.second = line;
	}
	NSLog(@"RGFX: %s", line.c_str());
}

/// The surface's YCbCr matrix as one of RGFX_YCBCR_MATRIX_*. CoreVideo keeps a pixel buffer's
/// propagated kCVImageBufferYCbCrMatrixKey attachment on its surface as the value
/// "IOSurfaceYCbCrMatrix", the attachment's string (measured on the M3 Max under macOS 27.0.1
/// with IOSurfaceCopyAllValues, for each of CoreVideo's six matrix constants). Read from the
/// surface itself: a CVPixelBuffer made over the surface reads it back too, but holds the
/// surface in use (IOSurfaceIsInUse, a use count of 1) while it lives, which Metal does not.
uint32_t ycbcrMatrix(IOSurfaceRef s)
{
	CFTypeRef value = IOSurfaceCopyValue(s, CFSTR("IOSurfaceYCbCrMatrix"));
	if (value == nullptr) {
		return RGFX_YCBCR_MATRIX_NONE;
	}
	static const struct {
		CFStringRef name;
		uint32_t code;
	} kMatrices[] = {
	    {CFSTR("ITU_R_601_4"), RGFX_YCBCR_MATRIX_ITU_R_601_4},
	    {CFSTR("ITU_R_709_2"), RGFX_YCBCR_MATRIX_ITU_R_709_2},
	    {CFSTR("ITU_R_2020"), RGFX_YCBCR_MATRIX_ITU_R_2020},
	    {CFSTR("SMPTE_240M_1995"), RGFX_YCBCR_MATRIX_SMPTE_240M_1995},
	    {CFSTR("DCI_P3"), RGFX_YCBCR_MATRIX_DCI_P3},
	    {CFSTR("P3_D65"), RGFX_YCBCR_MATRIX_P3_D65},
	};
	uint32_t code = RGFX_YCBCR_MATRIX_OTHER;
	if (CFGetTypeID(value) == CFStringGetTypeID()) {
		for (const auto &m : kMatrices) {
			if (CFEqual(value, m.name)) {
				code = m.code;
				break;
			}
		}
	}
	CFRelease(value);
	return code;
}

/// The process's copies between surfaces' pages and the GPU's view of them, by direction: a
/// diagnostic, an os_log line (info) at most every kCopyReportNs while copies run.
struct CopyCount {
	std::atomic<uint64_t> count{0};
	std::atomic<uint64_t> bytes{0};
	std::atomic<uint64_t> ns{0};
};

CopyCount gUploads;    // pages -> the GPU's view
CopyCount gWriteBacks; // the GPU's view -> pages

/// The GPU views the process made of surfaces: in place, or copies, by why.
std::atomic<uint64_t> gInPlace{0};
std::atomic<uint64_t> gCached{0};            // of them, cached on the host (their blocks')

/// The views alive in the process whose blocks are cached: a command buffer looks for its uses of
/// cached surfaces only while one lives (rgfxIOSurfaceNoteUse).
std::atomic<uint64_t> gCachedLive{0};
/// The process names its command buffers' order over IOSurfaces (rgfxIOSurfaceSetOrder): every
/// surface a buffer uses is noted, and its commit names the buffers of other queues it runs after.
std::atomic<bool> gSurfaceOrder{false};
/// The RgfxAskUncachedScopes alive on this thread.
thread_local unsigned tAskUncached = 0;
std::atomic<uint64_t> gCopiedAlignment{0};   // a plane where no host texture lies over it
std::atomic<uint64_t> gCopiedUnsupported{0}; // no view in place on the device, or of its pages
std::atomic<uint64_t> gCopiedRefused{0};     // the kext gave no block in place

const uint64_t kCopyReportNs = 10ull * 1000 * 1000 * 1000;

uint64_t nowNs()
{
	return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

/// Logs the copies and the views made since the last report, once kCopyReportNs has passed
/// since it.
void reportCopies(uint64_t now)
{
	static std::mutex lock;
	static uint64_t last = 0, upCount = 0, upBytes = 0, upNs = 0, wbCount = 0, wbBytes = 0,
	                wbNs = 0, inPlace = 0, cached = 0, forAlignment = 0, unsupported = 0,
	                refused = 0;
	std::unique_lock<std::mutex> g(lock, std::try_to_lock);
	if (!g.owns_lock()) {
		return;
	}
	if (last == 0) {
		last = now;
		return;
	}
	if (now - last < kCopyReportNs) {
		return;
	}
	uint64_t uc = gUploads.count, ub = gUploads.bytes, un = gUploads.ns;
	uint64_t wc = gWriteBacks.count, wb = gWriteBacks.bytes, wn = gWriteBacks.ns;
	uint64_t ip = gInPlace, cc = gCached, ca = gCopiedAlignment, cu = gCopiedUnsupported,
	         cr = gCopiedRefused;
	// The process's id and name are the log's own.
	os_log_info(OS_LOG_DEFAULT,
	            "RGFX: surface copies, %.1f s: %llu uploads %.1f MiB %.2f ms, %llu write-backs "
	            "%.1f MiB %.2f ms; views made: %llu in place (%llu cached), copies %llu for "
	            "alignment, %llu unsupported, %llu refused by the kext",
	            (now - last) / 1e9, uc - upCount, (ub - upBytes) / 1048576.0, (un - upNs) / 1e6,
	            wc - wbCount, (wb - wbBytes) / 1048576.0, (wn - wbNs) / 1e6, ip - inPlace,
	            cc - cached, ca - forAlignment, cu - unsupported, cr - refused);
	last = now;
	upCount = uc, upBytes = ub, upNs = un, wbCount = wc, wbBytes = wb, wbNs = wn;
	inPlace = ip, cached = cc, forAlignment = ca, unsupported = cu, refused = cr;
}

/// Counts a copy of `bytes` that started at `start`.
void countCopy(CopyCount &c, uint64_t bytes, uint64_t start)
{
	uint64_t now = nowNs();
	c.count.fetch_add(1, std::memory_order_relaxed);
	c.bytes.fetch_add(bytes, std::memory_order_relaxed);
	c.ns.fetch_add(now - start, std::memory_order_relaxed);
	reportCopies(now);
}

/// Counts a view of a surface made.
void countView(std::atomic<uint64_t> &c)
{
	c.fetch_add(1, std::memory_order_relaxed);
	reportCopies(nowNs());
}

} // namespace

/// What every process holding a surface's shared block agrees on, in the block past the planes'
/// rows, or at its start in place (RgfxSurfaceShadow::header): for a copy, the surface's seed
/// when the block last matched the surface's pages, valid where `serial` is the block's own (a
/// block made again holds an older one's bytes until its first holder writes it); for a cached
/// block in place, the surface's CPU generation, which the kext writes at each read-write CPU
/// lock of the surface (its kRgfxSurfaceCpuGenOffset) and each command buffer that uses the
/// surface names (rgfxIOSurfaceUses), and how many processes hold the surface, which the kext
/// writes as it caches the block, as a process holds the block again and as each client of the
/// surface goes (its kRgfxSurfaceClientsOffset; 0 where it wrote none), which WindowServer's keep
/// reads.
struct RgfxSurfaceHeader {
	uint64_t serial;
	uint32_t seed;
	uint32_t clients;
	uint64_t cpuGen;
};

static_assert(offsetof(RgfxSurfaceHeader, cpuGen) == kRgfxSurfaceCpuGenOffset,
              "where the kext writes the CPU generation");
static_assert(offsetof(RgfxSurfaceHeader, clients) == kRgfxSurfaceClientsOffset,
              "where the kext writes the surface's clients");

static_assert(sizeof(RgfxSurfaceHeader) % alignof(RgfxSurfaceWriters) == 0,
              "the writers follow the header, aligned");

/// The GPU's view of one surface, every plane of it: the surface's own pages, used in place,
/// where the host can (`inPlace`, RgfxRegion::acquireSurfacePages), else a copy in a block of the
/// device region: the surface's shared block, which every process holding the surface draws and
/// samples (the kext's, RgfxRegion::acquireSurface), or this process's own block where the kext
/// gives none. Its planes lie in it in order, as the surface's do in its own memory, each at its
/// own offset, so a texture over one plane that reads another (a two-plane YCbCr texture over
/// plane 0 reads plane 1's chroma) finds both in the one buffer, and every texture over the
/// surface reads and writes the same bytes, as Metal's over one surface do. In place, the shared
/// block holds only the header and the writers.
struct RgfxSurfaceShadow {
	IOSurfaceRef surface = nullptr; // retained
	uint32_t surfaceId = 0;
	std::vector<ShadowPlane> planes; // by plane; a surface without planes has the one plane 0
	RgfxRegion::Block block = {};
	RgfxRegion *region = nullptr; // where the block goes back at the copy's end
	id<MTLBuffer> buffer = nil; // the Shared buffer over the view: the pages, or the block
	std::mutex lock;            // one sync at a time in this process
	bool shared = false;        // the plane's shared block, else this process's own
	bool inPlace = false;       // the GPU uses the surface's pages; the block holds no copy
	bool cached = false;        // in place, and its GPU copy is the host's device cache
	std::atomic<bool> keepVolatile{false}; // a texture over it set it volatile or empty: unkept
	// The keep stamps (RgfxEncoder::keepStamp) of the command buffers that last named the view
	// as used (a cached one, rgfxIOSurfaceNoteUse) and as synced at its commit (a copy): each
	// command buffer names it once. Relaxed: a lost write only names it once more.
	std::atomic<uint64_t> usedBy{0};
	std::atomic<uint64_t> syncedBy{0};
	uint64_t serial = 0;        // the shared block's
	RgfxSurfaceHeader *header = nullptr; // the shared block's, past the planes' rows of a copy
	uint32_t seed = 0;          // this process's own block: the seed when it last matched
	RgfxSurfaceWriters *writers = nullptr; // past the header: the buffers a present waits for
	uint64_t writersAt = 0;                 // their offset in the block
	std::mutex writersLock;                 // one change of this process's slots at a time
	// By slot: the queue this process named there (0: none), and that queue while it lives.
	uint64_t slotQueue[RGFX_SURFACE_WRITER_SLOTS] = {};
	__weak id slotOwner[RGFX_SURFACE_WRITER_SLOTS];
	// Past the writers, where the block has room for them: each queue's latest buffer that only
	// reads the view, the same record, which only the bundles read (gSurfaceOrder); and this
	// process's slots of it, as the writers'.
	RgfxSurfaceWriters *readers = nullptr;
	uint64_t readerQueue[RGFX_SURFACE_WRITER_SLOTS] = {};
	__weak id readerOwner[RGFX_SURFACE_WRITER_SLOTS];

	~RgfxSurfaceShadow();

	/// Names the buffer at `seqno` on `queue` (id `queueId`) among the copy's writers.
	void noteWriter(id queue, uint64_t queueId, uint64_t seqno)
	{
		std::lock_guard<std::mutex> g(writersLock);
		if (!noteIn(writers, slotQueue, slotOwner, queue, queueId, seqno)) {
			static std::atomic<bool> logged{false};
			if (!logged.exchange(true)) {
				NSLog(@"RGFX: IOSurface %u: no writer slot left for queue 0x%llx: a present of "
				      @"the surface may show it before the queue's command buffers finish",
				      surfaceId, (unsigned long long)queueId);
			}
		}
	}

	/// Names the buffer at `seqno` on `queue` (id `queueId`) among the copy's readers, where the
	/// block has them.
	void noteReader(id queue, uint64_t queueId, uint64_t seqno)
	{
		if (readers == nullptr)
			return;
		std::lock_guard<std::mutex> g(writersLock);
		if (!noteIn(readers, readerQueue, readerOwner, queue, queueId, seqno)) {
			static std::atomic<bool> logged{false};
			if (!logged.exchange(true)) {
				NSLog(@"RGFX: IOSurface %u: no reader slot left for queue 0x%llx: a later write "
				      @"of the surface by another queue may not wait for its reads",
				      surfaceId, (unsigned long long)queueId);
			}
		}
	}

	/// Raises `queue`'s slot of `rec` to `seqno`: the slot it holds, else one of ours whose queue
	/// is gone, else an empty one; false where none is left. Under writersLock.
	static bool noteIn(RgfxSurfaceWriters *rec, uint64_t (&queues)[RGFX_SURFACE_WRITER_SLOTS],
	                   __weak id (&owners)[RGFX_SURFACE_WRITER_SLOTS], id queue, uint64_t queueId,
	                   uint64_t seqno)
	{
		for (uint32_t i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++) {
			if (queues[i] == queueId) {
				// Only this process writes the slot: its seqno only rises.
				uint64_t *s = &rec->writers[i].seqno;
				if (__atomic_load_n(s, __ATOMIC_RELAXED) < seqno)
					__atomic_store_n(s, seqno, __ATOMIC_RELEASE);
				return true;
			}
		}
		for (uint32_t i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++) {
			if (queues[i] != 0 && owners[i] == nil) {
				// A queue of ours that is gone, its buffers all finished: the slot is ours to give.
				give(rec, queues, owners, i, queue, queueId, seqno);
				return true;
			}
		}
		for (uint32_t i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++) {
			uint64_t empty = 0;
			if (queues[i] == 0 &&
			    __atomic_compare_exchange_n(&rec->writers[i].queue_obj_id, &empty, queueId, false,
			                                __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
				__atomic_store_n(&rec->writers[i].seqno, seqno, __ATOMIC_RELEASE);
				queues[i] = queueId;
				owners[i] = queue;
				return true;
			}
		}
		return false;
	}

	/// Gives slot `i` of `rec`, ours, to `queue`: its seqno emptied, then the queue, then the
	/// seqno.
	static void give(RgfxSurfaceWriters *rec, uint64_t (&queues)[RGFX_SURFACE_WRITER_SLOTS],
	                 __weak id (&owners)[RGFX_SURFACE_WRITER_SLOTS], uint32_t i, id queue,
	                 uint64_t queueId, uint64_t seqno)
	{
		__atomic_store_n(&rec->writers[i].seqno, 0, __ATOMIC_RELEASE);
		__atomic_store_n(&rec->writers[i].queue_obj_id, queueId, __ATOMIC_RELEASE);
		__atomic_store_n(&rec->writers[i].seqno, seqno, __ATOMIC_RELEASE);
		queues[i] = queueId;
		owners[i] = queue;
	}

	/// Empties this process's slots, of the writers and the readers: the seqno, then the queue.
	void emptyWriters()
	{
		std::lock_guard<std::mutex> g(writersLock);
		emptyIn(writers, slotQueue, slotOwner);
		if (readers != nullptr)
			emptyIn(readers, readerQueue, readerOwner);
	}

	static void emptyIn(RgfxSurfaceWriters *rec, uint64_t (&queues)[RGFX_SURFACE_WRITER_SLOTS],
	                    __weak id (&owners)[RGFX_SURFACE_WRITER_SLOTS])
	{
		for (uint32_t i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++) {
			if (queues[i] == 0)
				continue;
			__atomic_store_n(&rec->writers[i].seqno, 0, __ATOMIC_RELEASE);
			__atomic_store_n(&rec->writers[i].queue_obj_id, 0, __ATOMIC_RELEASE);
			queues[i] = 0;
			owners[i] = nil;
		}
	}

	uint8_t *copy() const { return const_cast<uint8_t *>(block.ptr); }

	/// What the planes' offsets count from: the surface's base in place, else the copy's.
	uint8_t *pixels() const
	{
		return inPlace ? static_cast<uint8_t *>(IOSurfaceGetBaseAddress(surface)) : copy();
	}

	/// The bytes of the surface's planes, as its CPU mapping lays them out.
	uint64_t planeBytes() const
	{
		uint64_t n = 0;
		for (const ShadowPlane &p : planes) {
			n += p.shape.rowBytes * p.shape.rows;
		}
		return n;
	}

	/// What a texture over the surface tells the host of it (RgfxSurfaceFacts), its YCbCr
	/// matrix as it is now.
	RgfxSurfaceFacts facts() const
	{
		RgfxSurfaceFacts f = {};
		f.surface_pixel_format = IOSurfaceGetPixelFormat(surface);
		f.plane_count = static_cast<uint32_t>(IOSurfaceGetPlaneCount(surface));
		f.width = static_cast<uint32_t>(IOSurfaceGetWidth(surface));
		f.height = static_cast<uint32_t>(IOSurfaceGetHeight(surface));
		if (planes.size() >= 2) {
			f.chroma_offset = planes[1].offset;
			f.chroma_bytes_per_row = planes[1].pitch;
		}
		f.ycbcr_matrix = ycbcrMatrix(surface);
		return f;
	}

	/// The surface's seed when the block last matched its pages; false where it never has.
	bool syncedSeed(uint32_t *out) const
	{
		if (!shared) {
			*out = seed;
			return true;
		}
		if (__atomic_load_n(&header->serial, __ATOMIC_ACQUIRE) != serial) {
			return false;
		}
		*out = __atomic_load_n(&header->seed, __ATOMIC_ACQUIRE);
		return true;
	}

	/// The block matches the surface's pages as of seed `s`.
	void setSynced(uint32_t s)
	{
		if (!shared) {
			seed = s;
			return;
		}
		__atomic_store_n(&header->seed, s, __ATOMIC_RELEASE);
		__atomic_store_n(&header->serial, serial, __ATOMIC_RELEASE);
	}

	/// Pages -> copy, under a read lock; records the seed. Caller holds `lock`. Nothing in
	/// place.
	void upload()
	{
		if (inPlace) {
			return;
		}
		uint32_t s = 0;
		uint64_t start = nowNs();
		if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, &s) != kIOReturnSuccess) {
			return;
		}
		for (size_t k = 0; k < planes.size(); k++) {
			const ShadowPlane &p = planes[k];
			const uint8_t *src = planeBase(surface, k);
			if (src != nullptr) {
				copyRows(p, copy() + p.offset, p.pitch, src, p.shape.rowBytes);
			}
		}
		IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, &s);
		setSynced(s);
		countCopy(gUploads, planeBytes(), start);
	}

	/// Copy -> pages. A CPU's write (`cpu`) goes under a write lock, and records the seed the
	/// unlock leaves, so the next commit does not read our own write back. A GPU's, at its
	/// buffer's completion, takes no lock: a lock would wait for the buffer's own writes, which
	/// the kext retires only after this, and the seed stays where the buffer's commit moved it,
	/// which the copy recorded then (rgfxIOSurfaceWritesPosted). Caller holds `lock`. Nothing in
	/// place.
	void writeBack(bool cpu)
	{
		if (inPlace) {
			return;
		}
		uint32_t s = 0;
		uint64_t start = nowNs();
		if (cpu && IOSurfaceLock(surface, 0, &s) != kIOReturnSuccess) {
			return;
		}
		for (size_t k = 0; k < planes.size(); k++) {
			const ShadowPlane &p = planes[k];
			uint8_t *dst = planeBase(surface, k);
			if (dst != nullptr) {
				copyRows(p, dst, p.shape.rowBytes, copy() + p.offset, p.pitch);
			}
		}
		if (cpu) {
			IOSurfaceUnlock(surface, 0, &s);
			setSynced(s);
		}
		countCopy(gWriteBacks, planeBytes(), start);
	}

	/// Uploads when the surface's seed moved since the block last matched it, in any process
	/// that holds the block: a CPU wrote the surface under a lock. Nothing in place.
	void uploadIfChanged()
	{
		if (inPlace) {
			return;
		}
		std::lock_guard<std::mutex> g(lock);
		uint32_t synced = 0;
		if (!syncedSeed(&synced) || IOSurfaceGetSeed(surface) != synced) {
			upload();
		}
	}

	/// Plane `p`'s rows from `src` to `dst`, each at its pitch.
	static void copyRows(const ShadowPlane &p, uint8_t *dst, uint64_t dstPitch, const uint8_t *src,
	                     uint64_t srcPitch)
	{
		if (dstPitch == srcPitch) {
			std::memcpy(dst, src, p.shape.rowBytes * p.shape.rows);
			return;
		}
		for (uint64_t r = 0; r < p.shape.rows; r++) {
			std::memcpy(dst + r * dstPitch, src + r * srcPitch, p.shape.rowBytes);
		}
	}
};

namespace {

/// The process's copies by IOSurfaceID. A dead copy's block goes back to the region
/// (RgfxRegion::release), which hands it to the accelerator kext to give out again.
struct Registry {
	std::mutex lock;
	std::map<uint32_t, std::weak_ptr<RgfxSurfaceShadow>> shadows;
};

Registry &registry()
{
	static Registry *r = new Registry(); // never destroyed: copies die at any time
	return *r;
}

} // namespace

RgfxSurfaceShadow::~RgfxSurfaceShadow()
{
	// The buffers that wrote the copy hold it until they finish, so none is left to wait for.
	if (writers != nullptr)
		emptyWriters();
	if (cached)
		gCachedLive.fetch_sub(1, std::memory_order_relaxed);
	// The buffer gives the block back once the host has dropped it (its deallocator); one
	// never made leaves the block to give back here.
	if (buffer != nil)
		buffer = nil;
	else if (block.ok && region != nullptr && shared)
		region->releaseSurface(block.offset);
	else if (block.ok && region != nullptr)
		region->release(block.offset);
	if (surface != nullptr) {
		CFRelease(surface);
	}
}

namespace {

/// The process's GPU view of every plane of `surface`, made (and, a copy, synced) on first use.
std::shared_ptr<RgfxSurfaceShadow> shadowFor(IOSurfaceRef surface, RgfxRegion *region,
                                             id<MTLDevice> device, RgfxSerializer *serializer)
{
	// Every holder lays a copy out alike, from the surface's shape alone: each plane's rows in
	// plane order, then the header, then the writers.
	std::vector<ShadowPlane> planes(std::max<size_t>(IOSurfaceGetPlaneCount(surface), 1));
	uint64_t end = 0;
	for (size_t k = 0; k < planes.size(); k++) {
		ShadowPlane &p = planes[k];
		if (!planeShape(surface, k, &p.shape) || planeBase(surface, k) == nullptr) {
			logNoTexture(surface, "plane shape",
			             format("plane %zu has no rows, or no CPU mapping in this process", k));
			return nullptr;
		}
		p.pitch = alignUp(p.shape.rowBytes, kShadowRowAlign);
		p.offset = alignUp(end, kShadowRowAlign);
		end = p.offset + p.pitch * p.shape.rows;
	}
	// In place, the planes stay where the surface's memory holds them, and the block holds the
	// header and the writers alone.
	std::vector<ShadowPlane> own = planes;
	const uint64_t ownWritersAt = sizeof(RgfxSurfaceHeader);
	const uint64_t ownSize =
	    alignUp(ownWritersAt + sizeof(RgfxSurfaceWriters), RgfxRegion::kDefaultBlockAlign);
	const uint64_t ownLength = alignUp(IOSurfaceGetAllocSize(surface), kSurfacePage);
	uint32_t surfaceId = IOSurfaceGetID(surface);
	Registry &r = registry();
	std::shared_ptr<RgfxSurfaceShadow> shadow;
	{
		std::lock_guard<std::mutex> g(r.lock);
		auto it = r.shadows.find(surfaceId);
		if (it != r.shadows.end()) {
			shadow = it->second.lock();
			// A live copy holds its surface, so its id still names that surface.
			if (shadow != nullptr) {
				return shadow;
			}
		}
		shadow = std::make_shared<RgfxSurfaceShadow>();
		shadow->surface = (IOSurfaceRef)CFRetain(surface);
		shadow->surfaceId = surfaceId;
		shadow->region = region;
		// The GPU uses the surface's own pages where the host's textures lie over its planes and
		// the kext gives the surface's block in place; else it copies them, as every holder of
		// the surface then does (the kext gives a surface's blocks in one mode).
		RgfxRegion::SurfaceBlock sharedBlock = {};
		std::string notInPlace; // why the view is a copy
		if (!region->hasSurfacePages()) {
			countView(gCopiedUnsupported);
			notInPlace = "the device uses no surface in place";
		} else if (!placeInPlace(surface, &own, region->surfacePagesAlignment(), &notInPlace)) {
			countView(gCopiedAlignment);
		} else {
			// The bundle asks a device cache for every surface, of any planes and format, as a
			// GPU of its own memory keeps a copy of each surface it uses: a video frame the GPU
			// writes stays in its memory as a window does (design.md, "The device cache"). The
			// outputs of the bundle's own video decoder ask none where the host lands no picture
			// in a cache (RgfxAskUncachedScope), its decode writing their pages.
			const bool cacheable = tAskUncached == 0;
			RgfxRegion::PagesAnswer answer = RgfxRegion::kPagesRefused;
			sharedBlock = region->acquireSurfacePages(IOSurfaceGetBaseAddress(surface), ownLength,
			                                          ownSize, surfaceId, cacheable, &answer);
			if (sharedBlock.block.ok) {
				shadow->inPlace = true;
				// Whoever made the block decided: a block another holder made cached is cached
				// for this process too.
				shadow->cached = sharedBlock.cached;
				if (shadow->cached) {
					gCachedLive.fetch_add(1, std::memory_order_relaxed);
					gCached.fetch_add(1, std::memory_order_relaxed);
				}
				countView(gInPlace);
			} else if (answer == RgfxRegion::kPagesUnsupported) {
				countView(gCopiedUnsupported);
				notInPlace = "the device uses no surface in place, or refused its pages";
			} else if (answer == RgfxRegion::kPagesNoRoom) {
				countView(gCopiedRefused);
				notInPlace = "the kext had no room for its block in place";
			} else {
				countView(gCopiedRefused);
				notInPlace = "the kext gave no block in place";
			}
		}
		uint64_t headerAt = 0, writersAt = ownWritersAt;
		if (shadow->inPlace) {
			shadow->planes = own;
		} else {
			shadow->planes = planes;
			headerAt = alignUp(end, alignof(RgfxSurfaceHeader));
			writersAt = headerAt + sizeof(RgfxSurfaceHeader);
			uint64_t size =
			    alignUp(writersAt + sizeof(RgfxSurfaceWriters), RgfxRegion::kDefaultBlockAlign);
			// The kext finds the surface's shared block by plane 0's pages, which no other
			// surface holds.
			const PlaneShape &first = planes[0].shape;
			sharedBlock =
			    region->acquireSurface(planeBase(surface, 0), first.rowBytes * first.rows, size);
			if (!sharedBlock.block.ok) {
				shadow->block = region->allocate(size);
			}
			if (!shadow->block.ok && !sharedBlock.block.ok) {
				uint64_t free = 0;
				std::string room = region->freeBytes(&free)
				                       ? format("%llu bytes free", (unsigned long long)free)
				                       : std::string("the kext keeps its free bytes");
				logNoTexture(surface, "region",
				             format("not in place (%s); the kext gave no shared copy block of "
				                    "%llu bytes; and the region has no room for one of the "
				                    "process's own (%s)",
				                    notInPlace.c_str(), (unsigned long long)size, room.c_str()));
			}
		}
		if (sharedBlock.block.ok) {
			shadow->block = sharedBlock.block;
			shadow->shared = true;
			shadow->serial = sharedBlock.serial;
			shadow->header = reinterpret_cast<RgfxSurfaceHeader *>(shadow->copy() + headerAt);
		}
		if (!shadow->block.ok) {
			return nullptr;
		}
		// A block made for this ask holds an older one's bytes: its writers start empty, before
		// any holder's texture names them.
		shadow->writersAt = writersAt;
		shadow->writers = reinterpret_cast<RgfxSurfaceWriters *>(shadow->copy() + writersAt);
		// The readers follow where the block's room past the writers holds them: every holder
		// asks the block's size from the writers' end alone, so a holder of a bundle without
		// readers asks the same block.
		const uint64_t readersEnd = writersAt + 2 * sizeof(RgfxSurfaceWriters);
		if (readersEnd <=
		    alignUp(writersAt + sizeof(RgfxSurfaceWriters), RgfxRegion::kDefaultBlockAlign))
			shadow->readers = shadow->writers + 1;
		const size_t slots =
		    (shadow->readers != nullptr ? 2 : 1) * sizeof(RgfxSurfaceWriters);
		if (!shadow->shared || sharedBlock.fresh)
			std::memset(static_cast<void *>(shadow->writers), 0, slots);
		r.shadows[surfaceId] = shadow;
	}
	uint64_t offset = shadow->block.offset;
	bool shared = shadow->shared;
	void (^letGo)(void) = ^{
		if (shared)
			region->releaseSurface(offset);
		else
			region->release(offset);
	};
	if (shadow->inPlace) {
		// Over the surface's own pages, from its base; the host's span is the block's.
		shadow->buffer = [RgfxBuffer bufferOverSurfacePagesWithContents:shadow->pixels()
		                                                    blockOffset:offset
		                                                      blockSize:shadow->block.size
		                                                         length:ownLength
		                                                        options:MTLResourceStorageModeShared
		                                                    deallocator:letGo
		                                                         device:device
		                                                     serializer:serializer];
	} else {
		shadow->buffer = [RgfxBuffer bufferOverHostMemoryWithContents:shadow->copy()
		                                                regionOffset:offset
		                                                      length:shadow->block.size
		                                                     options:MTLResourceStorageModeShared
		                                                 deallocator:letGo
		                                                      device:device
		                                                  serializer:serializer];
	}
	if (shadow->buffer == nil) {
		logNoTexture(surface, "buffer",
		             format("no buffer over its %s", shadow->inPlace ? "pages" : "copy block"));
		return nullptr;
	}
	// In place, the GPU's view is the pages: nothing to sync. This process's own block takes
	// the pages now; a shared one where no holder has synced it with them since their last CPU
	// write.
	if (shadow->inPlace) {
		return shadow;
	}
	if (shared) {
		shadow->uploadIfChanged();
	} else {
		std::lock_guard<std::mutex> g(shadow->lock);
		shadow->upload();
	}
	return shadow;
}

/// The wire shape of `d`, as RgfxTexture maps a descriptor.
RgfxTextureDescriptor wireDescriptor(MTLTextureDescriptor *d)
{
	RgfxTextureDescriptor out = {};
	out.pixel_format = static_cast<uint32_t>(d.pixelFormat);
	out.texture_type = static_cast<uint32_t>(d.textureType);
	out.width = static_cast<uint32_t>(d.width);
	out.height = static_cast<uint32_t>(d.height);
	out.depth = static_cast<uint32_t>(d.depth);
	out.mipmap_level_count = static_cast<uint32_t>(d.mipmapLevelCount);
	out.array_length = static_cast<uint32_t>(d.arrayLength);
	out.sample_count = static_cast<uint32_t>(d.sampleCount);
	out.usage = static_cast<uint32_t>(d.usage);
	out.options = static_cast<uint32_t>(d.resourceOptions);
	MTLTextureSwizzleChannels sw = d.swizzle;
	out.swizzle = static_cast<uint32_t>(sw.red) | (static_cast<uint32_t>(sw.green) << 8) |
	              (static_cast<uint32_t>(sw.blue) << 16) | (static_cast<uint32_t>(sw.alpha) << 24);
	return out;
}

/// An IOSurface lock of `options` of a cached view's surface, held for its scope, around the
/// texture's own CPU access: the kext's control function brings the GPU's writes from the
/// device cache into the pages, and a read-write lock moves the surface's CPU generation. Nothing
/// for a view not cached, whose pages are the GPU's.
class CachedLock {
public:
	CachedLock(const RgfxSurfaceShadow &s, IOSurfaceLockOptions options)
	    : m_surface(s.cached ? s.surface : nullptr), m_options(options)
	{
		if (m_surface != nullptr &&
		    IOSurfaceLock(m_surface, m_options, &m_seed) != kIOReturnSuccess)
			m_surface = nullptr;
	}
	~CachedLock()
	{
		if (m_surface != nullptr)
			IOSurfaceUnlock(m_surface, m_options, &m_seed);
	}
	CachedLock(const CachedLock &) = delete;
	CachedLock &operator=(const CachedLock &) = delete;

private:
	IOSurfaceRef m_surface;
	IOSurfaceLockOptions m_options;
	uint32_t m_seed = 0;
};

} // namespace

namespace {

// The keep: the views in place the process holds past their textures, so a texture made over the
// same surface soon after finds its view, and the view's block, held, and asks the kext for
// nothing. A compositor makes a texture over its flattened layers' surface each composite and
// lets it go at its end; without the keep each such texture asks the kext for the surface's block
// again, which wires and walks every page of the surface before it finds the block it holds
// (about 140 us for 12 MiB, measured on the RTX 5080 boot, 1.3 asks a composite). WebKit's WebGL
// makes a texture over its drawing buffer each frame (ANGLE's bindTexImage) and lets it go as the
// buffer goes to the display, and the buffer comes back for drawing two or more frames later. A
// block let go sleeps in the kext, but the buffer's trip to the display ends the sleep (as each
// other process lets go of the surface: the kext's dormant blocks, RgfxSurfaceBlocks.hpp), so
// without the keep each frame's texture makes the surface's block anew and waits for the host to
// map its pages (about 40 ms a frame for the 3840x2054 buffer, 15 frames a second, on the RTX
// 5080 boot).
//
// A view stays kept until kKeepNs past its last texture's making, three composites at 60 Hz, and
// while anything else in the process holds its surface (its retain count past the view's own),
// until kKeepHeldNs past it: a ring of buffers turning at a few frames a second keeps its views.
// A surface the view alone holds goes at the first sweep past kKeepNs, so the keep outlives
// neither the surface's last texture nor the process's last other hold of it by much more than
// kKeepNs. The process's commitAndWaitUntilSubmitted calls count no composites: WindowServer makes
// about three a composite (measured on that boot). The keep holds at most kKeepMaxViews views and
// kKeepMaxBytes of their surfaces, the least recently textured let go first, lets go of all of
// them at the process's memory pressure and where the kext has no room for a block (the region's
// reclaim, rgfxIOSurfaceKeepReclaim), and keeps none a texture over it set volatile or empty
// (setPurgeableState:). A kept view holds its surface (CFRetain) as a texture over it does, so
// its IOSurfaceID names no other surface, and its block, so the kext keeps the process's wire of
// the surface's pages as for a live texture: the pages the host maps are the surface's while the
// view is kept, and a texture made over it again is as one made while the last still lived. The
// surface's use count is never touched, so IOSurfaceIsInUse answers as before.
//
// WindowServer's keep also keeps a view of a cached block while another process holds its surface,
// by the kext's count of the surface's clients in the block's header (past WindowServer's own
// one), and lets it go as above once the count is down to WindowServer alone. WindowServer lets a
// layer's contents go as the next ones replace them, and takes them again when they come back: a
// ring of three 3840x2054 buffers turning at 30 Hz came back every 99 ms, and each texture asked
// the kext for the block again, 0.46 to 0.58 ms a frame, while the views went 50 ms past their
// textures (measured on the RTX 5080 boot). The other processes keep by their own holds alone:
// two processes that each kept a view while the other held the surface would keep each other's,
// each view the last hold of the surface in its process, until their caps let them go. A view of
// WindowServer's that another process's kept view alone holds goes once that keep lets go; while
// another process holds the surface of every view WindowServer keeps, its sweep comes kKeepHeldNs
// apart, not kKeepNs. Such views go at the region's reclaim as every kept view does, so the keep
// costs no block a refusal: a texture over one of their surfaces then asks the kext for its block
// again.
constexpr size_t kKeepMaxViews = 8;
constexpr uint64_t kKeepMaxBytes = 256ull << 20;
constexpr uint64_t kKeepNs = 50ull * 1000 * 1000;
constexpr uint64_t kKeepHeldNs = 1000ull * 1000 * 1000;

struct Kept {
	std::shared_ptr<RgfxSurfaceShadow> view;
	uint64_t until = 0;     // nowNs() at which it goes where the view alone holds its surface
	uint64_t heldUntil = 0; // nowNs() at which it goes all the same, unless heldElsewhere
	uint64_t bytes = 0;     // its surface's
};

/// Whether nothing in the process but `view` holds its surface: the view's is its one retain
/// (the textures over it hold the view, not the surface).
bool viewAloneHolds(const RgfxSurfaceShadow &view)
{
	return CFGetRetainCount(view.surface) <= 1;
}

/// Whether the keep keeps a cached block's view while another process holds its surface: in
/// WindowServer, or as the tests set it (rgfxIOSurfaceKeepCountClients); -1 until first asked.
std::atomic<int> gCountsClients{-1};

bool countsClients()
{
	int on = gCountsClients.load(std::memory_order_relaxed);
	if (on < 0) {
		const char *name = getprogname();
		on = name != nullptr && strcmp(name, "WindowServer") == 0;
		int unset = -1;
		if (!gCountsClients.compare_exchange_strong(unset, on, std::memory_order_relaxed))
			on = unset;
	}
	return on != 0;
}

/// Whether another process holds `view`'s surface, where the keep counts them: the kext's count
/// of its clients in the block's header (RgfxSurfaceHeader::clients) is past this process's one.
/// A block not cached has no count.
bool heldElsewhere(const RgfxSurfaceShadow &view)
{
	return countsClients() && view.cached && view.header != nullptr &&
	       __atomic_load_n(&view.header->clients, __ATOMIC_ACQUIRE) > 1;
}

/// Whether `kept` is due to go at `now` (the keep).
bool keepDue(const Kept &kept, uint64_t now)
{
	const RgfxSurfaceShadow &view = *kept.view;
	if (heldElsewhere(view))
		return false;
	return now >= kept.heldUntil || (now >= kept.until && viewAloneHolds(view));
}

struct Keep {
	std::mutex lock;
	std::vector<Kept> views; // the least recently textured first
	uint64_t bytes = 0;
	bool sweeping = false; // a sweep is due
	dispatch_source_t pressure = nil;
};

Keep &keep()
{
	static Keep *k = new Keep(); // never destroyed: views go at any time
	return *k;
}

/// Moves out of `k` (locked) into `gone` the views due to go at `now`, or all of them.
void expire(Keep &k, uint64_t now, bool all, std::vector<std::shared_ptr<RgfxSurfaceShadow>> *gone)
{
	for (auto it = k.views.begin(); it != k.views.end();) {
		if (all || keepDue(*it, now) || it->view->keepVolatile.load(std::memory_order_relaxed)) {
			k.bytes -= it->bytes;
			gone->push_back(std::move(it->view));
			it = k.views.erase(it);
		} else {
			++it;
		}
	}
}

/// Lets go of the views due to go at `now`, or all of them; their blocks go back outside the
/// keep's lock, as the views die.
void sweep(uint64_t now, bool all)
{
	Keep &k = keep();
	std::vector<std::shared_ptr<RgfxSurfaceShadow>> gone;
	{
		std::lock_guard<std::mutex> g(k.lock);
		expire(k, now, all, &gone);
	}
}

/// Schedules a sweep kKeepNs on, where `k` (locked) holds views and none is due, or kKeepHeldNs
/// on where another process holds the surface of every view it holds: a process that stops making
/// textures lets its kept views go all the same.
void armSweep(Keep &k)
{
	if (k.sweeping || k.views.empty())
		return;
	k.sweeping = true;
	const bool allElsewhere = std::all_of(k.views.begin(), k.views.end(),
	                                      [](const Kept &e) { return heldElsewhere(*e.view); });
	const uint64_t in = allElsewhere ? kKeepHeldNs : kKeepNs;
	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(in)),
	               dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
		Keep &kk = keep();
		std::vector<std::shared_ptr<RgfxSurfaceShadow>> gone;
		{
			std::lock_guard<std::mutex> g(kk.lock);
			kk.sweeping = false;
			expire(kk, nowNs(), false, &gone);
			armSweep(kk);
		}
	});
}

/// Keeps `view`, whose texture was just made, past its textures (the keep); a copy is not kept.
void keepView(const std::shared_ptr<RgfxSurfaceShadow> &view)
{
	if (!view->inPlace)
		return;
	Keep &k = keep();
	std::vector<std::shared_ptr<RgfxSurfaceShadow>> gone;
	std::lock_guard<std::mutex> g(k.lock);
	const uint64_t now = nowNs();
	auto held = std::find_if(k.views.begin(), k.views.end(),
	                         [&](const Kept &e) { return e.view == view; });
	if (held != k.views.end()) {
		k.bytes -= held->bytes;
		gone.push_back(std::move(held->view));
		k.views.erase(held);
	}
	const uint64_t bytes = IOSurfaceGetAllocSize(view->surface);
	if (!view->keepVolatile.load(std::memory_order_relaxed) && bytes <= kKeepMaxBytes) {
		k.views.push_back({view, now + kKeepNs, now + kKeepHeldNs, bytes});
		k.bytes += bytes;
	}
	expire(k, now, false, &gone);
	while (k.views.size() > kKeepMaxViews || k.bytes > kKeepMaxBytes) {
		k.bytes -= k.views.front().bytes;
		gone.push_back(std::move(k.views.front().view));
		k.views.erase(k.views.begin());
	}
	if (k.pressure == nil) {
		k.pressure = dispatch_source_create(
		    DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
		    DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
		    dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
		if (k.pressure != nil) {
			dispatch_source_set_event_handler(k.pressure, ^{
				rgfxIOSurfaceKeepFlush();
			});
			dispatch_resume(k.pressure);
		}
	}
	armSweep(k);
	// `gone` may hold the views' last references: they die once the lock is released (its
	// guard goes first, as it was declared last).
}

} // namespace

void rgfxIOSurfaceKeepFlush(void)
{
	sweep(0, true);
}

bool rgfxIOSurfaceKeepReclaim(void)
{
	Keep &k = keep();
	std::vector<std::shared_ptr<RgfxSurfaceShadow>> gone;
	{
		std::lock_guard<std::mutex> g(k.lock);
		expire(k, 0, true, &gone);
	}
	// The views die here, the least recently textured first, each giving its block back.
	const bool any = !gone.empty();
	for (auto &view : gone)
		view.reset();
	return any;
}

void rgfxIOSurfaceKeepSweepAt(uint64_t now)
{
	sweep(now, false);
}

void rgfxIOSurfaceKeepCountClients(bool on)
{
	gCountsClients.store(on ? 1 : 0, std::memory_order_relaxed);
}

@interface RgfxIOSurfaceTexture ()
- (std::shared_ptr<RgfxSurfaceShadow>)rgfxShadow;
@end

@implementation RgfxIOSurfaceTexture {
	std::shared_ptr<RgfxSurfaceShadow> _shadow;
	NSUInteger _plane;
	// Bytes from the plane's first byte to the texture's (textureWithDescriptor:...origin:).
	uint64_t _offset;
}

+ (instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                            iosurface:(IOSurfaceRef)surface
                                plane:(NSUInteger)plane
                               region:(RgfxRegion *)region
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	return [self textureWithDescriptor:descriptor
	                         iosurface:surface
	                             plane:plane
	                            origin:MTLOriginMake(0, 0, 0)
	                            region:region
	                            device:device
	                        serializer:serializer];
}

+ (instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                            iosurface:(IOSurfaceRef)surface
                                plane:(NSUInteger)plane
                               origin:(MTLOrigin)origin
                               region:(RgfxRegion *)region
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	if (descriptor == nil || surface == nullptr || region == nullptr || serializer == nullptr) {
		logNoTexture(surface, "arguments",
		             descriptor == nil  ? "no descriptor"
		             : surface == nullptr ? "no surface"
		                                  : "no device region or transport");
		return nil;
	}
	if (plane >= std::max<size_t>(IOSurfaceGetPlaneCount(surface), 1)) {
		logNoTexture(surface, "plane", format("plane %lu, past its planes", (unsigned long)plane));
		return nil;
	}
	// shadowFor logs why it made no view.
	std::shared_ptr<RgfxSurfaceShadow> shadow = shadowFor(surface, region, device, serializer);
	if (shadow == nullptr) {
		return nil;
	}
	// The guest sees the shape it asked for; the host's texture lies over the Shared
	// buffer of its view, so it is Shared.
	RgfxTextureDescriptor desc = wireDescriptor(descriptor);
	RgfxTextureDescriptor hostDesc = desc;
	hostDesc.options = (hostDesc.options & ~static_cast<uint32_t>(MTLResourceStorageModeMask)) |
	                   static_cast<uint32_t>(MTLResourceStorageModeShared);
	const ShadowPlane &p = shadow->planes[plane];
	// A texture inside its plane lies in the view, in the plane's rows and elements, each pixel
	// an element.
	const uint64_t offset = origin.y * p.pitch + origin.x * p.shape.bytesPerElement;
	const uint64_t width = descriptor.width, height = descriptor.height;
	if (offset != 0 &&
	    (width == 0 || height == 0 || p.shape.elementWidth != 1 || p.shape.elementHeight != 1 ||
	     origin.y + height > p.shape.rows ||
	     (origin.x + width) * p.shape.bytesPerElement > p.shape.rowBytes)) {
		logNoTexture(surface, "origin",
		             format("%llux%llu at (%lu, %lu) of plane %lu, past the plane",
		                    (unsigned long long)width, (unsigned long long)height,
		                    (unsigned long)origin.x, (unsigned long)origin.y, (unsigned long)plane));
		return nil;
	}
	RgfxCreated created = serializer->createTextureFromIOSurface(
	    [(RgfxResource *)shadow->buffer rgfxObjId], p.offset + offset, p.pitch, shadow->surfaceId,
	    static_cast<uint32_t>(plane), hostDesc, shadow->writersAt, shadow->facts());
	if (created.resource_id == 0) {
		// No resource id left: no texture, as Metal answers one it cannot make.
		logNoTexture(surface, "resource id", "every texture resource id is taken");
		return nil;
	}
	if (serializer->refused(created)) {
		logNoTexture(surface, "create", "the device never got its create");
		return nil;
	}
	RgfxIOSurfaceTexture *t = [[RgfxIOSurfaceTexture alloc] initWithObjId:created.obj_id
	                                                           gpuAddress:0
	                                                           resourceId:created.resource_id
	                                                           descriptor:desc
	                                                               device:device
	                                                           serializer:serializer];
	if (t == nil) {
		logNoTexture(surface, "texture", "the texture object was not made");
		return nil;
	}
	keepView(shadow);
	t->_shadow = std::move(shadow);
	t->_plane = plane;
	t->_offset = offset;
	t->_rgfxSurfaceRoot = t;
	[t rgfxTakeDescriptor:descriptor];
	return t;
}

// The state answers as any texture's (RgfxTexture); a volatile or empty one, which lets the
// surface's pages be purged, also lets its view go from the keep at once, and keeps it out until
// a texture over the surface is set non-volatile again.
- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
	if (_shadow != nullptr &&
	    (state == MTLPurgeableStateVolatile || state == MTLPurgeableStateEmpty ||
	     state == MTLPurgeableStateNonVolatile)) {
		_shadow->keepVolatile.store(state != MTLPurgeableStateNonVolatile,
		                            std::memory_order_relaxed);
		if (state != MTLPurgeableStateNonVolatile)
			sweep(nowNs(), false);
	}
	return [super setPurgeableState:state];
}

- (std::shared_ptr<RgfxSurfaceShadow>)rgfxShadow
{
	return _shadow;
}

// The view of `t`, read with no message and no copy, for the encoders' notes below.
static const std::shared_ptr<RgfxSurfaceShadow> &rgfxShadowOfRoot(
    __unsafe_unretained RgfxIOSurfaceTexture *t)
{
	return t->_shadow;
}

- (IOSurfaceRef)iosurface
{
	return _shadow->surface;
}

// The texture lies at its plane's offset in its view's buffer, in the buffer's allocation, so the
// textures over one plane share both, as Metal's over one surface do (measured on the M3 Max under
// macOS 27: one allocationID and one gpuAddress for two textures and a view over a surface, of any
// storage mode); plane 0's lie at the buffer's start, and one inside its plane past the plane's
// offset by its own. A buffer over the surface's pages in place has no GPU address (its create
// names none), and nor do its textures.
- (uint64_t)gpuAddress
{
	uint64_t base = _shadow->buffer.gpuAddress;
	return base == 0 ? 0 : base + _shadow->planes[_plane].offset + _offset;
}

- (uint64_t)rgfxAllocationId
{
	return static_cast<RgfxResource *>(_shadow->buffer).rgfxAllocationId;
}

- (NSUInteger)iosurfacePlane
{
	return _plane;
}

// The bytes the texture takes: its copy of the surface in the region, or the surface's pages its
// buffer lies over in place (SecurityAgent's RenderBox asks as it accounts a surface's texture).
- (NSUInteger)allocatedSize
{
	return _shadow->inPlace ? _shadow->buffer.length : static_cast<NSUInteger>(_shadow->block.size);
}

// The texture's own CPU access reads and writes its view, for formats of one element per pixel
// at level 0 / slice 0: the copy, a write also going to the surface's pages, or in place the
// pages alone. A texture inside its plane (the video driver's, which nothing reads on the CPU)
// has none.

- (BOOL)rgfxCpuRegion:(MTLRegion)region level:(NSUInteger)level slice:(NSUInteger)slice
{
	const PlaneShape &s = _shadow->planes[_plane].shape;
	return _offset == 0 && level == 0 && slice == 0 && s.elementWidth == 1 &&
	       s.elementHeight == 1 &&
	       region.origin.y + region.size.height <= s.rows &&
	       (region.origin.x + region.size.width) * s.bytesPerElement <= s.rowBytes;
}

- (void)getBytes:(void *)pixelBytes
      bytesPerRow:(NSUInteger)bytesPerRow
    bytesPerImage:(NSUInteger)bytesPerImage
       fromRegion:(MTLRegion)region
      mipmapLevel:(NSUInteger)level
            slice:(NSUInteger)slice
{
	if (![self rgfxCpuRegion:region level:level slice:slice]) {
		return;
	}
	_shadow->uploadIfChanged();
	// A cached surface's GPU writes are in the host's device cache: a read lock has the kext
	// bring them into the pages first. Taken before the view's own lock, which a completion
	// takes while the lock may wait for that completion's writes.
	CachedLock cpu(*_shadow, kIOSurfaceLockReadOnly);
	std::lock_guard<std::mutex> g(_shadow->lock);
	const ShadowPlane &p = _shadow->planes[_plane];
	uint64_t bpe = p.shape.bytesPerElement;
	const uint8_t *rows = _shadow->pixels() + p.offset;
	for (NSUInteger r = 0; r < region.size.height; r++) {
		std::memcpy(static_cast<uint8_t *>(pixelBytes) + r * bytesPerRow,
		            rows + (region.origin.y + r) * p.pitch + region.origin.x * bpe,
		            region.size.width * bpe);
	}
}

- (void)getBytes:(void *)pixelBytes
     bytesPerRow:(NSUInteger)bytesPerRow
      fromRegion:(MTLRegion)region
     mipmapLevel:(NSUInteger)level
{
	[self getBytes:pixelBytes
	      bytesPerRow:bytesPerRow
	    bytesPerImage:0
	       fromRegion:region
	      mipmapLevel:level
	            slice:0];
}

- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
                slice:(NSUInteger)slice
            withBytes:(const void *)pixelBytes
          bytesPerRow:(NSUInteger)bytesPerRow
        bytesPerImage:(NSUInteger)bytesPerImage
{
	if (![self rgfxCpuRegion:region level:level slice:slice]) {
		return;
	}
	_shadow->uploadIfChanged();
	// A cached surface's GPU copy is the host's device cache: a read-write lock brings the
	// GPU's writes into the pages, and has the next command buffer that uses the surface take
	// the CPU's. Taken before the view's own lock, as in getBytes.
	CachedLock cpu(*_shadow, 0);
	std::lock_guard<std::mutex> g(_shadow->lock);
	const ShadowPlane &p = _shadow->planes[_plane];
	uint64_t bpe = p.shape.bytesPerElement;
	uint8_t *rows = _shadow->pixels() + p.offset;
	for (NSUInteger r = 0; r < region.size.height; r++) {
		std::memcpy(rows + (region.origin.y + r) * p.pitch + region.origin.x * bpe,
		            static_cast<const uint8_t *>(pixelBytes) + r * bytesPerRow,
		            region.size.width * bpe);
	}
	_shadow->writeBack(true);
}

- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
            withBytes:(const void *)pixelBytes
          bytesPerRow:(NSUInteger)bytesPerRow
{
	[self replaceRegion:region
	        mipmapLevel:level
	              slice:0
	          withBytes:pixelBytes
	        bytesPerRow:bytesPerRow
	      bytesPerImage:0];
}

@end

void rgfxIOSurfaceSyncBeforeCommit(RgfxEncoder *enc)
{
	for (const std::shared_ptr<void> &p : enc->takeSurfaceSyncs())
		static_cast<RgfxSurfaceShadow *>(p.get())->uploadIfChanged();
}

RgfxAskUncachedScope::RgfxAskUncachedScope()
{
	tAskUncached++;
}

RgfxAskUncachedScope::~RgfxAskUncachedScope()
{
	tAskUncached--;
}

// The view of the texture over an IOSurface that `resource` is, or views through any views (the
// root each texture records as it is made); null for any other.
static const std::shared_ptr<RgfxSurfaceShadow> *rgfxSurfaceView(__unsafe_unretained id resource)
{
	if (!RgfxIsResource(resource))
		return nullptr;
	__unsafe_unretained RgfxResource *r = (RgfxResource *)resource;
	__unsafe_unretained RgfxIOSurfaceTexture *root = r->_rgfxSurfaceRoot;
	return root != nil ? &rgfxShadowOfRoot(root) : nullptr;
}

// Records on `enc`, once its command buffer, that its commit syncs `view` where it is a copy.
static void rgfxNoteSync(RgfxEncoder *enc, const std::shared_ptr<RgfxSurfaceShadow> &view)
{
	if (view->inPlace)
		return;
	const uint64_t stamp = enc->keepStamp();
	if (view->syncedBy.load(std::memory_order_relaxed) == stamp)
		return;
	view->syncedBy.store(stamp, std::memory_order_relaxed);
	enc->noteSurfaceSync(view);
}

void rgfxIOSurfaceNoteWrite(RgfxEncoder *enc, __unsafe_unretained id texture)
{
	if (enc == nullptr)
		return;
	if (const std::shared_ptr<RgfxSurfaceShadow> *view = rgfxSurfaceView(texture)) {
		enc->noteSurfaceWrite(*view);
		rgfxNoteSync(enc, *view);
	}
}

void rgfxIOSurfaceNoteUse(RgfxEncoder *enc, __unsafe_unretained id resource)
{
	if (enc == nullptr)
		return;
	const std::shared_ptr<RgfxSurfaceShadow> *view = rgfxSurfaceView(resource);
	if (view == nullptr)
		return;
	rgfxNoteSync(enc, *view);
	const uint64_t stamp = enc->keepStamp();
	const bool named = (*view)->cached || gSurfaceOrder.load(std::memory_order_relaxed);
	if (named && (*view)->usedBy.load(std::memory_order_relaxed) != stamp) {
		(*view)->usedBy.store(stamp, std::memory_order_relaxed);
		enc->noteSurfaceUse(*view);
	}
}

void rgfxIOSurfaceSetOrder(bool on)
{
	gSurfaceOrder.store(on, std::memory_order_relaxed);
}

/// Each slot of `rec` that names a queue other than `own`, read as the host reads a writer (its
/// queue, its seqno, then its queue again, a slot whose queue changed between reading as free),
/// added to `waits`: once a queue, at its highest seqno.
static void nameSlots(const RgfxSurfaceWriters *rec, uint64_t own,
                      std::vector<RgfxSurfaceOrderWait> &waits)
{
	for (uint32_t i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++) {
		const RgfxSurfaceWriter &slot = rec->writers[i];
		const uint64_t queue = __atomic_load_n(&slot.queue_obj_id, __ATOMIC_ACQUIRE);
		if (queue == 0 || queue == own)
			continue;
		const uint64_t seqno = __atomic_load_n(&slot.seqno, __ATOMIC_ACQUIRE);
		if (seqno == 0 || __atomic_load_n(&slot.queue_obj_id, __ATOMIC_ACQUIRE) != queue)
			continue;
		auto it = std::find_if(waits.begin(), waits.end(), [queue](const RgfxSurfaceOrderWait &w) {
			return w.queue_obj_id == queue;
		});
		if (it == waits.end())
			waits.push_back({queue, seqno});
		else if (it->seqno < seqno)
			it->seqno = seqno;
	}
}

std::vector<std::shared_ptr<void>> rgfxIOSurfaceNameUses(
    RgfxEncoder *enc, const std::vector<std::shared_ptr<void>> &writes, uint64_t queueObjId)
{
	const std::vector<std::shared_ptr<void>> uses = enc->takeSurfaceUses();
	std::vector<std::shared_ptr<void>> reads;
	const bool order = gSurfaceOrder.load(std::memory_order_relaxed);
	if (gCachedLive.load(std::memory_order_relaxed) == 0 && !order)
		return reads;
	if (gCachedLive.load(std::memory_order_relaxed) != 0) {
		std::vector<RgfxSurfaceShadow *> named;
		std::vector<RgfxOpSurfaceUse> ops;
		auto name = [&](const std::shared_ptr<void> &p, bool writes) {
			auto *s = static_cast<RgfxSurfaceShadow *>(p.get());
			if (!s->cached)
				return;
			auto it = std::find(named.begin(), named.end(), s);
			if (it != named.end()) {
				if (writes)
					ops[it - named.begin()].flags |= RGFX_SURFACE_USE_WRITES;
				return;
			}
			named.push_back(s);
			RgfxOpSurfaceUse op = {};
			op.block_offset = s->block.offset;
			// As the kext left it at the surface's last read-write CPU lock, in any process.
			op.cpu_gen = __atomic_load_n(&s->header->cpuGen, __ATOMIC_ACQUIRE);
			op.flags = writes ? RGFX_SURFACE_USE_WRITES : 0;
			ops.push_back(op);
		};
		for (const auto &w : writes)
			name(w, true);
		for (const auto &u : uses)
			name(u, false);
		if (!ops.empty())
			enc->prependSurfaceUses(ops.data(), ops.size());
	}
	if (!order || (uses.empty() && writes.empty()))
		return reads;
	// The buffers of other queues it runs after, as the surfaces' blocks name them now: of each
	// surface it writes, the writers and the readers; of each it only reads, the writers. Each
	// was posted before its slot named it, so before this commit.
	std::vector<RgfxSurfaceShadow *> seen;
	std::vector<RgfxSurfaceOrderWait> waits;
	for (const auto &w : writes) {
		auto *s = static_cast<RgfxSurfaceShadow *>(w.get());
		if (std::find(seen.begin(), seen.end(), s) != seen.end())
			continue;
		seen.push_back(s);
		if (s->writers != nullptr)
			nameSlots(s->writers, queueObjId, waits);
		if (s->readers != nullptr)
			nameSlots(s->readers, queueObjId, waits);
	}
	for (const auto &u : uses) {
		auto *s = static_cast<RgfxSurfaceShadow *>(u.get());
		if (std::find(seen.begin(), seen.end(), s) != seen.end())
			continue;
		seen.push_back(s);
		if (s->writers != nullptr)
			nameSlots(s->writers, queueObjId, waits);
		reads.push_back(u);
	}
	enc->prependSurfaceOrder(waits.data(), waits.size());
	return reads;
}

void RgfxSurfaceBindings::rebind(NSUInteger index, __unsafe_unretained id texture, bool overSurface)
{
	auto it = std::find_if(m_bound.begin(), m_bound.end(),
	                       [index](const Bound &b) { return b.index == index; });
	if (!overSurface) {
		if (it != m_bound.end()) {
			m_bound.erase(it);
		}
		return;
	}
	if (it == m_bound.end()) {
		m_bound.push_back({index, texture, false});
		m_changed = true;
	} else if (it->texture != texture) {
		it->texture = texture;
		it->noted = false;
		m_changed = true;
	}
}

void RgfxSurfaceBindings::resident(__unsafe_unretained id resource)
{
	if (resource == nil)
		return;
	if (rgfxSurfaceView(resource) == nullptr &&
	    !(RgfxMemoryShadow::anyLive() && rgfxShadowOf(resource) != nullptr))
		return;
	for (const Resident &r : m_resident) {
		if (r.resource == resource)
			return;
	}
	const bool texture = [resource conformsToProtocol:@protocol(MTLTexture)];
	m_resident.push_back({resource, texture, false});
	m_changed = true;
}

void RgfxSurfaceBindings::noteChangedWrites(RgfxEncoder *enc, const RgfxAirFunction *fn)
{
	m_changed = false;
	m_lastFn = fn;
	if (m_bound.empty() && m_resident.empty())
		return;
	const bool anywhere = fn != nullptr && rgfxAirWritesTextureAnywhere(*fn);
	for (Bound &b : m_bound) {
		if (b.noted) {
			continue;
		}
		bool writes = fn != nullptr && rgfxAirWritesTexture(*fn, static_cast<uint32_t>(b.index));
		if (!writes && (fn == nullptr || anywhere)) {
			const MTLTextureUsage usage = [(id<MTLTexture>)b.texture usage];
			writes = usage == MTLTextureUsageUnknown || (usage & MTLTextureUsageShaderWrite) != 0;
		}
		if (writes) {
			rgfxIOSurfaceNoteWrite(enc, b.texture);
			b.noted = true;
		}
	}
	if (m_resident.empty())
		return;
	bool textures = fn == nullptr, buffers = fn == nullptr;
	if (fn != nullptr) {
		for (const RgfxAirArgument &a : fn->arguments) {
			if (a.kind == RgfxAirKind::IndirectBuffer) {
				buffers = true;
				textures = textures || a.writesTextures;
			}
		}
	}
	for (Resident &r : m_resident) {
		if (r.noted || !(r.texture ? textures : buffers))
			continue;
		rgfxIOSurfaceNoteWrite(enc, r.resource);
		if (enc != nullptr && RgfxMemoryShadow::anyLive()) {
			if (std::shared_ptr<RgfxMemoryShadow> shadow = rgfxShadowOf(r.resource))
				enc->noteShadowUse(shadow, true);
		}
		r.noted = true;
	}
}

RgfxRegion *rgfxIOSurfaceWritesRegion(const std::vector<std::shared_ptr<void>> &writes)
{
	for (const auto &w : writes) {
		auto *s = static_cast<RgfxSurfaceShadow *>(w.get());
		if (s->shared && s->region != nullptr && s->region->hasSurfaceWrites()) {
			return s->region;
		}
	}
	return nullptr;
}

void rgfxIOSurfaceWritesPosted(RgfxRegion *region, const std::vector<std::shared_ptr<void>> &writes,
                               uint64_t queueObjId, uint64_t seqno)
{
	// Two words a surface: its shared block and its IOSurfaceID. A process's own block names no
	// surface the kext knows, and no other process reads it.
	std::vector<RgfxSurfaceShadow *> named;
	std::vector<uint64_t> words;
	words.reserve(writes.size() * 2);
	for (const auto &w : writes) {
		auto *s = static_cast<RgfxSurfaceShadow *>(w.get());
		if (!s->shared || s->region != region ||
		    std::find(named.begin(), named.end(), s) != named.end()) {
			continue;
		}
		named.push_back(s);
		words.push_back(s->block.offset);
		words.push_back(s->surfaceId);
	}
	region->noteSurfaceWrites(queueObjId, seqno, words.data(),
	                          static_cast<uint32_t>(words.size() / 2));
	// The kext moved each surface's seed for the GPU's write, which a copy holds: the copy now
	// stands for the surface as of that seed.
	for (RgfxSurfaceShadow *s : named) {
		if (s->inPlace) {
			continue;
		}
		std::lock_guard<std::mutex> g(s->lock);
		s->setSynced(IOSurfaceGetSeed(s->surface));
	}
}

void rgfxIOSurfaceRetire(RgfxRegion *region, uint64_t queueObjId, uint64_t seqno)
{
	region->retireSurfaceWrites(queueObjId, seqno);
}

void rgfxIOSurfaceNoteCommitted(const std::vector<std::shared_ptr<void>> &writes, id queue,
                                uint64_t queueObjId, uint64_t seqno)
{
	std::vector<RgfxSurfaceShadow *> done;
	for (const auto &w : writes) {
		auto *s = static_cast<RgfxSurfaceShadow *>(w.get());
		if (std::find(done.begin(), done.end(), s) != done.end()) {
			continue;
		}
		done.push_back(s);
		s->noteWriter(queue, queueObjId, seqno);
	}
}

void rgfxIOSurfaceNoteRead(const std::vector<std::shared_ptr<void>> &reads, id queue,
                           uint64_t queueObjId, uint64_t seqno)
{
	for (const auto &r : reads)
		static_cast<RgfxSurfaceShadow *>(r.get())->noteReader(queue, queueObjId, seqno);
}

uint64_t rgfxIOSurfaceNilLog(IOSurfaceRef surface, const char *cause, std::string *line)
{
	NilLog &l = nilLog();
	std::lock_guard<std::mutex> g(l.lock);
	auto it = l.seen.find(NilLog::key(surface, cause));
	if (it == l.seen.end()) {
		line->clear();
		return 0;
	}
	*line = it->second.second;
	return it->second.first;
}

bool rgfxIOSurfaceWritesInPlace(const std::vector<std::shared_ptr<void>> &writes)
{
	for (const auto &w : writes) {
		if (!static_cast<RgfxSurfaceShadow *>(w.get())->inPlace) {
			return false;
		}
	}
	return true;
}

void rgfxIOSurfaceWriteBack(const std::vector<std::shared_ptr<void>> &writes)
{
	std::vector<RgfxSurfaceShadow *> done;
	for (const auto &w : writes) {
		auto *s = static_cast<RgfxSurfaceShadow *>(w.get());
		if (std::find(done.begin(), done.end(), s) != done.end()) {
			continue;
		}
		done.push_back(s);
		std::lock_guard<std::mutex> g(s->lock);
		s->writeBack(false);
	}
}
