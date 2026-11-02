/*
 * RgfxIOSurface: see RgfxIOSurface.h. Our own ObjC++.
 */

#import "RgfxIOSurface.h"

#import "RgfxBuffer.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <tuple>
#include <unistd.h>
#include <utility>

#include "RgfxEncoder.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

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
/// diagnostic, a line in kCopyLog at most every kCopyReportNs while copies run.
struct CopyCount {
	std::atomic<uint64_t> count{0};
	std::atomic<uint64_t> bytes{0};
	std::atomic<uint64_t> ns{0};
};

CopyCount gUploads;    // pages -> the GPU's view
CopyCount gWriteBacks; // the GPU's view -> pages

/// The GPU views the process made of surfaces: in place, or copies, by why.
std::atomic<uint64_t> gInPlace{0};
std::atomic<uint64_t> gCopiedAlignment{0};   // a plane where no host texture lies over it
std::atomic<uint64_t> gCopiedUnsupported{0}; // no view in place on the device, or of its pages
std::atomic<uint64_t> gCopiedRefused{0};     // the kext gave no block in place

const uint64_t kCopyReportNs = 10ull * 1000 * 1000 * 1000;
const char *const kCopyLog = "/tmp/rgfx-surface-copies.log";

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
	                wbNs = 0, inPlace = 0, forAlignment = 0, unsupported = 0, refused = 0;
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
	uint64_t ip = gInPlace, ca = gCopiedAlignment, cu = gCopiedUnsupported, cr = gCopiedRefused;
	char line[384];
	int n = snprintf(line, sizeof(line),
	                 "RGFX: surface copies, pid %d %s, %.1f s: %llu uploads %.1f MiB %.2f ms, "
	                 "%llu write-backs %.1f MiB %.2f ms; views made: %llu in place, copies "
	                 "%llu for alignment, %llu unsupported, %llu refused by the kext\n",
	                 getpid(), getprogname(), (now - last) / 1e9, uc - upCount,
	                 (ub - upBytes) / 1048576.0, (un - upNs) / 1e6, wc - wbCount,
	                 (wb - wbBytes) / 1048576.0, (wn - wbNs) / 1e6, ip - inPlace,
	                 ca - forAlignment, cu - unsupported, cr - refused);
	int fd = open(kCopyLog, O_WRONLY | O_CREAT | O_APPEND, 0666);
	if (fd >= 0 && n > 0) {
		(void)write(fd, line, std::min(static_cast<size_t>(n), sizeof(line) - 1));
	}
	if (fd >= 0) {
		close(fd);
	}
	last = now;
	upCount = uc, upBytes = ub, upNs = un, wbCount = wc, wbBytes = wb, wbNs = wn;
	inPlace = ip, forAlignment = ca, unsupported = cu, refused = cr;
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
/// rows, or at its start in place (RgfxSurfaceShadow::header): the surface's seed when the block
/// last matched the surface's pages, valid where `serial` is the block's own (a block made again
/// holds an older one's bytes until its first holder writes it).
struct RgfxSurfaceHeader {
	uint64_t serial;
	uint32_t seed;
	uint32_t reserved;
};

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
	uint64_t serial = 0;        // the shared block's
	RgfxSurfaceHeader *header = nullptr; // the shared block's, past the planes' rows of a copy
	uint32_t seed = 0;          // this process's own block: the seed when it last matched
	RgfxSurfaceWriters *writers = nullptr; // past the header: the buffers a present waits for
	uint64_t writersAt = 0;                 // their offset in the block
	std::mutex writersLock;                 // one change of this process's slots at a time
	// By slot: the queue this process named there (0: none), and that queue while it lives.
	uint64_t slotQueue[RGFX_SURFACE_WRITER_SLOTS] = {};
	__weak id slotOwner[RGFX_SURFACE_WRITER_SLOTS];

	~RgfxSurfaceShadow();

	/// Names the buffer at `seqno` on `queue` (id `queueId`) among the copy's writers.
	void noteWriter(id queue, uint64_t queueId, uint64_t seqno)
	{
		std::lock_guard<std::mutex> g(writersLock);
		for (uint32_t i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++) {
			if (slotQueue[i] == queueId) {
				// Only this process writes the slot: its seqno only rises.
				uint64_t *s = &writers->writers[i].seqno;
				if (__atomic_load_n(s, __ATOMIC_RELAXED) < seqno)
					__atomic_store_n(s, seqno, __ATOMIC_RELEASE);
				return;
			}
		}
		for (uint32_t i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++) {
			if (slotQueue[i] != 0 && slotOwner[i] == nil) {
				// A queue of ours that is gone, its buffers all finished: the slot is ours to give.
				give(i, queue, queueId, seqno);
				return;
			}
		}
		for (uint32_t i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++) {
			uint64_t empty = 0;
			if (slotQueue[i] == 0 &&
			    __atomic_compare_exchange_n(&writers->writers[i].queue_obj_id, &empty, queueId,
			                                false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
				__atomic_store_n(&writers->writers[i].seqno, seqno, __ATOMIC_RELEASE);
				slotQueue[i] = queueId;
				slotOwner[i] = queue;
				return;
			}
		}
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			NSLog(@"RGFX: IOSurface %u: no writer slot left for queue 0x%llx: a present of the "
			      @"surface may show it before the queue's command buffers finish",
			      surfaceId, (unsigned long long)queueId);
		}
	}

	/// Gives slot `i`, ours, to `queue`: its seqno emptied, then the queue, then the seqno.
	void give(uint32_t i, id queue, uint64_t queueId, uint64_t seqno)
	{
		__atomic_store_n(&writers->writers[i].seqno, 0, __ATOMIC_RELEASE);
		__atomic_store_n(&writers->writers[i].queue_obj_id, queueId, __ATOMIC_RELEASE);
		__atomic_store_n(&writers->writers[i].seqno, seqno, __ATOMIC_RELEASE);
		slotQueue[i] = queueId;
		slotOwner[i] = queue;
	}

	/// Empties this process's slots: the seqno, then the queue.
	void emptyWriters()
	{
		std::lock_guard<std::mutex> g(writersLock);
		for (uint32_t i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++) {
			if (slotQueue[i] == 0)
				continue;
			__atomic_store_n(&writers->writers[i].seqno, 0, __ATOMIC_RELEASE);
			__atomic_store_n(&writers->writers[i].queue_obj_id, 0, __ATOMIC_RELEASE);
			slotQueue[i] = 0;
			slotOwner[i] = nil;
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

	/// Copy -> pages, under a write lock; records the seed the unlock leaves, so the
	/// next commit does not read our own write back. Caller holds `lock`. Nothing in place.
	void writeBack()
	{
		if (inPlace) {
			return;
		}
		uint32_t s = 0;
		uint64_t start = nowNs();
		if (IOSurfaceLock(surface, 0, &s) != kIOReturnSuccess) {
			return;
		}
		for (size_t k = 0; k < planes.size(); k++) {
			const ShadowPlane &p = planes[k];
			uint8_t *dst = planeBase(surface, k);
			if (dst != nullptr) {
				copyRows(p, dst, p.shape.rowBytes, copy() + p.offset, p.pitch);
			}
		}
		IOSurfaceUnlock(surface, 0, &s);
		setSynced(s);
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
			RgfxRegion::PagesAnswer answer = RgfxRegion::kPagesRefused;
			sharedBlock = region->acquireSurfacePages(IOSurfaceGetBaseAddress(surface), ownLength,
			                                          ownSize, &answer);
			if (sharedBlock.block.ok) {
				shadow->inPlace = true;
				countView(gInPlace);
			} else if (answer == RgfxRegion::kPagesUnsupported) {
				countView(gCopiedUnsupported);
				notInPlace = "the device uses no surface in place, or refused its pages";
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
		if (!shadow->shared || sharedBlock.fresh)
			std::memset(static_cast<void *>(shadow->writers), 0, sizeof(RgfxSurfaceWriters));
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

} // namespace

@interface RgfxIOSurfaceTexture ()
- (std::shared_ptr<RgfxSurfaceShadow>)rgfxShadow;
@end

@implementation RgfxIOSurfaceTexture {
	std::shared_ptr<RgfxSurfaceShadow> _shadow;
	NSUInteger _plane;
}

+ (instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                            iosurface:(IOSurfaceRef)surface
                                plane:(NSUInteger)plane
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
	RgfxCreated created = serializer->createTextureFromIOSurface(
	    [(RgfxResource *)shadow->buffer rgfxObjId], p.offset, p.pitch, shadow->surfaceId,
	    static_cast<uint32_t>(plane), hostDesc, shadow->writersAt, shadow->facts());
	if (created.resource_id == 0) {
		// No resource id left: no texture, as Metal answers one it cannot make.
		logNoTexture(surface, "resource id", "every texture resource id is taken");
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
	t->_shadow = std::move(shadow);
	t->_plane = plane;
	[t rgfxTakeDescriptor:descriptor];
	return t;
}

- (std::shared_ptr<RgfxSurfaceShadow>)rgfxShadow
{
	return _shadow;
}

- (IOSurfaceRef)iosurface
{
	return _shadow->surface;
}

// The texture lies at its plane's offset in its view's buffer, in the buffer's allocation, so the
// textures over one plane share both, as Metal's over one surface do (measured on the M3 Max under
// macOS 27: one allocationID and one gpuAddress for two textures and a view over a surface, of any
// storage mode); plane 0's lie at the buffer's start. A buffer over the surface's pages in place
// has no GPU address (its create names none), and nor do its textures.
- (uint64_t)gpuAddress
{
	uint64_t base = _shadow->buffer.gpuAddress;
	return base == 0 ? 0 : base + _shadow->planes[_plane].offset;
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
// pages alone.

- (BOOL)rgfxCpuRegion:(MTLRegion)region level:(NSUInteger)level slice:(NSUInteger)slice
{
	const PlaneShape &s = _shadow->planes[_plane].shape;
	return level == 0 && slice == 0 && s.elementWidth == 1 && s.elementHeight == 1 &&
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
	std::lock_guard<std::mutex> g(_shadow->lock);
	const ShadowPlane &p = _shadow->planes[_plane];
	uint64_t bpe = p.shape.bytesPerElement;
	uint8_t *rows = _shadow->pixels() + p.offset;
	for (NSUInteger r = 0; r < region.size.height; r++) {
		std::memcpy(rows + (region.origin.y + r) * p.pitch + region.origin.x * bpe,
		            static_cast<const uint8_t *>(pixelBytes) + r * bytesPerRow,
		            region.size.width * bpe);
	}
	_shadow->writeBack();
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

void rgfxIOSurfaceSyncBeforeCommit(void)
{
	std::vector<std::shared_ptr<RgfxSurfaceShadow>> live;
	{
		Registry &r = registry();
		std::lock_guard<std::mutex> g(r.lock);
		for (auto it = r.shadows.begin(); it != r.shadows.end();) {
			std::shared_ptr<RgfxSurfaceShadow> s = it->second.lock();
			if (s == nullptr) {
				it = r.shadows.erase(it);
				continue;
			}
			live.push_back(std::move(s));
			++it;
		}
	}
	for (const auto &s : live) {
		s->uploadIfChanged();
	}
}

void rgfxIOSurfaceNoteWrite(RgfxEncoder *enc, id texture)
{
	// A view writes the texels of the texture it views, through every view between.
	id root = texture;
	while ([root isKindOfClass:[RgfxTexture class]] && [(RgfxTexture *)root parentTexture] != nil) {
		root = [(RgfxTexture *)root parentTexture];
	}
	if (enc != nullptr && [root isKindOfClass:[RgfxIOSurfaceTexture class]]) {
		enc->noteSurfaceWrite([(RgfxIOSurfaceTexture *)root rgfxShadow]);
	}
}

void rgfxIOSurfaceNoteShaderWrite(RgfxEncoder *enc, id texture)
{
	if ([texture isKindOfClass:[RgfxTexture class]] &&
	    ([(id<MTLTexture>)texture usage] & MTLTextureUsageShaderWrite) != 0) {
		rgfxIOSurfaceNoteWrite(enc, texture);
	}
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
		s->writeBack();
	}
}
