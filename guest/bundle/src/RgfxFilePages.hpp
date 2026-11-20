/*
 * RgfxFilePages: whether the read-only memory under a no-copy buffer cannot change, as the guest's
 * OS answers once, at the buffer's make (RgfxRegion::acquirePages, kRgfxPagesUnchanging,
 * RGFX_FEATURE_UNCHANGING_PAGES), and why not where it can.
 *
 * UNCHANGING. The memory cannot change where all of these hold at the make:
 *   - the range lies in one VM region of the process, mapped from a file (no submap, no anonymous
 *     memory, no device memory: the region's object has a pager of its own, and the kernel names
 *     a vnode for it, proc_pidinfo's PROC_PIDREGIONPATHINFO);
 *   - the mapping's maximum protection has no write, so the process can never write it: no
 *     mprotect or vm_protect raises a protection past the maximum. A shared mapping of the file
 *     (kRgfxFileMaxNoWrite) or a private copy-on-write one (share mode SM_COW,
 *     kRgfxFileCopyOnWrite), whose maximum protection was taken down with vm_protect's
 *     set_maximum, as mmap gives a private mapping write in its maximum;
 *   - every page the process sees there is the file's own page: the region's object is the file's
 *     (mach_vm_region_recurse's shadow depth 0), or every page of the range resolves to the
 *     file's object below the private objects over it (mach_vm_page_range_query's
 *     VM_PAGE_QUERY_PAGE_EXTERNAL on each). A private mapping of a file always has one such
 *     object over the file's, at a depth of 1, from the make on: xnu enters it through
 *     vm_object_copy_strategically, which for a vnode's object (MEMORY_OBJECT_COPY_DELAY, the
 *     vnode pager's strategy) makes the delayed copy object vm_object_copy_delayed answers. A page
 *     the process wrote before its maximum protection lost write, or one paged out to the
 *     compressor, lies in such an object, and a shadow over it, and is not the file's;
 *   - and the file's volume is mounted read-only (the region's vnode's file system, matched by
 *     its fsid among getfsstat's, has MNT_RDONLY), so nothing writes the file. No path is opened
 *     or looked up: every Metal process loads the bundle, sandboxed ones too, which report a
 *     refused file operation to the kernel's log (tests/bundle_files_check.sh).
 *
 * THE PAGES THE KEXT WIRES are the app's own, as for any pages block. A shared mapping's are the
 * file's pages. A private mapping needs a copy (xnu's vm_map_create_upl resolves it, and
 * vm_object_iopl_request faults each page into the process's private object for the wire), so the
 * wire makes the process a copy of the file's bytes, as a Mac's driver's wire of the app's memory
 * does; no one writes it after, as the process's maximum protection has no write. Either way the
 * GPU reads what the app reads, the file's bytes as of the make.
 *
 * COUNTS. Every make that classifies is counted (rgfxFilePagesNote): its bytes by clause, and by
 * why for memory that is neither, how long the classification took, and the unchanging blocks the
 * kext refused. They are logged at the info level, as totals for the process, within a minute of
 * a change. The why of memory that is neither goes to the host with its block too
 * (kRgfxPagesWhyShift, RGFX_SURFACE_PAGES_WHY), a diagnostic the host counts its copies by.
 */

#ifndef RGFX_FILE_PAGES_HPP_
#define RGFX_FILE_PAGES_HPP_

#include <cstddef>
#include <cstdint>

#include "reims_graphics_wire.h"

/// Why the memory under a no-copy buffer cannot change (RgfxFilePages.hpp), or that it can.
enum RgfxFileClause : uint8_t {
	// A shared mapping of the file whose maximum protection has no write.
	kRgfxFileMaxNoWrite = 0,
	// A private copy-on-write mapping of the file whose maximum protection has no write.
	kRgfxFileCopyOnWrite,
	// Neither: the memory is not provably unchanging.
	kRgfxFileNeither,
	kRgfxFileClauses,
};

/// Why memory is neither (kRgfxFileNeither): the host's RGFX_PAGES_WHY_* values, which the block
/// carries to it.
enum RgfxFileNeither : uint8_t {
	kRgfxNeitherNone = RGFX_PAGES_WHY_UNSAID,
	// Anonymous or device memory, a submap, or a region the kernel names no vnode for.
	kRgfxNeitherNotAFile = RGFX_PAGES_WHY_NOT_A_FILE,
	// The range's pages lie in more than one VM region.
	kRgfxNeitherSpansRegions = RGFX_PAGES_WHY_SPANS_REGIONS,
	// A shared mapping whose maximum protection has write (a file opened for writing).
	kRgfxNeitherWritableMapping = RGFX_PAGES_WHY_WRITABLE_MAPPING,
	// A private copy-on-write mapping whose maximum protection has write, as mmap makes one.
	kRgfxNeitherWritableMaxProtection = RGFX_PAGES_WHY_WRITABLE_MAX_PROTECTION,
	// A page the process sees is not the file's own (it wrote it, or it was paged out).
	kRgfxNeitherShadowed = RGFX_PAGES_WHY_SHADOWED,
	// The file's volume is not mounted read-only.
	kRgfxNeitherWritableVolume = RGFX_PAGES_WHY_WRITABLE_VOLUME,
	// A call failed (the range is not mapped, the region changed meanwhile, no volume matched).
	kRgfxNeitherOther = RGFX_PAGES_WHY_OTHER,
	kRgfxNeitherKinds,
};

/// What rgfxFilePagesOf answered.
struct RgfxFilePages {
	RgfxFileClause clause = kRgfxFileNeither;
	RgfxFileNeither why = kRgfxNeitherOther;
};

/// The clause of the `length` bytes at `address` in this process, now.
RgfxFilePages rgfxFilePagesOf(const void *address, uint64_t length);

/// The process's totals (rgfxFilePagesNote).
struct RgfxFilePagesCounts {
	// Makes that classified.
	uint64_t classified;
	// Their blocks and bytes by clause, and by why for kRgfxFileNeither.
	uint64_t count[kRgfxFileClauses];
	uint64_t bytes[kRgfxFileClauses];
	uint64_t neitherCount[kRgfxNeitherKinds];
	uint64_t neitherBytes[kRgfxNeitherKinds];
	// Unchanging blocks the kext refused, which took the app's memory as read-only instead.
	uint64_t kextRefused;
	uint64_t kextRefusedBytes;
	// The time the classifications took, in all.
	uint64_t ns;
	// How many took each time, four steps an octave: bucket i holds the times from
	// rgfxFilePagesBucketNs(i) up to the next bucket's.
	static const unsigned kBuckets = 256;
	uint64_t buckets[kBuckets];
};

/// The least time, in nanoseconds, of bucket `i` (RgfxFilePagesCounts::buckets).
uint64_t rgfxFilePagesBucketNs(unsigned i);

/// Counts one make that classified: `pages`' clause and why, the `length` bytes asked, the `ns`
/// its classification took, and whether the kext refused the unchanging block. Logs the totals
/// within a minute.
void rgfxFilePagesNote(const RgfxFilePages &pages, uint64_t length, uint64_t ns, bool kextRefused);

/// The process's totals so far.
RgfxFilePagesCounts rgfxFilePagesCounts();

#endif // RGFX_FILE_PAGES_HPP_
