/*
 * RgfxResidencySet: the reims-graphics bundle's MTLResidencySet (macOS 15), the arm64 device's
 * answer to newResidencySetWithDescriptor:error:, as Apple silicon's Metal makes one; the x86
 * device makes none, as the x86 Macs make none (RgfxMakesResidencySets).
 *
 * The device keeps every allocation resident, so a set is bookkeeping alone: it sends nothing to
 * the host, requestResidency and endResidency do nothing, and a queue or a command buffer holds
 * the sets it is given. One thing more: a command buffer that uses a set, or is committed to a
 * queue that holds one, names at its commit each texture over an IOSurface, or view of one, the
 * set holds committed, as a resource it uses (rgfxSurfaces), so that the surface's copy or
 * device cache takes the CPU's writes before the GPU reads it through the set. What the set
 * answers is what the M3 Max's answers
 * (docs/graphics-kext/dgpu-oracle.md, Q31; measured: macOS 27.0.1, tools/oracle/oracle_residency):
 *  - It holds its allocations as three groups: the committed ones, the adds since the last
 *    commit (none of them committed), and the committed ones a remove since then marks. An add
 *    of a marked one takes its mark away, and a remove of an uncommitted add takes the add back;
 *    an add of a committed one, or a remove of one the set does not hold, does nothing.
 *    removeAllAllocations takes back every add and marks every committed one. A commit drops
 *    the marked ones and commits the adds.
 *  - allocationCount counts the committed allocations and the adds, the marked ones among them
 *    until the commit drops them; containsAllocation: and allAllocations answer the adds and the
 *    committed ones not marked (allAllocations in the order they were committed, then added).
 *  - The set retains each allocation it holds, a marked one until the commit drops it, and lets
 *    the dropped ones go at that commit, not when the caller's autorelease pool drains.
 *  - allocatedSize is what the last commit left: 0 before one, then the bytes of the memory the
 *    committed allocations lie in, each piece of memory once however many of them lie there: a
 *    heap's resources count their heap, a view its texture, a texture over a buffer its buffer,
 *    and every other allocation its own allocatedSize. An allocation of no device of ours
 *    counts nothing. (The M3's buffers count the block of 128 KiB its small buffers share; the
 *    device's buffers lie in slabs, and each counts its own pages.)
 *  - A nil allocation added or removed ends the process, as the M3's set does (SIGSEGV there);
 *    containsAllocation: of nil answers NO.
 *  - The label is the descriptor's, as Metal's documentation gives it ("The label specified at
 *    creation"), and setLabel: changes it, as the M3's set answers it. The M3's set answers nil
 *    whatever the descriptor's label: a documented answer is kept whatever a Mac does (class 1).
 *  - The descriptor's initialCapacity is a hint the set does not need, and a nil descriptor makes
 *    a set of none, as the M3 makes one. (The M3 spins without end at a capacity of
 *    NSUIntegerMax.)
 *
 * Its members are safe to call from any thread, as a CAMetalLayer's set is changed from
 * QuartzCore's threads while the app reads it.
 *
 * Our own ObjC++: conforms to MTLResidencySet but subclasses only NSObject, so it instantiates
 * and is unit-tested on the Mac.
 */

#ifndef RGFX_RESIDENCY_SET_H_
#define RGFX_RESIDENCY_SET_H_

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Whether the device makes residency sets: the arm64 device does, as Apple silicon's Metal makes
// them on macOS 15 and later; the x86 device does not, as the x86 Macs answer nil (measured:
// macOS 26.7.1, the AMD Radeon Pro 5300M and the Intel UHD 630), so no queue, command buffer or
// encoder of it holds one.
#if defined(__arm64__)
#define RgfxMakesResidencySets 1
#else
#define RgfxMakesResidencySets 0
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxResidencySet : NSObject <MTLResidencySet>

/// A set of `device` (weak, nil in a unit test) labelled with `descriptor`'s label, holding no
/// allocation. `descriptor` may be nil. Its selector is the one Metal's base device inits the
/// object of allocResidencySet with.
- (instancetype)initWithDevice:(nullable id<MTLDevice>)device
                    descriptor:(nullable MTLResidencySetDescriptor *)descriptor;

- (instancetype)init NS_UNAVAILABLE;

/// Changes the set's label (the descriptor's until then) to a copy of `label`, as Metal's set
/// answers it (_MTLObjectWithLabel's).
- (void)setLabel:(nullable NSString *)label;

/// The committed allocations that are textures over an IOSurface or views of one, in the order
/// they were committed, as the set's last commit left them (a remove since marks one, which
/// stays until that commit drops it); nil where there is none. A command buffer that uses the set,
/// or is committed to a queue that holds it, names each at its commit as a resource it uses
/// (rgfxIOSurfaceNoteUse): it names a cached surface to the host (RGFX_OP_SURFACE_USE) and syncs
/// a copy, where nothing else the buffer encodes may name the surface.
- (nullable NSArray *)rgfxSurfaces;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_RESIDENCY_SET_H_
