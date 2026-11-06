// ReimsGraphicsLoader, a Lilu plugin: it binds ReimsGraphics, which OpenCore
// injects into the Boot kernel collection, to the System collection's
// libraries it uses (README.md; docs/graphics-kext/design.md, item 13).
// OpenCore links ReimsGraphics dormant, with no personality, against
// generated shims in place of those libraries (tools/gen-shims.py). When a
// catalogue send runs, this loads and retains the real libraries, checks
// that the library classes ReimsGraphics derives from are laid out as it was
// built against, points every site of it from its shim's placeholder to the
// real symbol, and adds its personalities without matching. The send's own
// matching then matches them with the System collection's, so probe score
// decides. ReimsGraphics binds on its own PCI function (1234:5247), which
// only its personalities match, so this depends on no other driver. It
// checks no library version: the OpenCore entries' MinKernel and MaxKernel
// bound the releases the set is injected on, and the layout checks refuse a
// release whose classes differ.
#include <Headers/kern_api.hpp>
#include <Headers/kern_patcher.hpp>
#include <Headers/kern_util.hpp>
#include <Headers/plugin_start.hpp>
#include <Headers/kern_mach.hpp>
#include <IOKit/IOCatalogue.h>
#include <IOKit/IOPlatformExpert.h>
#include <IOKit/IOService.h>
#include <libkern/c++/OSArray.h>
#include <libkern/c++/OSDictionary.h>
#include <libkern/c++/OSCollectionIterator.h>
#include <libkern/c++/OSString.h>
#include <libkern/c++/OSSymbol.h>
#include <libkern/c++/OSBoolean.h>
#include <libkern/c++/OSNumber.h>
#include <libkern/c++/OSUnserialize.h>
#include <libkern/OSKextLib.h>
#include <kern/clock.h>
#include <kern/thread.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include "Fixup.h"
#include "Log.h"
#include "Rows.h"

extern IOCatalogue *gIOCatalogue;

// The log (Log.h): IOLog with the uptime, Lilu's always-on SYSLOG, so each
// line reaches `log show` with no boot argument, and the whole log in the
// plugin service's LoaderLog property, so a line from before the service
// started shows once it has. Callers run on whichever thread Lilu or xnu
// calls them on, so each line reserves its place in the text before writing.
namespace {
constexpr const char *kLog = "rgfxload";
char logText[8192];
size_t logLength = 0;
} // namespace

void note(const char *fmt, ...) {
    char line[320];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    uint64_t now, nanos;
    clock_get_uptime(&now);
    absolutetime_to_nanoseconds(now, &nanos);
    IOLog("%s: [%llu ms] %s\n", kLog, nanos / 1000000, line);
    SYSLOG(kLog, "[%llu ms] %s", nanos / 1000000, line);
    size_t length = strlen(line);
    size_t at = __atomic_fetch_add(&logLength, length + 1, __ATOMIC_RELAXED);
    if (at + length + 1 < sizeof(logText)) {  // the last byte stays NUL
        memcpy(logText + at, line, length);
        logText[at + length] = '\n';
    }
    if (auto service = ADDPR(selfInstance))
        service->setProperty("LoaderLog", logText);
}

namespace {
// The generated tables' rows (tools/gen-tables.py), in Rows.h.
using namespace Bridge;
#include "tables.h"

// What the code below indexes by: each consumer has the next run of
// libraries, its sites and classes name only those, its classes' slots
// follow one another in its slots, and each kernel method is in
// kernelSymbols.
constexpr bool tablesConsistent() {
    size_t next = 0;
    for (size_t c = 0; c < arrsize(consumers); ++c) {
        auto &consumer = consumers[c];
        if (consumer.firstLibrary != next || !consumer.libraryCount)
            return false;
        next += consumer.libraryCount;
        for (size_t s = 0; s < consumer.siteCount; ++s)
            if (consumer.sites[s].library - consumer.firstLibrary >= consumer.libraryCount)
                return false;
        size_t slot = 0;
        for (size_t l = 0; l < consumer.layoutCount; ++l) {
            auto &layout = consumer.layouts[l];
            if (layout.library - consumer.firstLibrary >= consumer.libraryCount ||
                layout.firstSlot != slot || !layout.slotCount)
                return false;
            slot += layout.slotCount;
        }
        if (!consumer.layoutCount || slot != consumer.slotTotal)
            return false;
        for (size_t s = 0; s < consumer.slotTotal; ++s)
            if (consumer.slots[s].kernel >= static_cast<int32_t>(arrsize(kernelSymbols)))
                return false;
    }
    return next == arrsize(libraries);
}
static_assert(arrsize(consumers) <= 32, "published has a bit per consumer");
static_assert(tablesConsistent(), "tables.h does not index as Loader.cpp does");

// xnu libkern/libkern/OSKextLibPrivate.h:167; not in the kernel SDK.
constexpr const char *kBundleStartedKey = "OSBundleStarted";

// Private static OSKext entry points, resolved with Lilu, so no OSKext
// layout or vtable slot is assumed (xnu libkern/libkern/c++/OSKext.h,
// OSKextLibPrivate.h). The three declared to return OSPtr return a retained
// raw pointer: OSSharedPtr is trivial_abi (intrusive_shared_ptr.h:124).
using Lookup = OSObject *(*)(const char *);
using CopyInfo = OSDictionary *(*)(OSArray *, OSArray *);
using SendAll = void (*)(bool);
using CopyAll = OSArray *(*)(bool);
Lookup lookup = nullptr;
CopyInfo copyInfo = nullptr;
SendAll sendAll = nullptr;  // the originals, once routed
CopyAll copyAll = nullptr;
bool patcherLoaded = false; // onPatcherLoad has run (startOnGoplessBoot)

// Per library: Lilu's watch for its load, and what binding to it needs.
KernelPatcher::KextInfo watched[arrsize(libraries)];
const char *watchedPaths[arrsize(libraries)];
struct LibraryState {
    mach_vm_address_t base;  // from its load callback
    OSObject *hold;          // its OSKext, retained for the loader's lifetime
    MachInfo *image;         // its symbols, while its consumer binds
    uint64_t shimBase;
};
LibraryState state[arrsize(libraries)];
// Each kernel method a library class's vtable holds, solved by name when Lilu
// loads the patcher; 0 if the kernel has none.
uint64_t kernelAddresses[arrsize(kernelSymbols)];

// The first load command of mh that matches; null if none does, or at the
// first one that is malformed.
template <typename Match>
const load_command *findCommand(const mach_header_64 *mh, Match matches) {
    auto at = reinterpret_cast<const uint8_t *>(mh + 1);
    auto end = at + mh->sizeofcmds;
    for (uint32_t i = 0; i < mh->ncmds; ++i) {
        auto lc = reinterpret_cast<const load_command *>(at);
        auto left = static_cast<size_t>(end - at);
        if (left < sizeof(*lc) || lc->cmdsize < sizeof(*lc) || lc->cmdsize > left)
            return nullptr;
        if (matches(lc))
            return lc;
        at += lc->cmdsize;
    }
    return nullptr;
}

// The address of the Boot KC's fileset entry `id`, else 0. xnu has slid the
// entries and their segments (osfmk/mach/dyld_kernel_fixups.h,
// kernel_collection_slide), so this finds our consumers and shims, which are
// never loaded or initialized.
uint64_t filesetEntry(const char *id) {
    auto kernel = MachInfo::create(true);
    if (!kernel)
        return 0;
    auto mh = reinterpret_cast<const mach_header_64 *>(kernel->findKernelBase());
    kernel->deinit();
    MachInfo::deleter(kernel);
    if (!mh || mh->magic != MH_MAGIC_64 || mh->filetype != MH_FILESET)
        return 0;
    auto lc = findCommand(mh, [id](const load_command *lc) {
        if (lc->cmd != LC_FILESET_ENTRY || lc->cmdsize < sizeof(fileset_entry_command))
            return false;
        auto offset = reinterpret_cast<const fileset_entry_command *>(lc)->entry_id.offset;
        if (offset < sizeof(fileset_entry_command) || offset >= lc->cmdsize)
            return false;
        auto name = reinterpret_cast<const char *>(lc) + offset;
        return memchr(name, 0, lc->cmdsize - offset) && !strcmp(name, id);
    });
    return lc ? reinterpret_cast<const fileset_entry_command *>(lc)->vmaddr : 0;
}

// Whether [at, at + width) lies in one segment, not __LINKEDIT, of the image
// at base.
bool inImage(uint64_t base, uint64_t at, size_t width) {
    auto mh = reinterpret_cast<const mach_header_64 *>(base);
    if (!mh || mh->magic != MH_MAGIC_64 || at < base || at + width < at)
        return false;
    return findCommand(mh, [at, width](const load_command *lc) {
        if (lc->cmd != LC_SEGMENT_64 || lc->cmdsize < sizeof(segment_command_64))
            return false;
        auto seg = reinterpret_cast<const segment_command_64 *>(lc);
        return strncmp(seg->segname, SEG_LINKEDIT, sizeof(seg->segname)) && at >= seg->vmaddr &&
               width <= seg->vmsize && at - seg->vmaddr <= seg->vmsize - width;
    }) != nullptr;
}

// Whether the loaded image at `base` has a symbol `name` at `address`. A
// library's local method can have several copies of one name, one per unit
// that compiled it (`nm` lists four of IOGPUMemory::getLength() const in
// 25G241's IOGPUFamily), and Lilu's solveSymbol gives the first. This reads
// the image's symbol table where a kernel collection's kext has it in
// memory, as Lilu's MachInfo::kcGetRunningAddresses finds it (Lilu 1.7.2
// kern_mach.cpp:846-941): its __LINKEDIT segment and LC_SYMTAB, each symbol
// at its final address (MachInfo::solveSymbol, :331-378).
bool namedAt(uint64_t base, const char *name, uint64_t address) {
    auto mh = reinterpret_cast<const mach_header_64 *>(base);
    if (!mh || mh->magic != MH_MAGIC_64)
        return false;
    const segment_command_64 *linkedit = nullptr;
    const symtab_command *symtab = nullptr;
    findCommand(mh, [&](const load_command *lc) {
        if (lc->cmd == LC_SEGMENT_64 && lc->cmdsize >= sizeof(segment_command_64) &&
            !strncmp(reinterpret_cast<const segment_command_64 *>(lc)->segname, SEG_LINKEDIT, 16))
            linkedit = reinterpret_cast<const segment_command_64 *>(lc);
        else if (lc->cmd == LC_SYMTAB && lc->cmdsize >= sizeof(symtab_command))
            symtab = reinterpret_cast<const symtab_command *>(lc);
        return false;
    });
    if (!linkedit || !symtab || symtab->symoff < linkedit->fileoff || symtab->stroff < linkedit->fileoff)
        return false;
    uint64_t symbols = symtab->symoff - linkedit->fileoff, strings = symtab->stroff - linkedit->fileoff;
    if (symbols + uint64_t(symtab->nsyms) * sizeof(nlist_64) > linkedit->vmsize ||
        strings + symtab->strsize > linkedit->vmsize)
        return false;
    auto list = reinterpret_cast<const nlist_64 *>(linkedit->vmaddr + symbols);
    auto text = reinterpret_cast<const char *>(linkedit->vmaddr + strings);
    size_t length = strlen(name) + 1;
    for (uint32_t i = 0; i < symtab->nsyms; ++i) {
        auto &entry = list[i];
        if (entry.n_value != address || (entry.n_type & N_STAB) || (entry.n_type & N_TYPE) != N_SECT)
            continue;
        uint32_t at = entry.n_un.n_strx;
        if (at < symtab->strsize && symtab->strsize - at >= length && !memcmp(text + at, name, length))
            return true;
    }
    return false;
}

// OSKext's information about the loaded kext `id`, retained; null if there
// is none: all of it, or only `keys` (OSKext::copyLoadedKextInfo).
OSDictionary *kextInfo(const char *id, const char *const *keys = nullptr, size_t keyCount = 0) {
    OSString *name = OSString::withCString(id);
    const OSObject *names[] {name};
    OSArray *ids = name ? OSArray::withObjects(names, 1) : nullptr;
    OSArray *wanted = keyCount ? OSArray::withCapacity(static_cast<unsigned>(keyCount)) : nullptr;
    bool complete = ids && (!keyCount || wanted);
    for (size_t i = 0; complete && i < keyCount; ++i) {
        auto key = OSString::withCString(keys[i]);
        complete = key && wanted->setObject(key);
        OSSafeReleaseNULL(key);
    }
    OSDictionary *all = complete ? copyInfo(ids, wanted) : nullptr;
    auto info = all ? OSDynamicCast(OSDictionary, all->getObject(id)) : nullptr;
    if (info)
        info->retain();
    OSSafeReleaseNULL(all);
    OSSafeReleaseNULL(wanted);
    OSSafeReleaseNULL(ids);
    OSSafeReleaseNULL(name);
    return info;
}

// The loaded kext `id`'s in-memory load address (OSBundleLoadAddress), or 0.
// A library that loaded before the loader registered its watch gets no load
// callback, so this recovers its base when state[].base is still 0.
uint64_t loadAddress(const char *id) {
    const char *keys[] {"OSBundleLoadAddress"};
    auto info = kextInfo(id, keys, arrsize(keys));
    auto number = info ? OSDynamicCast(OSNumber, info->getObject("OSBundleLoadAddress")) : nullptr;
    uint64_t address = number ? number->unsigned64BitValue() : 0;
    OSSafeReleaseNULL(info);
    return address;
}

// Whether the loaded library has started.
bool started(const Library &lib) {
    const char *keys[] {kBundleStartedKey};
    auto info = kextInfo(lib.id, keys, arrsize(keys));
    auto flag = info ? OSDynamicCast(OSBoolean, info->getObject(kBundleStartedKey)) : nullptr;
    bool running = flag && flag->isTrue();
    if (!running)
        note("%s: loaded but not started", lib.id);
    OSSafeReleaseNULL(info);
    return running;
}

// Loads library `index` and opens its image and its shim's for binding.
// OSKextLoadKextWithIdentifier starts no matching (xnu OSKext.h:809). Before
// the System KC is registered, the library has no OSKext, and this refuses
// until a later send.
bool openLibrary(size_t index) {
    auto &lib = libraries[index];
    auto &s = state[index];
    // Hold before requesting the load, closing an unload window after it.
    s.hold = lookup(lib.id);
    if (!s.hold) {
        note("%s: no OSKext to retain", lib.id);
        return false;
    }
    auto status = OSKextLoadKextWithIdentifier(lib.id);
    if (status != kOSReturnSuccess) {
        note("%s: load failed %#x", lib.id, status);
        return false;
    }
    if (!started(lib))
        return false;
    auto base = __atomic_load_n(&s.base, __ATOMIC_ACQUIRE);
    if (!base) {
        // No load callback (the library loaded before the loader's watch):
        // take its load address from OSKext instead.
        base = loadAddress(lib.id);
        if (base) {
            __atomic_store_n(&s.base, base, __ATOMIC_RELEASE);
            note("%s: no load callback; OSBundleLoadAddress %#llx", lib.id, base);
        }
    }
    if (!base) {
        note("%s: no load callback and no OSBundleLoadAddress", lib.id);
        return false;
    }
    s.shimBase = filesetEntry(lib.shim);
    if (!s.shimBase)
        note("%s: shim %s has no Boot-KC fileset entry", lib.id, lib.shim);
    s.image = MachInfo::create(false, lib.id);
    if (!s.shimBase || !s.image || s.image->init(&lib.path) != KERN_SUCCESS ||
        s.image->getRunningAddresses(base) != KERN_SUCCESS) {
        note("%s: cannot open loaded image or find shim (shim base %#llx)", lib.id, s.shimBase);
        return false;
    }
    note("%s: loaded at %#llx and retained", lib.id, base);
    return true;
}

// Frees the consumer's library images. Their OSKext references stay held for
// the loader's lifetime, also after a later check or publication failed.
void closeLibraries(const Consumer &consumer) {
    for (size_t i = consumer.firstLibrary; i < consumer.firstLibrary + consumer.libraryCount; ++i) {
        auto &image = state[i].image;
        if (image) {
            image->deinit();
            MachInfo::deleter(image);
            image = nullptr;
        }
    }
}

// The consumer's Boot-KC address, if it is there and OSKext has not loaded it.
uint64_t dormantBase(const Consumer &consumer) {
    auto base = filesetEntry(consumer.id);
    auto loaded = kextInfo(consumer.id);
    bool dormant = !loaded;
    OSSafeReleaseNULL(loaded);
    if (!base || !dormant) {
        note("%s: missing Boot-KC image or already loaded", consumer.name);
        return 0;
    }
    return base;
}

// Each site's new value, once every site passes: its symbol resolves, it and
// a branch's opcode lie in the consumer, its placeholder lies in the shim,
// and it holds what OpenCore linked there. No trampoline: an out-of-range
// rel32 leaves the whole consumer dormant.
bool plan(const Consumer &consumer, uint64_t base, uint64_t *values) {
    for (size_t i = 0; i < consumer.siteCount; ++i) {
        auto &site = consumer.sites[i];
        auto &lib = state[site.library];
        auto at = base + site.offset;
        auto placeholder = lib.shimBase + site.placeholder;
        auto real = lib.image->solveSymbol(site.symbol);
        if (!real || (site.width != 4 && site.width != 8) || !inImage(base, at, site.width) ||
            !inImage(lib.shimBase, placeholder, 1) || (site.width == 4 && !inImage(base, at - 1, 1))) {
            note("%s: invalid %s site +%#x or missing %s", consumer.name, site.kind, site.offset, site.symbol);
            return false;
        }
        uint64_t old = 0;
        uint8_t opcode = 0;
        lilu_os_memcpy(&old, reinterpret_cast<void *>(at), site.width);
        if (site.width == 4)
            lilu_os_memcpy(&opcode, reinterpret_cast<void *>(at - 1), 1);
        auto result = Bridge::replacement(site.width, site.opcode, opcode, at, old, placeholder, real,
                                          site.addend, values[i]);
        if (result != Bridge::OK) {
            note("%s: %s +%#x %s failed the %s check (old %#llx)", consumer.name, site.kind,
                 site.offset, site.symbol, Bridge::describe(result), old);
            return false;
        }
    }
    note("%s: all %zu sites resolved and placeholders checked", consumer.name, consumer.siteCount);
    return true;
}

// Writes every planned value, then reads each back.
bool write(const Consumer &consumer, uint64_t base, const uint64_t *values) {
    auto status = MachInfo::setKernelWriting(true, KernelPatcher::kernelWriteLock);
    bool written = status == KERN_SUCCESS;
    if (written) {
        for (size_t i = 0; i < consumer.siteCount; ++i)
            lilu_os_memcpy(reinterpret_cast<void *>(base + consumer.sites[i].offset), &values[i],
                           consumer.sites[i].width);
        status = MachInfo::setKernelWriting(false, KernelPatcher::kernelWriteLock);
        written = status == KERN_SUCCESS;
        for (size_t i = 0; written && i < consumer.siteCount; ++i)
            written = !memcmp(reinterpret_cast<void *>(base + consumer.sites[i].offset), &values[i],
                              consumer.sites[i].width);
    }
    note("%s: %s writing %zu sites (status %#x)", consumer.name,
         written ? "finished" : "FAILED", consumer.siteCount, status);
    return written;
}

// Whether each library class the consumer derives from is laid out as the
// table's KDK lays it out, compared by Rows.h's compareLayout:
// - its size: OSMetaClass::getClassSize (xnu OSMetaClass.cpp:772) of the
//   class's OSMetaClass, which OSMetaClass::getMetaClassWithName (:1233)
//   finds once the library has loaded;
// - its vtable in the loaded library, `vtable` solved in its image and read
//   from its address point, two words in: each slot the address of the
//   method the KDK has there: the slot's own address if the library's
//   symbol table names it so (namedAt), else the method's address solved in
//   the library's image (Lilu's MachInfo::solveSymbol) or, for a kernel
//   method, in the kernel when the patcher loaded; then the zero word that
//   follows the KDK's last slot.
// The first difference refuses the consumer, before any site is written.
bool layoutsMatch(const Consumer &consumer) {
    for (size_t i = 0; i < consumer.layoutCount; ++i) {
        auto &layout = consumer.layouts[i];
        auto &lib = state[layout.library];
        auto name = OSSymbol::withCString(layout.name);
        auto meta = name ? OSMetaClass::getMetaClassWithName(name) : nullptr;
        OSSafeReleaseNULL(name);
        auto vtable = lib.image->solveSymbol(layout.vtable);
        uint64_t point = vtable ? vtable + 2 * sizeof(uint64_t) : 0;
        uint32_t at = 0;
        uint64_t got = 0, want = 0;
        auto size = [meta] { return meta ? meta->getClassSize() : 0u; };
        auto read = [&lib, point](uint32_t slot, uint64_t &value) {
            auto address = point + slot * sizeof(uint64_t);
            if (!point || !inImage(lib.base, address, sizeof(value)))
                return false;
            lilu_os_memcpy(&value, reinterpret_cast<void *>(address), sizeof(value));
            return true;
        };
        auto resolve = [&lib](const Slot &slot, uint64_t value) -> uint64_t {
            if (slot.kernel >= 0)
                return kernelAddresses[slot.kernel];
            return namedAt(lib.base, slot.symbol, value) ? value : lib.image->solveSymbol(slot.symbol);
        };
        auto result = compareLayout(layout, consumer.slots, size, read, resolve, at, got, want);
        if (result == Mismatch::Size)
            note("%s: REFUSED: %s's size is %llu, built against %llu", consumer.name, layout.name, got, want);
        else if (result == Mismatch::Slot)
            note("%s: REFUSED: %s's vtable slot %u holds %#llx, built against %s at %#llx", consumer.name,
                 layout.name, at, got, consumer.slots[layout.firstSlot + at].symbol, want);
        else if (result == Mismatch::Symbol)
            note("%s: REFUSED: %s's vtable slot %u, built against %s, which is not found", consumer.name,
                 layout.name, at, consumer.slots[layout.firstSlot + at].symbol);
        else if (result == Mismatch::Length)
            note("%s: REFUSED: %s's vtable has more than %u slots (%#llx)", consumer.name, layout.name, at, got);
        else if (result != Mismatch::None)
            note("%s: REFUSED: %s: %s (slot %u)", consumer.name, layout.name, describe(result), at);
        if (result != Mismatch::None)
            return false;
    }
    note("%s: %zu library classes laid out as built: their sizes and %zu vtable slots", consumer.name,
         consumer.layoutCount, consumer.slotTotal);
    return true;
}

// Binds a dormant consumer to its loaded libraries: no site is written
// unless every site of it can be, and every library class it derives from
// is laid out as it was built against.
bool bind(const Consumer &consumer) {
    auto base = dormantBase(consumer);
    if (!base)
        return false;
    for (size_t i = consumer.firstLibrary; i < consumer.firstLibrary + consumer.libraryCount; ++i)
        if (!openLibrary(i))
            return false;
    if (!layoutsMatch(consumer))
        return false;
    auto size = consumer.siteCount * sizeof(uint64_t);
    auto values = static_cast<uint64_t *>(IOMalloc(size));
    if (!values) {
        note("%s: cannot allocate fixup plan", consumer.name);
        return false;
    }
    bool bound = plan(consumer, base, values) && write(consumer, base, values);
    IOFree(values, size);
    return bound;
}

// The dictionary's values, as addDrivers takes personalities; null unless
// there is at least one and each was taken.
OSArray *personalityArray(OSDictionary *personalities) {
    auto iterator = personalities ? OSCollectionIterator::withCollection(personalities) : nullptr;
    auto array = iterator ? OSArray::withCapacity(personalities->getCount()) : nullptr;
    bool complete = array != nullptr;
    while (complete) {
        auto key = OSDynamicCast(OSSymbol, iterator->getNextObject());
        if (!key)
            break;
        complete = array->setObject(personalities->getObject(key));
    }
    if (!complete || !array->getCount() || array->getCount() != personalities->getCount())
        OSSafeReleaseNULL(array);
    OSSafeReleaseNULL(iterator);
    return array;
}

// Removes every personality of the bundle `id` from the catalogue.
void withdraw(const char *id) {
    auto identifier = OSString::withCString(id);
    auto matching = identifier ? OSDictionary::withCapacity(1) : nullptr;
    if (matching && matching->setObject(kCFBundleIdentifierKey, identifier) && gIOCatalogue)
        gIOCatalogue->removeDrivers(matching, false);
    OSSafeReleaseNULL(matching);
    OSSafeReleaseNULL(identifier);
}

// Adds a bound consumer's personalities without matching: the catalogue
// send that follows matches them in its own pass. On a failure, none of
// them.
bool publish(const Consumer &consumer) {
    OSString *error = nullptr;
    auto parsed = OSUnserializeXML(consumer.personality, &error);
    auto personalities = personalityArray(OSDynamicCast(OSDictionary, parsed));
    bool published = false;
    if (personalities && gIOCatalogue) {
        published = gIOCatalogue->addDrivers(personalities, false);
        // A failed addDrivers starts no matching, but may already have added
        // some of them (xnu IOCatalogue.cpp:411). Remove those too.
        if (!published)
            withdraw(consumer.id);
    }
    note("%s: %s personalities%s%s", consumer.name, published ? "published" : "FAILED to publish",
         error ? ": " : "", error ? error->getCStringNoCopy() : "");
    OSSafeReleaseNULL(personalities);
    OSSafeReleaseNULL(parsed);
    OSSafeReleaseNULL(error);
    return published;
}

bool interfacesResolved() {
    return lookup && copyInfo;
}

// The consumers bound and published, one bit each.
uint32_t published = 0;
bool publishing = false;

// Binds and publishes each consumer once, as soon as its libraries are
// loaded and started so it can be bound. Called from both send hooks, so
// every catalogue send retries the consumers not yet published, until each
// binds: the first sends run before the System KC is registered, when its
// libraries have no OSKext.
void publishAll() {
    if (!interfacesResolved()) {
        note("publish: kernel interfaces not resolved yet; waiting");
        return;
    }
    if (__atomic_exchange_n(&publishing, true, __ATOMIC_ACQ_REL)) {
        note("publish: already running on another thread");
        return;
    }
    for (size_t i = 0; i < arrsize(consumers); ++i) {
        auto &consumer = consumers[i];
        if (published >> i & 1)
            continue;
        note("publish: binding %s", consumer.name);
        bool bound = bind(consumer);
        if (bound && publish(consumer)) {
            published |= 1u << i;
            note("publish: %s PUBLISHED", consumer.name);
        } else
            note("publish: %s not ready (%s); retrying on the next send", consumer.name,
                 bound ? "publication failed" : "bind failed");
        closeLibraries(consumer);
    }
    __atomic_store_n(&publishing, false, __ATOMIC_RELEASE);
}

// OSKext::loadFileSetKexts calls this after registering each collection,
// holding its recursive kext lock (OSKext.cpp:13258, :13429). The
// consumers publish before the send, so its matching includes them.
void onSendAll(bool startMatching) {
    note("catalogue send: matching %u", startMatching);
    publishAll();
    sendAll(startMatching);
    note("catalogue send returned");
}

// The send collects its personalities here
// (OSKext::sendAllKextPersonalitiesToCatalog, OSKext.cpp:14944), so binding
// here is still before its matching. The library loads run on the send's
// thread, outside Lilu's load callback.
OSArray *onCopyAll(bool filterSafeBootFlag) {
    note("personalities collected");
    publishAll();
    return copyAll(filterSafeBootFlag);
}

template <typename Function>
Function solve(KernelPatcher &patcher, const char *symbol) {
    auto address = patcher.solveSymbol(KernelPatcher::KernelID, symbol);
    patcher.clearError();
    return reinterpret_cast<Function>(address);
}

void onPatcherLoad(void *, KernelPatcher &patcher) {
    __atomic_store_n(&patcherLoaded, true, __ATOMIC_RELEASE);
    lookup = solve<Lookup>(patcher, "__ZN6OSKext24lookupKextWithIdentifierEPKc");
    copyInfo = solve<CopyInfo>(patcher, "__ZN6OSKext18copyLoadedKextInfoEP7OSArrayS1_");
    note("kernel interfaces: lookup %s, info %s", lookup ? "yes" : "NO", copyInfo ? "yes" : "NO");
    if (!interfacesResolved())
        return;
    // A method not found here refuses the classes whose vtables hold it
    // (layoutsMatch), naming it.
    size_t solved = 0;
    for (size_t i = 0; i < arrsize(kernelSymbols); ++i)
        solved += (kernelAddresses[i] = solve<uint64_t>(patcher, kernelSymbols[i])) != 0;
    note("kernel methods in library vtables: %zu of %zu solved", solved, arrsize(kernelSymbols));
    KernelPatcher::RouteRequest routes[] {
        {"__ZN6OSKext24copyAllKextPersonalitiesEb", onCopyAll, copyAll},
        {"__ZN6OSKext33sendAllKextPersonalitiesToCatalogEb", onSendAll, sendAll},
    };
    bool installed = patcher.routeMultiple(KernelPatcher::KernelID, routes, arrsize(routes));
    patcher.clearError();
    note("catalogue send hooks: %s", installed ? "installed" : "FAILED; consumers stay dormant");
}

void onKextLoad(void *, KernelPatcher &, size_t id, mach_vm_address_t base, size_t) {
    for (size_t i = 0; i < arrsize(watched); ++i) {
        if (id == watched[i].loadIndex) {
            __atomic_store_n(&state[i].base, base, __ATOMIC_RELEASE);
            note("%s: library load callback", watched[i].id);
        }
    }
}

// Lilu 1.7.2 on an x86_64 kernel of Big Sur or later calls its plugins'
// onPatcherLoad only from its route of PE_initialize_console, on the first
// kPEEnableScreen (kern_start.cpp, Configuration::registerPolicy, which then
// registers no MAC policy, and Configuration::initConsole). A framebuffer
// sends that op when it takes the boot console. A boot with no boot
// framebuffer never sends it: OpenCore logs "OCC: Missing compatible GOP",
// PE_state.video has no base address, and Lilu initialises no plugin, so
// the loader would hook no send and publish nothing. Measured on msi-laptop,
// 2026-10-30, macOS 26.6.1 (25G76) with only reims-graphics-pci (1234:5247)
// and no option ROM: the serial log held the loader's "started" line and no
// later one, and IONDRVFramebuffer took 1234:5247. So on such a boot, once
// BSD is up (IOKitBSDInit publishes IOBSD, before kernelmanagerd can
// register the System KC), this sends kPEEnableScreen through the platform
// expert, as a framebuffer would (IOPlatformExpert::setConsoleInfo, xnu
// IOPlatformExpert.cpp:412-416), and Lilu initialises its plugins. With no
// info, xnu's PE_initialize_console leaves PE_state.video as it is
// (pexpert/i386/pe_init.c), and initialize_screen only resumes an acquired
// console (osfmk/console/video_console.c:2980-2984), which a boot with no
// framebuffer has never paused. A boot with a boot framebuffer is left to
// its framebuffer.
void startOnGoplessBoot(void *, wait_result_t) {
    auto matching = IOService::resourceMatching("IOBSD");
    auto bsd = matching ? IOService::waitForMatchingService(matching) : nullptr;
    OSSafeReleaseNULL(bsd);
    OSSafeReleaseNULL(matching);
    PE_Video video {};
    auto platform = IOService::getPlatform();
    bool console = platform && platform->getConsoleInfo(&video) == kIOReturnSuccess;
    bool loaded = __atomic_load_n(&patcherLoaded, __ATOMIC_ACQUIRE);
    note("BSD is up: Lilu's patcher %s; boot framebuffer %s %#lx (%lux%lu)", loaded ? "loaded" : "not loaded",
         console ? "at" : "unknown,", video.v_baseAddr, video.v_width, video.v_height);
    if (loaded || !console || video.v_baseAddr)
        return;
    note("no boot framebuffer: sending kPEEnableScreen so Lilu initialises its plugins");
    platform->setConsoleInfo(nullptr, kPEEnableScreen);
}

void start() {
    note("started; waiting for a catalogue send");
    // Every library in one array, registered once. In a kernel collection
    // Lilu reads the image from memory and only checks that the path names
    // a kext.
    for (size_t i = 0; i < arrsize(libraries); ++i) {
        watchedPaths[i] = libraries[i].path;
        watched[i].id = libraries[i].id;
        watched[i].paths = &watchedPaths[i];
        watched[i].pathNum = 1;
        watched[i].sys[KernelPatcher::KextInfo::Loaded] = true;
    }
    lilu.onPatcherLoadForce(onPatcherLoad);
    lilu.onKextLoadForce(watched, arrsize(watched), onKextLoad);
    thread_t waiter = nullptr;
    if (kernel_thread_start(startOnGoplessBoot, nullptr, &waiter) == KERN_SUCCESS)
        thread_deallocate(waiter);
    else
        note("FAILED to start the BSD waiter; a boot with no boot framebuffer stays dormant");
}
const char *bootArgOff[] {"-rgfxloadoff"};
const char *bootArgDebug[] {"-rgfxloaddbg"};
} // namespace

PluginConfiguration ADDPR(config) {
    xStringify(PRODUCT_NAME), parseModuleVersion(xStringify(MODULE_VERSION)), LiluAPI::AllowNormal,
    bootArgOff, arrsize(bootArgOff), bootArgDebug, arrsize(bootArgDebug), nullptr, 0,
    KernelVersion::Tahoe, KernelVersion::Tahoe, start,
};
// The kmod_info OpenCore and xnu look for.
extern "C" kern_return_t ADDPR(kern_start)(kmod_info_t *, void *);
extern "C" kern_return_t ADDPR(kern_stop)(kmod_info_t *, void *);
KMOD_EXPLICIT_DECL(com.steelbrain.ReimsGraphicsLoader, xStringify(MODULE_VERSION), ADDPR(kern_start),
                   ADDPR(kern_stop))
