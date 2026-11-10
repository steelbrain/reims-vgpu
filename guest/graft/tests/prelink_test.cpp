// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of the prelink check (sdk/tool/prelink.h) on synthetic collections
// built from the staged x86-64 fixture consumer, which make test builds with
// graft-kext build.
//
// A small test linker places the staged consumer into an x86-64 fileset at a
// kernel address, as OpenCore does: segment file offsets become
// collection-absolute, PC-relative imports are resolved, and every pointer
// becomes a chained rebase (DYLD_CHAINED_PTR_X86_64_KERNEL_CACHE). Rebase
// targets count from the lowest segment, region offsets from the header, and the
// injected region's starts sit after the table's declared size. All values come
// from the staged image's own relocations, not its manifest. The collections
// hold only the fixture; no Apple binary is read.
//
//     prelink_test GRAFT_KEXT DIRECTORY
#include "ToolTest.h"
#include "prelink.h"
#include "process.h"

#include <sys/wait.h>

#include <algorithm>
#include <random>
#include <set>
#include <tuple>

using namespace graft;
using namespace graft::tool;
using graft_test::headers;
using graft_test::manifestOf;
using graft_test::refuses;

namespace {

const std::string Identifier = "ai.aneesiqbal.graft.test.ConsumerFixture";
constexpr uint64_t Base = 0xffffff8001234000;  // the consumer's fileset entry
constexpr uint64_t Header = Base - 0x20000;    // the collection header, at file offset 0
constexpr uint64_t Lowest = Header - 0x100000; // a lower vm-only segment: the rebase base
constexpr uint64_t Kernel = Base - 0x10000;    // where the test linker puts kernel imports
constexpr uint64_t Entry = 0x4000;             // the consumer's collection file offset
constexpr uint64_t Page = 0x1000;
// Offsets of the fields the test linker moves, in segment_command_64 and section_64.
constexpr uint64_t SegmentAddress = 24, SegmentFileOffset = 40;
constexpr uint64_t SectionAddress = 32, SectionFileOffset = 48;
// <mach-o/x86_64/reloc.h>: a PC-relative reference to data, which the fixture
// has and graft-kext refuses for an import.
constexpr uint32_t X86_64_RELOC_SIGNED = 1;
// <mach-o/fixup-chains.h>
constexpr uint16_t StartNone = 0xffff;
constexpr unsigned NextShift = 51;
constexpr uint64_t NextMask = 0xfff;
constexpr uint16_t PointerX86KernelCache = 11;

uint64_t pageAlign(uint64_t size) {
    return (size + Page - 1) & ~(Page - 1);
}

void append32(Bytes &out, uint32_t value) {
    uint8_t bytes[4];
    store32(bytes, value);
    out.insert(out.end(), bytes, bytes + 4);
}
void append64(Bytes &out, uint64_t value) {
    uint8_t bytes[8];
    store64(bytes, value);
    out.insert(out.end(), bytes, bytes + 8);
}
void append16(Bytes &out, uint16_t value) {
    uint8_t bytes[2];
    store16(bytes, value);
    out.insert(out.end(), bytes, bytes + 2);
}

// The consumer relocated to Base, and the chained rebases it needs.
struct Linked {
    // A segment as linked: its address, vm size, file offset and file size.
    struct Placed {
        uint64_t address, vmsize, offset, filesize;
    };

    Bytes data;
    std::map<uint64_t, uint64_t> rebases;
    std::vector<Placed> segments;

    explicit Linked(const Bytes &kext) : data(kext) {
        Image image(kext);
        graft_test::Headers commands = headers(kext);
        uint64_t slide = Base - image.base;
        for (size_t i = 0; i < image.segments.size(); ++i) {
            const Segment &segment = image.segments[i];
            store64(&data[commands.segments[i] + SegmentAddress], segment.address + slide);
            store64(&data[commands.segments[i] + SegmentFileOffset], segment.offset + Entry);
            segments.push_back({segment.address + slide, segment.vmsize, Entry + segment.offset,
                                segment.filesize});
        }
        for (size_t i = 0; i < image.sections.size(); ++i) {
            const Section &section = image.sections[i];
            store64(&data[commands.sections[i] + SectionAddress], section.address + slide);
            if (section.offset)
                store32(&data[commands.sections[i] + SectionFileOffset],
                        static_cast<uint32_t>(section.offset + Entry));
        }
        for (const Relocation &relocation : image.relocations()) {
            uint64_t at = image.fileOffset(relocation.address, relocation.width);
            uint64_t site = relocation.address + slide;
            const Symbol *symbol =
                    relocation.external ? &image.symbols.at(relocation.symbolIndex) : nullptr;
            uint64_t target = 0;
            if (symbol)
                target = symbol->defined() ? symbol->value + slide
                                           : Kernel + 16 * uint64_t(relocation.symbolIndex);
            bool pointer = relocation.width == 8 && !relocation.pcrel &&
                           relocation.kind == X86_64_RELOC_UNSIGNED;
            bool relative = relocation.width == 4 && relocation.pcrel &&
                            (relocation.kind == X86_64_RELOC_SIGNED ||
                             relocation.kind == X86_64_RELOC_BRANCH);
            // Every relocation of the fixture is one of these two forms.
            CHECK(pointer || relative);
            if (pointer) {
                uint64_t value = load64(&data[at]);
                rebases[site] = symbol ? target + value : value + slide;
            } else if (symbol) {
                // A local PC-relative reference moves with its whole image.
                auto addend = static_cast<int32_t>(load32(&data[at]));
                store32(&data[at], static_cast<uint32_t>(target + addend - (site + 4)));
            }
        }
    }

    uint64_t fileOffset(uint64_t address) const {
        auto segment = std::find_if(segments.begin(), segments.end(), [&](const Placed &placed) {
            return placed.address <= address && address < placed.address + placed.filesize;
        });
        CHECK(segment != segments.end());
        return segment->offset + address - segment->address;
    }
};

struct FilesetEntry {
    std::string name;
    uint64_t address, offset;
};

struct CollectionSegment {
    std::string name;
    uint64_t address, vmsize, offset, filesize;
};

Bytes segmentCommand(const CollectionSegment &segment) {
    Bytes out;
    append32(out, LC_SEGMENT_64);
    append32(out, SEGMENT_COMMAND_SIZE);
    Bytes name(16);
    std::copy(segment.name.begin(), segment.name.end(), name.begin());
    out.insert(out.end(), name.begin(), name.end());
    for (uint64_t field : {segment.address, segment.vmsize, segment.offset, segment.filesize})
        append64(out, field);
    for (uint32_t field : {7u, 7u, 0u, 0u}) // maxprot, initprot, nsects, flags
        append32(out, field);
    return out;
}

Bytes entryCommand(const FilesetEntry &entry) {
    uint32_t size = (32 + static_cast<uint32_t>(entry.name.size()) + 1 + 7) & ~7u;
    Bytes out;
    append32(out, LC_FILESET_ENTRY);
    append32(out, size);
    append64(out, entry.address);
    append64(out, entry.offset);
    append32(out, 32); // the name's offset in the command
    append32(out, 0);
    out.insert(out.end(), entry.name.begin(), entry.name.end());
    out.resize(size);
    return out;
}

// Write the chains of one segment's rebases into data, page by page in address
// order, and return its page starts, or nothing when it has no rebase.
std::optional<std::vector<uint16_t>> chainPages(Bytes &data, const Linked &linked,
                                                const std::map<uint64_t, uint64_t> &rebases,
                                                uint64_t address, uint64_t vmsize) {
    std::vector<uint64_t> inside;
    for (const auto &entry : rebases) {
        if (address <= entry.first && entry.first < address + vmsize)
            inside.push_back(entry.first);
    }
    if (inside.empty())
        return std::nullopt;
    std::vector<uint16_t> starts((vmsize + Page - 1) / Page, StartNone);
    for (size_t page = 0; page < starts.size(); ++page) {
        uint64_t low = address + page * Page;
        std::vector<uint64_t> chain;
        for (uint64_t at : inside) {
            if (low <= at && at < low + Page)
                chain.push_back(at);
        }
        if (chain.empty())
            continue;
        starts[page] = static_cast<uint16_t>(chain[0] - low);
        for (size_t i = 0; i < chain.size(); ++i) {
            uint64_t step = i + 1 < chain.size() ? chain[i + 1] - chain[i] : 0;
            store64(&data[linked.fileOffset(chain[i])],
                    (rebases.at(chain[i]) - Lowest) | step << NextShift);
        }
    }
    return starts;
}

// How a test changes the collection the test linker builds.
struct Layout {
    std::vector<FilesetEntry> entries = {{"com.apple.kernel", Kernel, 0},
                                         {Identifier, Base, Entry}};
    // The pointers to chain as rebases, by address, in place of the linker's.
    std::optional<std::map<uint64_t, uint64_t>> rebases;
    // Further chain entries, each holding Lowest.
    std::vector<uint64_t> extra;
    uint16_t pointerFormat = PointerX86KernelCache;
    // List the regions opposite to load-command order.
    bool reverse = false;
};

// A collection holding kext, laid out as layout says.
Bytes collection(const Bytes &kext, const Layout &layout = {}) {
    Linked linked(kext);
    std::map<uint64_t, uint64_t> rebases = layout.rebases ? *layout.rebases : linked.rebases;
    for (uint64_t address : layout.extra)
        rebases[address] = Lowest;
    std::vector<CollectionSegment> segments = {{"__HIB", Lowest, Page, 0, 0},
                                               {"__TEXT", Header, Entry, 0, Entry}};
    for (size_t i = 0; i < linked.segments.size(); ++i) {
        const Linked::Placed &placed = linked.segments[i];
        segments.push_back({"__REGION" + std::to_string(i), placed.address, placed.vmsize,
                            placed.offset, placed.filesize});
    }
    Bytes data(pageAlign(Entry + linked.data.size()));
    std::copy(linked.data.begin(), linked.data.end(), data.begin() + Entry);
    std::vector<std::pair<size_t, Bytes>> regions;
    for (size_t i = 0; i < segments.size(); ++i) {
        auto starts = chainPages(data, linked, rebases, segments[i].address, segments[i].vmsize);
        if (!starts)
            continue;
        Bytes body;
        append32(body, 22 + 2 * static_cast<uint32_t>(starts->size()));
        append16(body, static_cast<uint16_t>(Page));
        append16(body, layout.pointerFormat);
        append64(body, segments[i].address - Header);
        append32(body, 0);
        append16(body, static_cast<uint16_t>(starts->size()));
        for (uint16_t start : *starts)
            append16(body, start);
        regions.emplace_back(i, body);
    }
    // The table's own __LINKEDIT segment is last. Segments without fixups have
    // no region, so there can be fewer regions than slots.
    size_t count = segments.size() + 1;
    if (layout.reverse)
        std::reverse(regions.begin(), regions.end());
    std::vector<uint32_t> infos(count);
    Bytes pool;
    for (size_t slot = 0; slot < regions.size(); ++slot) {
        size_t index = layout.reverse ? slot : regions[slot].first;
        infos[index] = static_cast<uint32_t>(4 + 4 * count + pool.size());
        pool.insert(pool.end(), regions[slot].second.begin(), regions[slot].second.end());
    }
    Bytes startsInImage;
    append32(startsInImage, static_cast<uint32_t>(count));
    for (uint32_t info : infos)
        append32(startsInImage, info);
    // Declare only the header and the slot table; the regions follow, as
    // OpenCore appends them after the declared size.
    auto declared = static_cast<uint32_t>(28 + startsInImage.size());
    Bytes table;
    for (uint32_t field : {0u, 28u, declared, declared, 0u, 1u, 0u})
        append32(table, field);
    table.insert(table.end(), startsInImage.begin(), startsInImage.end());
    table.insert(table.end(), pool.begin(), pool.end());
    uint64_t linkedit = data.size();
    data.resize(data.size() + pageAlign(table.size()));
    std::copy(table.begin(), table.end(), data.begin() + static_cast<ptrdiff_t>(linkedit));
    uint64_t size = data.size() - linkedit;
    segments.push_back({"__LINKEDIT", Header + 0x10000000, size, linkedit, size});
    Bytes commands;
    for (const CollectionSegment &segment : segments) {
        Bytes command = segmentCommand(segment);
        commands.insert(commands.end(), command.begin(), command.end());
    }
    for (uint32_t field : {LC_DYLD_CHAINED_FIXUPS, 16u, static_cast<uint32_t>(linkedit), declared})
        append32(commands, field);
    for (const FilesetEntry &entry : layout.entries) {
        Bytes command = entryCommand(entry);
        commands.insert(commands.end(), command.begin(), command.end());
    }
    Bytes header;
    for (uint32_t field : {MH_MAGIC_64, CPU_TYPE_X86_64, 3u, MH_FILESET,
                           static_cast<uint32_t>(segments.size() + 1 + layout.entries.size()),
                           static_cast<uint32_t>(commands.size()), 0u, 0u})
        append32(header, field);
    header.insert(header.end(), commands.begin(), commands.end());
    CHECK(header.size() <= Entry);
    std::copy(header.begin(), header.end(), data.begin());
    return data;
}

// The staged fixture, its linked form and its manifest's sites.
struct Fixture {
    std::string bundle;
    Bytes kext;
    Image staged;
    Linked linked;
    Bytes good;
    Bytes section;
    manifest::View view;
    std::vector<manifest::Site> sites;
    // The manifest's image offset, which is also its offset from Base once linked.
    uint64_t manifest;

    explicit Fixture(const std::string &directory)
        : bundle(directory + "/Fixture.kext"), kext(readFile(bundle + "/Contents/MacOS/Fixture")),
          staged(kext), linked(kext), good(collection(kext)),
          section(graft_test::graftSection(staged)), view(manifestOf(section)),
          manifest(staged.section("__DATA", "__graft").address - staged.base) {
        for (uint32_t i = 0; i < view.header().sites.count; ++i)
            sites.push_back(view.row<manifest::Site>(view.header().sites, i));
    }

    size_t check(const Bytes &data) const {
        return checkPrelinked(Collection(data), Identifier, staged);
    }
    // The collection file offset of a consumer image offset.
    uint64_t at(uint64_t offset) const { return linked.fileOffset(Base + offset); }
    const manifest::Site &site(uint8_t width) const {
        auto found = std::find_if(sites.begin(), sites.end(),
                                  [&](const manifest::Site &site) { return site.width == width; });
        CHECK(found != sites.end());
        return *found;
    }
    uint32_t declared() const { return view.header().size; }
};

struct Ran {
    int status;
    std::string out, err;
};

Ran runTool(const std::string &tool, const std::vector<std::string> &arguments,
            const std::string &scratch) {
    std::vector<std::string> command = {tool};
    command.insert(command.end(), arguments.begin(), arguments.end());
    std::string out = scratch + "/stdout", err = scratch + "/stderr";
    int raw = std::system(
            (shellJoin(command) + " >" + shellJoin({out}) + " 2>" + shellJoin({err})).c_str());
    return {WIFEXITED(raw) ? WEXITSTATUS(raw) : -1, readText(out), readText(err)};
}

void faithfulPrelinkPasses(const Fixture &fixture) {
    std::set<uint8_t> widths;
    for (const manifest::Site &site : fixture.sites)
        widths.insert(site.width);
    CHECK((widths == std::set<uint8_t>{4, 8})); // the fixture covers both site forms
    CHECK(fixture.check(fixture.good) == fixture.sites.size());
    Layout reversed;
    reversed.reverse = true;
    // Regions are matched by offset, not by order.
    CHECK(fixture.check(collection(fixture.kext, reversed)) == fixture.sites.size());
    Bytes data = fixture.good;
    data[fixture.at(fixture.manifest + fixture.declared())] ^= 0xff;
    CHECK(fixture.check(data) == fixture.sites.size()); // the reserved tail is not compared
    puts("faithful prelink: pass (regions in either order, the tail not compared)");
}

// A consumer's build runs the check as a command and acts on its exit status: 0
// when every consumer is bindable, 1 on the first defect, 2 for a bad command
// line.
void commandReportsAndExits(const Fixture &fixture, const std::string &tool,
                            const std::string &scratch) {
    std::string good = scratch + "/good.bin";
    writeFile(good, fixture.good);
    Ran passed = runTool(tool, {"check-prelinked", good, fixture.bundle}, scratch);
    CHECK(passed.status == 0 && passed.err.empty());
    CHECK(passed.out == "prelink: " + Identifier + ": manifest intact, " +
                                std::to_string(fixture.sites.size()) +
                                " sites at their placeholders\n"
                                "prelink: OK: every listed consumer is bindable in good.bin\n");
    Bytes changed = fixture.good;
    changed[fixture.at(fixture.manifest)] ^= 1;
    std::string bad = scratch + "/bad.bin";
    writeFile(bad, changed);
    Ran refused = runTool(tool, {"check-prelinked", bad, fixture.bundle}, scratch);
    CHECK(refused.status == 1 && refused.out.empty());
    CHECK(refused.err ==
          "prelink: " + Identifier + ": manifest differs from the staged consumer\n");
    std::string missing = scratch + "/missing.bin";
    Ran absent = runTool(tool, {"check-prelinked", missing, fixture.bundle}, scratch);
    CHECK(absent.status == 1 && absent.out.empty());
    CHECK(absent.err == "prelink: cannot read " + missing + ": No such file or directory\n");
    Ran usage = runTool(tool, {"check-prelinked", good}, scratch);
    CHECK(usage.status == 2 && usage.out.empty());
    CHECK(usage.err.rfind("usage: ", 0) == 0);
    puts("check-prelinked: pass (its output and exit status: 0, 1 and 2)");
}

void changedManifestIsRefused(const Fixture &fixture) {
    for (uint64_t offset :
         {uint64_t(0), uint64_t(sizeof(manifest::Header)), uint64_t(fixture.declared() - 1)}) {
        Bytes data = fixture.good;
        data[fixture.at(fixture.manifest + offset)] ^= 1;
        CHECK(refuses([&] { fixture.check(data); }, "manifest differs"));
    }
    puts("changed manifest: pass (refused)");
}

void pointerSites(const Fixture &fixture) {
    uint64_t address = Base + fixture.site(8).offset;
    std::map<uint64_t, uint64_t> moved = fixture.linked.rebases;
    moved[address] += 8;
    Layout layout;
    layout.rebases = moved;
    CHECK(refuses([&] { fixture.check(collection(fixture.kext, layout)); }, "not its placeholder"));

    // The right absolute value, but no chain entry: the booter would never slide it.
    std::map<uint64_t, uint64_t> unchained = fixture.linked.rebases;
    uint64_t value = unchained.at(address);
    unchained.erase(address);
    layout.rebases = unchained;
    Bytes data = collection(fixture.kext, layout);
    store64(&data[fixture.linked.fileOffset(address)], value);
    CHECK(refuses([&] { fixture.check(data); }, "not in the collection's fixup chains"));

    uint64_t at = fixture.at(fixture.site(8).offset);
    for (unsigned bit : {30u, 32u, 48u, 49u, 63u}) { // level, diversity, addrDiv, key, auth
        Bytes flagged = fixture.good;
        store64(&flagged[at], load64(&flagged[at]) | uint64_t(1) << bit);
        CHECK(refuses([&] { fixture.check(flagged); }, "not a plain chained rebase"));
    }
    puts("pointer sites: pass (a moved, unchained or non-plain pointer is refused)");
}

void branchSites(const Fixture &fixture) {
    uint64_t branch = Base + fixture.site(4).offset;
    Layout layout;
    layout.extra = {branch - 4};
    CHECK(refuses([&] { fixture.check(collection(fixture.kext, layout)); }, "overlaps the branch"));
    layout.extra = {Base + fixture.manifest + 16};
    CHECK(refuses([&] { fixture.check(collection(fixture.kext, layout)); },
                  "overlaps the manifest"));

    uint64_t at = fixture.at(fixture.site(4).offset);
    Bytes moved = fixture.good;
    store32(&moved[at], load32(&moved[at]) + 1);
    CHECK(refuses([&] { fixture.check(moved); }, "not its placeholder"));
    Bytes opcode = fixture.good;
    opcode[at - 1] ^= 0xe8 ^ 0xe9;
    CHECK(refuses([&] { fixture.check(opcode); }, "branch opcode"));
    puts("branch sites: pass (a chain entry over one, a moved one, another opcode)");
}

// Point a pointer site at the image header in both copies, so the manifests
// agree and the value matches: only the tail rule fails.
void placeholderOutsideTailIsRefused(const Fixture &fixture) {
    const manifest::Site &site = fixture.site(8);
    size_t index = static_cast<size_t>(&site - fixture.sites.data());
    Bytes kext = fixture.kext;
    uint64_t sites =
            fixture.staged.section("__DATA", "__graft").offset + fixture.view.header().sites.offset;
    store64(&kext[sites + index * sizeof(manifest::Site) + offsetof(manifest::Site, placeholder)],
            0);
    Image staged(kext);
    Layout layout;
    layout.rebases = fixture.linked.rebases;
    (*layout.rebases)[Base + site.offset] = Base + static_cast<uint64_t>(site.addend);
    Bytes data = collection(kext, layout);
    CHECK(refuses([&] { checkPrelinked(Collection(data), Identifier, staged); },
                  "outside the reserved manifest tail"));
    puts("placeholder outside the tail: pass (refused)");
}

void malformedChains(const Fixture &fixture) {
    Layout layout;
    layout.pointerFormat = 8;
    Bytes otherFormat = collection(fixture.kext, layout);
    CHECK(refuses([&] { Collection made(otherFormat); }, "unsupported chained segment starts"));
    Bytes data = fixture.good;
    uint64_t at = fixture.linked.fileOffset(fixture.linked.rebases.rbegin()->first);
    store64(&data[at], load64(&data[at]) | NextMask << NextShift);
    CHECK(refuses([&] { fixture.check(data); }, "leaves its region"));
    puts("malformed chains: pass (another pointer format, a chain leaving its region)");
}

void entries(const Fixture &fixture) {
    Bytes data = fixture.good;
    uint64_t command = 0;
    for (size_t i = 0; i < fixture.staged.segments.size(); ++i) {
        if (fixture.staged.segments[i].name == "__DATA")
            command = headers(fixture.kext).segments[i];
    }
    uint64_t at = Entry + command + SegmentAddress;
    store64(&data[at], load64(&data[at]) + 0x1000);
    CHECK(refuses([&] { fixture.check(data); }, "segment __DATA changed its layout"));

    Layout layout;
    layout.entries = {{Identifier, Base + 0x1000, Entry}};
    CHECK(refuses([&] { fixture.check(collection(fixture.kext, layout)); },
                  "does not start its __TEXT"));
    layout.entries = {{"com.apple.kernel", Kernel, 0}};
    CHECK(refuses([&] { fixture.check(collection(fixture.kext, layout)); }, "no fileset entry"));
    layout.entries = {{Identifier, Base, Entry}, {Identifier, Base, Entry}};
    Bytes twice = collection(fixture.kext, layout);
    CHECK(refuses([&] { Collection made(twice); }, "duplicate fileset entry"));
    layout.entries = {{Identifier, Base, uint64_t(1) << 40}};
    CHECK(refuses([&] { fixture.check(collection(fixture.kext, layout)); }, "outside file"));
    CHECK(refuses([&] { Collection made(fixture.kext); }, "expected an x86-64 kernel collection"));
    puts("fileset entries: pass (a moved segment, a misplaced, missing, duplicate or outside "
         "entry, not a collection)");
}

void truncatedCollections(const Fixture &fixture) {
    uint64_t end = fixture.good.size();
    for (uint64_t length : {uint64_t(0), uint64_t(31), uint64_t(32), uint64_t(40), uint64_t(100),
                            Entry, Entry + 31, Entry + 100, end / 2, end - 1}) {
        Bytes cut(fixture.good.begin(), fixture.good.begin() + static_cast<ptrdiff_t>(length));
        CHECK(refuses([&] { fixture.check(cut); }, ""));
    }
    puts("truncated collections: pass (every cut is refused)");
}

// Corrupt bytes in either header and its load commands, or in the chained-fixups
// table: the check passes or refuses with Error, and never reads past the file
// or throws anything else.
void corruptionThrowsOnlyCheckErrors(const Fixture &fixture) {
    std::mt19937 random(7);
    auto below = [&](uint64_t bound) { return random() % bound; };
    uint32_t commands = load32(&fixture.good[20]);
    uint32_t inner = load32(&fixture.good[Entry + 20]);
    // collection() writes the chained-fixups table after the consumer's pages.
    uint64_t table = pageAlign(Entry + fixture.linked.data.size());
    const std::pair<uint64_t, uint64_t> spans[] = {
            {0, 32 + commands}, {Entry, Entry + 32 + inner}, {table, table + 256}};
    size_t refused = 0;
    for (int round = 0; round < 900; ++round) {
        Bytes data = fixture.good;
        auto [start, stop] = spans[below(3)];
        for (uint64_t n = 1 + below(4); n; --n)
            data[start + below(stop - start)] = static_cast<uint8_t>(below(256));
        try {
            fixture.check(data);
        } catch (const Error &) {
            ++refused;
        }
    }
    CHECK(refused);
    puts("corruption: pass (900 corrupted collections, each checked or refused)");
}

} // namespace

int main(int argc, char **argv) {
    CHECK(argc == 3);
    Fixture fixture(argv[2]);
    TemporaryDirectory scratch("graft-prelink-test-");
    faithfulPrelinkPasses(fixture);
    commandReportsAndExits(fixture, argv[1], scratch.path().string());
    changedManifestIsRefused(fixture);
    pointerSites(fixture);
    branchSites(fixture);
    placeholderOutsideTailIsRefused(fixture);
    malformedChains(fixture);
    entries(fixture);
    truncatedCollections(fixture);
    corruptionThrowsOnlyCheckErrors(fixture);
}
