// The GL functions' checks (tests/shell_tests.mm's main calls rgfxGLTests()):
//   - the .metallib writer (src/RgfxMetallibWriter.*): each function of the library Apple's Metal
//     compiler builds from tests/air_arguments.metal (RGFX_AIR_ARGUMENTS), written again from its
//     bitcode, is a library this Mac's Metal loads, with the one function of that name and stage,
//     whose bitcode the bundle's reader finds as it was and reads to the same entry points; its
//     header, tags and hash are the layout's; the same function gives the same bytes; the bitcode
//     unwrapper takes LLVM's wrapper off, and leaves other bytes as they are; all of them written
//     into one library are what this Mac's Metal loads too, each in its place as `metal` places a
//     library's functions;
//   - the .metallib of the kernels in the container Metal's compiler answers for OpenCL C source,
//     over a container written here in the measured layout from that library's kernels, and over
//     this Mac's own where its Metal compiles OpenCL C; other bytes, and a container that does not
//     read whole, give none;
//   - a function of GL IR (src/RgfxGLFunction.*): what it answers, as Metal's own function of GL IR
//     does (measured: macOS 26.7.1), the IR it keeps, and the variants' return data;
//   - the pipeline reflection (src/RgfxPipelineReflection.mm): a GL variant's return data is what
//     the render reflection answers for its stage;
//   - the stage metadata's normalization (src/RgfxGLLowering.*), over modules' text written here
//     in the generator's shape: an integer user() varying named unsigned, a node without a type
//     name given its signature's name and a name, and every other line as it was.
// A variant is not lowered here: the GL code generator lowers the IR the renderer gives a device,
// which only a GL context makes; tools/gl_probe.m, tools/gl_inject.mm and tools/gl_cases.m check
// that on a device.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <objc/message.h>
#include <objc/runtime.h>

#include <CommonCrypto/CommonDigest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#import "RgfxGLFunction.h"
#import "RgfxPipelineReflection.h"

#include "RgfxAirArguments.hpp"
#include "RgfxGLLowering.hpp"
#include "RgfxLibraryKey.hpp"
#include "RgfxMetallibWriter.hpp"
#include "rgfx_test_ring.hpp" // CHECK, g_checks

void rgfxGLTests();

namespace {

uint64_t le64(const uint8_t *p)
{
	uint64_t v = 0;
	std::memcpy(&v, p, sizeof v);
	return v;
}

uint32_t le32(const uint8_t *p)
{
	uint32_t v = 0;
	std::memcpy(&v, p, sizeof v);
	return v;
}

// One function of a .metallib's list: its tags' data by name.
struct Listed {
	std::string name;
	uint8_t type = 0xff;
	std::vector<uint8_t> vers, hash, offt, mdsz;
};

std::vector<Listed> functionsOf(const std::vector<uint8_t> &lib)
{
	std::vector<Listed> out;
	if (lib.size() < 0x58 || std::memcmp(lib.data(), "MTLB", 4) != 0)
		return out;
	const uint8_t *p = lib.data();
	const uint64_t list = le64(p + 0x18);
	const uint32_t count = le32(p + list);
	uint64_t q = list + 4;
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t group = le32(p + q);
		Listed f;
		for (uint64_t t = q + 4; t + 6 <= q + group && std::memcmp(p + t, "ENDT", 4) != 0;) {
			const uint16_t n = static_cast<uint16_t>(p[t + 4] | p[t + 5] << 8);
			const uint8_t *d = p + t + 6;
			if (std::memcmp(p + t, "NAME", 4) == 0)
				f.name.assign(reinterpret_cast<const char *>(d), strnlen(reinterpret_cast<const char *>(d), n));
			else if (std::memcmp(p + t, "TYPE", 4) == 0)
				f.type = d[0];
			else if (std::memcmp(p + t, "VERS", 4) == 0)
				f.vers.assign(d, d + n);
			else if (std::memcmp(p + t, "HASH", 4) == 0)
				f.hash.assign(d, d + n);
			else if (std::memcmp(p + t, "OFFT", 4) == 0)
				f.offt.assign(d, d + n);
			else if (std::memcmp(p + t, "MDSZ", 4) == 0)
				f.mdsz.assign(d, d + n);
			t += 6 + n;
		}
		out.push_back(f);
		q += group;
	}
	return out;
}

std::vector<uint8_t> readFile(const char *path)
{
	std::ifstream f(path, std::ios::binary);
	return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

MTLFunctionType metalTypeOf(uint8_t type)
{
	return type == 0 ? MTLFunctionTypeVertex : type == 1 ? MTLFunctionTypeFragment : MTLFunctionTypeKernel;
}

void test_metallib_writer()
{
	const char *path = getenv("RGFX_AIR_ARGUMENTS");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	const std::vector<uint8_t> lib = readFile(path);
	const std::vector<Listed> listed = functionsOf(lib);
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	CHECK(metal != nil);
	size_t written = 0;
	for (const Listed &f : listed) {
		// Its stages only: a stitchable or visible function's TYPE is none of the three.
		if (f.type > 2)
			continue;
		const uint8_t *slice = nullptr;
		size_t sliceSize = 0;
		CHECK(rgfxMetallibBitcode(lib.data(), lib.size(), f.name, &slice, &sliceSize));
		size_t bcSize = sliceSize;
		const uint8_t *bc = RgfxUnwrapBitcode(slice, &bcSize);
		CHECK(bc != slice && bcSize < sliceSize);
		RgfxMetallibFunction w;
		w.name = f.name;
		w.stage = static_cast<RgfxMetallibStage>(f.type);
		CHECK(f.vers.size() == 8);
		std::memcpy(&w.airMajor, f.vers.data(), 2);
		std::memcpy(&w.airMinor, f.vers.data() + 2, 2);
		std::memcpy(&w.languageMajor, f.vers.data() + 4, 2);
		std::memcpy(&w.languageMinor, f.vers.data() + 6, 2);
		w.bitcode = bc;
		w.bitcodeLength = bcSize;
		const std::vector<uint8_t> out = RgfxWriteMetallib(w);
		CHECK(RgfxWriteMetallib(w) == out);

		// The header: its size, the list after it, the section at its end, 16-byte padded.
		CHECK(le64(out.data() + 0x10) == out.size());
		CHECK(le64(out.data() + 0x18) == 0x58);
		const uint64_t section = le64(out.data() + 0x48), sectionSize = le64(out.data() + 0x50);
		CHECK(section + sectionSize == out.size() && sectionSize % 16 == 0);
		const std::vector<Listed> back = functionsOf(out);
		CHECK(back.size() == 1);
		if (back.size() != 1)
			continue;
		CHECK(back[0].name == f.name && back[0].type == f.type && back[0].vers == f.vers);
		uint8_t hash[CC_SHA256_DIGEST_LENGTH];
		CC_SHA256(out.data() + section, static_cast<CC_LONG>(sectionSize), hash);
		CHECK(back[0].hash == std::vector<uint8_t>(hash, hash + sizeof hash));
		CHECK(back[0].offt == std::vector<uint8_t>(24, 0));
		CHECK(back[0].mdsz.size() == 8 && le64(back[0].mdsz.data()) == sectionSize);
		uint8_t key[32];
		CHECK(rgfxLibraryKey(out.data(), out.size(), key));

		// The bundle's reader finds the same bitcode and reads the same entry points from it.
		const uint8_t *again = nullptr;
		size_t againSize = 0;
		CHECK(rgfxMetallibBitcode(out.data(), out.size(), f.name, &again, &againSize));
		size_t inner = againSize;
		const uint8_t *innerAt = RgfxUnwrapBitcode(again, &inner);
		CHECK(inner == bcSize && std::memcmp(innerAt, bc, bcSize) == 0);
		std::vector<RgfxAirFunction> was, now;
		const bool read = rgfxReadAirFunctions(slice, sliceSize, was);
		CHECK(rgfxReadAirFunctions(again, againSize, now) == read);
		CHECK(was.size() == now.size());
		for (size_t i = 0; i < was.size() && i < now.size(); i++)
			CHECK(was[i].name == now[i].name && was[i].arguments.size() == now[i].arguments.size());

		// This Mac's Metal loads it: one function, of the name and the stage.
		NSData *bytes = [NSData dataWithBytes:out.data() length:out.size()];
		dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
		NSError *error = nil;
		id<MTLLibrary> loaded = [metal newLibraryWithData:data error:&error];
		if (loaded == nil)
			fprintf(stderr, "gl_tests: %s: %s\n", f.name.c_str(), error.localizedDescription.UTF8String);
		CHECK(loaded != nil);
		CHECK([loaded.functionNames isEqualToArray:@[ @(f.name.c_str()) ]]);
		id<MTLFunction> fn = [loaded newFunctionWithName:@(f.name.c_str())];
		CHECK(fn != nil && fn.functionType == metalTypeOf(f.type));
		written++;
	}
	CHECK(written >= 8);

	// The unwrapper: bytes with no wrapper, or a wrapper naming bytes past its buffer, stay.
	const uint8_t raw[24] = {'B', 'C', 0xc0, 0xde};
	size_t n = sizeof raw;
	CHECK(RgfxUnwrapBitcode(raw, &n) == raw && n == sizeof raw);
	uint8_t wrapped[40] = {};
	const uint32_t header[5] = {0x0B17C0DE, 0, 20, 30, 0xffffffffu};
	std::memcpy(wrapped, header, sizeof header);
	n = sizeof wrapped;
	CHECK(RgfxUnwrapBitcode(wrapped, &n) == wrapped && n == sizeof wrapped);
	const uint32_t fits[5] = {0x0B17C0DE, 0, 20, 12, 0xffffffffu};
	std::memcpy(wrapped, fits, sizeof fits);
	n = sizeof wrapped;
	CHECK(RgfxUnwrapBitcode(wrapped, &n) == wrapped + 20 && n == 12);
	n = 8;
	CHECK(RgfxUnwrapBitcode(wrapped, &n) == wrapped && n == 8);
}

// The stage functions of the RGFX_AIR_ARGUMENTS library, their bitcode unwrapped, as
// RgfxWriteMetallib takes them; `lib` holds the bytes they point into.
std::vector<RgfxMetallibFunction> stageFunctions(const std::vector<uint8_t> &lib)
{
	std::vector<RgfxMetallibFunction> out;
	for (const Listed &f : functionsOf(lib)) {
		const uint8_t *slice = nullptr;
		size_t sliceSize = 0;
		if (f.type > 2 || f.vers.size() != 8 ||
		    !rgfxMetallibBitcode(lib.data(), lib.size(), f.name, &slice, &sliceSize))
			continue;
		RgfxMetallibFunction w;
		w.name = f.name;
		w.stage = static_cast<RgfxMetallibStage>(f.type);
		std::memcpy(&w.airMajor, f.vers.data(), 2);
		std::memcpy(&w.airMinor, f.vers.data() + 2, 2);
		std::memcpy(&w.languageMajor, f.vers.data() + 4, 2);
		std::memcpy(&w.languageMinor, f.vers.data() + 6, 2);
		size_t n = sliceSize;
		w.bitcode = RgfxUnwrapBitcode(slice, &n);
		w.bitcodeLength = n;
		out.push_back(w);
	}
	return out;
}

// What this Mac's Metal makes of `out`: its function names, sorted, with each one's type checked
// against `fs`; empty where it does not load.
NSArray<NSString *> *metalLoads(id<MTLDevice> metal, const std::vector<uint8_t> &out,
                                const std::vector<RgfxMetallibFunction> &fs)
{
	dispatch_data_t data = dispatch_data_create(out.data(), out.size(), nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
	NSError *error = nil;
	id<MTLLibrary> loaded = [metal newLibraryWithData:data error:&error];
	if (loaded == nil) {
		fprintf(stderr, "gl_tests: %s\n", error.localizedDescription.UTF8String);
		return @[];
	}
	for (const RgfxMetallibFunction &f : fs) {
		id<MTLFunction> fn = [loaded newFunctionWithName:@(f.name.c_str())];
		CHECK(fn != nil && fn.functionType == metalTypeOf(static_cast<uint8_t>(f.stage)));
	}
	return [loaded.functionNames sortedArrayUsingSelector:@selector(compare:)];
}

NSArray<NSString *> *namesOf(const std::vector<RgfxMetallibFunction> &fs)
{
	NSMutableArray<NSString *> *names = [NSMutableArray array];
	for (const RgfxMetallibFunction &f : fs)
		[names addObject:@(f.name.c_str())];
	return [names sortedArrayUsingSelector:@selector(compare:)];
}

// Each of `fs` in `out` in its place: its tags, its bitcode where its OFFT and MDSZ say, and the
// bitcode the bundle's reader finds by its name.
void checkPlaced(const std::vector<uint8_t> &out, const std::vector<RgfxMetallibFunction> &fs)
{
	const uint8_t *p = out.data();
	CHECK(le64(p + 0x10) == out.size() && le64(p + 0x18) == 0x58);
	const uint64_t section = le64(p + 0x48), sectionSize = le64(p + 0x50);
	CHECK(section + sectionSize == out.size() && sectionSize % 16 == 0);
	// The metadata: 8 bytes a function, each; the public after the list, UUID and ENDT.
	CHECK(le64(p + 0x30) == 8 * fs.size() && le64(p + 0x40) == 8 * fs.size());
	CHECK(le64(p + 0x18) + 4 + le64(p + 0x20) + 26 == le64(p + 0x28));
	CHECK(le64(p + 0x38) == le64(p + 0x28) + 8 * fs.size() && section == le64(p + 0x38) + 8 * fs.size());
	const std::vector<Listed> back = functionsOf(out);
	CHECK(back.size() == fs.size());
	uint64_t at = 0;
	for (size_t i = 0; i < back.size() && i < fs.size(); i++) {
		CHECK(back[i].name == fs[i].name && back[i].type == static_cast<uint8_t>(fs[i].stage));
		CHECK(back[i].offt.size() == 24 && back[i].mdsz.size() == 8);
		if (back[i].offt.size() != 24 || back[i].mdsz.size() != 8)
			continue;
		CHECK(le64(back[i].offt.data()) == 8 * i && le64(back[i].offt.data() + 8) == 8 * i);
		CHECK(le64(back[i].offt.data() + 16) == at);
		const uint64_t size = le64(back[i].mdsz.data());
		CHECK(size % 16 == 0 && at + size <= sectionSize);
		uint8_t hash[CC_SHA256_DIGEST_LENGTH];
		CC_SHA256(p + section + at, static_cast<CC_LONG>(size), hash);
		CHECK(back[i].hash == std::vector<uint8_t>(hash, hash + sizeof hash));
		size_t n = static_cast<size_t>(size);
		const uint8_t *bc = RgfxUnwrapBitcode(p + section + at, &n);
		CHECK(n == fs[i].bitcodeLength && std::memcmp(bc, fs[i].bitcode, n) == 0);
		const uint8_t *found = nullptr;
		size_t foundSize = 0;
		CHECK(rgfxMetallibBitcode(p, out.size(), fs[i].name, &found, &foundSize));
		CHECK(found == p + section + at);
		at += size;
	}
	CHECK(at == sectionSize);
}

void test_metallib_writer_several()
{
	const char *path = getenv("RGFX_AIR_ARGUMENTS");
	if (path == nullptr)
		return; // test_metallib_writer says so
	const std::vector<uint8_t> lib = readFile(path);
	const std::vector<RgfxMetallibFunction> fs = stageFunctions(lib);
	CHECK(fs.size() >= 8);
	const std::vector<uint8_t> out = RgfxWriteMetallib(fs);
	CHECK(RgfxWriteMetallib(fs) == out);
	checkPlaced(out, fs);
	uint8_t key[32];
	CHECK(rgfxLibraryKey(out.data(), out.size(), key));
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	CHECK([metalLoads(metal, out, fs) isEqualToArray:namesOf(fs)]);
	// One function in a list is the library of that function alone.
	CHECK(RgfxWriteMetallib(std::vector<RgfxMetallibFunction>{fs[0]}) == RgfxWriteMetallib(fs[0]));
}

void put32(std::vector<uint8_t> &d, uint32_t v)
{
	for (int i = 0; i < 4; i++)
		d.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void set32(std::vector<uint8_t> &d, size_t at, uint32_t v)
{
	for (int i = 0; i < 4; i++)
		d[at + i] = static_cast<uint8_t>(v >> (8 * i));
}

// A container in the layout RgfxMetallibOfClLibrary reads, as Metal's compiler answers it for
// OpenCL C source (measured: macOS 26.7.1), of `fs`' modules under their names.
std::vector<uint8_t> clContainer(const std::vector<RgfxMetallibFunction> &fs)
{
	std::vector<uint8_t> table = {8, 0, 2, 0, 1, 0, 1, 0};
	put32(table, static_cast<uint32_t>(fs.size()));
	std::vector<uint8_t> modules, names;
	put32(names, static_cast<uint32_t>(fs.size()));
	for (size_t i = 0; i < fs.size(); i++) {
		const uint32_t entry[7] = {2, 0, static_cast<uint32_t>(8 * i), 0, static_cast<uint32_t>(modules.size()), 0, 0xfc};
		for (uint32_t w : entry)
			put32(table, w);
		table.insert(table.end(), fs[i].name.begin(), fs[i].name.end());
		table.push_back(0);
		names.insert(names.end(), fs[i].name.begin(), fs[i].name.end());
		names.push_back(0);
		put32(modules, static_cast<uint32_t>(20 + fs[i].bitcodeLength));
		const uint32_t wrapper[5] = {0x0B17C0DE, 0, 20, static_cast<uint32_t>(fs[i].bitcodeLength), 0xffffffffu};
		for (uint32_t w : wrapper)
			put32(modules, w);
		modules.insert(modules.end(), fs[i].bitcode, fs[i].bitcode + fs[i].bitcodeLength);
	}
	const std::string defaultName("default.metallib");
	std::vector<std::vector<uint8_t>> sections = {table,
	                                              std::vector<uint8_t>(8 * fs.size(), 0),
	                                              {},
	                                              modules,
	                                              {},
	                                              std::vector<uint8_t>(defaultName.begin(), defaultName.end()),
	                                              names,
	                                              {0, 0, 0, 0},
	                                              {0, 0, 0, 0},
	                                              {0, 0, 0, 0}};
	sections[5].push_back(0);
	std::vector<uint8_t> out(0x68, 0);
	out[0] = 1;
	for (size_t i = 0; i < sections.size(); i++) {
		const bool empty = sections[i].empty() && i == 4;
		set32(out, 0x10 + 8 * i, empty ? 0 : static_cast<uint32_t>(out.size()));
		set32(out, 0x14 + 8 * i, static_cast<uint32_t>(sections[i].size()));
		out.insert(out.end(), sections[i].begin(), sections[i].end());
	}
	return out;
}

// Metal's compiler over OpenCL C source, as the GL renderer's cl2Metal compiles a compute program
// (MTLCompileOptions' languageVersion 0x20002, sourceLanguage 1 and its additional arguments,
// read from its code): the library's bytes, or none where this Mac's Metal makes none.
NSData *clLibraryBytes(id<MTLDevice> metal, NSString *source, NSArray<NSString *> **names)
{
	MTLCompileOptions *o = [MTLCompileOptions new];
	SEL language = sel_registerName("setSourceLanguage:"), arguments = sel_registerName("setAdditionalCompilerArguments:");
	if (![o respondsToSelector:language] || ![o respondsToSelector:arguments])
		return nil;
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(o, sel_registerName("setLanguageVersion:"), 0x20002);
	((void (*)(id, SEL, NSInteger))objc_msgSend)(o, language, 1);
	((void (*)(id, SEL, id))objc_msgSend)(o, arguments,
	                                      @"-cl-single-precision-constant -fapplecl-lax-vector-conversions "
	                                      @"-fapplecl-functional-casts -fapplecl-inline-definitions -cl-std=CL1.2");
	NSError *error = nil;
	id<MTLLibrary> lib = [metal newLibraryWithSource:source options:o error:&error];
	SEL contents = sel_registerName("libraryDataContents");
	if (lib == nil || ![(id)lib respondsToSelector:contents])
		return nil;
	*names = [lib.functionNames sortedArrayUsingSelector:@selector(compare:)];
	return ((id(*)(id, SEL))objc_msgSend)(lib, contents);
}

void test_cl_library()
{
	const char *path = getenv("RGFX_AIR_ARGUMENTS");
	if (path == nullptr)
		return; // test_metallib_writer says so
	const std::vector<uint8_t> lib = readFile(path);
	std::vector<RgfxMetallibFunction> kernels;
	for (RgfxMetallibFunction &f : stageFunctions(lib))
		if (f.stage == RgfxMetallibStage::Kernel && kernels.size() < 2)
			kernels.push_back(f);
	CHECK(kernels.size() == 2);
	if (kernels.size() != 2)
		return;
	kernels[0].name = "__entry_first";
	kernels[1].name = "__entry_second";
	const std::vector<uint8_t> container = clContainer(kernels);
	std::vector<uint8_t> out;
	CHECK(RgfxMetallibOfClLibrary(container.data(), container.size(), out));
	// The writer's VERS: the container states none.
	for (RgfxMetallibFunction &f : kernels) {
		const RgfxMetallibFunction defaults;
		f.airMajor = defaults.airMajor;
		f.airMinor = defaults.airMinor;
		f.languageMajor = defaults.languageMajor;
		f.languageMinor = defaults.languageMinor;
	}
	CHECK(out == RgfxWriteMetallib(kernels));
	checkPlaced(out, kernels);
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	CHECK([metalLoads(metal, out, kernels) isEqualToArray:namesOf(kernels)]);

	// Bytes that are not the container, or a container that does not read whole, give none.
	std::vector<uint8_t> none = {7};
	CHECK(!RgfxMetallibOfClLibrary(lib.data(), lib.size(), none) && none == std::vector<uint8_t>{7});
	CHECK(!RgfxMetallibOfClLibrary(container.data(), 0x60, none));
	const auto broken = [&](size_t at, uint32_t value) {
		std::vector<uint8_t> c = container;
		set32(c, at, value);
		return !RgfxMetallibOfClLibrary(c.data(), c.size(), none);
	};
	const size_t table = le32(container.data() + 0x10), modules = le32(container.data() + 0x28);
	CHECK(broken(0, 2));                                   // the leading 1
	CHECK(broken(0x10, static_cast<uint32_t>(container.size()))); // the table past the end
	CHECK(broken(table, 0x00030008));                      // the table's head
	CHECK(broken(table + 8, 0));                           // no functions
	CHECK(broken(table + 12, 1));                          // a stage not a kernel
	CHECK(broken(table + 12 + 16, 0x7fffffff));            // a module past the section
	CHECK(broken(modules, 0x7fffffff));                    // a module's size past the section
	CHECK(broken(modules + 4, 0x12345678));                // no wrapper
	CHECK(broken(0x14, 12 + 28 + 5));                      // a table that ends inside a name
	CHECK(none == std::vector<uint8_t>{7});
	// The table's first u16 is a release's own (macOS 27's 9).
	std::vector<uint8_t> c9 = container, out9;
	set32(c9, table, 0x00020009);
	CHECK(RgfxMetallibOfClLibrary(c9.data(), c9.size(), out9) && out9 == out);

	// This Mac's own container, where its Metal compiles OpenCL C (two kernels, one program).
	NSArray<NSString *> *names = nil;
	NSData *bytes = clLibraryBytes(metal,
	                               @"__kernel void add(__global const float *a, __global float *b) {\n"
	                               @"  b[get_global_id(0)] += a[get_global_id(0)];\n}\n"
	                               @"__kernel void twice(__global float *b) { b[get_global_id(0)] *= 2.0f; }\n",
	                               &names);
	if (bytes.length < 4 || le32(static_cast<const uint8_t *>(bytes.bytes)) != 1) {
		fprintf(stderr, "gl_tests: this Mac's Metal answers no OpenCL container (%lu bytes): the written one only\n",
		        (unsigned long)bytes.length);
		return;
	}
	out.clear();
	CHECK(RgfxMetallibOfClLibrary(static_cast<const uint8_t *>(bytes.bytes), bytes.length, out));
	CHECK([metalLoads(metal, out, {}) isEqualToArray:names]);
	CHECK(names.count == 2);
}

void test_gl_function_answers()
{
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	CHECK([[RgfxGLFunction alloc] initWithIR:nullptr functionType:MTLFunctionTypeVertex inputsDescription:nil
	                                  device:metal] == nil);
	uint8_t ir[4 + 6] = {6, 0, 0, 0, 3, 0, 0, 0, 0x81, 0};
	NSObject *inputs = [NSObject new];
	RgfxGLFunction *f = [[RgfxGLFunction alloc] initWithIR:ir
	                                          functionType:MTLFunctionTypeFragment
	                                     inputsDescription:inputs
	                                                device:metal];
	CHECK(f != nil);
	ir[8] = 0; // the function keeps a copy
	CHECK(f.rgfxIR.length == sizeof ir && static_cast<const uint8_t *>(f.rgfxIR.bytes)[8] == 0x81);
	CHECK(f.rgfxInputsDescription == inputs);
	CHECK([f.name isEqualToString:@"__main"]);
	CHECK(f.device == metal);
	CHECK(f.functionType == MTLFunctionTypeFragment);
	CHECK(f.patchType == MTLPatchTypeNone && f.patchControlPointCount == 0);
	CHECK(f.vertexAttributes == nil && f.stageInputAttributes == nil);
	CHECK(f.functionConstantsDictionary.count == 0 && f.options == MTLFunctionOptionNone);
	CHECK(f.label == nil);
	f.label = @"shader";
	CHECK([f.label isEqualToString:@"shader"]);
	RgfxGLFunction *v = [[RgfxGLFunction alloc] initWithIR:ir
	                                          functionType:MTLFunctionTypeVertex
	                                     inputsDescription:nil
	                                                device:metal];
	CHECK(v.functionType == MTLFunctionTypeVertex && v.rgfxInputsDescription == nil);

	// The return data a variant keeps.
	CHECK(RgfxGLPluginReturnData(f) == nil && RgfxGLPluginReturnData(nil) == nil);
	NSData *ret = [@"bplist00" dataUsingEncoding:NSUTF8StringEncoding];
	RgfxGLSetPluginReturnData(f, ret);
	CHECK([RgfxGLPluginReturnData(f) isEqualToData:ret]);
	RgfxGLSetPluginReturnData(f, nil);
	CHECK(RgfxGLPluginReturnData(f) == nil);
}

void test_reflection_answers_return_data()
{
	NSData *vertexData = [@"vertex" dataUsingEncoding:NSUTF8StringEncoding];
	NSData *fragmentData = [@"fragment" dataUsingEncoding:NSUTF8StringEncoding];
	RgfxStageReflection vertex, fragment;
	vertex.pluginReturnData = vertexData;
	fragment.pluginReturnData = fragmentData;
	NSString *why = nil;
	id r = RgfxRenderPipelineReflectionMake(&vertex, &fragment, nullptr, nullptr, &why);
	CHECK(r != nil);
	CHECK([[r valueForKey:@"vertexPluginReturnData"] isEqualToData:vertexData]);
	CHECK([[r valueForKey:@"fragmentPluginReturnData"] isEqualToData:fragmentData]);
	RgfxStageReflection plain;
	id none = RgfxRenderPipelineReflectionMake(&plain, &plain, nullptr, nullptr, &why);
	CHECK(none != nil && [none valueForKey:@"vertexPluginReturnData"] == nil &&
	      [none valueForKey:@"fragmentPluginReturnData"] == nil);
	// A kernel's: where Metal's class keeps none (macOS 27), the reflection is still made.
	RgfxStageReflection kernel;
	kernel.pluginReturnData = vertexData;
	CHECK(RgfxComputePipelineReflectionMake(kernel, &why) != nil);
}

// A fragment stage in the shape the GL code generator writes one (measured: macOS 26.7.1, a
// GLSL shader writing gl_FragDepth and reading gl_FragCoord and a flat int): our own text.
const char kFragmentText[] = R"AIR(; ModuleID = 'fragment'
define { float, <4 x float> } @__main(<4 x float> %0, i32 %1) {
  ret { float, <4 x float> } undef
}

!air.fragment = !{!0}
!air.version = !{!7, !12}
!air.language_version = !{!9, !13}
!air.compile_options = !{!10, !11, !14, !10, !19, !15}
!llvm.module.flags = !{!16, !17, !18}
!gld.data = !{!8}

!0 = !{{ float, <4 x float> } (<4 x float>, i32)* @__main, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.depth", !"air.depth_qualifier", !"air.any"}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.position", !"air.center", !"air.perspective", !"air.arg_type_name", !"float4"}
!6 = !{i32 1, !"air.fragment_input", !"user(slot8)", !"air.center", !"air.flat", !"air.arg_type_name", !"int", !"air.arg_name", !"user(slot8)"}
!7 = !{i32 2, i32 3, i32 0}
!8 = !{!"gld.text", !"a, {b} \22(c)\22"}
!9 = !{!"OpenGL GLSL", i32 1, i32 0, i32 0}
!10 = !{!"air.compile.fast_math_enable"}
!11 = !{!"air.compile.native_double_disable"}
!12 = !{i32 2, i32 8, i32 0}
!13 = !{!"Metal", i32 4, i32 0, i32 0}
!14 = !{!"air.compile.framebuffer_fetch_disable"}
!15 = !{!"air.compile.framebuffer_fetch_enable"}
!16 = !{i32 7, !"air.max_textures", i32 128}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 7, !"air.max_textures", i32 16}
!19 = !{!"air.compile.fast_math_disable"}
)AIR";

// kFragmentText as the rules make it: the generator's list gone; one version, the module's own,
// its language Metal's; the module's own compile options and module flags, once each; the flat
// input without its sampling key, and named unsigned; the depth output named.
const char kFragmentNormalized[] = R"AIR(; ModuleID = 'fragment'
define { float, <4 x float> } @__main(<4 x float> %0, i32 %1) {
  ret { float, <4 x float> } undef
}

!air.fragment = !{!0}
!air.version = !{!7}
!air.language_version = !{!9}
!air.compile_options = !{!10, !14}
!llvm.module.flags = !{!16, !17}

!0 = !{{ float, <4 x float> } (<4 x float>, i32)* @__main, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"#air_depth"}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.position", !"air.center", !"air.perspective", !"air.arg_type_name", !"float4"}
!6 = !{i32 1, !"air.fragment_input", !"user(slot8)", !"air.flat", !"air.arg_type_name", !"uint", !"air.arg_name", !"user(slot8)"}
!7 = !{i32 2, i32 3, i32 0}
!8 = !{!"gld.text", !"a, {b} \22(c)\22"}
!9 = !{!"Metal", i32 2, i32 3, i32 0}
!10 = !{!"air.compile.fast_math_enable"}
!11 = !{!"air.compile.native_double_disable"}
!12 = !{i32 2, i32 8, i32 0}
!13 = !{!"Metal", i32 4, i32 0, i32 0}
!14 = !{!"air.compile.framebuffer_fetch_disable"}
!15 = !{!"air.compile.framebuffer_fetch_enable"}
!16 = !{i32 7, !"air.max_textures", i32 128}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 7, !"air.max_textures", i32 16}
!19 = !{!"air.compile.fast_math_disable"}
)AIR";

// A vertex stage: integer varyings named signed (the rule makes them unsigned) and unsigned, a
// float varying, an integer attribute (not a varying: its type name is left as it is, its bare
// attribute name dropped), and outputs and inputs without type names, one a distinct node.
const char kVertexText[] = R"AIR(!air.vertex = !{!10}
!10 = !{{ <4 x float>, <2 x i32>, i32, <4 x float>, i32 } (<2 x float>, i32, i16)* @__main, !11, !17}
!11 = !{!12, !13, !14, !15, !16}
!12 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"#air_position"}
!13 = !{!"air.vertex_output", !"user(slot9)", !"air.arg_type_name", !"int2", !"air.arg_name", !"user(slot9)"}
!14 = !{!"air.vertex_output", !"user(slot8)", !"air.arg_type_name", !"uint", !"air.arg_name", !"user(slot8)"}
!15 = !{!"air.vertex_output", !"user(slot10)", !"air.arg_type_name", !"float4", !"air.arg_name", !"user(slot10)"}
!16 = distinct !{!"air.render_target", i32 1, i32 0}
!17 = !{!18, !19, !20}
!18 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"attrib_0", !"air.arg_type_name", !"int2", !"air.arg_name", !"attrib_0"}
!19 = !{i32 1, !"air.vertex_id"}
!20 = !{i32 2, !"air.instance_id", !"air.arg_name", !"instance"})AIR";

const char kVertexNormalized[] = R"AIR(!air.vertex = !{!10}
!10 = !{{ <4 x float>, <2 x i32>, i32, <4 x float>, i32 } (<2 x float>, i32, i16)* @__main, !11, !17}
!11 = !{!12, !13, !14, !15, !16}
!12 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"#air_position"}
!13 = !{!"air.vertex_output", !"user(slot9)", !"air.arg_type_name", !"uint2", !"air.arg_name", !"user(slot9)"}
!14 = !{!"air.vertex_output", !"user(slot8)", !"air.arg_type_name", !"uint", !"air.arg_name", !"user(slot8)"}
!15 = !{!"air.vertex_output", !"user(slot10)", !"air.arg_type_name", !"float4", !"air.arg_name", !"user(slot10)"}
!16 = distinct !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"uint", !"air.arg_name", !"#air_render_target_1"}
!17 = !{!18, !19, !20}
!18 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"int2", !"air.arg_name", !"attrib_0"}
!19 = !{i32 1, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"#air_vertex_id"}
!20 = !{i32 2, !"air.instance_id", !"air.arg_name", !"instance", !"air.arg_type_name", !"ushort"})AIR";

// A fragment stage's buffers in the shape the generator writes them (measured: macOS 26.7.1, the
// uniform buffers of gl_cases' shaders): no air.address_space, and one of a known size with no
// pointee size either; a texture beside them. Our own text.
const char kBufferText[] = R"AIR(!air.fragment = !{!0}
!0 = !{{ <4 x float> } (i8 addrspace(2)*, i8 addrspace(1)*, %struct._texture_2d_t addrspace(1)*)* @__main, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"fs_uniform_buffer"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"uchar", !"air.arg_name", !"u"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"})AIR";

// kBufferText as the rules make it: each buffer's address space from its parameter's pointer, the
// sized one's pointee size and alignment from its type name; the texture as it was.
const char kBufferNormalized[] = R"AIR(!air.fragment = !{!0}
!0 = !{{ <4 x float> } (i8 addrspace(2)*, i8 addrspace(1)*, %struct._texture_2d_t addrspace(1)*)* @__main, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"fs_uniform_buffer"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"u"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"})AIR";

void test_stage_metadata_normalized()
{
	{
		RgfxGLSignature buffers;
		buffers.outputs = {"float4"};
		buffers.parameters = {"", "", ""};
		std::string text = kBufferText;
		CHECK(RgfxGLNormalizeStageText(text, buffers));
		CHECK(text == kBufferNormalized);
		CHECK(!RgfxGLNormalizeStageText(text, buffers));
	}
	RgfxGLSignature fragment;
	fragment.outputs = {"float", "float4"};
	fragment.parameters = {"float4", "uint"};
	std::string text = kFragmentText;
	CHECK(RgfxGLNormalizeStageText(text, fragment));
	CHECK(text == kFragmentNormalized);
	// Normalized text stays as it is.
	CHECK(!RgfxGLNormalizeStageText(text, fragment));
	CHECK(text == kFragmentNormalized);
	// Where the signature names no type, a node without a type name is left as it is; the
	// varying is still made unsigned.
	RgfxGLSignature unnamed;
	unnamed.parameters = {"float4", "uint"};
	text = kFragmentText;
	CHECK(RgfxGLNormalizeStageText(text, unnamed));
	CHECK(text.find("!2 = !{!\"air.depth\", !\"air.depth_qualifier\", !\"air.any\"}\n") != std::string::npos);
	CHECK(text.find("!\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"user(slot8)\"}") != std::string::npos);

	RgfxGLSignature vertex;
	vertex.outputs = {"float4", "uint2", "uint", "float4", "uint"};
	vertex.parameters = {"float2", "uint", "ushort"};
	text = kVertexText;
	CHECK(RgfxGLNormalizeStageText(text, vertex));
	CHECK(text == kVertexNormalized);

	// No vertex or fragment entry point: a kernel's module, or none at all.
	const std::string kernel = "!air.kernel = !{!0}\n!0 = !{void ()* @__main, !1, !2}\n!1 = !{}\n!2 = "
	                           "!{!3}\n!3 = !{!\"air.depth\"}\n";
	text = kernel;
	CHECK(!RgfxGLNormalizeStageText(text, vertex) && text == kernel);
	// The module's rules hold for a kernel too: the generator's lists go.
	text = kernel + "!gld.plugin_info = !{!1}\n";
	CHECK(RgfxGLNormalizeStageText(text, vertex) && text == kernel);
	text = "";
	CHECK(!RgfxGLNormalizeStageText(text, vertex) && text.empty());
	// An entry node naming nodes the text lacks.
	const std::string dangling = "!air.vertex = !{!0}\n!0 = !{void ()* @__main, !5, !6}\n";
	text = dangling;
	CHECK(!RgfxGLNormalizeStageText(text, vertex) && text == dangling);
}

} // namespace

void rgfxGLTests()
{
	@autoreleasepool {
		test_metallib_writer();
		test_metallib_writer_several();
		test_cl_library();
		test_gl_function_answers();
		test_reflection_answers_return_data();
		test_stage_metadata_normalized();
	}
}
