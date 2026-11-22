// The native test of RgfxDeviceName.hpp: the `model` bytes the kext states on the PCI device, from
// the host's device name, as each guest's kext takes it (built twice: device_name_test for the
// x86-64 kext, device_name_arm64_test with RGFX_GUEST_ARM64 for the arm64e kext). Both cut the
// host's name at its length, at the field and at a NUL, and end it with a NUL, as an Intel Mac's
// Intel UHD Graphics 630 states its own (measured, macOS 26.7.1). The arm64e kext states the
// host's name whole, and none for no name. The x86-64 kext states the x86-64 name, "AMD - " and
// the host's name without "Intel" and "NVIDIA", and "AMD - reims-graphics" for no name.
#include "RgfxDeviceName.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "device_name_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// `model` as the kext reads it: out of a whole device info's bytes, the host's layout, every byte
// of it not the name's set to 0xa5.
size_t fromDeviceInfo(uint32_t nameLen, const void *name, size_t bytes, uint8_t (&out)[kRgfxDeviceModelMax])
{
	static RgfxDeviceInfo info;
	std::memset(&info, 0xa5, sizeof(info));
	info.name_len = nameLen;
	std::memset(info.name, 0, sizeof(info.name));
	std::memcpy(info.name, name, bytes);
	std::memset(out, 0x5a, kRgfxDeviceModelMax);
	return rgfxDeviceModelFrom(reinterpret_cast<const uint8_t *>(&info) + kRgfxDeviceNameFirst, out);
}

// Whether `model`, `n` bytes, is `want` and its NUL.
bool is(const uint8_t *model, size_t n, const char *want)
{
	const size_t len = std::strlen(want);
	return n == len + 1 && !std::memcmp(model, want, len) && model[len] == 0;
}

#if !RGFX_GUEST_ARM64
// Whether `s` holds "intel" or "nvidia" in any ASCII case.
bool holdsVendorWord(const uint8_t *s, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		for (const char *vendor : kRgfxVendorWords) {
			if (rgfxMatchAnyCase(s + i, n - i, vendor))
				return true;
		}
	}
	return false;
}

// The x86-64 name of `host` through the device info.
bool x86NameIs(const char *host, const char *want)
{
	uint8_t model[kRgfxDeviceModelMax];
	const size_t n = fromDeviceInfo(static_cast<uint32_t>(std::strlen(host)), host, std::strlen(host), model);
	return is(model, n, want) && !holdsVendorWord(model, n);
}
#endif
} // namespace

int main()
{
	uint8_t model[kRgfxDeviceModelMax];
	const char intel[] = "Intel UHD Graphics 630";
	const char nvidia[] = "NVIDIA GeForce RTX 5080 Laptop GPU";
	uint8_t full[RGFX_DEVICE_NAME_LEN];
	for (size_t i = 0; i < sizeof(full); i++)
		full[i] = static_cast<uint8_t>('A' + i % 26);
	size_t n = 0;

#if RGFX_GUEST_ARM64
	// A host GPU's name: its bytes and a NUL, as the Intel GPU's 23 bytes of `model`.
	n = fromDeviceInfo(22, intel, 22, model);
	check(n == 23, "a 22-byte name is 23 bytes of model");
	check(!std::memcmp(model, intel, 23), "the name's bytes, then its NUL");
	n = fromDeviceInfo(sizeof(nvidia) - 1, nvidia, sizeof(nvidia) - 1, model);
	check(n == sizeof(nvidia) && !std::memcmp(model, nvidia, sizeof(nvidia)),
	      "the host's GPU name, NUL-ended");

	// No name: name_len 0 states none, whatever the field holds; so does a NUL first.
	check(fromDeviceInfo(0, intel, 22, model) == 0, "name_len 0 is no name");
	check(fromDeviceInfo(0, "", 0, model) == 0, "an empty field is no name");
	check(fromDeviceInfo(5, "\0abcd", 5, model) == 0, "a NUL first is no name");

	// A whole field, 256 bytes and no NUL in it: all of them, then a NUL past the field.
	n = fromDeviceInfo(RGFX_DEVICE_NAME_LEN, full, sizeof(full), model);
	check(n == RGFX_DEVICE_NAME_LEN + 1, "a full field is 256 bytes and a NUL");
	check(!std::memcmp(model, full, sizeof(full)) && model[RGFX_DEVICE_NAME_LEN] == 0,
	      "every byte of a full field, NUL-ended");
	// A length past the field reads the field alone, never the bytes after it (architecture_name).
	n = fromDeviceInfo(UINT32_MAX, full, sizeof(full), model);
	check(n == RGFX_DEVICE_NAME_LEN + 1 && model[RGFX_DEVICE_NAME_LEN] == 0,
	      "a length past the field is cut to it");
	n = fromDeviceInfo(RGFX_DEVICE_NAME_LEN + 1, full, sizeof(full), model);
	check(n == RGFX_DEVICE_NAME_LEN + 1, "257 is cut to the field too");

	// A length short of the name: its first bytes, NUL-ended.
	n = fromDeviceInfo(5, intel, 22, model);
	check(is(model, n, "Intel"), "name_len bounds the name");
	// A NUL inside the length: the name up to it.
	n = fromDeviceInfo(10, "AMD\0Radeon", 10, model);
	check(is(model, n, "AMD"), "a NUL inside the length ends the name");
#else
	// A host GPU's name: "AMD - " and the name without its vendor words, then a NUL.
	check(x86NameIs(intel, "AMD - UHD Graphics 630"), "Intel's word and its space go");
	check(x86NameIs(nvidia, "AMD - GeForce RTX 5080 Laptop GPU"), "NVIDIA's word and its space go");
	check(x86NameIs("Intel(R) Iris(R) Xe Graphics (RPL-P)", "AMD - Iris(R) Xe Graphics (RPL-P)"),
	      "a mark after the word goes with it; a later one stays");
	check(x86NameIs("Intel(R) UHD Graphics 630 (CFL GT2)", "AMD - UHD Graphics 630 (CFL GT2)"),
	      "the UHD 630 on Linux");
	check(x86NameIs("Intel(R) Arc(tm) A770 Graphics (DG2)", "AMD - Arc(tm) A770 Graphics (DG2)"),
	      "the A770");
	check(x86NameIs("NVIDIA GeForce RTX 3090", "AMD - GeForce RTX 3090"), "the 3090");
	check(x86NameIs("AMD Radeon 780M Graphics (RADV PHOENIX)",
	                "AMD - AMD Radeon 780M Graphics (RADV PHOENIX)"),
	      "an AMD GPU's name whole");
	check(x86NameIs("intel(tm) nViDiA\xc2\xae X\xe2\x84\xa2", "AMD - X\xe2\x84\xa2"),
	      "any case and either UTF-8 mark; a mark after no word stays");
	check(x86NameIs("InNVIDIAtel(R) X", "AMD - X"), "a word that taking one out leaves goes too");
	check(x86NameIs("Mesa Intel", "AMD - Mesa "), "a word at the end; the space before it stays");

	// No name, or nothing left: "reims-graphics" after the prefix; a NUL first is no name.
	check(fromDeviceInfo(0, intel, 22, model) == 21 && is(model, 21, "AMD - reims-graphics"),
	      "name_len 0 is AMD - reims-graphics");
	n = fromDeviceInfo(5, "\0abcd", 5, model);
	check(is(model, n, "AMD - reims-graphics"), "a NUL first is no name");
	check(x86NameIs("NVIDIA", "AMD - reims-graphics"), "a name of a vendor word alone leaves none");
	check(x86NameIs("Intel(R) ", "AMD - reims-graphics"), "a word, its mark and its space leave none");

	// A whole field, 256 bytes and no NUL in it: the prefix, all of them, then a NUL.
	n = fromDeviceInfo(RGFX_DEVICE_NAME_LEN, full, sizeof(full), model);
	check(n == kRgfxDeviceModelMax, "a full field is the prefix, 256 bytes and a NUL");
	check(!std::memcmp(model, "AMD - ", 6) && !std::memcmp(model + 6, full, sizeof(full)) &&
	          model[kRgfxDeviceModelMax - 1] == 0,
	      "every byte of a full field after the prefix, NUL-ended");
	// A length past the field reads the field alone, never the bytes after it (architecture_name).
	n = fromDeviceInfo(UINT32_MAX, full, sizeof(full), model);
	check(n == kRgfxDeviceModelMax && model[kRgfxDeviceModelMax - 1] == 0, "a length past the field is cut to it");

	// A length short of the name: its first bytes; a NUL inside the length ends it.
	n = fromDeviceInfo(11, nvidia, sizeof(nvidia) - 1, model);
	check(is(model, n, "AMD - GeFo"), "name_len bounds the name before its words go");
	n = fromDeviceInfo(10, "AMD\0Radeon", 10, model);
	check(is(model, n, "AMD - AMD"), "a NUL inside the length ends the name");
#endif

	// The span the kext reads holds name_len and the whole field, and lies inside the device info.
	check(kRgfxDeviceNameFirst == offsetof(RgfxDeviceInfo, name_len), "the span starts at name_len");
	check(kRgfxDeviceNameEnd == offsetof(RgfxDeviceInfo, name) + sizeof(RgfxDeviceInfo{}.name),
	      "the span ends at the name field's end");
	check(kRgfxDeviceNameEnd <= sizeof(RgfxDeviceInfo), "the span ends inside the device info");

	std::printf("device_name_test: OK (%d checks, %s kext)\n", checks, RGFX_GUEST_ARM64 ? "arm64e" : "x86-64");
	return 0;
}
