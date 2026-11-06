// The native test of RgfxDeviceName.hpp: the `model` bytes the kext states on the PCI device, from
// the host's device name: a name and its NUL, as an Intel Mac's Intel UHD Graphics 630 states its
// own (measured, macOS 26.7.1); none for no name; a whole field with no NUL in it, NUL-ended; a
// length past the field, cut to it; and a NUL inside the length, cut there.
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
} // namespace

int main()
{
	uint8_t model[kRgfxDeviceModelMax];

	// A host GPU's name: its bytes and a NUL, as the Intel GPU's 23 bytes of `model`.
	const char intel[] = "Intel UHD Graphics 630";
	size_t n = fromDeviceInfo(22, intel, 22, model);
	check(n == 23, "a 22-byte name is 23 bytes of model");
	check(!std::memcmp(model, intel, 23), "the name's bytes, then its NUL");
	const char nvidia[] = "NVIDIA GeForce RTX 5080 Laptop GPU";
	n = fromDeviceInfo(sizeof(nvidia) - 1, nvidia, sizeof(nvidia) - 1, model);
	check(n == sizeof(nvidia) && !std::memcmp(model, nvidia, sizeof(nvidia)),
	      "the host's GPU name, NUL-ended");

	// No name: name_len 0 states none, whatever the field holds; so does a NUL first.
	check(fromDeviceInfo(0, intel, 22, model) == 0, "name_len 0 is no name");
	check(fromDeviceInfo(0, "", 0, model) == 0, "an empty field is no name");
	check(fromDeviceInfo(5, "\0abcd", 5, model) == 0, "a NUL first is no name");

	// A whole field, 256 bytes and no NUL in it: all of them, then a NUL past the field.
	uint8_t full[RGFX_DEVICE_NAME_LEN];
	for (size_t i = 0; i < sizeof(full); i++)
		full[i] = static_cast<uint8_t>('A' + i % 26);
	n = fromDeviceInfo(RGFX_DEVICE_NAME_LEN, full, sizeof(full), model);
	check(n == kRgfxDeviceModelMax, "a full field is 256 bytes and a NUL");
	check(!std::memcmp(model, full, sizeof(full)) && model[RGFX_DEVICE_NAME_LEN] == 0,
	      "every byte of a full field, NUL-ended");
	// A length past the field reads the field alone, never the bytes after it (architecture_name).
	n = fromDeviceInfo(UINT32_MAX, full, sizeof(full), model);
	check(n == kRgfxDeviceModelMax && model[RGFX_DEVICE_NAME_LEN] == 0, "a length past the field is cut to it");
	n = fromDeviceInfo(RGFX_DEVICE_NAME_LEN + 1, full, sizeof(full), model);
	check(n == kRgfxDeviceModelMax, "257 is cut to the field too");

	// A length short of the name: its first bytes, NUL-ended.
	n = fromDeviceInfo(5, intel, 22, model);
	check(n == 6 && !std::memcmp(model, "Intel", 6), "name_len bounds the name");
	// A NUL inside the length: the name up to it.
	n = fromDeviceInfo(10, "AMD\0Radeon", 10, model);
	check(n == 4 && !std::memcmp(model, "AMD", 4), "a NUL inside the length ends the name");

	// The span the kext reads holds name_len and the whole field, and lies inside the device info.
	check(kRgfxDeviceNameFirst == offsetof(RgfxDeviceInfo, name_len), "the span starts at name_len");
	check(kRgfxDeviceNameEnd == offsetof(RgfxDeviceInfo, name) + sizeof(RgfxDeviceInfo{}.name),
	      "the span ends at the name field's end");
	check(kRgfxDeviceNameEnd <= sizeof(RgfxDeviceInfo), "the span ends inside the device info");

	std::printf("device_name_test: OK (%d checks)\n", checks);
	return 0;
}
