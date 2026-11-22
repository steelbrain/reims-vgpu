// RgfxDeviceName: the GPU name the kext states to the guest, from the host's device name in the
// device info (RgfxDeviceInfo's name_len and name, written at RGFX_DEVICE_INFO_OFFSET when the
// device is made). On an arm64 guest it is the host's name whole. On an x86-64 guest it is the
// x86-64 name (rgfxX86Name): "AMD - " and the host's name without the vendor words "Intel" and
// "NVIDIA", on every host GPU. The bundle's Metal device answers the same name as its `name`
// on either guest (rgfxMetalNameOf, guest/bundle/src/RgfxCaps.hpp).
//
// A Mac's GPU states its name as its IOPCIDevice's `model`, an OSData of the name's bytes, which
// System Information shows as the GPU's "Chipset Model"; without one it shows "Graphics" and the
// memory instead. Measured on an Intel Mac under macOS 26.7.1, both GPUs: the Intel UHD Graphics
// 630's `model` is its 22 bytes and a NUL, the AMD Radeon Pro 5300M's its 20 bytes with none,
// and System Information names both. So the kext sets `model` as the Intel's is: the name's
// bytes, then a NUL. The host's name is its first name_len bytes (at most RGFX_DEVICE_NAME_LEN),
// cut at a NUL among them. On an arm64 guest, a host that gave no name (name_len 0, or a NUL
// first) has none stated; on an x86-64 guest it has "AMD - reims-graphics", as the bundle
// answers.
//
// Why the x86-64 name holds no vendor word: WindowServer's compositor (SkyLight, read from its
// code) takes a Metal device whose name holds "Intel" for an Intel GPU, and then scales a
// full-screen video's YUV layer through Intel's video pixel processor (AppleGVACore). That
// processor keeps only plugins of Intel's renderer IDs, so with the device's own ID it has none,
// and WindowServer ends on its assert. Measured: a 4K video full screen in QuickTime Player
// killed WindowServer on a 780M host under the name "Intel - AMD Radeon 780M Graphics (RADV
// PHOENIX)" and on an Iris Xe host under "Intel - Intel(R) Iris(R) Xe Graphics (RPL-P)", on
// macOS 26.7.2 and 15.8.2. The compositor's other test, a name holding "NVIDIA", is the other
// vendor word: without either, every host takes the path of an AMD GPU's Mac.
//
// Header-only and free of the kernel's headers, so its native tests (tests/device_name_test.cpp,
// built for the x86-64 kext's rules and, as device_name_arm64_test, the arm64e kext's) build it
// as the kext does, and the bundle builds rgfxX86Name as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_DEVICE_NAME_HPP_
#define RGFX_DEVICE_NAME_HPP_

#include "reims_graphics_wire.h"

#include <stddef.h>
#include <stdint.h>

#ifndef RGFX_GUEST_ARM64
#define RGFX_GUEST_ARM64 0
#endif

// The bytes of RgfxDeviceInfo the kext reads for the name: from name_len to the name's end.
static constexpr uint64_t kRgfxDeviceNameFirst = offsetof(RgfxDeviceInfo, name_len);
static constexpr uint64_t kRgfxDeviceNameEnd = offsetof(RgfxDeviceInfo, name) + RGFX_DEVICE_NAME_LEN;

// The x86-64 name's prefix, and the name after it where the host's name leaves none.
static constexpr char kRgfxX86NamePrefix[] = "AMD - ";
static constexpr char kRgfxNoHostName[] = "reims-graphics";
static constexpr size_t kRgfxX86NamePrefixLen = sizeof(kRgfxX86NamePrefix) - 1;

// The most bytes of an x86-64 name: the prefix, a whole name field and a NUL.
static constexpr size_t kRgfxX86NameMax = kRgfxX86NamePrefixLen + RGFX_DEVICE_NAME_LEN + 1;

// The most bytes of `model`: the x86-64 name's, the longer of the two guests'.
static constexpr size_t kRgfxDeviceModelMax = kRgfxX86NameMax;

// The vendor words the x86-64 name holds none of, and the marks it takes out with one directly
// before them: "(R)" and "(TM)", and the UTF-8 of "®" and "™". In lower case: each matches in
// any ASCII case.
static constexpr const char *kRgfxVendorWords[] = {"intel", "nvidia"};
static constexpr const char *kRgfxVendorMarks[] = {"(r)", "(tm)", "\xc2\xae", "\xe2\x84\xa2"};

// `c` in ASCII's lower case; every other byte as it is.
inline uint8_t rgfxAsciiLower(uint8_t c)
{
	return c >= 'A' && c <= 'Z' ? static_cast<uint8_t>(c + ('a' - 'A')) : c;
}

// How many of the `n` bytes at `s` `lower` (lower case, NUL-ended) matches in any ASCII case:
// its length, or 0 where it does not match.
inline size_t rgfxMatchAnyCase(const uint8_t *s, size_t n, const char *lower)
{
	size_t i = 0;
	for (; lower[i]; i++) {
		if (i >= n || rgfxAsciiLower(s[i]) != static_cast<uint8_t>(lower[i]))
			return 0;
	}
	return i;
}

// One pass over the `n` bytes at `s`, in place: each vendor word taken out, with the mark
// directly after it, if any, and then one space, if one follows. The bytes left.
inline size_t rgfxDropVendorWordsOnce(uint8_t *s, size_t n)
{
	size_t out = 0;
	for (size_t i = 0; i < n;) {
		size_t word = 0;
		for (const char *vendor : kRgfxVendorWords) {
			if ((word = rgfxMatchAnyCase(s + i, n - i, vendor)))
				break;
		}
		if (!word) {
			s[out++] = s[i++];
			continue;
		}
		i += word;
		for (const char *mark : kRgfxVendorMarks) {
			if (const size_t m = rgfxMatchAnyCase(s + i, n - i, mark)) {
				i += m;
				break;
			}
		}
		if (i < n && s[i] == ' ')
			i++;
	}
	return out;
}

// The x86-64 name of a host GPU whose name is the `length` bytes at `name`, cut at a NUL among
// them and at RGFX_DEVICE_NAME_LEN, into `out`, NUL-ended: its length, the NUL not counted.
// "AMD - " and the host's name, with every vendor word taken out as rgfxDropVendorWordsOnce
// takes them, pass after pass until none is left (so taking one out never leaves another);
// every other byte of the host's is kept, its spaces too. "AMD - reims-graphics" where the
// host's name leaves nothing. Nothing else in the name is read: the device's type and vendor
// decide nothing, so every GPU gets the prefix.
inline size_t rgfxX86Name(const uint8_t *name, size_t length, char (&out)[kRgfxX86NameMax])
{
	for (size_t i = 0; i < kRgfxX86NamePrefixLen; i++)
		out[i] = kRgfxX86NamePrefix[i];
	uint8_t *host = reinterpret_cast<uint8_t *>(out + kRgfxX86NamePrefixLen);
	size_t n = 0;
	while (n < length && n < RGFX_DEVICE_NAME_LEN && name[n]) {
		host[n] = name[n];
		n++;
	}
	for (size_t before = n + 1; n != before;) {
		before = n;
		n = rgfxDropVendorWordsOnce(host, n);
	}
	if (!n) {
		for (; kRgfxNoHostName[n]; n++)
			host[n] = static_cast<uint8_t>(kRgfxNoHostName[n]);
	}
	host[n] = 0;
	return kRgfxX86NamePrefixLen + n;
}

// `model`'s bytes on an arm64 guest into `out`, from a name_len and the name field's
// RGFX_DEVICE_NAME_LEN bytes: the host's name and a NUL. How many, its NUL counted, or 0 when
// the host gave no name.
inline size_t rgfxDeviceModel(uint32_t nameLen, const uint8_t *name, uint8_t (&out)[kRgfxDeviceModelMax])
{
	const size_t limit = nameLen < RGFX_DEVICE_NAME_LEN ? nameLen : RGFX_DEVICE_NAME_LEN;
	size_t n = 0;
	while (n < limit && name[n]) {
		out[n] = name[n];
		n++;
	}
	if (!n)
		return 0;
	out[n] = 0;
	return n + 1;
}

// `model`'s bytes on an x86-64 guest into `out`, from a name_len and the name field's
// RGFX_DEVICE_NAME_LEN bytes: the x86-64 name (rgfxX86Name) and a NUL. How many, its NUL
// counted; never 0.
inline size_t rgfxDeviceModelX86(uint32_t nameLen, const uint8_t *name, uint8_t (&out)[kRgfxDeviceModelMax])
{
	return rgfxX86Name(name, nameLen, reinterpret_cast<char (&)[kRgfxX86NameMax]>(out)) + 1;
}

// `model`'s bytes out of the device info's bytes from kRgfxDeviceNameFirst to kRgfxDeviceNameEnd,
// as the kext's guest takes them (RGFX_GUEST_ARM64).
inline size_t rgfxDeviceModelFrom(const uint8_t *span, uint8_t (&out)[kRgfxDeviceModelMax])
{
	uint32_t nameLen = 0;
	for (unsigned i = 0; i < sizeof(nameLen); i++)
		nameLen |= static_cast<uint32_t>(span[i]) << (8 * i);
	const uint8_t *name = span + (offsetof(RgfxDeviceInfo, name) - kRgfxDeviceNameFirst);
	return RGFX_GUEST_ARM64 ? rgfxDeviceModel(nameLen, name, out) : rgfxDeviceModelX86(nameLen, name, out);
}

#endif // RGFX_DEVICE_NAME_HPP_
