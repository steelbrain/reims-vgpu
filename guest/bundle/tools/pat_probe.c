// pat_probe: the guest PAT on every CPU, as the x86-64 kext reads it (kRgfxAccelMethodCpuPats,
// guest/kext/src/RgfxPat.hpp), and what a write-back block of the device's region costs this
// process's CPU, on the kext's own selectors (guest/kext/src/RgfxUserClientABI.hpp), with no
// Metal.
//   pat_probe check        as root: every CPU's PAT, one line each. Prints "PAT OK" where
//                          every online CPU answered (hw.activecpu), none past the kernel's
//                          CPU count, and each one holds the kext's PAT: entry 2 write-back,
//                          entry 6 write-combining. Else "PAT FAIL: why", and exits 1.
//   pat_probe speed [MiB]  a block of the region (64 MiB by default) in this process's window,
//                          mapped write-back as any block the bundle asks so, beside as many bytes
//                          of the process's own memory: the MB/s of 64-bit stores over each and of
//                          loads, the best of three passes, and whether the loads read back what
//                          was stored. Prints "SPEED OK", or "SPEED FAIL: why" and exits 1. Any
//                          kext, the one before CpuPats included.
// Build: xcrun clang -arch x86_64 -O2 -mmacosx-version-min=11.0 -framework IOKit pat_probe.c
//        -o pat_probe
#include <IOKit/IOKitLib.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

#include "../../kext/src/RgfxPat.hpp"
#include "../../kext/src/RgfxUserClientABI.hpp"

// The most CPU numbers one read takes: an inline struct out's 4096 bytes.
enum { kMaxCpus = 512 };

static io_connect_t openClient(void)
{
	io_service_t svc =
	    IOServiceGetMatchingService(MACH_PORT_NULL, IOServiceMatching("RgfxAccelerator"));
	if (!svc)
		return 0;
	io_connect_t conn = 0;
	kern_return_t kr = IOServiceOpen(svc, mach_task_self(), kRgfxAccelUserClientType, &conn);
	IOObjectRelease(svc);
	return kr == KERN_SUCCESS ? conn : 0;
}

static int check(void)
{
	io_connect_t conn = openClient();
	if (!conn) {
		printf("PAT FAIL: no RgfxAccelerator to open\n");
		return 1;
	}
	uint64_t features[3] = {0};
	uint32_t n = 3;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodFeatures, NULL, 0, features, &n);
	if (kr != KERN_SUCCESS || !(features[1] & kRgfxKextFeatureCpuPats)) {
		printf("PAT FAIL: the kext reads no PAT (Features 0x%x, its word 0x%llx)\n", kr,
		       (unsigned long long)features[1]);
		IOServiceClose(conn);
		return 1;
	}
	static uint64_t pats[kMaxCpus];
	size_t bytes = sizeof(pats);
	uint64_t out[3] = {0};
	n = 3;
	kr = IOConnectCallMethod(conn, kRgfxAccelMethodCpuPats, NULL, 0, NULL, 0, out, &n, pats,
	                         &bytes);
	IOServiceClose(conn);
	if (kr != KERN_SUCCESS) {
		printf("PAT FAIL: CpuPats 0x%x%s\n", kr,
		       kr == kIOReturnNotPrivileged ? " (run it as root)" : "");
		return 1;
	}
	int active = 0;
	size_t len = sizeof(active);
	if (sysctlbyname("hw.activecpu", &active, &len, NULL, 0) != 0)
		active = -1;
	const uint64_t answered = out[0], count = out[1], outside = out[2];
	const size_t words = bytes / sizeof(uint64_t);
	uint64_t held = 0;
	for (size_t c = 0; c < words; c++) {
		if (pats[c] == RGFX_PAT_NONE) {
			printf("cpu%zu: no answer\n", c);
			continue;
		}
		const int ok = rgfxPatHeld(pats[c]);
		held += ok != 0;
		printf("cpu%zu: 0x%016llx entry 2 0x%02x entry 6 0x%02x%s\n", c,
		       (unsigned long long)pats[c], rgfxPatEntry(pats[c], RGFX_PAT_DEVICE_ENTRY),
		       rgfxPatEntry(pats[c], RGFX_PAT_WRITE_COMBINED_ENTRY), ok ? "" : " NOT HELD");
	}
	printf("%llu CPUs answered of %llu the kernel counts, %d active, %llu past its count\n",
	       (unsigned long long)answered, (unsigned long long)count, active,
	       (unsigned long long)outside);
	if (outside)
		printf("PAT FAIL: %llu CPUs answered past the kernel's count\n",
		       (unsigned long long)outside);
	else if (words < count)
		printf("PAT FAIL: %llu CPUs, past the %d this tool reads\n", (unsigned long long)count,
		       kMaxCpus);
	else if (active < 0 || answered != (uint64_t)active)
		printf("PAT FAIL: %llu CPUs answered, %d active\n", (unsigned long long)answered, active);
	else if (held != answered)
		printf("PAT FAIL: %llu of %llu CPUs hold entry 2 write-back and entry 6 "
		       "write-combining\n",
		       (unsigned long long)held, (unsigned long long)answered);
	else {
		printf("PAT OK: %llu CPUs, entry 2 write-back, entry 6 write-combining\n",
		       (unsigned long long)held);
		return 0;
	}
	return 1;
}

static uint64_t nowNs(void)
{
	static mach_timebase_info_data_t tb;
	if (!tb.denom)
		mach_timebase_info(&tb);
	return mach_absolute_time() * tb.numer / tb.denom;
}

// The best MB/s of three passes of 64-bit stores of a pattern over `p`, then of loads, and
// whether every load read what the last pass stored.
static int timeSpan(volatile uint64_t *p, size_t bytes, double *storeMBs, double *loadMBs)
{
	const size_t n = bytes / sizeof(uint64_t);
	uint64_t best = UINT64_MAX;
	for (int pass = 0; pass < 3; pass++) {
		const uint64_t seed = 0x9e3779b97f4a7c15ull * (uint64_t)(pass + 1);
		const uint64_t t0 = nowNs();
		for (size_t i = 0; i < n; i++)
			p[i] = seed ^ i;
		const uint64_t t = nowNs() - t0;
		best = t < best ? t : best;
	}
	*storeMBs = (double)bytes / ((double)best / 1e9) / 1e6;
	// What the last pass stored, xored over the span.
	const uint64_t last = 0x9e3779b97f4a7c15ull * 3;
	uint64_t want = 0;
	for (size_t i = 0; i < n; i++)
		want ^= last ^ i;
	int same = 1;
	best = UINT64_MAX;
	for (int pass = 0; pass < 3; pass++) {
		uint64_t a = 0, b = 0, c = 0, d = 0;
		const uint64_t t0 = nowNs();
		for (size_t i = 0; i + 3 < n; i += 4) {
			a ^= p[i];
			b ^= p[i + 1];
			c ^= p[i + 2];
			d ^= p[i + 3];
		}
		const uint64_t t = nowNs() - t0;
		best = t < best ? t : best;
		same = same && (a ^ b ^ c ^ d) == want;
	}
	*loadMBs = (double)bytes / ((double)best / 1e9) / 1e6;
	for (size_t i = 0; i < n && same; i += 4096)
		same = p[i] == (last ^ i);
	return same;
}

static int speed(size_t mib)
{
	const size_t bytes = mib << 20;
	io_connect_t conn = openClient();
	if (!conn) {
		printf("SPEED FAIL: no RgfxAccelerator to open\n");
		return 1;
	}
	uint64_t ctx[5] = {0};
	uint32_t n = 5;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodOpenContext, NULL, 0, ctx, &n);
	if (kr != KERN_SUCCESS) {
		printf("SPEED FAIL: OpenContext 0x%x\n", kr);
		IOServiceClose(conn);
		return 1;
	}
	const uint64_t in[2] = {bytes, 0x1000};
	uint64_t offset = 0;
	n = 1;
	kr = IOConnectCallScalarMethod(conn, kRgfxAccelMethodAlloc, in, 2, &offset, &n);
	if (kr != KERN_SUCCESS) {
		printf("SPEED FAIL: Alloc of %zu MiB 0x%x\n", mib, kr);
		IOServiceClose(conn);
		return 1;
	}
	volatile uint64_t *block = (volatile uint64_t *)(uintptr_t)(ctx[3] + offset);
	uint64_t *ram = malloc(bytes);
	if (!ram) {
		printf("SPEED FAIL: no %zu MiB of memory\n", mib);
		IOServiceClose(conn);
		return 1;
	}
	double bs = 0, bl = 0, rs = 0, rl = 0;
	const int blockSame = timeSpan(block, bytes, &bs, &bl);
	const int ramSame = timeSpan(ram, bytes, &rs, &rl);
	printf("region block (write-back, %zu MiB at region offset 0x%llx): store %.0f MB/s, load %.0f "
	       "MB/s%s\n",
	       mib, (unsigned long long)offset, bs, bl, blockSame ? "" : ", READ BACK WRONG");
	printf("process memory (%zu MiB): store %.0f MB/s, load %.0f MB/s%s\n", mib, rs, rl,
	       ramSame ? "" : ", READ BACK WRONG");
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodFree, &offset, 1, NULL, NULL);
	IOServiceClose(conn);
	free(ram);
	if (!blockSame || !ramSame) {
		printf("SPEED FAIL: a load read other than what was stored\n");
		return 1;
	}
	printf("SPEED OK\n");
	return 0;
}

int main(int argc, char **argv)
{
	if (argc >= 2 && !strcmp(argv[1], "check"))
		return check();
	if (argc >= 2 && !strcmp(argv[1], "speed")) {
		const long mib = argc >= 3 ? strtol(argv[2], NULL, 0) : 64;
		if (mib <= 0 || mib > 512) {
			fprintf(stderr, "pat_probe speed: MiB from 1 to 512\n");
			return 2;
		}
		return speed((size_t)mib);
	}
	fprintf(stderr, "usage: pat_probe check | speed [MiB]\n");
	return 2;
}
