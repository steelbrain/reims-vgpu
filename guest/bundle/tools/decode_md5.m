// decode_md5: decodes Annex B H.264 and HEVC files with VideoToolbox's hardware decoder, hashes
// every decoded frame as the CPU reads it, and compares the hashes with a reference made off the
// device (ffmpeg's MD5 of each frame's NV12 or P010 planes, one a line, beside the stream).
//   decode_md5 [-l LOOPS] [-f] [-i t|b] [-a] FILE...
//     FILE    NAME.264 (H.264) or NAME.265 (HEVC); its reference is NAME.md5 in the same
//             directory, the frames' MD5s in any order.
//     -l N    after the hashed pass, decodes the stream N more times without reading the frames on
//             the CPU, and prints the frames decoded a second (default 0, none).
//     -a      lets VideoToolbox pick any decoder, hardware where it can, as a player does: a
//             stream the hardware refuses at its session's create goes to the software decoder,
//             and the line says which one decoded: "decoder=hardware" or "decoder=software" as
//             the session answers kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
//             "decoder=unknown" where it answers nothing (VideoToolbox's software H.264 decoder
//             does not, measured on an x86-64 Mac on macOS 26.7.1 and an Apple-silicon one on 27).
//     -i t|b  makes each format description name two fields, top (t) or bottom (b) first, as a
//             QuickTime file's 'fiel' atom does for interlaced video; the line says "fields=2".
//     -f      decodes every unit under one format description, of the stream's first parameter
//             sets, the later ones dropped with the rest: what a client that misses a change of
//             the sets hands VideoToolbox.
// The stream's parameter sets go in format descriptions, never in the samples: a unit after a
// VPS, SPS or PPS unlike the one before it of its kind is decoded under a new description of the
// sets then current. The session takes it where VTDecompressionSessionCanAcceptFormatDescription
// says it can; otherwise its frames are waited for and a new session is made with it.
// A frame's MD5 is over its visible planes as ffmpeg writes them: the luma rows, then the
// interleaved chroma rows, each cropped to the frame's width (two bytes a sample for 10-bit).
// The comparison is of the sorted lists: the decoder's output order is not the file's order.
// One line a file: "decode_md5: NAME OK frames=N lock_ms_mean=M [fps=F]", "decode_md5: NAME FAIL
// why", or "decode_md5: NAME SKIP why" where the system has no hardware decoder for the codec. A
// stream of more than one format description says so on its line: "formats=F sessions=S", the
// descriptions and the sessions its hashed pass made.
// A FAIL of the hashed pass's frames names them, by the access unit each was decoded from (its
// place in the file, from 0, which the tool stamps as its presentation time in 1/30 s): on the
// line, "wrong=U,..." (a frame whose MD5 is none of the reference's, or one more of an MD5 than
// the reference holds), "missing=U,..." (a unit no frame came back for) and "failed=U,..." (a
// unit the decoder answered an error for), each at most 32 then "+N"; then a line a frame,
// "decode_md5: NAME frame wrong|missing|failed unit=U type=T pts=U/30 [out=I md5=M]
// [status=S]", T the unit's first VCL NAL unit type and I the frame's place in the output, at
// most 100 lines. Scattered single frames read as lost writes, a run from one unit to the
// next random access point as a reference gone wrong.
// Exits 1 where a file failed, else 0.
// Build (-arch arm64 for an Apple-silicon Mac):
//   xcrun clang -arch x86_64 -mmacosx-version-min=14.0 -fobjc-arc -O2 -Wall -Wextra -Werror
//       -Wno-unused-parameter -framework Foundation -framework CoreMedia -framework CoreVideo
//       -framework VideoToolbox decode_md5.m -o decode_md5
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <VideoToolbox/VideoToolbox.h>

#include <CommonCrypto/CommonDigest.h>
#include <mach/mach_time.h>
#include <unistd.h>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

static double nowMs(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

// What the output callback gathers for one pass.
@interface DecodePass : NSObject
@property BOOL hashes;
@property NSMutableArray<NSString *> *md5s;
// The hashed frames in the order they came back: each its MD5 and its stamp (its unit), and each
// error its stamp and status.
@property NSMutableArray<NSArray *> *outputs;
@property NSMutableArray<NSArray *> *failures;
@property int frames;
@property int errors;
@property double lockSum;
@property OSType format;
@end
@implementation DecodePass
@end

static NSString *frameMd5(CVPixelBufferRef pb)
{
	OSType format = CVPixelBufferGetPixelFormatType(pb);
	size_t sample = (format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange ||
	                 format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange)
	                    ? 2
	                    : 1;
	size_t w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb);
	CC_MD5_CTX md;
	CC_MD5_Init(&md);
	for (size_t p = 0; p < CVPixelBufferGetPlaneCount(pb) && p < 2; p++) {
		const uint8_t *base = CVPixelBufferGetBaseAddressOfPlane(pb, p);
		size_t stride = CVPixelBufferGetBytesPerRowOfPlane(pb, p);
		size_t rows = p == 0 ? h : (h + 1) / 2;
		size_t bytes = (p == 0 ? w : (w + 1) / 2 * 2) * sample;
		for (size_t r = 0; r < rows; r++)
			CC_MD5_Update(&md, base + r * stride, (CC_LONG)bytes);
	}
	unsigned char out[CC_MD5_DIGEST_LENGTH];
	CC_MD5_Final(out, &md);
	NSMutableString *s = [NSMutableString stringWithCapacity:32];
	for (int i = 0; i < CC_MD5_DIGEST_LENGTH; i++)
		[s appendFormat:@"%02x", out[i]];
	return s;
}

// A pixel format's four characters.
static NSString *fourcc(OSType v)
{
	char c[5] = {(char)(v >> 24), (char)(v >> 16), (char)(v >> 8), (char)v, 0};
	for (int i = 0; i < 4; i++)
		if (c[i] < 32 || c[i] > 126)
			return [NSString stringWithFormat:@"0x%08x", (unsigned)v];
	return @(c);
}

static void decoded(void *ref, void *frameRef, OSStatus status, VTDecodeInfoFlags flags,
                    CVImageBufferRef image, CMTime pts, CMTime duration)
{
	DecodePass *pass = (__bridge DecodePass *)ref;
	if (status != noErr || image == NULL) {
		@synchronized(pass) {
			pass.errors++;
			[pass.failures addObject:@[ @(pts.value), @(status) ]];
		}
		return;
	}
	NSString *md5 = nil;
	double lock = 0;
	if (pass.hashes) {
		double t0 = nowMs();
		CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
		lock = nowMs() - t0;
		md5 = frameMd5(image);
		CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
	}
	@synchronized(pass) {
		if (pass.frames == 0)
			pass.format = CVPixelBufferGetPixelFormatType(image);
		pass.frames++;
		pass.lockSum += lock;
		if (md5 != nil) {
			[pass.md5s addObject:md5];
			[pass.outputs addObject:@[ md5, @(pts.value) ]];
		}
	}
}

// The type of the first VCL NAL unit of `unit` (4-byte-length NAL units), -1 for none.
static int unitType(NSData *unit, BOOL hevc)
{
	const uint8_t *b = unit.bytes;
	for (size_t at = 0; at + 5 <= unit.length;) {
		const uint32_t len = CFSwapInt32BigToHost(*(const uint32_t *)(b + at));
		const int t = hevc ? (b[at + 4] >> 1) & 0x3f : b[at + 4] & 0x1f;
		if (hevc ? t < 32 : t == 1 || t == 5)
			return t;
		at += 4 + (size_t)len;
	}
	return -1;
}

// `list`'s first 32 numbers joined by commas, then "+N" for the rest.
static NSString *capped(NSArray<NSNumber *> *list)
{
	NSMutableString *s = [NSMutableString string];
	for (NSUInteger i = 0; i < list.count && i < 32; i++)
		[s appendFormat:@"%@%@", i ? @"," : @"", list[i]];
	if (list.count > 32)
		[s appendFormat:@"+%lu", (unsigned long)(list.count - 32)];
	return s;
}

// The hashed pass's frames that are not the reference's, named on `line` and printed one a line
// after it (the header says how).
static void nameFrames(const char *n, NSArray<NSString *> *ref, NSArray<NSData *> *units, BOOL hevc,
                       DecodePass *pass, NSMutableString *line, NSMutableArray<NSString *> *detail)
{
	NSCountedSet *want = [[NSCountedSet alloc] initWithArray:ref];
	NSMutableIndexSet *back = [NSMutableIndexSet indexSet];
	NSMutableArray<NSNumber *> *wrong = [NSMutableArray array], *missing = [NSMutableArray array],
	                           *failed = [NSMutableArray array];
	for (NSUInteger i = 0; i < pass.outputs.count; i++) {
		NSString *md5 = pass.outputs[i][0];
		NSNumber *unit = pass.outputs[i][1];
		[back addIndex:unit.unsignedIntegerValue];
		if ([want countForObject:md5] > 0) {
			[want removeObject:md5];
			continue;
		}
		[wrong addObject:unit];
		[detail addObject:[NSString stringWithFormat:@"decode_md5: %s frame wrong unit=%@ type=%d pts=%@/30 out=%lu md5=%@",
		                                             n, unit, unitType(units[unit.unsignedIntegerValue % units.count], hevc),
		                                             unit, (unsigned long)i, md5]];
	}
	for (NSArray *f in pass.failures) {
		NSNumber *unit = f[0];
		[back addIndex:unit.unsignedIntegerValue];
		[failed addObject:unit];
		[detail addObject:[NSString stringWithFormat:@"decode_md5: %s frame failed unit=%@ type=%d pts=%@/30 status=%@", n,
		                                             unit, unitType(units[unit.unsignedIntegerValue % units.count], hevc),
		                                             unit, f[1]]];
	}
	for (NSUInteger u = 0; u < units.count; u++)
		if (![back containsIndex:u]) {
			[missing addObject:@(u)];
			[detail addObject:[NSString stringWithFormat:@"decode_md5: %s frame missing unit=%lu type=%d pts=%lu/30", n,
			                                             (unsigned long)u, unitType(units[u], hevc), (unsigned long)u]];
		}
	if (wrong.count)
		[line appendFormat:@" wrong=%@", capped(wrong)];
	if (missing.count)
		[line appendFormat:@" missing=%@", capped(missing)];
	if (failed.count)
		[line appendFormat:@" failed=%@", capped(failed)];
}

// The NAL units of an Annex B stream, without their start codes or trailing zero bytes.
static NSArray<NSData *> *nalUnits(NSData *file)
{
	const uint8_t *b = file.bytes;
	size_t n = file.length, start = SIZE_MAX;
	NSMutableArray<NSData *> *out = [NSMutableArray array];
	for (size_t i = 0; i + 3 <= n; i++) {
		if (b[i] != 0 || b[i + 1] != 0 || b[i + 2] != 1)
			continue;
		if (start != SIZE_MAX) {
			size_t end = i;
			while (end > start && b[end - 1] == 0)
				end--;
			[out addObject:[NSData dataWithBytes:b + start length:end - start]];
		}
		start = i + 3;
		i += 2;
	}
	if (start != SIZE_MAX && start < n)
		[out addObject:[NSData dataWithBytes:b + start length:n - start]];
	return out;
}

// With -a, sessions take any decoder VideoToolbox picks, hardware where it can (else only the
// hardware one); g_hardware is whether the first session's is hardware (-1 not known).
static BOOL g_anyDecoder;
static int g_hardware = -1;

// With -i t or -i b, each format description names two fields, top or bottom first (0: as
// CoreMedia makes it from the parameter sets).
static int g_fields;

// A format description of the parameter sets `sets` holds (VPS, SPS and PPS, each by its id: HEVC's
// the latest of each kind, H.264's every SPS and PPS by seq_ and pic_parameter_set_id, as an avcC
// record holds them); NULL where the system takes none of them.
static CMVideoFormatDescriptionRef describe(NSDictionary<NSNumber *, NSData *> *const sets[3], BOOL hevc)
{
	NSMutableArray<NSData *> *list = [NSMutableArray array];
	for (int k = hevc ? 0 : 1; k < 3; k++) {
		if (sets[k].count == 0)
			return NULL;
		for (NSNumber *id in [sets[k].allKeys sortedArrayUsingSelector:@selector(compare:)])
			[list addObject:sets[k][id]];
	}
	const uint8_t *bytes[600];
	size_t sizes[600];
	if (list.count > 600)
		return NULL;
	for (NSUInteger i = 0; i < list.count; i++) {
		bytes[i] = list[i].bytes;
		sizes[i] = list[i].length;
	}
	CMVideoFormatDescriptionRef fd = NULL;
	OSStatus st = hevc ? CMVideoFormatDescriptionCreateFromHEVCParameterSets(NULL, list.count, bytes, sizes, 4,
	                                                                         NULL, &fd)
	                   : CMVideoFormatDescriptionCreateFromH264ParameterSets(NULL, list.count, bytes, sizes, 4,
	                                                                         &fd);
	if (st != noErr)
		return NULL;
	if (g_fields == 0)
		return fd;
	// The description a QuickTime file's 'fiel' atom of two fields gives, top or bottom first.
	NSMutableDictionary *ext = [(__bridge NSDictionary *)CMFormatDescriptionGetExtensions(fd) mutableCopy];
	ext[(__bridge NSString *)kCMFormatDescriptionExtension_FieldCount] = @2;
	ext[(__bridge NSString *)kCMFormatDescriptionExtension_FieldDetail] =
	    (__bridge NSString *)(g_fields == 't' ? kCMFormatDescriptionFieldDetail_TemporalTopFirst
	                                          : kCMFormatDescriptionFieldDetail_TemporalBottomFirst);
	const CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(fd);
	CMVideoFormatDescriptionRef fields = NULL;
	st = CMVideoFormatDescriptionCreate(NULL, CMFormatDescriptionGetMediaSubType(fd), dims.width,
	                                    dims.height, (__bridge CFDictionaryRef)ext, &fields);
	CFRelease(fd);
	return st == noErr ? fields : NULL;
}

// The field count a format description names (its FieldCount extension), 0 for none.
static int descriptionFields(CMVideoFormatDescriptionRef fd)
{
	NSNumber *n = (__bridge NSNumber *)CMFormatDescriptionGetExtension(fd, kCMFormatDescriptionExtension_FieldCount);
	return [n isKindOfClass:NSNumber.class] ? n.intValue : 0;
}

// Bits of an H.264 NAL unit's RBSP: the bytes after its header, emulation prevention removed.
typedef struct {
	uint8_t b[64];
	size_t n, bit;
	BOOL over;
} Bits;

static void rbspHead(NSData *nal, Bits *r)
{
	const uint8_t *b = nal.bytes;
	r->n = r->bit = 0;
	r->over = NO;
	int zeros = 0;
	for (size_t i = 1; i < nal.length && r->n < sizeof r->b; i++) {
		if (zeros >= 2 && b[i] == 3) {
			zeros = 0;
			continue;
		}
		zeros = b[i] == 0 ? zeros + 1 : 0;
		r->b[r->n++] = b[i];
	}
}

static uint32_t bitsU(Bits *r, int k)
{
	uint32_t v = 0;
	for (int i = 0; i < k; i++) {
		if (r->bit >= r->n * 8) {
			r->over = YES;
			return 0;
		}
		v = v << 1 | ((r->b[r->bit / 8] >> (7 - r->bit % 8)) & 1);
		r->bit++;
	}
	return v;
}

static uint32_t bitsUe(Bits *r)
{
	int zeros = 0;
	while (!r->over && bitsU(r, 1) == 0)
		if (++zeros > 31) {
			r->over = YES;
			return 0;
		}
	return (uint32_t)((1ull << zeros) - 1 + bitsU(r, zeros));
}

static int32_t bitsSe(Bits *r)
{
	const uint32_t k = bitsUe(r);
	return k & 1 ? (int32_t)((k + 1) / 2) : -(int32_t)(k / 2);
}

// What an H.264 slice header's start needs of its SPS (7.3.2.1.1), by seq_parameter_set_id.
typedef struct {
	BOOL valid, frameMbsOnly, separateColourPlane;
	uint32_t log2MaxFrameNum;
} SpsHead;

static void readSps(NSData *nal, SpsHead sps[32])
{
	Bits r;
	rbspHead(nal, &r);
	const uint32_t profile = bitsU(&r, 8);
	bitsU(&r, 16);
	const uint32_t id = bitsUe(&r);
	SpsHead s = {NO, YES, NO, 4};
	if (profile == 100 || profile == 110 || profile == 122 || profile == 244 || profile == 44 ||
	    profile == 83 || profile == 86 || profile == 118 || profile == 128 || profile == 138 ||
	    profile == 139 || profile == 134 || profile == 135) {
		const uint32_t chroma = bitsUe(&r);
		if (chroma == 3)
			s.separateColourPlane = bitsU(&r, 1);
		bitsUe(&r);
		bitsUe(&r);
		bitsU(&r, 1);
		if (bitsU(&r, 1))
			for (int i = 0; i < (chroma != 3 ? 8 : 12) && !r.over; i++)
				if (bitsU(&r, 1)) {
					int32_t last = 8, next = 8;
					for (int j = 0; j < (i < 6 ? 16 : 64) && !r.over; j++) {
						if (next != 0)
							next = (last + bitsSe(&r) + 256) % 256;
						last = next == 0 ? last : next;
					}
				}
	}
	s.log2MaxFrameNum = bitsUe(&r) + 4;
	const uint32_t pocType = bitsUe(&r);
	if (pocType == 0) {
		bitsUe(&r);
	} else if (pocType == 1) {
		bitsU(&r, 1);
		bitsSe(&r);
		bitsSe(&r);
		const uint32_t cycle = bitsUe(&r);
		for (uint32_t i = 0; i < cycle && !r.over; i++)
			bitsSe(&r);
	}
	bitsUe(&r);
	bitsU(&r, 1);
	bitsUe(&r);
	bitsUe(&r);
	s.frameMbsOnly = bitsU(&r, 1);
	s.valid = !r.over && id < 32 && s.log2MaxFrameNum <= 16;
	if (s.valid)
		sps[id] = s;
}

// Whether the H.264 slice `nal` is of a field picture, and its parity and frame_num (7.3.3), by
// the SPS its PPS names (`ppsSps`, by pic_parameter_set_id; 0xff for none yet). False where the
// header names no SPS read, or ends early.
static BOOL sliceField(NSData *nal, const SpsHead sps[32], const uint8_t ppsSps[256], BOOL *field,
                       BOOL *bottom, uint32_t *frameNum)
{
	Bits r;
	rbspHead(nal, &r);
	bitsUe(&r);
	bitsUe(&r);
	const uint32_t pps = bitsUe(&r);
	if (r.over || pps > 255 || ppsSps[pps] >= 32 || !sps[ppsSps[pps]].valid)
		return NO;
	const SpsHead *s = &sps[ppsSps[pps]];
	if (s->separateColourPlane)
		bitsU(&r, 2);
	*frameNum = bitsU(&r, (int)s->log2MaxFrameNum);
	*field = !s->frameMbsOnly && bitsU(&r, 1);
	*bottom = *field && bitsU(&r, 1);
	return !r.over;
}

// The id a parameter set of `kind` (0 VPS, 1 SPS, 2 PPS) is kept by: H.264's seq_ or
// pic_parameter_set_id; 0 for HEVC's, whose latest of each kind is kept.
static NSNumber *setId(NSData *d, int kind, BOOL hevc)
{
	if (hevc)
		return @0;
	Bits r;
	rbspHead(d, &r);
	if (kind == 1)
		bitsU(&r, 24);
	return @(bitsUe(&r));
}

// The stream's access units as samples of 4-byte-length NAL units, into `units`, and the format
// description each is decoded under, into `formats` (one an entry of `formats` names by its
// index, `unitFormats`). H.264: a slice whose first_mb_in_slice is 0, or an SEI, SPS, PPS or AUD
// after a slice, starts an access unit, and only slices go in the samples; the second field of a
// field pair (a field picture of the other parity and the same frame_num as a field alone in the
// sample before it) goes in its first field's sample, as a QuickTime or MP4 file holds the two
// fields of a frame in one sample (ffmpeg's muxers write them so). HEVC: a VCL unit with
// first_slice_segment_in_pic_flag, or a parameter set, AUD, prefix SEI or reserved non-VCL unit
// after a VCL one, starts an access unit, and parameter sets, AUDs and end units stay out of the
// samples. A unit's description is of the latest VPS, SPS and PPS before it (for H.264, the
// latest SPS and PPS of each id); a set unlike the one before it of its kind and id makes a new one
// for the units after it. With `firstOnly`, every unit is under one description, of the stream's
// first sets. False where the system takes no description a unit needs.
static BOOL split(NSArray<NSData *> *nals, BOOL hevc, BOOL firstOnly, NSMutableArray<NSData *> *units,
                  NSMutableArray *formats, NSMutableArray<NSNumber *> *unitFormats)
{
	// VPS, SPS, PPS: the latest of each by its id (setId).
	NSMutableDictionary<NSNumber *, NSData *> *set[3] = {[NSMutableDictionary dictionary],
	                                                    [NSMutableDictionary dictionary],
	                                                    [NSMutableDictionary dictionary]};
	BOOL changed = YES;
	if (firstOnly) {
		for (NSData *d in nals) {
			uint8_t b0 = ((const uint8_t *)d.bytes)[0];
			int t = hevc ? (b0 >> 1) & 0x3f : b0 & 0x1f;
			int k = hevc ? (t >= 32 && t <= 34 ? t - 32 : -1) : (t == 7 ? 1 : t == 8 ? 2 : -1);
			if (k >= 0 && set[k][setId(d, k, hevc)] == nil)
				set[k][setId(d, k, hevc)] = d;
		}
		CMVideoFormatDescriptionRef fd = describe(set, hevc);
		if (fd == NULL)
			return NO;
		[formats addObject:(__bridge_transfer id)fd];
		changed = NO;
	}
	NSMutableData *cur = nil;
	BOOL picture = NO;
	NSUInteger curFormat = 0;
	// H.264's fields: the SPSs and PPSs read, and the last sample's field where it holds one alone
	// (its parity, -1 for none) and its frame_num.
	SpsHead spsHeads[32] = {{NO, YES, NO, 4}};
	uint8_t ppsSps[256];
	memset(ppsSps, 0xff, sizeof ppsSps);
	int loneField = -1;
	uint32_t loneFrameNum = 0;
	for (NSData *d in nals) {
		const uint8_t *b = d.bytes;
		BOOL vcl, first, starts, keep;
		int kind;
		if (hevc) {
			int t = (b[0] >> 1) & 0x3f;
			vcl = t < 32;
			first = vcl && d.length > 2 && (b[2] & 0x80);
			starts = first || (t >= 32 && t <= 35) || t == 39 || (t >= 41 && t <= 44) || (t >= 48 && t <= 55);
			keep = !(t >= 32 && t <= 37);
			kind = t >= 32 && t <= 34 ? t - 32 : -1;
		} else {
			int t = b[0] & 0x1f;
			vcl = t == 1 || t == 5;
			first = vcl && d.length > 1 && (b[1] & 0x80);
			starts = first || (t >= 6 && t <= 9);
			keep = vcl;
			kind = t == 7 ? 1 : t == 8 ? 2 : -1;
			if (t == 7) {
				readSps(d, spsHeads);
			} else if (t == 8) {
				Bits r;
				rbspHead(d, &r);
				const uint32_t pps = bitsUe(&r), sps = bitsUe(&r);
				if (!r.over && pps < 256 && sps < 32)
					ppsSps[pps] = (uint8_t)sps;
			}
		}
		if (picture && starts) {
			[units addObject:cur];
			[unitFormats addObject:@(curFormat)];
			cur = nil;
			picture = NO;
		}
		BOOL field = NO, bottom = NO;
		uint32_t frameNum = 0;
		if (!hevc && first && sliceField(d, spsHeads, ppsSps, &field, &bottom, &frameNum)) {
			if (field && loneField >= 0 && (int)bottom != loneField && frameNum == loneFrameNum &&
			    units.count > 0) {
				// The second field of the last sample's.
				cur = [units.lastObject mutableCopy];
				curFormat = unitFormats.lastObject.unsignedIntegerValue;
				[units removeLastObject];
				[unitFormats removeLastObject];
				picture = YES;
				loneField = -1;
			} else {
				loneField = field ? (int)bottom : -1;
				loneFrameNum = frameNum;
			}
		}
		if (kind >= 0 && !firstOnly && ![set[kind][setId(d, kind, hevc)] isEqualToData:d]) {
			set[kind][setId(d, kind, hevc)] = d;
			changed = YES;
		}
		if (!keep)
			continue;
		// A unit is under the description of the sets current at its first VCL unit.
		if (vcl && !picture) {
			if (changed) {
				CMVideoFormatDescriptionRef fd = describe(set, hevc);
				if (fd == NULL)
					return NO;
				[formats addObject:(__bridge_transfer id)fd];
				changed = NO;
			}
			curFormat = formats.count - 1;
		}
		if (cur == nil)
			cur = [NSMutableData data];
		uint32_t len = CFSwapInt32HostToBig((uint32_t)d.length);
		[cur appendBytes:&len length:4];
		[cur appendData:d];
		picture = picture || vcl;
	}
	if (cur != nil && picture) {
		[units addObject:cur];
		[unitFormats addObject:@(curFormat)];
	}
	return formats.count > 0 && units.count > 0;
}

// The pixel format the frames are asked in, the one ffmpeg's reference is of: 10-bit 4:2:0 for an
// HEVC stream of the Main 10 profile (general_profile_idc 2, in the SPS's first profile byte), else
// 8-bit 4:2:0. It is the decoder's own on the guest's device; a decoder that answers another
// (Apple silicon's packed 10-bit) converts.
static OSType outputFormat(NSArray<NSData *> *nals, BOOL hevc)
{
	for (NSData *d in nals) {
		const uint8_t *b = d.bytes;
		if (hevc && d.length > 3 && ((b[0] >> 1) & 0x3f) == 33)
			return (b[3] & 0x1f) == 2 ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
			                          : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
	}
	return kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
}

// A decompression session of `fd` whose frames go to `pass`, into `out`; its OSStatus.
static OSStatus openSession(CMVideoFormatDescriptionRef fd, OSType format, DecodePass *pass,
                            VTDecompressionSessionRef *out)
{
	NSDictionary *attrs = @{(__bridge NSString *)kCVPixelBufferPixelFormatTypeKey: @(format)};
	NSDictionary *spec = @{
		(__bridge NSString *)(g_anyDecoder ? kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder
		                                   : kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder): @YES
	};
	VTDecompressionOutputCallbackRecord cb = {decoded, (__bridge void *)pass};
	OSStatus st = VTDecompressionSessionCreate(NULL, fd, (__bridge CFDictionaryRef)spec,
	                                           (__bridge CFDictionaryRef)attrs, &cb, out);
	CFBooleanRef hw = NULL;
	if (st == noErr && g_hardware < 0 &&
	    VTSessionCopyProperty(*out, kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder, NULL,
	                          &hw) == noErr &&
	    hw != NULL) {
		g_hardware = CFBooleanGetValue(hw) ? 1 : 0;
		CFRelease(hw);
	}
	return st;
}

// Ends `session` once its frames are out.
static void closeSession(VTDecompressionSessionRef session)
{
	VTDecompressionSessionWaitForAsynchronousFrames(session);
	VTDecompressionSessionInvalidate(session);
	CFRelease(session);
}

// Decodes every unit LOOPS times, each under its format description (split): a session takes a
// new description where it says it can, else a new session is made for it once the last one's
// frames are out. Returns the pass's wall time in ms, or a negative OSStatus; the sessions made
// go in `sessions`.
static double decodeAll(NSArray *formats, NSArray<NSData *> *units, NSArray<NSNumber *> *unitFormats,
                        int loops, OSType format, DecodePass *pass, int *sessions)
{
	CMVideoFormatDescriptionRef current = (__bridge CMVideoFormatDescriptionRef)formats[0];
	VTDecompressionSessionRef session = NULL;
	OSStatus st = openSession(current, format, pass, &session);
	if (st != noErr) {
		*sessions = 0;
		return st < 0 ? st : -st;
	}
	*sessions = 1;
	double t0 = nowMs();
	int64_t i = 0;
	for (int loop = 0; loop < loops && st == noErr; loop++) {
		for (NSUInteger k = 0; k < units.count; k++) {
			NSData *u = units[k];
			CMVideoFormatDescriptionRef fd =
			    (__bridge CMVideoFormatDescriptionRef)formats[unitFormats[k].unsignedIntegerValue];
			if (fd != current) {
				if (!VTDecompressionSessionCanAcceptFormatDescription(session, fd)) {
					closeSession(session);
					session = NULL;
					st = openSession(fd, format, pass, &session);
					if (st != noErr)
						break;
					++*sessions;
				}
				current = fd;
			}
			CMBlockBufferRef bb = NULL;
			CMBlockBufferCreateWithMemoryBlock(NULL, NULL, u.length, NULL, NULL, 0, u.length,
			                                   kCMBlockBufferAssureMemoryNowFlag, &bb);
			CMBlockBufferReplaceDataBytes(u.bytes, bb, 0, u.length);
			CMSampleTimingInfo timing = {CMTimeMake(1, 30), CMTimeMake(i++, 30), kCMTimeInvalid};
			size_t size = u.length;
			CMSampleBufferRef sb = NULL;
			CMSampleBufferCreateReady(NULL, bb, fd, 1, 1, &timing, 1, &size, &sb);
			CFRelease(bb);
			st = VTDecompressionSessionDecodeFrame(session, sb, kVTDecodeFrame_EnableAsynchronousDecompression,
			                                       NULL, NULL);
			CFRelease(sb);
			if (st != noErr)
				break;
		}
	}
	if (session != NULL)
		closeSession(session);
	double ms = nowMs() - t0;
	if (st != noErr)
		return st < 0 ? st : -st;
	return ms;
}

static BOOL checkFile(NSString *path, int loops, BOOL firstOnly)
{
	NSString *name = path.lastPathComponent.stringByDeletingPathExtension;
	NSString *ext = path.pathExtension.lowercaseString;
	BOOL hevc = [ext isEqualToString:@"265"] || [ext isEqualToString:@"hevc"];
	const char *n = name.UTF8String;
	g_hardware = -1;
	if (!g_anyDecoder && !VTIsHardwareDecodeSupported(hevc ? kCMVideoCodecType_HEVC : kCMVideoCodecType_H264)) {
		printf("decode_md5: %s SKIP no hardware %s decoder\n", n, hevc ? "HEVC" : "H.264");
		return YES;
	}
	NSData *file = [NSData dataWithContentsOfFile:path];
	NSString *refText = [NSString stringWithContentsOfFile:[path.stringByDeletingPathExtension
	                                                           stringByAppendingPathExtension:@"md5"]
	                                              encoding:NSUTF8StringEncoding
	                                                 error:NULL];
	if (file == nil || refText == nil) {
		printf("decode_md5: %s FAIL cannot read the stream or its reference\n", n);
		return NO;
	}
	NSMutableArray<NSString *> *ref = [NSMutableArray array];
	for (NSString *line in [refText componentsSeparatedByCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet])
		if (line.length == 32)
			[ref addObject:line.lowercaseString];
	NSMutableArray<NSData *> *units = [NSMutableArray array];
	NSMutableArray *formats = [NSMutableArray array];
	NSMutableArray<NSNumber *> *unitFormats = [NSMutableArray array];
	NSArray<NSData *> *nals = nalUnits(file);
	OSType format = outputFormat(nals, hevc);
	if (!split(nals, hevc, firstOnly, units, formats, unitFormats)) {
		printf("decode_md5: %s FAIL no parameter sets the system takes\n", n);
		return NO;
	}
	DecodePass *pass = [DecodePass new];
	pass.hashes = YES;
	pass.md5s = [NSMutableArray array];
	pass.outputs = [NSMutableArray array];
	pass.failures = [NSMutableArray array];
	int sessions = 0;
	double ms = decodeAll(formats, units, unitFormats, 1, format, pass, &sessions);
	if (ms < 0) {
		printf("decode_md5: %s FAIL %s: OSStatus %d\n", n, sessions ? "decoding" : "creating its session",
		       (int)ms);
		return NO;
	}
	[ref sortUsingSelector:@selector(compare:)];
	[pass.md5s sortUsingSelector:@selector(compare:)];
	NSCountedSet *want = [[NSCountedSet alloc] initWithArray:ref];
	int common = 0;
	for (NSString *m in pass.md5s) {
		if ([want countForObject:m] > 0) {
			[want removeObject:m];
			common++;
		}
	}
	BOOL ok = pass.errors == 0 && [pass.md5s isEqualToArray:ref];
	double lockMean = pass.frames ? pass.lockSum / pass.frames : 0;
	NSMutableString *line = [NSMutableString string];
	NSMutableArray<NSString *> *detail = [NSMutableArray array];
	if (ok)
		[line appendFormat:@"decode_md5: %s OK frames=%d units=%lu format=%@ lock_ms_mean=%.3f hashed_ms=%.1f", n,
		                   pass.frames, (unsigned long)units.count, fourcc(pass.format), lockMean, ms];
	else
	{
		[line appendFormat:@"decode_md5: %s FAIL frames=%d expected=%lu matching=%d errors=%d format=%@", n,
		                   pass.frames, (unsigned long)ref.count, common, pass.errors, fourcc(pass.format)];
		nameFrames(n, ref, units, hevc, pass, line, detail);
	}
	if (formats.count > 1)
		[line appendFormat:@" formats=%lu sessions=%d", (unsigned long)formats.count, sessions];
	if (g_anyDecoder)
		[line appendFormat:@" decoder=%s", g_hardware == 1 ? "hardware" : g_hardware == 0 ? "software" : "unknown"];
	if (descriptionFields((__bridge CMVideoFormatDescriptionRef)formats[0]) != 0)
		[line appendFormat:@" fields=%d", descriptionFields((__bridge CMVideoFormatDescriptionRef)formats[0])];
	if (ok && loops > 0) {
		DecodePass *timed = [DecodePass new];
		int timedSessions = 0;
		double tms = decodeAll(formats, units, unitFormats, loops, format, timed, &timedSessions);
		if (tms > 0 && timed.errors == 0)
			[line appendFormat:@" fps=%.1f timed_frames=%d", timed.frames * 1000.0 / tms, timed.frames];
		else {
			[line appendFormat:@" timed_pass_failed=%d", (int)tms];
			ok = NO;
			[line replaceOccurrencesOfString:@" OK " withString:@" FAIL " options:0 range:NSMakeRange(0, line.length)];
		}
	}
	printf("%s\n", line.UTF8String);
	for (NSUInteger i = 0; i < detail.count && i < 100; i++)
		printf("%s\n", detail[i].UTF8String);
	return ok;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		int c, loops = 0;
		BOOL firstOnly = NO;
		while ((c = getopt(argc, argv, "l:fi:a")) != -1) {
			if (c == 'a')
				g_anyDecoder = YES;
			else if (c == 'l')
				loops = atoi(optarg);
			else if (c == 'f')
				firstOnly = YES;
			else if (c == 'i' && (optarg[0] == 't' || optarg[0] == 'b'))
				g_fields = optarg[0];
			else {
				fprintf(stderr, "usage: decode_md5 [-l LOOPS] [-f] [-i t|b] [-a] FILE...\n");
				return 2;
			}
		}
		setvbuf(stdout, NULL, _IOLBF, 0);
		BOOL ok = YES;
		for (int i = optind; i < argc; i++)
			ok = checkFile(@(argv[i]), loops, firstOnly) && ok;
		return ok ? 0 : 1;
	}
}
