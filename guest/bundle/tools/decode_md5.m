// decode_md5: decodes Annex B H.264 and HEVC files with VideoToolbox's hardware decoder, hashes
// every decoded frame as the CPU reads it, and compares the hashes with a reference made off the
// device (ffmpeg's MD5 of each frame's NV12 or P010 planes, one a line, beside the stream).
//   decode_md5 [-l LOOPS] [-f] FILE...
//     FILE    NAME.264 (H.264) or NAME.265 (HEVC); its reference is NAME.md5 in the same
//             directory, the frames' MD5s in any order.
//     -l N    after the hashed pass, decodes the stream N more times without reading the frames on
//             the CPU, and prints the frames decoded a second (default 0, none).
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

// A format description of the parameter sets `vps` (HEVC alone), `sps` and `pps`; NULL where the
// system takes none of them.
static CMVideoFormatDescriptionRef describe(NSData *vps, NSData *sps, NSData *pps, BOOL hevc)
{
	if (sps == nil || pps == nil || (hevc && vps == nil))
		return NULL;
	CMVideoFormatDescriptionRef fd = NULL;
	OSStatus st;
	if (hevc) {
		const uint8_t *sets[3] = {vps.bytes, sps.bytes, pps.bytes};
		size_t sizes[3] = {vps.length, sps.length, pps.length};
		st = CMVideoFormatDescriptionCreateFromHEVCParameterSets(NULL, 3, sets, sizes, 4, NULL, &fd);
	} else {
		const uint8_t *sets[2] = {sps.bytes, pps.bytes};
		size_t sizes[2] = {sps.length, pps.length};
		st = CMVideoFormatDescriptionCreateFromH264ParameterSets(NULL, 2, sets, sizes, 4, &fd);
	}
	return st == noErr ? fd : NULL;
}

// The stream's access units as samples of 4-byte-length NAL units, into `units`, and the format
// description each is decoded under, into `formats` (one an entry of `formats` names by its
// index, `unitFormats`). H.264: a slice whose first_mb_in_slice is 0, or an SEI, SPS, PPS or AUD
// after a slice, starts an access unit, and only slices go in the samples. HEVC: a VCL unit with
// first_slice_segment_in_pic_flag, or a parameter set, AUD, prefix SEI or reserved non-VCL unit
// after a VCL one, starts an access unit, and parameter sets, AUDs and end units stay out of the
// samples. A unit's description is of the latest VPS, SPS and PPS before it; a set unlike the one
// before it of its kind makes a new one for the units after it. With `firstOnly`, every unit is
// under one description, of the stream's first sets. False where the system takes no description
// a unit needs.
static BOOL split(NSArray<NSData *> *nals, BOOL hevc, BOOL firstOnly, NSMutableArray<NSData *> *units,
                  NSMutableArray *formats, NSMutableArray<NSNumber *> *unitFormats)
{
	NSData *set[3] = {nil, nil, nil}; // VPS, SPS, PPS: the latest of each
	BOOL changed = YES;
	if (firstOnly) {
		for (NSData *d in nals) {
			uint8_t b0 = ((const uint8_t *)d.bytes)[0];
			int t = hevc ? (b0 >> 1) & 0x3f : b0 & 0x1f;
			int k = hevc ? (t >= 32 && t <= 34 ? t - 32 : -1) : (t == 7 ? 1 : t == 8 ? 2 : -1);
			if (k >= 0 && set[k] == nil)
				set[k] = d;
		}
		CMVideoFormatDescriptionRef fd = describe(set[0], set[1], set[2], hevc);
		if (fd == NULL)
			return NO;
		[formats addObject:(__bridge_transfer id)fd];
		changed = NO;
	}
	NSMutableData *cur = nil;
	BOOL picture = NO;
	NSUInteger curFormat = 0;
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
		}
		if (picture && starts) {
			[units addObject:cur];
			[unitFormats addObject:@(curFormat)];
			cur = nil;
			picture = NO;
		}
		if (kind >= 0 && !firstOnly && ![set[kind] isEqualToData:d]) {
			set[kind] = d;
			changed = YES;
		}
		if (!keep)
			continue;
		// A unit is under the description of the sets current at its first VCL unit.
		if (vcl && !picture) {
			if (changed) {
				CMVideoFormatDescriptionRef fd = describe(set[0], set[1], set[2], hevc);
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
		(__bridge NSString *)kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder: @YES
	};
	VTDecompressionOutputCallbackRecord cb = {decoded, (__bridge void *)pass};
	return VTDecompressionSessionCreate(NULL, fd, (__bridge CFDictionaryRef)spec,
	                                    (__bridge CFDictionaryRef)attrs, &cb, out);
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
	if (st != noErr)
		return st < 0 ? st : -st;
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
	if (!VTIsHardwareDecodeSupported(hevc ? kCMVideoCodecType_HEVC : kCMVideoCodecType_H264)) {
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
		printf("decode_md5: %s FAIL decoding: OSStatus %d\n", n, (int)ms);
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
		while ((c = getopt(argc, argv, "l:f")) != -1) {
			if (c == 'l')
				loops = atoi(optarg);
			else if (c == 'f')
				firstOnly = YES;
			else {
				fprintf(stderr, "usage: decode_md5 [-l LOOPS] [-f] FILE...\n");
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
