// Native tests of the AppleGVA preference edit (src/rgfx_gva_pref.h): an
// absent file is made, the key is set in an XML plist's top-level dict with
// every other byte kept, and in a binary plist's with every object kept, as
// CoreFoundation, which writes the binary inputs here, reads the outputs back;
// unreadable files are left. Build and run with `make test`.
#include "rgfx_gva_pref.h"
#include <CoreFoundation/CoreFoundation.h>
#include <cassert>
#include <cstdio>
#include <string>

// The edit of `in` (nullptr: absent), and the bytes it writes.
static RgfxGvaPref edit(const char *in, std::string *written, size_t cap = 4096) {
	std::string out(cap, '\0');
	size_t len = 0;
	RgfxGvaPref verdict = rgfx_gva_pref_edit(in, in ? strlen(in) : 0, &out[0], cap, &len);
	if (verdict == kRgfxGvaPrefWrite)
		*written = out.substr(0, len);
	return verdict;
}

// The edit of the bytes `in`, and the bytes it writes.
static RgfxGvaPref editBytes(const std::string &in, std::string *written, size_t cap = 1 << 20) {
	std::string out(cap, '\0');
	size_t len = 0;
	RgfxGvaPref verdict = rgfx_gva_pref_edit(in.data(), in.size(), &out[0], cap, &len);
	if (verdict == kRgfxGvaPrefWrite)
		*written = out.substr(0, len);
	return verdict;
}

// `plist` as CoreFoundation writes it in binary, as cfprefsd does.
static std::string binary(CFPropertyListRef plist) {
	CFDataRef data =
	    CFPropertyListCreateData(nullptr, plist, kCFPropertyListBinaryFormat_v1_0, 0, nullptr);
	assert(data != nullptr);
	std::string bytes(reinterpret_cast<const char *>(CFDataGetBytePtr(data)),
	                  static_cast<size_t>(CFDataGetLength(data)));
	CFRelease(data);
	return bytes;
}

// What CoreFoundation reads `bytes` as, and in which format; nullptr where it
// reads nothing.
static CFPropertyListRef parse(const std::string &bytes, CFPropertyListFormat *format) {
	CFDataRef data = CFDataCreate(nullptr, reinterpret_cast<const UInt8 *>(bytes.data()),
	                              static_cast<CFIndex>(bytes.size()));
	CFPropertyListRef plist =
	    CFPropertyListCreateWithData(nullptr, data, kCFPropertyListImmutable, format, nullptr);
	CFRelease(data);
	return plist;
}

static CFStringRef str(const char *s) {
	return CFStringCreateWithCString(nullptr, s, kCFStringEncodingUTF8);
}

// Edits `dict` written in binary, and checks that CoreFoundation reads the
// edit as a binary plist of `dict` with the key true, that the kext's check
// passes it, and that a second edit leaves it.
static void binaryEditKeeps(CFDictionaryRef dict) {
	std::string in = binary(dict), w;
	assert(editBytes(in, &w) == kRgfxGvaPrefWrite);
	CFPropertyListFormat format = kCFPropertyListXMLFormat_v1_0;
	CFPropertyListRef read = parse(w, &format);
	assert(read != nullptr && format == kCFPropertyListBinaryFormat_v1_0);
	CFMutableDictionaryRef want = CFDictionaryCreateMutableCopy(nullptr, 0, dict);
	CFStringRef key = str("gvaForceAMDKE");
	CFDictionarySetValue(want, key, kCFBooleanTrue);
	assert(CFEqual(read, want));
	assert(rgfx_gva_pref_binary_kept(reinterpret_cast<const uint8_t *>(in.data()), in.size(),
	                                 reinterpret_cast<const uint8_t *>(w.data()), w.size()));
	std::string again;
	assert(editBytes(w, &again) == kRgfxGvaPrefSet);
	// What cfprefsd would write next: the edit as CoreFoundation writes it again.
	assert(editBytes(binary(read), &again) == kRgfxGvaPrefSet);
	CFRelease(key);
	CFRelease(want);
	CFRelease(read);
}

static void binaryTests() {
	CFStringRef key = str("gvaForceAMDKE");
	// What `defaults delete com.apple.AppleGVA` leaves: an empty dict.
	CFMutableDictionaryRef dict = CFDictionaryCreateMutable(
	    nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	assert(binary(dict).size() == 42);
	binaryEditKeeps(dict);

	// Other keys of every kind kept.
	int n = 3;
	double r = 2.5;
	CFNumberRef integer = CFNumberCreate(nullptr, kCFNumberIntType, &n);
	CFNumberRef real = CFNumberCreate(nullptr, kCFNumberDoubleType, &r);
	CFDateRef date = CFDateCreate(nullptr, 800000000.0);
	const UInt8 raw[] = {0, 1, 2, 0xff};
	CFDataRef data = CFDataCreate(nullptr, raw, sizeof raw);
	CFStringRef text = str("x\xc3\xa9");
	const void *items[] = {integer, text, kCFBooleanFalse};
	CFArrayRef array = CFArrayCreate(nullptr, items, 3, &kCFTypeArrayCallBacks);
	CFMutableDictionaryRef nested = CFDictionaryCreateMutable(
	    nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	CFDictionarySetValue(nested, key, kCFBooleanFalse);
	const char *names[] = {"gvaDebug", "r", "d", "data", "s", "a", "n", "off"};
	const void *values[] = {integer, real, date, data, text, array, nested, kCFBooleanFalse};
	for (size_t i = 0; i < 8; i++) {
		CFStringRef name = str(names[i]);
		CFDictionarySetValue(dict, name, values[i]);
		CFRelease(name);
	}
	binaryEditKeeps(dict);

	// The key false, or a number: true in its place.
	CFDictionarySetValue(dict, key, kCFBooleanFalse);
	binaryEditKeeps(dict);
	CFDictionarySetValue(dict, key, integer);
	binaryEditKeeps(dict);
	CFDictionarySetValue(dict, key, kCFBooleanTrue);
	std::string w;
	assert(editBytes(binary(dict), &w) == kRgfxGvaPrefSet);
	CFDictionaryRemoveValue(dict, key);

	// 15 entries and more, whose count follows the dict's marker.
	for (int i = 0; i < 20; i++) {
		char name[16];
		snprintf(name, sizeof name, "k%d", i);
		CFStringRef k = str(name);
		CFDictionarySetValue(dict, k, kCFBooleanTrue);
		CFRelease(k);
	}
	binaryEditKeeps(dict);

	// Objects past 255: references of 2 bytes. Up to there, where the new
	// objects' references no longer fit in 1 byte, the file is left.
	CFMutableDictionaryRef many = CFDictionaryCreateMutable(
	    nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	int left = 0, written = 0;
	for (int i = 0; i < 300; i++) {
		char name[16];
		snprintf(name, sizeof name, "key%03d", i);
		CFStringRef k = str(name);
		CFDictionarySetValue(many, k, kCFBooleanFalse);
		CFRelease(k);
		std::string in = binary(many);
		struct RgfxBplist b;
		assert(rgfx_bplist_read(reinterpret_cast<const uint8_t *>(in.data()), in.size(), &b));
		RgfxGvaPref verdict = editBytes(in, &w);
		if (b.ref_size == 1 && b.objects + 2 > 0xff) {
			assert(verdict == kRgfxGvaPrefBinary);
			left++;
		} else {
			assert(verdict == kRgfxGvaPrefWrite);
			written++;
			if (i % 50 == 0 || b.ref_size == 2)
				binaryEditKeeps(many);
		}
	}
	assert(left > 0 && written > 250);

	// Not a dict, or no well-formed binary plist: left.
	assert(editBytes(binary(array), &w) == kRgfxGvaPrefBinary);
	std::string cut = binary(dict);
	assert(editBytes(cut.substr(0, cut.size() - 1), &w) == kRgfxGvaPrefBinary);
	assert(editBytes(std::string("bplist00\xd1\x01\x02", 11), &w) == kRgfxGvaPrefBinary);
	std::string bad = binary(dict);
	bad[bad.size() - 8] = '\x7f'; // the table's offset, the trailer's last 8 bytes, past the file
	assert(editBytes(bad, &w) == kRgfxGvaPrefBinary);
	// No room for the edit.
	CFDictionaryRef none = CFDictionaryCreate(nullptr, nullptr, nullptr, 0, nullptr, nullptr);
	std::string empty = binary(none);
	CFRelease(none);
	std::string out(empty.size(), '\0');
	size_t len = 0;
	assert(rgfx_gva_pref_edit(empty.data(), empty.size(), &out[0], out.size(), &len) ==
	       kRgfxGvaPrefUnreadable);

	// The kext's check refuses an edit that changed another entry.
	std::string in = binary(dict);
	assert(editBytes(in, &w) == kRgfxGvaPrefWrite);
	struct RgfxBplist kept;
	assert(rgfx_bplist_read(reinterpret_cast<const uint8_t *>(in.data()), in.size(), &kept));
	std::string tampered = w;
	tampered[kept.table - 1] ^= 0x01; // the last byte of the objects the edit keeps
	assert(!rgfx_gva_pref_binary_kept(reinterpret_cast<const uint8_t *>(in.data()), in.size(),
	                                  reinterpret_cast<const uint8_t *>(tampered.data()),
	                                  tampered.size()));

	CFRelease(many);
	CFRelease(nested);
	CFRelease(array);
	CFRelease(text);
	CFRelease(data);
	CFRelease(date);
	CFRelease(real);
	CFRelease(integer);
	CFRelease(dict);
	CFRelease(key);
}

static const char kHead[] =
	"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
	"<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
	"\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
	"<plist version=\"1.0\">\n";

int main() {
	std::string w;

	// An absent file: a fresh plist of the key alone, which an edit leaves.
	assert(edit(nullptr, &w) == kRgfxGvaPrefWrite);
	assert(w == kRgfxGvaFresh);
	assert(edit(w.c_str(), &w) == kRgfxGvaPrefSet);
	// No room for it.
	assert(edit(nullptr, &w, 16) == kRgfxGvaPrefUnreadable);

	// Another key kept, the key put first in the dict.
	std::string other = std::string(kHead) +
	                    "<dict>\n\t<key>gvaDebug</key>\n\t<integer>3</integer>\n</dict>\n</plist>\n";
	assert(edit(other.c_str(), &w) == kRgfxGvaPrefWrite);
	assert(w == std::string(kHead) +
	                "<dict>\n\t<key>gvaForceAMDKE</key>\n\t<true/>\n\t<key>gvaDebug</key>\n"
	                "\t<integer>3</integer>\n</dict>\n</plist>\n");
	assert(edit(w.c_str(), &w) == kRgfxGvaPrefSet);

	// An empty dict.
	std::string empty = std::string(kHead) + "<dict/>\n</plist>\n";
	assert(edit(empty.c_str(), &w) == kRgfxGvaPrefWrite);
	assert(w == std::string(kHead) +
	                "<dict>\n\t<key>gvaForceAMDKE</key>\n\t<true/>\n</dict>\n</plist>\n");

	// The key false, or of another type, becomes true in place.
	std::string off = std::string(kHead) +
	                  "<dict>\n\t<key>a</key>\n\t<string>x</string>\n"
	                  "\t<key>gvaForceAMDKE</key>\n\t<false/>\n</dict>\n</plist>\n";
	assert(edit(off.c_str(), &w) == kRgfxGvaPrefWrite);
	assert(w == std::string(kHead) +
	                "<dict>\n\t<key>a</key>\n\t<string>x</string>\n"
	                "\t<key>gvaForceAMDKE</key>\n\t<true/>\n</dict>\n</plist>\n");
	std::string number = std::string(kHead) +
	                     "<dict><key>gvaForceAMDKE</key><integer>0</integer></dict></plist>";
	assert(edit(number.c_str(), &w) == kRgfxGvaPrefWrite);
	assert(w == std::string(kHead) + "<dict><key>gvaForceAMDKE</key><true/></dict></plist>");
	std::string text = std::string(kHead) +
	                   "<dict><key>gvaForceAMDKE</key><string>NO</string></dict></plist>";
	assert(edit(text.c_str(), &w) == kRgfxGvaPrefWrite);
	assert(w == std::string(kHead) + "<dict><key>gvaForceAMDKE</key><true/></dict></plist>");

	// The key in a nested dict is another key: the top-level one is added.
	std::string nested = std::string(kHead) +
	                     "<dict><key>n</key><dict><key>gvaForceAMDKE</key><false/></dict>"
	                     "</dict></plist>";
	assert(edit(nested.c_str(), &w) == kRgfxGvaPrefWrite);
	assert(w == std::string(kHead) +
	                "<dict>\n\t<key>gvaForceAMDKE</key>\n\t<true/><key>n</key><dict>"
	                "<key>gvaForceAMDKE</key><false/></dict></dict></plist>");

	// A value of a kind it does not edit, no plist, a top-level array, a
	// cut-off file: left as they are.
	std::string data = std::string(kHead) +
	                   "<dict><key>gvaForceAMDKE</key><data>AA==</data></dict></plist>";
	assert(edit(data.c_str(), &w) == kRgfxGvaPrefUnreadable);
	assert(edit("{ gvaForceAMDKE = 1; }", &w) == kRgfxGvaPrefUnreadable);
	std::string array = std::string(kHead) + "<array/></plist>";
	assert(edit(array.c_str(), &w) == kRgfxGvaPrefUnreadable);
	assert(edit("<?xml version=\"1.0\"?><plist", &w) == kRgfxGvaPrefUnreadable);
	std::string cut = std::string(kHead) + "<dict><key>gvaForceAMDKE</key><integer>1";
	assert(edit(cut.c_str(), &w) == kRgfxGvaPrefUnreadable);
	// No room for the edit.
	assert(edit(other.c_str(), &w, other.size()) == kRgfxGvaPrefUnreadable);

	binaryTests();

	printf("gva_pref_test: all checks passed\n");
	return 0;
}
