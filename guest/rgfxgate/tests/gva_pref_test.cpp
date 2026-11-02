// Native tests of the AppleGVA preference edit (src/rgfx_gva_pref.h): an
// absent file is made, the key is set in an XML plist's top-level dict with
// every other byte kept, and binary or unreadable files are left. Build and
// run with `make test`.
#include "rgfx_gva_pref.h"
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

	// A value of a kind it does not edit, a binary plist, no plist, a
	// top-level array, a cut-off file: left as they are.
	std::string data = std::string(kHead) +
	                   "<dict><key>gvaForceAMDKE</key><data>AA==</data></dict></plist>";
	assert(edit(data.c_str(), &w) == kRgfxGvaPrefUnreadable);
	assert(edit("bplist00\xd1\x01\x02", &w) == kRgfxGvaPrefBinary);
	assert(edit("{ gvaForceAMDKE = 1; }", &w) == kRgfxGvaPrefUnreadable);
	std::string array = std::string(kHead) + "<array/></plist>";
	assert(edit(array.c_str(), &w) == kRgfxGvaPrefUnreadable);
	assert(edit("<?xml version=\"1.0\"?><plist", &w) == kRgfxGvaPrefUnreadable);
	std::string cut = std::string(kHead) + "<dict><key>gvaForceAMDKE</key><integer>1";
	assert(edit(cut.c_str(), &w) == kRgfxGvaPrefUnreadable);
	// No room for the edit.
	assert(edit(other.c_str(), &w, other.size()) == kRgfxGvaPrefUnreadable);

	printf("gva_pref_test: all checks passed\n");
	return 0;
}
