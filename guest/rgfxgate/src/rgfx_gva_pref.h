// Pure editing logic of the AppleGVA preference RgfxGate keeps set, shared by
// the kext (RgfxGatePrefs.cpp) and the native tests. No kernel or Lilu
// dependency.
//
// AppleGVA picks the vendor video driver of a Mac with AMD graphics only where
// its preferences domain (/Library/Preferences/com.apple.AppleGVA.plist) has
// gvaForceAMDKE true; our accelerator answers as AMD's would, so the key sends
// VideoToolbox's hardware sessions to our bundle's video driver. The edit sets
// that one key in the file's text and keeps every other byte: an absent file
// gets a fresh XML plist; an XML plist whose top-level dict lacks the key, or
// holds another value for it, gets the key as true; a binary plist, or text
// that is no XML plist with a top-level dict, is left as it is.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef KERNEL
#include <libkern/libkern.h> // memcmp, memcpy, strlen in the kext
#else
#include <string.h>
#endif

// What to do with the preferences file.
enum RgfxGvaPref {
	kRgfxGvaPrefSet,        // the key is already true: nothing to write
	kRgfxGvaPrefWrite,      // write the edited bytes
	kRgfxGvaPrefBinary,     // a binary plist: left as it is
	kRgfxGvaPrefUnreadable, // no XML plist with a top-level dict, or no room: left as it is
};

static const char kRgfxGvaKeyElement[] = "<key>gvaForceAMDKE</key>";
static const char kRgfxGvaFresh[] =
	"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
	"<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
	"\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
	"<plist version=\"1.0\">\n"
	"<dict>\n"
	"\t<key>gvaForceAMDKE</key>\n"
	"\t<true/>\n"
	"</dict>\n"
	"</plist>\n";

#define RGFX_NOT_FOUND ((size_t)-1)

// Where `needle` first occurs in s[from, len); RGFX_NOT_FOUND where it does not.
static inline size_t rgfx_find(const char *s, size_t len, const char *needle, size_t from) {
	size_t n = strlen(needle);
	if (n == 0 || len < n)
		return RGFX_NOT_FOUND;
	for (size_t i = from; i + n <= len; i++)
		if (memcmp(s + i, needle, n) == 0)
			return i;
	return RGFX_NOT_FOUND;
}

// Whether s[at, len) starts with `prefix`.
static inline bool rgfx_starts(const char *s, size_t len, size_t at, const char *prefix) {
	size_t n = strlen(prefix);
	return at <= len && len - at >= n && memcmp(s + at, prefix, n) == 0;
}

// The first byte at or after `at` that is no XML whitespace.
static inline size_t rgfx_skip_space(const char *s, size_t len, size_t at) {
	while (at < len && (s[at] == ' ' || s[at] == '\t' || s[at] == '\r' || s[at] == '\n'))
		at++;
	return at;
}

// Writes in[0, cut) + insert + in[resume, len) into out; false where out is too small.
static inline bool rgfx_splice(const char *in, size_t len, size_t cut, size_t resume,
                               const char *insert, char *out, size_t cap, size_t *out_len) {
	size_t n = strlen(insert);
	size_t total = cut + n + (len - resume);
	if (total > cap)
		return false;
	memcpy(out, in, cut);
	memcpy(out + cut, insert, n);
	memcpy(out + cut + n, in + resume, len - resume);
	*out_len = total;
	return true;
}

// The end of the value element at `at` (a boolean, an integer, a real or a
// string); RGFX_NOT_FOUND for any other element.
static inline size_t rgfx_value_end(const char *s, size_t len, size_t at) {
	if (rgfx_starts(s, len, at, "<true/>"))
		return at + 7;
	if (rgfx_starts(s, len, at, "<false/>"))
		return at + 8;
	static const char *const kOpen[] = {"<integer>", "<real>", "<string>"};
	static const char *const kClose[] = {"</integer>", "</real>", "</string>"};
	for (size_t i = 0; i < 3; i++) {
		if (!rgfx_starts(s, len, at, kOpen[i]))
			continue;
		size_t close = rgfx_find(s, len, kClose[i], at);
		return close == RGFX_NOT_FOUND ? RGFX_NOT_FOUND : close + strlen(kClose[i]);
	}
	return RGFX_NOT_FOUND;
}

// How deep `at` lies in dicts from `from`: the dicts opened before it, less
// those closed.
static inline long rgfx_dict_depth(const char *s, size_t len, size_t from, size_t at) {
	long depth = 0;
	for (size_t i = from; i < at && i < len; i++) {
		if (rgfx_starts(s, len, i, "<dict>"))
			depth++;
		else if (rgfx_starts(s, len, i, "</dict>"))
			depth--;
	}
	return depth;
}

// What to write for the file's bytes `in` (in_len of them; nullptr for an
// absent file): on kRgfxGvaPrefWrite, the bytes are out[0, *out_len).
static inline enum RgfxGvaPref rgfx_gva_pref_edit(const char *in, size_t in_len, char *out,
                                                  size_t cap, size_t *out_len) {
	if (in == NULL) {
		size_t n = strlen(kRgfxGvaFresh);
		if (n > cap)
			return kRgfxGvaPrefUnreadable;
		memcpy(out, kRgfxGvaFresh, n);
		*out_len = n;
		return kRgfxGvaPrefWrite;
	}
	if (rgfx_starts(in, in_len, 0, "bplist"))
		return kRgfxGvaPrefBinary;
	// The top-level dict: the first element inside the plist element.
	size_t plist = rgfx_find(in, in_len, "<plist", 0);
	if (plist == RGFX_NOT_FOUND)
		return kRgfxGvaPrefUnreadable;
	size_t plist_end = rgfx_find(in, in_len, ">", plist);
	if (plist_end == RGFX_NOT_FOUND)
		return kRgfxGvaPrefUnreadable;
	size_t root = rgfx_skip_space(in, in_len, plist_end + 1);
	if (rgfx_starts(in, in_len, root, "<dict/>")) {
		bool ok = rgfx_splice(in, in_len, root, root + 7,
		                      "<dict>\n\t<key>gvaForceAMDKE</key>\n\t<true/>\n</dict>", out, cap,
		                      out_len);
		return ok ? kRgfxGvaPrefWrite : kRgfxGvaPrefUnreadable;
	}
	if (!rgfx_starts(in, in_len, root, "<dict>"))
		return kRgfxGvaPrefUnreadable;
	// The key in the top-level dict, if there: its value set to true.
	for (size_t key = rgfx_find(in, in_len, kRgfxGvaKeyElement, root); key != RGFX_NOT_FOUND;
	     key = rgfx_find(in, in_len, kRgfxGvaKeyElement, key + 1)) {
		if (rgfx_dict_depth(in, in_len, root, key) != 1)
			continue;
		size_t value = rgfx_skip_space(in, in_len, key + strlen(kRgfxGvaKeyElement));
		if (rgfx_starts(in, in_len, value, "<true/>"))
			return kRgfxGvaPrefSet;
		size_t end = rgfx_value_end(in, in_len, value);
		if (end == RGFX_NOT_FOUND)
			return kRgfxGvaPrefUnreadable;
		bool ok = rgfx_splice(in, in_len, value, end, "<true/>", out, cap, out_len);
		return ok ? kRgfxGvaPrefWrite : kRgfxGvaPrefUnreadable;
	}
	// No such key: the key first in the dict.
	size_t after = root + 6;
	bool ok = rgfx_splice(in, in_len, after, after,
	                      "\n\t<key>gvaForceAMDKE</key>\n\t<true/>", out, cap, out_len);
	return ok ? kRgfxGvaPrefWrite : kRgfxGvaPrefUnreadable;
}
