// Pure editing logic of the AppleGVA preference RgfxGate keeps set, shared by
// the kext (RgfxGatePrefs.cpp) and the native tests. No kernel or Lilu
// dependency.
//
// AppleGVA picks the vendor video driver of a Mac with AMD graphics only where
// its preferences domain (/Library/Preferences/com.apple.AppleGVA.plist) has
// gvaForceAMDKE true; our accelerator answers as AMD's would, so the key sends
// VideoToolbox's hardware sessions to our bundle's video driver. The edit sets
// that one key and keeps everything else: an absent file gets a fresh XML
// plist; an XML plist whose top-level dict lacks the key, or holds another
// value for it, gets the key as true in its text, every other byte kept; a
// binary plist of such a dict, as `defaults` and cfprefsd write (an empty one
// after `defaults delete`), gets the key as true with every object kept
// (rgfx_gva_pref_edit_binary). A binary plist the edit cannot read, or text
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
	kRgfxGvaPrefBinary,     // a binary plist the edit cannot take: left as it is
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

// Binary plists (bplist00). A file is its header, "bplist00", its objects, a
// table of each object's offset, and a 32-byte trailer: 6 bytes unused, the
// size of a table entry and of an object reference, then the object count, the
// top object's index and the table's offset, 8 bytes each, all big-endian. An
// object's marker byte holds its kind in the high nibble and its count in the
// low one, or 0xF and an integer object after it holding the count. A dict
// (0xD) is its marker, its keys' references, then its values'; an ASCII string
// (0x5) its bytes; true is 0x09.

struct RgfxBplist {
	uint8_t offset_size; // bytes a table entry has
	uint8_t ref_size;    // bytes an object reference has
	uint64_t objects;    // how many objects there are
	uint64_t top;        // the top object's index
	uint64_t table;      // the offset table's offset
};

static const char kRgfxGvaKeyName[] = "gvaForceAMDKE";

// The big-endian number of p[0, n).
static inline uint64_t rgfx_be(const uint8_t *p, size_t n) {
	uint64_t v = 0;
	for (size_t i = 0; i < n; i++)
		v = (v << 8) | p[i];
	return v;
}

// Writes v into p[0, n), big-endian.
static inline void rgfx_put_be(uint8_t *p, size_t n, uint64_t v) {
	for (size_t i = n; i-- > 0;) {
		p[i] = (uint8_t)(v & 0xff);
		v >>= 8;
	}
}

// The bytes a number up to v takes, from 1 to 8.
static inline uint8_t rgfx_be_size(uint64_t v) {
	uint8_t n = 1;
	while (n < 8 && (v >> (8 * n)) != 0)
		n++;
	return n;
}

// The trailer of s[0, len); false where it is no bplist00 whose table and top
// object lie inside it.
static inline bool rgfx_bplist_read(const uint8_t *s, size_t len, struct RgfxBplist *b) {
	if (len < 8 + 1 + 32 || memcmp(s, "bplist00", 8) != 0)
		return false;
	const uint8_t *t = s + len - 32;
	b->offset_size = t[6];
	b->ref_size = t[7];
	b->objects = rgfx_be(t + 8, 8);
	b->top = rgfx_be(t + 16, 8);
	b->table = rgfx_be(t + 24, 8);
	if (b->offset_size < 1 || b->offset_size > 8 || b->ref_size < 1 || b->ref_size > 8)
		return false;
	if (b->objects == 0 || b->top >= b->objects || b->table < 9 || b->table > len - 32)
		return false;
	return (len - 32 - b->table) / b->offset_size >= b->objects;
}

// Where object `index` starts; 0 where it is no object or starts outside the
// objects.
static inline uint64_t rgfx_bplist_at(const uint8_t *s, const struct RgfxBplist *b,
                                      uint64_t index) {
	if (index >= b->objects)
		return 0;
	uint64_t at = rgfx_be(s + b->table + index * b->offset_size, b->offset_size);
	return at >= 8 && at < b->table ? at : 0;
}

// The count the object at `at` holds and where what it holds starts; false
// where its count runs past the objects.
static inline bool rgfx_bplist_count(const uint8_t *s, const struct RgfxBplist *b, uint64_t at,
                                     uint64_t *count, uint64_t *body) {
	uint8_t low = s[at] & 0x0f;
	if (low != 0x0f) {
		*count = low;
		*body = at + 1;
		return true;
	}
	if (at + 2 > b->table || (s[at + 1] & 0xf0) != 0x10 || (s[at + 1] & 0x0f) > 3)
		return false;
	uint64_t n = (uint64_t)1 << (s[at + 1] & 0x0f);
	if (at + 2 + n > b->table)
		return false;
	*count = rgfx_be(s + at + 2, (size_t)n);
	*body = at + 2 + n;
	return true;
}

// The top-level dict's entries and where its key references start; false
// where the top object is no dict whose references lie inside the objects.
static inline bool rgfx_bplist_dict(const uint8_t *s, const struct RgfxBplist *b,
                                    uint64_t *count, uint64_t *refs) {
	uint64_t at = rgfx_bplist_at(s, b, b->top);
	if (at == 0 || (s[at] & 0xf0) != 0xd0 || !rgfx_bplist_count(s, b, at, count, refs))
		return false;
	return *count <= (b->table - *refs) / (2u * b->ref_size);
}

// Reference `i` of the run at `refs`.
static inline uint64_t rgfx_bplist_ref(const uint8_t *s, const struct RgfxBplist *b,
                                       uint64_t refs, uint64_t i) {
	return rgfx_be(s + refs + i * b->ref_size, b->ref_size);
}

// The entry of the top-level dict whose key is gvaForceAMDKE; `count` where
// none is.
static inline uint64_t rgfx_bplist_key_entry(const uint8_t *s, const struct RgfxBplist *b,
                                             uint64_t count, uint64_t refs) {
	for (uint64_t i = 0; i < count; i++) {
		uint64_t at = rgfx_bplist_at(s, b, rgfx_bplist_ref(s, b, refs, i)), n = 0, body = 0;
		if (at == 0 || (s[at] & 0xf0) != 0x50 || !rgfx_bplist_count(s, b, at, &n, &body))
			continue;
		if (n == sizeof(kRgfxGvaKeyName) - 1 && n <= b->table - body &&
		    memcmp(s + body, kRgfxGvaKeyName, (size_t)n) == 0)
			return i;
	}
	return count;
}

// Whether the value of entry `entry` is true.
static inline bool rgfx_bplist_value_true(const uint8_t *s, const struct RgfxBplist *b,
                                          uint64_t count, uint64_t refs, uint64_t entry) {
	uint64_t at = rgfx_bplist_at(s, b, rgfx_bplist_ref(s, b, refs, count + entry));
	return at != 0 && s[at] == 0x09;
}

// The edit of a binary plist: every object kept byte for byte, and after them
// the key's string and a true where the top-level dict lacks the key, a true
// alone where it holds another value, and a copy of the dict with them, which
// the new trailer names as the top object. The old dict stays, referenced by
// nothing, as CoreFoundation's reader allows. Left (kRgfxGvaPrefBinary) where
// the file does not read as a dict, or where its references are too small to
// name the new objects.
static inline enum RgfxGvaPref rgfx_gva_pref_edit_binary(const uint8_t *s, size_t len,
                                                         uint8_t *out, size_t cap,
                                                         size_t *out_len) {
	struct RgfxBplist b;
	uint64_t count = 0, refs = 0;
	if (!rgfx_bplist_read(s, len, &b) || !rgfx_bplist_dict(s, &b, &count, &refs))
		return kRgfxGvaPrefBinary;
	uint64_t entry = rgfx_bplist_key_entry(s, &b, count, refs);
	bool add = entry == count;
	if (!add && rgfx_bplist_value_true(s, &b, count, refs, entry))
		return kRgfxGvaPrefSet;
	uint64_t key_index = b.objects, true_index = b.objects + (add ? 1 : 0);
	uint64_t dict_index = true_index + 1, entries = count + (add ? 1 : 0);
	if (rgfx_be_size(dict_index) > b.ref_size)
		return kRgfxGvaPrefBinary;
	size_t rs = b.ref_size;
	size_t key_len = add ? 1 + sizeof(kRgfxGvaKeyName) - 1 : 0;
	size_t count_bytes = entries < 15 ? 0 : entries < 0x100 ? 1 : entries < 0x10000 ? 2 : 4;
	size_t marker_len = count_bytes == 0 ? 1 : 2 + count_bytes;
	uint64_t table = b.table + key_len + 1 + marker_len + entries * 2 * rs;
	uint8_t os = rgfx_be_size(table);
	if (os < b.offset_size)
		os = b.offset_size;
	uint64_t objects = dict_index + 1;
	uint64_t total = table + objects * os + 32;
	if (total > cap)
		return kRgfxGvaPrefUnreadable;
	memcpy(out, s, (size_t)b.table);
	size_t at = (size_t)b.table;
	uint64_t key_at = at;
	if (add) {
		out[at++] = (uint8_t)(0x50 | (sizeof(kRgfxGvaKeyName) - 1));
		memcpy(out + at, kRgfxGvaKeyName, sizeof(kRgfxGvaKeyName) - 1);
		at += sizeof(kRgfxGvaKeyName) - 1;
	}
	uint64_t true_at = at;
	out[at++] = 0x09;
	uint64_t dict_at = at;
	if (count_bytes == 0) {
		out[at++] = (uint8_t)(0xd0 | entries);
	} else {
		out[at++] = 0xdf;
		out[at++] = (uint8_t)(0x10 | (count_bytes == 1 ? 0 : count_bytes == 2 ? 1 : 2));
		rgfx_put_be(out + at, count_bytes, entries);
		at += count_bytes;
	}
	memcpy(out + at, s + refs, (size_t)count * rs);
	at += (size_t)count * rs;
	if (add) {
		rgfx_put_be(out + at, rs, key_index);
		at += rs;
	}
	for (uint64_t i = 0; i < count; i++) {
		if (i == entry)
			rgfx_put_be(out + at, rs, true_index);
		else
			memcpy(out + at, s + refs + (count + i) * rs, rs);
		at += rs;
	}
	if (add) {
		rgfx_put_be(out + at, rs, true_index);
		at += rs;
	}
	for (uint64_t i = 0; i < b.objects; i++) {
		rgfx_put_be(out + at, os, rgfx_be(s + b.table + i * b.offset_size, b.offset_size));
		at += os;
	}
	if (add) {
		rgfx_put_be(out + at, os, key_at);
		at += os;
	}
	rgfx_put_be(out + at, os, true_at);
	at += os;
	rgfx_put_be(out + at, os, dict_at);
	at += os;
	memcpy(out + at, s + len - 32, 6);
	out[at + 6] = os;
	out[at + 7] = (uint8_t)rs;
	rgfx_put_be(out + at + 8, 8, objects);
	rgfx_put_be(out + at + 16, 8, dict_index);
	rgfx_put_be(out + at + 24, 8, table);
	at += 32;
	*out_len = at;
	return kRgfxGvaPrefWrite;
}

// Whether `edited` reads as the binary plist `in` with the key true: its
// objects the same bytes, and its top-level dict the same entries but for the
// key's value, the key added where `in` lacked it. The kext's check of a
// binary edit before it writes it, as the kernel's XML parser checks an XML
// one.
static inline bool rgfx_gva_pref_binary_kept(const uint8_t *in, size_t in_len,
                                             const uint8_t *edited, size_t edited_len) {
	struct RgfxBplist a, b;
	uint64_t count_a = 0, refs_a = 0, count_b = 0, refs_b = 0;
	if (!rgfx_bplist_read(in, in_len, &a) || !rgfx_bplist_dict(in, &a, &count_a, &refs_a) ||
	    !rgfx_bplist_read(edited, edited_len, &b) ||
	    !rgfx_bplist_dict(edited, &b, &count_b, &refs_b))
		return false;
	if (b.table < a.table || b.objects < a.objects || memcmp(in, edited, (size_t)a.table) != 0)
		return false;
	for (uint64_t i = 0; i < a.objects; i++)
		if (rgfx_bplist_at(in, &a, i) != rgfx_bplist_at(edited, &b, i))
			return false;
	uint64_t entry_a = rgfx_bplist_key_entry(in, &a, count_a, refs_a);
	uint64_t entry_b = rgfx_bplist_key_entry(edited, &b, count_b, refs_b);
	if (count_b != count_a + (entry_a == count_a ? 1 : 0) || entry_b == count_b ||
	    !rgfx_bplist_value_true(edited, &b, count_b, refs_b, entry_b))
		return false;
	for (uint64_t i = 0; i < count_a; i++) {
		if (i == entry_a)
			continue;
		if (rgfx_bplist_ref(in, &a, refs_a, i) != rgfx_bplist_ref(edited, &b, refs_b, i) ||
		    rgfx_bplist_ref(in, &a, refs_a, count_a + i) !=
		        rgfx_bplist_ref(edited, &b, refs_b, count_b + i))
			return false;
	}
	return true;
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
		return rgfx_gva_pref_edit_binary(reinterpret_cast<const uint8_t *>(in), in_len,
		                                 reinterpret_cast<uint8_t *>(out), cap, out_len);
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
