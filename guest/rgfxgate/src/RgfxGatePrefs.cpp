// RgfxGate's AppleGVA preference. AppleGVA, the framework behind VideoToolbox's
// hardware sessions, picks a Mac's vendor video driver from its accelerator,
// and on a Mac with AMD graphics only where its preferences domain has
// gvaForceAMDKE true; our accelerator answers as AMD's would, so with the key
// the sessions reach our bundle's video driver. On every normal boot the
// install thread checks the domain's file and sets the key where it is not
// true, keeping the file's other keys (rgfx_gva_pref.h): an absent file is
// made, an XML plist edited in place, a binary one (cfprefsd's, as `defaults
// delete` leaves it) edited with every object kept; an unreadable one is logged
// and left. An edited XML plist must parse, as the kernel's XML parser reads
// it, to a dict with the key true and every other key kept, and an edited
// binary one must read back so (rgfx_gva_pref_binary_kept), or nothing is
// written. The write runs in kernel context, as root, through the ordinary VFS
// path: /Library/Preferences is no SIP-restricted path.
#include "RgfxGateFile.hpp"

#include <IOKit/IOLib.h>
#include <libkern/c++/OSBoolean.h>
#include <libkern/c++/OSDictionary.h>
#include <libkern/c++/OSUnserialize.h>
#include <sys/fcntl.h>
#include <sys/vnode.h>

#include "RgfxGate.hpp"
#include "RgfxGatePrefs.hpp"
#include "rgfx_gva_pref.h"
#include "rgfx_match.h"

namespace rgfxgate {

static const char kGvaPrefPath[] = "/Library/Preferences/com.apple.AppleGVA.plist";
static const char kGvaKey[] = "gvaForceAMDKE";
// A larger file is no preferences file of AppleGVA's: left as it is.
static constexpr size_t kMaxPrefBytes = 64 * 1024;
// Room the edit adds at most: a key and its value. A binary plist's edit also
// copies its top-level dict's references and writes its offset table again, in
// entries one byte wider at most: within 4 times the file, and the room.
static constexpr size_t kEditRoom = 512;
static constexpr size_t kBinaryGrowth = 4;

// The dict `bytes` holds as an XML plist, retained; nullptr where it holds
// none.
static OSDictionary *parseDict(const char *bytes, size_t len) {
	char *z = static_cast<char *>(IOMalloc(len + 1));
	if (z == nullptr)
		return nullptr;
	memcpy(z, bytes, len);
	z[len] = '\0';
	OSObject *obj = OSUnserializeXML(z, len + 1);
	IOFree(z, len + 1);
	OSDictionary *dict = OSDynamicCast(OSDictionary, obj);
	if (dict == nullptr && obj != nullptr)
		obj->release();
	return dict;
}

// Whether `edited` reads as `before` (nullptr: an absent file) with the key
// true and nothing else changed in its count of keys.
static bool editKept(OSDictionary *before, const char *edited, size_t len) {
	OSDictionary *after = parseDict(edited, len);
	if (after == nullptr)
		return false;
	bool hadKey = before != nullptr && before->getObject(kGvaKey) != nullptr;
	unsigned int expected = (before ? before->getCount() : 0) + (hadKey ? 0 : 1);
	bool kept = after->getObject(kGvaKey) == kOSBooleanTrue && after->getCount() == expected;
	after->release();
	return kept;
}

void keepVideoPreference() {
	vfs_context_t ctx = vfs_context_create(nullptr);
	if (ctx == nullptr)
		return;
	vnode_t vp = nullptr;
	// Installer/Recovery: do nothing.
	if (vnode_lookup("/System/Installation", 0, &vp, ctx) == 0) {
		vnode_put(vp);
		vfs_context_rele(ctx);
		return;
	}
	int found = vnode_lookup(kGvaPrefPath, 0, &vp, ctx);
	if (found == 0)
		vnode_put(vp);
	vfs_context_rele(ctx);
	if (found != 0 && found != ENOENT) {
		graft_client_log(kClientID, "AppleGVA preferences: lookup failed (%d); left", found);
		return;
	}

	uint8_t *file = nullptr;
	size_t size = 0;
	if (found == 0) {
		file = readFile(kGvaPrefPath, size, kMaxPrefBytes);
		if (file == nullptr || size > kMaxPrefBytes) {
			graft_client_log(kClientID, "AppleGVA preferences unreadable (%zu bytes); left", size);
			if (file != nullptr)
				IOFree(file, size);
			return;
		}
	}
	const char *in = reinterpret_cast<const char *>(file);
	bool binary = in != nullptr && rgfx_starts(in, size, 0, "bplist");
	OSDictionary *before = nullptr;
	size_t cap = (binary ? kBinaryGrowth * size : size) + kEditRoom;
	char *out = static_cast<char *>(IOMalloc(cap));
	size_t outLen = 0;
	RgfxGvaPref verdict = kRgfxGvaPrefUnreadable;
	if (out != nullptr)
		verdict = rgfx_gva_pref_edit(in, size, out, cap, &outLen);
	// An XML file the kernel's parser does not read as a dict is left too.
	if (verdict == kRgfxGvaPrefWrite && in != nullptr && !binary) {
		before = parseDict(in, size);
		if (before == nullptr)
			verdict = kRgfxGvaPrefUnreadable;
	}
	switch (verdict) {
	case kRgfxGvaPrefSet:
		graft_client_log(kClientID, "AppleGVA preferences already set");
		break;
	case kRgfxGvaPrefBinary:
		graft_client_log(kClientID, "AppleGVA preferences are a binary plist the edit cannot take; left");
		break;
	case kRgfxGvaPrefUnreadable:
		graft_client_log(kClientID, "AppleGVA preferences are no XML plist with a dict; left");
		break;
	case kRgfxGvaPrefWrite:
		if (binary ? !rgfx_gva_pref_binary_kept(file, size, reinterpret_cast<uint8_t *>(out), outLen)
		           : !editKept(before, out, outLen)) {
			graft_client_log(kClientID, "AppleGVA preferences edit does not read back; left");
			break;
		}
		{
			// Write a temp sibling, then rename it over the file, so a torn write
			// or a concurrent cfprefsd write never leaves AppleGVA a half-written
			// plist. /Library/Preferences is not SIP-restricted, so the rename
			// needs no hook (rgfxRenameNode, shared with the install's swap).
			const char *what = found == 0 ? "edit" : "create";
			char tmp[sizeof(kGvaPrefPath) + 8];
			if (!rgfx_sibling_path(kGvaPrefPath, ".rgfxtmp", tmp, sizeof(tmp))) {
				graft_client_log(kClientID, "AppleGVA preferences temp path too long; left");
				break;
			}
			int err = writeFile(tmp, out, outLen,
			                                    O_CREAT | O_TRUNC | O_WRONLY | O_NOFOLLOW, 0644);
			if (err != 0) {
				graft_client_log(kClientID, "AppleGVA preferences %s FAILED (%d)", what, err);
				break;
			}
			vfs_context_t wctx = vfs_context_create(nullptr);
			bool renamed = wctx != nullptr && rgfxRenameNode(tmp, kGvaPrefPath, false, wctx);
			if (wctx != nullptr)
				vfs_context_rele(wctx);
			graft_client_log(kClientID, "AppleGVA preferences %s %s", what,
			       renamed ? "written" : "rename FAILED; left");
		}
		break;
	}
	if (before != nullptr)
		before->release();
	if (out != nullptr)
		IOFree(out, cap);
	if (file != nullptr)
		IOFree(file, size);
}

} // namespace rgfxgate
