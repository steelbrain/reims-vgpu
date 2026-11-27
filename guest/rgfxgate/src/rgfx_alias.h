// The link the guest's HEVC encoder plug-in loads the bundle's executable by
// (RgfxGateInstall.cpp's linkHevcDriver). AppleGVAHEVCEncoder, finding the
// accelerator's IntelAccelerator nub, dlopens a fixed path under
// /Library/GPUBundles, which must pass its trust check: the link is a hard link
// to the installed executable, one inode, so it carries the executable's
// signature and rootless tag, and dyld maps one image whichever path a process
// loads it by. Each install writes the executable as a new inode, so the link
// is made again whenever it names another file than the executable.
//
// No kernel or Graft dependency: the update takes the file system as a
// parameter, which the kext gives its VNOP calls and the native tests
// (tests/alias_test.cpp) a real directory, so the tests run the exact code the
// kext runs.
#pragma once
#include <stddef.h>
#include <stdint.h>

// What an update did.
enum RgfxAliasResult {
	// The link already names the executable's file: nothing changed.
	kRgfxAliasCurrent = 0,
	// The link was made, after any other file at its path was removed.
	kRgfxAliasLinked,
	// There is no executable to link to: nothing changed.
	kRgfxAliasNoTarget,
	// A directory above the link, or the link itself, could not be made.
	kRgfxAliasFailed,
};

// Makes `link` a hard link to `target` where it names another file or none:
// the file at `link` (not followed, so a symbolic link is replaced itself) is
// removed, the directories `dirs` (`ndirs` of them, outermost first, each
// holding the next and the last holding `link`) are made where absent, the link
// is made, and then the directories are tagged, deepest first, as the
// bundle's tree is. `Fs` answers file_id(path, &id) (a lookup that does not
// follow a final symbolic link; false where nothing is there), and does
// remove(path), ensure_dir(path) (true where a directory is there after),
// link(target, path) (true where made) and tag(path).
template <typename Fs>
static inline RgfxAliasResult rgfx_alias_update(Fs &fs, const char *target,
                                                const char *const *dirs, size_t ndirs,
                                                const char *link) {
	uint64_t want = 0, have = 0;
	if (!fs.file_id(target, &want))
		return kRgfxAliasNoTarget;
	if (fs.file_id(link, &have)) {
		if (have == want)
			return kRgfxAliasCurrent;
		fs.remove(link);
	}
	for (size_t i = 0; i < ndirs; i++)
		if (!fs.ensure_dir(dirs[i]))
			return kRgfxAliasFailed;
	if (!fs.link(target, link))
		return kRgfxAliasFailed;
	for (size_t i = ndirs; i-- > 0;)
		fs.tag(dirs[i]);
	return kRgfxAliasLinked;
}
