/*
 * RgfxGLStore: the GL shader variants the bundle lowered (RgfxGLLowering.hpp), kept on disk
 * across an app's runs, as Metal's compiler service keeps what it compiled for an app in the
 * app's own cache directory.
 *
 * Metal's devices lower a GL variant in their compiler service and keep the result; the device
 * lowers it here, in the app's process (RgfxLowerGLIR), and without a store each run of an app
 * lowered every variant again on the app's thread (measured: Heaven on macOS 26.7.2, an RTX 3090,
 * lowers 411 variants a run, 2 ms at the median, 16 ms at the 90th percentile and up to 37 ms,
 * 2.1 s in all, each run the same). An entry holds only what the lowering answered before, under
 * the identity of everything the lowering reads (RgfxGLVariantKey), so it changes no byte the
 * guest reads, only how soon the guest has it.
 *
 * An entry is a file of its own, named for the SHA-256 of its key: "RGFXGLS1", the key's and
 * the value's u64 lengths, the key, the value, and the SHA-256 of everything before it. A read
 * takes an entry only where it reads whole, its hash is the bytes' and its key is the one asked
 * for; any other entry is removed and the variant lowered again. A write goes to a file of its
 * own and is renamed into place (Foundation's atomic write), so no process reads half an entry.
 * The entries' bytes are capped: past the cap the least recently used are dropped, oldest first,
 * until three quarters of it are left; a read marks its entry used.
 *
 * Every failure (no directory, a refused read or write, an entry that does not read whole) leaves
 * the variant to the lowering, as without a store; the store says the first one once
 * (takeFirstFailure), and the caller logs it. A read or write the system refuses turns the store
 * off for the process, so a sandbox that refuses its directory reports one refusal, not one a
 * variant.
 *
 * Our own ObjC++ (RgfxGLStore.mm), through Foundation's file calls, as the bundle writes an app's
 * binary archive: it imports none of open(2)'s family (tests/bundle_files_check.sh).
 */
#ifndef RGFX_GL_STORE_HPP_
#define RGFX_GL_STORE_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "RgfxGLLowering.hpp"

/// The key of a variant (newFunctionWithPluginData:bitcodeType:): `identity`
/// (RgfxGLLoweringIdentity: the bundle's build, the system's generator and LLVM, the OS build,
/// the generator's builtins and environment), the GL IR's `irLength` bytes at `ir` (after its
/// length word), the plugin data's `pluginLength` bytes at `plugin`, the bitcode type and the
/// function's MTLFunctionType. Each field is its tag, its u64 length and its bytes, so no two
/// lists of fields make one key.
std::string RgfxGLVariantKey(const std::string &identity, const uint8_t *ir, size_t irLength, const uint8_t *plugin,
                             size_t pluginLength, uint8_t bitcodeType, uint64_t functionType);

/// The key of a kernel of a variant's module under a stage input descriptor
/// (RgfxGLSpecializeStageIn): the variant's key and the descriptor's (RgfxGLStageIn::key).
std::string RgfxGLStageInKey(const std::string &variantKey, const std::string &stageInKey);

/// A lowered variant as an entry's value: "RGFXGLV1", its stage, and its metallib, entry, return
/// data and stage input text, each after its u64 length.
std::vector<uint8_t> RgfxGLEncodeLowered(const RgfxGLLowered &lowered);

/// The variant `value` holds, in `out`; false where it is not one RgfxGLEncodeLowered wrote.
bool RgfxGLDecodeLowered(const uint8_t *value, size_t length, RgfxGLLowered &out);

class RgfxGLStore {
public:
	/// A store in `directory`, made with its parents at the first write, of at most `capBytes`
	/// bytes of entries, for lowerings of `identity` (RgfxGLLoweringIdentity).
	RgfxGLStore(std::string directory, uint64_t capBytes, std::string identity = std::string());

	/// The value kept under `key`, in `value`, and marks the entry used. False where there is
	/// none, or where the entry does not read whole, with its hash and the key asked for (the
	/// entry is then removed).
	bool get(const std::string &key, std::vector<uint8_t> &value);

	/// Keeps `value` under `key`, in place of any entry there, and drops the least recently used
	/// entries while they hold more than the cap. False where it was not kept.
	bool put(const std::string &key, const std::vector<uint8_t> &value);

	/// True, with the first failure in `why`, the first time it is called after a failure; false
	/// before and after. A missing entry is no failure.
	bool takeFirstFailure(std::string &why);

	/// Records a failure the caller saw (an entry whose value did not decode).
	void fail(const std::string &why);

	/// Records a failure of the system's (a refused read or write), and turns the store off: its
	/// reads find nothing and its writes keep nothing from then on.
	void refuse(const std::string &why);

	/// The file an entry under `key` lives in.
	std::string pathOf(const std::string &key) const;

	const std::string &directory() const { return directory_; }
	const std::string &identity() const { return identity_; }

private:
	void trim(); // with lock_ held
	bool off();

	const std::string directory_;
	const uint64_t cap_;
	const std::string identity_;
	std::mutex lock_;          // guards everything below
	bool made_ = false;        // the directory is there
	bool off_ = false;         // refused: nothing is read or written
	bool counted_ = false;     // bytes_ counts the entries in the directory
	uint64_t bytes_ = 0;       // the entries' bytes, as last counted and written since
	std::string failure_;      // the first failure
	bool failureTaken_ = false;
};

/// The variants a process lowered or read from its store, by key, so that a variant the process
/// asks for again is answered from memory, the bytes it answered before: a GL app asks for some
/// variants more than once a run (measured: Heaven on macOS 26.7.2, an RTX 3090, asks for 75 of
/// its 298 variants again, 113 lowerings, 490 ms of its 2.1 s of lowering; and for 8 of its 34
/// stage-in kernels, 10 lowerings). Holds at most `capBytes` of keys and values; past it the
/// oldest go first.
class RgfxGLMemo {
public:
	explicit RgfxGLMemo(uint64_t capBytes);

	/// The variant kept under `key`, in `out`; false where there is none.
	bool get(const std::string &key, RgfxGLLowered &out);

	/// Keeps `lowered` under `key`, where no variant is kept under it already.
	void put(const std::string &key, const RgfxGLLowered &lowered);

	uint64_t bytes();

private:
	const uint64_t cap_;
	std::mutex lock_; // guards everything below
	std::unordered_map<std::string, RgfxGLLowered> entries_;
	std::deque<const std::string *> order_; // entries_' keys, oldest first
	uint64_t bytes_ = 0;
};

/// The variant under `key`: the memo's, where `memo` holds one; else the store's, where `store`
/// holds one that reads and decodes whole; else what `lower` makes, which the store then keeps.
/// The memo keeps what the store or `lower` gave. `*hit` (where given) says whether `lower` was
/// spared. False, with `why`, where `lower` fails. A null `store` or `memo` is not asked.
bool RgfxGLLowerKept(RgfxGLStore *store, const std::string &key,
                     const std::function<bool(RgfxGLLowered &, std::string &)> &lower, RgfxGLLowered &out,
                     std::string &why, bool *hit = nullptr, RgfxGLMemo *memo = nullptr);

#endif // RGFX_GL_STORE_HPP_
