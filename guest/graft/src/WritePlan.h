// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Shared transaction logic. The kernel supplies memory operations and a
// write window; host tests exercise the same precheck, verification and
// rollback path with controlled failures. This header never allocates.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace graft {

struct Patch {
    uint8_t *address;
    const uint8_t *before;
    const uint8_t *after;
    size_t size;
};

enum class WriteError { None, Unavailable, Declaration, Changed, Verify, Rollback };

namespace write {
namespace detail {

struct Range {
    uintptr_t first, last;
};
inline bool range(const void *address, size_t size, Range &out) {
    uintptr_t first = reinterpret_cast<uintptr_t>(address);
    if (!address || !size || size - 1 > UINTPTR_MAX - first)
        return false;
    out = {first, first + size - 1};
    return true;
}
inline bool overlaps(Range a, Range b) {
    return a.first <= b.last && b.first <= a.last;
}

} // namespace detail

// Validate arithmetic and non-aliasing, not mapping access or permissions.
// The caller keeps declarations and snapshots immutable through execution,
// and guarantees that every range is accessible for its full lifetime.
// Targets cannot alias snapshots or the declaration array: such writes
// would destroy inputs needed for a later write or rollback.
inline WriteError validate(const Patch *patches, size_t count) {
    if (!count)
        return WriteError::None;
    detail::Range declarations;
    if (count > SIZE_MAX / sizeof(Patch) ||
        !detail::range(patches, count * sizeof(Patch), declarations))
        return WriteError::Declaration;
    for (size_t i = 0; i < count; ++i) {
        const auto &p = patches[i];
        detail::Range target, before, after;
        if (!detail::range(p.address, p.size, target) || !detail::range(p.before, p.size, before) ||
            !detail::range(p.after, p.size, after) || detail::overlaps(target, declarations))
            return WriteError::Declaration;
        for (size_t j = 0; j < count; ++j) {
            detail::Range snapshot;
            if (!detail::range(patches[j].before, patches[j].size, snapshot) ||
                detail::overlaps(target, snapshot) ||
                !detail::range(patches[j].after, patches[j].size, snapshot) ||
                detail::overlaps(target, snapshot))
                return WriteError::Declaration;
            if (j < i) {
                detail::Range other;
                if (!detail::range(patches[j].address, patches[j].size, other) ||
                    detail::overlaps(target, other))
                    return WriteError::Declaration;
            }
        }
    }
    return WriteError::None;
}

// Requires validate() == None and unchanged declarations/snapshots. Memory
// operations must not allocate, block, log or throw. matches() must actually
// read the target; store() must not change declarations or snapshots.
// enter()/leave() bracket every store, including rollback. A Rollback result
// explicitly means restoration could not be verified, not a safe recovery.
template <typename Memory>
WriteError applyValidated(const Patch *patches, size_t count, Memory &memory) {
    if (!count)
        return WriteError::None;
    for (size_t i = 0; i < count; ++i) {
        const auto &p = patches[i];
        if (!memory.matches(p.address, p.before, p.size))
            return WriteError::Changed;
    }
    memory.enter();
    for (size_t i = 0; i < count; ++i) {
        const auto &p = patches[i];
        memory.store(p.address, p.after, p.size);
    }
    bool verified = true;
    for (size_t i = 0; i < count; ++i) {
        const auto &p = patches[i];
        if (!memory.matches(p.address, p.after, p.size))
            verified = false;
    }
    WriteError error = WriteError::None;
    if (!verified) {
        for (size_t i = 0; i < count; ++i) {
            const auto &p = patches[i];
            memory.store(p.address, p.before, p.size);
        }
        bool restored = true;
        for (size_t i = 0; i < count; ++i) {
            const auto &p = patches[i];
            if (!memory.matches(p.address, p.before, p.size))
                restored = false;
        }
        error = restored ? WriteError::Verify : WriteError::Rollback;
    }
    memory.leave();
    return error;
}

template <typename Memory>
WriteError apply(const Patch *patches, size_t count, Memory &memory) {
    auto error = validate(patches, count);
    return error == WriteError::None ? applyValidated(patches, count, memory) : error;
}

} // namespace write
} // namespace graft
