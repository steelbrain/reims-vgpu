// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Small helpers to allocate and free a value-initialized array through the
// kernel allocator, with overflow checks and placement construction. The kext
// has no standard library, so the binding code uses these for its scratch.
#pragma once
#include <IOKit/IOLib.h>

namespace graft {
struct ConstructAt {};
} // namespace graft
inline void *operator new(size_t, graft::ConstructAt, void *storage) noexcept {
    return storage;
}

namespace graft {
template <typename T>
T *allocateArray(size_t count) {
    if (!count || count > SIZE_MAX / sizeof(T))
        return nullptr;
    auto storage = static_cast<T *>(IOMalloc(count * sizeof(T)));
    if (storage)
        for (size_t i = 0; i < count; ++i)
            new (ConstructAt{}, storage + i) T{};
    return storage;
}
template <typename T>
void freeArray(T *storage, size_t count) {
    if (storage) {
        for (size_t i = 0; i < count; ++i)
            storage[i].~T();
        IOFree(storage, count * sizeof(T));
    }
}
} // namespace graft
