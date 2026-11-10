// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Compare the consumer compiler's superclass requirements with the running
// class. These are build requirements, never per-release layout tables.
#pragma once
#include <stdint.h>

namespace graft {
namespace layout {

struct Requirement {
    const char *name;
    uint32_t instanceSize;
    uint32_t parentSlots;
};

// The reasons a class's layout differs from its requirement. Unreadable and
// Method are the vtable planner's (Vtable.h): a vtable it cannot read, and a
// slot that does not hold the method the consumer names.
enum class Error { None, Declaration, MissingClass, Size, Length, Unreadable, Method };

// Whether the running class, of instance size actualSize and actualSlots vtable
// method slots, has exactly the size and vtable length the consumer was
// compiled against. An actualSize of 0 is a class that is not registered.
inline Error compare(const Requirement &r, uint32_t actualSize, uint32_t actualSlots) {
    if (!r.name || !r.instanceSize || !r.parentSlots)
        return Error::Declaration;
    if (!actualSize)
        return Error::MissingClass;
    if (actualSize != r.instanceSize)
        return Error::Size;
    if (actualSlots != r.parentSlots)
        return Error::Length;
    return Error::None;
}

} // namespace layout
} // namespace graft
