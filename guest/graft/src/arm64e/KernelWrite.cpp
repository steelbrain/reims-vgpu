// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The arm64e kernel-write backend (src/KernelWrite.h's interface). Graft binds
// a dormant consumer by writing its own `__DATA`, which the collection maps
// read-write, so there is no write-protect window to open and no
// stop-the-world: the consumer is not running, and nothing else reads those
// bytes until Graft opens its activation gate. A data barrier after the
// stores makes them visible before the consumer runs. The transaction logic
// (precheck, verify, rollback) is the neutral src/WritePlan.h.
//
// Every target must be writable (`__DATA`, not `__DATA_CONST`, which the kernel
// locks read-only before Graft's kmod start): the binding layer refuses a target
// that is not before any store, through writableConsumerTarget (the arm64e
// BindingPlacement, which rejects a read-only or non-writable segment), since a
// store to read-only memory faults. `_mp_rendezvous_no_intrs`, which the x86-64
// backend needs, is not exported on vmapple, and this backend does not use it.
#include "KernelWrite.h"
#include <IOKit/IOLib.h>
#include <string.h>

#if !defined(__arm64__)
#error Graft's arm64e kernel writes require the arm64e architecture implementation
#endif

namespace graft {
namespace {
IOLock *writeLock;

struct KernelMemory {
    bool matches(const uint8_t *address, const uint8_t *expected, size_t size) {
        return memcmp(address, expected, size) == 0;
    }
    void enter() {}
    void store(uint8_t *address, const uint8_t *bytes, size_t size) {
        memcpy(address, bytes, size);
    }
    // Complete the stores and make them visible before the consumer runs.
    void leave() { __asm__ volatile("dsb ish" ::: "memory"); }
};
} // namespace

bool initializeWrites(const RuntimeImage &) {
    writeLock = IOLockAlloc();
    return writeLock != nullptr;
}

WriteError applyWrites(const Patch *patches, size_t count) {
    if (!writeLock)
        return WriteError::Unavailable;
    auto declaration = write::validate(patches, count);
    if (declaration != WriteError::None)
        return declaration;
    if (!count)
        return WriteError::None;
    KernelMemory memory;
    IOLockLock(writeLock);
    auto error = write::applyValidated(patches, count, memory);
    IOLockUnlock(writeLock);
    return error;
}

} // namespace graft
