// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The x86-64 kernel-write window (src/KernelWrite.h's interface). One CPU
// clears CR0.WP for the transaction while the others are stopped at a
// rendezvous, then restores it and serializes instruction fetch before they
// resume. The transaction logic itself (precheck, verify, rollback) is the
// architecture-neutral src/WritePlan.h; this file supplies only the x86-64
// memory operations and the stop-the-world; src/arm64e/KernelWrite.cpp is the
// arm64e sibling.
#include "KernelWrite.h"
#include <IOKit/IOLib.h>
#include <mach/vm_prot.h>

#if !defined(__x86_64__)
#error Graft kernel writes require the x86-64 architecture implementation
#endif

namespace graft {
namespace {
using Rendezvous = void (*)(void (*)(void *), void *);
Rendezvous rendezvous;
IOLock *writeLock;
struct Transaction {
    const Patch *patches;
    size_t count;
    uint32_t claimed = 0;
    uint32_t complete = 0;
    WriteError error = WriteError::None;
};

void serialize() {
    uint32_t a = 0, b, c, d;
    __asm__ volatile("mfence; cpuid" : "+a"(a), "=b"(b), "=c"(c), "=d"(d)::"memory");
}

struct KernelMemory {
    uint64_t cr0 = 0;
    bool matches(const uint8_t *address, const uint8_t *expected, size_t size) {
        return memcmp(address, expected, size) == 0;
    }
    void enter() {
        __asm__ volatile("mov %%cr0, %0" : "=r"(cr0));
        uint64_t writable = cr0 & ~(uint64_t{1} << 16);
        __asm__ volatile("mov %0, %%cr0" ::"r"(writable) : "memory");
    }
    void store(uint8_t *address, const uint8_t *bytes, size_t size) {
        memcpy(address, bytes, size);
    }
    void leave() { __asm__ volatile("mov %0, %%cr0" ::"r"(cr0) : "memory"); }
};

void perform(void *argument) {
    auto &t = *static_cast<Transaction *>(argument);
    uint32_t expected = 0;
    if (__atomic_compare_exchange_n(&t.claimed, &expected, 1, false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_RELAXED)) {
        KernelMemory memory;
        t.error = write::applyValidated(t.patches, t.count, memory);
        __atomic_store_n(&t.complete, 1, __ATOMIC_RELEASE);
    } else {
        while (!__atomic_load_n(&t.complete, __ATOMIC_ACQUIRE))
            __asm__ volatile("pause" ::: "memory");
    }
    serialize();
}
} // namespace

bool initializeWrites(const RuntimeImage &kernel) {
    uint64_t address = kernel.resolve("_mp_rendezvous_no_intrs");
    if (!address || !kernel.contains(address, 16, VM_PROT_EXECUTE))
        return false;
    writeLock = IOLockAlloc();
    if (!writeLock)
        return false;
    rendezvous = reinterpret_cast<Rendezvous>(address);
    return true;
}

WriteError applyWrites(const Patch *patches, size_t count) {
    if (!rendezvous || !writeLock)
        return WriteError::Unavailable;
    auto declaration = write::validate(patches, count);
    if (declaration != WriteError::None)
        return declaration;
    if (!count)
        return WriteError::None;
    Transaction transaction{patches, count};
    IOLockLock(writeLock);
    rendezvous(perform, &transaction);
    IOLockUnlock(writeLock);
    return transaction.error;
}

} // namespace graft
