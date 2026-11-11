// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Graft's always-on diagnostics: one tagged log that goes to IOLog and to the
// registry root, so ioreg shows Graft's state and why a consumer stayed
// dormant (docs/design.md, "What it reports"). It publishes GraftState,
// GraftReason, a bounded GraftLog and a GraftClient.<identifier> property per
// client. No boot-arg turns it on. The bounded buffer drops oldest lines first.
// On x86-64 the kernel keeps the IOLog lines of a kext OpenCore injected out of
// the unified log, so there GraftLog is the only record of the steps.
#include "Log.h"
#include <IOKit/IOLib.h>
#include <IOKit/IORegistryEntry.h>
#include <libkern/c++/OSString.h>

namespace graft {
namespace {
IOLock *lock;
IOLock *publicationLock;
char text[8192];
size_t length;
char currentState[64];
char currentReason[320];
} // namespace

bool initializeLog() {
    lock = IOLockAlloc();
    publicationLock = IOLockAlloc();
    if (!lock || !publicationLock) {
        if (lock)
            IOLockFree(lock);
        if (publicationLock)
            IOLockFree(publicationLock);
        lock = publicationLock = nullptr;
        return false;
    }
    return true;
}

void publishLog() {
    auto root = IORegistryEntry::getRegistryRoot();
    if (!root || !lock)
        return;
    IOLockLock(publicationLock);
    IOLockLock(lock);
    auto log = OSString::withCString(text);
    auto state = OSString::withCString(currentState);
    auto reason = OSString::withCString(currentReason);
    IOLockUnlock(lock);
    if (log) {
        root->setProperty("GraftLog", log);
        log->release();
    }
    if (state) {
        root->setProperty("GraftState", state);
        state->release();
    }
    if (reason) {
        root->setProperty("GraftReason", reason);
        reason->release();
    }
    IOLockUnlock(publicationLock);
}

void note(const char *format, ...) {
    char line[320];
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    IOLog("graft: %s\n", line);
    if (!lock)
        return;
    size_t added = strlen(line);
    IOLockLock(lock);
    while (length + added + 2 > sizeof(text)) {
        size_t drop = 0;
        while (drop < length && text[drop++] != '\n') {
        }
        memmove(text, text + drop, length - drop);
        length -= drop;
    }
    memcpy(text + length, line, added);
    length += added;
    text[length++] = '\n';
    text[length] = 0;
    IOLockUnlock(lock);
    publishLog();
}

void setState(const char *state, const char *reason) {
    if (lock) {
        IOLockLock(lock);
        strlcpy(currentState, state, sizeof(currentState));
        strlcpy(currentReason, reason, sizeof(currentReason));
        IOLockUnlock(lock);
    }
    note("%s: %s", state, reason);
}

void clientState(const char *identifier, const char *state, const char *reason) {
    note("%s: %s: %s", identifier, state, reason);
    auto root = IORegistryEntry::getRegistryRoot();
    if (root) {
        char key[160], value[384];
        snprintf(key, sizeof(key), "GraftClient.%s", identifier);
        snprintf(value, sizeof(value), "%s: %s", state, reason);
        root->setProperty(key, value);
    }
}

} // namespace graft
