// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The kernel-write window: the architecture-neutral interface over one
// transaction that precheck-writes-verifies a batch of patches. The transaction
// logic is WritePlan.h. Each architecture supplies the memory operations:
// src/x86_64/KernelWrite.cpp stops the other CPUs and lifts write protection,
// and src/arm64e/KernelWrite.cpp stores into the dormant consumer's __DATA.
#pragma once
#include "RuntimeImage.h"
#include "WritePlan.h"

namespace graft {
bool initializeWrites(const RuntimeImage &kernel);
// The caller owns and validates every target mapping and its lifetime.
// All old values are checked before the first write, and a verification
// failure restores every old value. On x86-64 both happen while the other CPUs
// are stopped.
WriteError applyWrites(const Patch *patches, size_t count);
} // namespace graft
