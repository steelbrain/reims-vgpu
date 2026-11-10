// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The catalogue binding's interface (KernelBinding.cpp). Start installs the
// two catalogue routes once; each catalogue send then calls bindConsumers to
// retry the dormant consumer's bind until it succeeds or is refused.
#pragma once
#include "RuntimeImage.h"

namespace graft {
// Resolve the OSKext lookup symbols and install the two catalogue-send routes.
bool initializeBinding(const RuntimeImage &kernel, const mach_header_64 *collection);
// Attempt the consumer bind once, under a single-entry guard; safe to call
// repeatedly. Does nothing after the bind has finished (bound or refused).
void bindConsumers();
} // namespace graft
