// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Graft's diagnostics interface (Log.cpp): the tagged log and the registry
// state properties (GraftState, GraftReason, GraftLog, GraftClient.<id>).
#pragma once

namespace graft {
bool initializeLog();
void note(const char *format, ...) __attribute__((format(printf, 1, 2)));
void setState(const char *state, const char *reason);
void clientState(const char *identifier, const char *state, const char *reason);
void publishLog();
} // namespace graft
