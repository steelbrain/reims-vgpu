// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// A tiny, dependency-free check for the host tests. Like assert, but always
// on and it names the file, line and failed expression on stderr before it
// aborts, so a failure says what broke, not only where. It evaluates its
// expression exactly once. Used in place of bare assert across the host tests.
#pragma once
#include <stdio.h>
#include <stdlib.h>

namespace graft_test {
inline void checkFailed(const char *expression, const char *file, int line) {
    fprintf(stderr, "%s:%d: CHECK failed: %s\n", file, line, expression);
    fflush(stderr);
    abort();
}
} // namespace graft_test

#define CHECK(expression)                                                                          \
    ((expression) ? (void)0 : graft_test::checkFailed(#expression, __FILE__, __LINE__))
