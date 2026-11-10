// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The route installer's interface (x86_64/Route.cpp, arm64e/Route.cpp): install
// one or more entry routes to a wrapper that can call the original, in a single
// transaction.
#pragma once
#include "RuntimeImage.h"

namespace graft {
struct RouteRequest {
    const char *symbol;
    void *replacement;
    void **original;
};
bool initializeRoutes(const RuntimeImage &kernel);
// All requests install in one transaction. The caller retains replacement
// code and original-pointer storage for the lifetime of the installed routes.
// For an owned client (null for Graft's own routes), its registry state says
// "routed", or "refused" naming the route and the reason.
bool routeKernel(const char *client, const RouteRequest *requests, size_t count);
} // namespace graft
