// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The architecture's binding fixup: how an import slot's replacement value is
// built. x86_64/Fixup.h builds pointers and branch displacements; arm64e/Fixup.h
// builds authenticated, signed pointers. Binding.h reaches the right one here.
#pragma once
#if defined(__arm64__)
#include "arm64e/Fixup.h"
#elif defined(__x86_64__)
#include "x86_64/Fixup.h"
#else
#error Graft's binding fixup requires the x86-64 or arm64e implementation
#endif
