// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// A tiny dylib for tests/macho_test.cpp to read: it is a real linked Mach-O,
// built from our own source, so the test exercises MachO.h's whole parse path
// (header, load commands, __LINKEDIT, LC_SYMTAB, the string table) without any
// Apple binary. The test only reads this image's symbol table; it never calls
// these functions.
int graft_fixture_a(int x) {
    return x + 1;
}
int graft_fixture_b(int x) {
    return x * 2;
}
int graft_fixture_c(int x) {
    return x - 3;
}
