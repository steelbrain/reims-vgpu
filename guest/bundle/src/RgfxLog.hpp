// RgfxLog: the os_log object the bundle's os_log lines go to. Its subsystem is the bundle's
// identifier, so `log config --subsystem ai.aneesiqbal.reims-graphics` sets the level those lines
// are kept at and persisted to, on every release: a line on OS_LOG_DEFAULT has no subsystem, and
// no mode persists its debug lines (docs/lessons.md). The bundle's NSLog lines are at the default
// level, which the log store keeps, and carry no subsystem. tests/bundle_log_check.sh keeps the
// subsystem equal to Info.plist's CFBundleIdentifier and the bundle off OS_LOG_DEFAULT.

#pragma once

#include <os/log.h>

// The subsystem of the bundle's lines: its CFBundleIdentifier.
#define RGFX_LOG_SUBSYSTEM "ai.aneesiqbal.reims-graphics"
// Their category: one for the whole bundle, Metal device and video driver alike.
#define RGFX_LOG_CATEGORY "bundle"

// The bundle's log object, made at the first call in a process and never released.
os_log_t rgfxLog();
