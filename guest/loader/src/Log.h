// Ours: the loader's log, which Loader.cpp defines: IOLog with the uptime,
// and the whole log in the plugin service's LoaderLog property.
#pragma once

void note(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
