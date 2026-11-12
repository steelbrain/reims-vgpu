/*
 * RgfxLog: the bundle's log object (RgfxLog.hpp). Defined in C++, outside ARC, so the object is
 * a plain pointer that nothing releases.
 */

#include "RgfxLog.hpp"

os_log_t rgfxLog()
{
	// Never released: a thread may log while the process exits.
	static os_log_t log = os_log_create(RGFX_LOG_SUBSYSTEM, RGFX_LOG_CATEGORY);
	return log;
}
