/*
 * RgfxLog: the bundle's log object (RgfxLog.hpp). Defined in C++, outside ARC, so the object is
 * a plain pointer that nothing releases.
 */

#include "RgfxLog.hpp"

#include <sys/stat.h>
#include <unistd.h>

os_log_t rgfxLog()
{
	// Never released: a thread may log while the process exits.
	static os_log_t log = os_log_create(RGFX_LOG_SUBSYSTEM, RGFX_LOG_CATEGORY);
	return log;
}

bool rgfxLogToStderr()
{
	struct stat st;
	if (fstat(STDERR_FILENO, &st) != 0)
		return false;
	const mode_t type = st.st_mode & S_IFMT;
	return type == S_IFREG || type == S_IFSOCK || type == S_IFIFO || isatty(STDERR_FILENO);
}
