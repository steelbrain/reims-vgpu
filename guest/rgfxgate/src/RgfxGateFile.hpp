// Narrow file helpers used only by the bundle installer and preference writer.
#pragma once
#include <IOKit/IOLib.h>
#include <sys/errno.h>
#include <sys/fcntl.h>
#include <sys/uio.h>
#include <sys/vnode.h>

namespace rgfxgate {

static inline uint8_t *readFile(const char *path, size_t &size, size_t limit) {
    size = 0;
    vfs_context_t ctx = vfs_context_create(nullptr);
    if (ctx == nullptr)
        return nullptr;
    vnode_t vp = nullptr;
    uint8_t *bytes = nullptr;
    if (vnode_open(path, FREAD | O_NOFOLLOW, 0, 0, &vp, ctx) == 0) {
        struct vnode_attr attr;
        VATTR_INIT(&attr);
        VATTR_WANTED(&attr, va_data_size);
        if (vnode_isreg(vp) && vnode_getattr(vp, &attr, ctx) == 0 &&
            VATTR_IS_SUPPORTED(&attr, va_data_size) && attr.va_data_size > 0 &&
            attr.va_data_size <= limit && attr.va_data_size <= INT_MAX) {
            size = static_cast<size_t>(attr.va_data_size);
            bytes = static_cast<uint8_t *>(IOMalloc(size));
            int residual = 0;
            if (bytes != nullptr &&
                (vn_rdwr(UIO_READ, vp, reinterpret_cast<caddr_t>(bytes),
                         static_cast<int>(size), 0, UIO_SYSSPACE, 0,
                         vfs_context_ucred(ctx), &residual, vfs_context_proc(ctx)) != 0 || residual != 0)) {
                IOFree(bytes, size);
                bytes = nullptr;
            }
        }
        vnode_close(vp, FREAD, ctx);
    }
    vfs_context_rele(ctx);
    return bytes;
}

static inline int writeFile(const char *path, const void *bytes, size_t size,
                            int flags, int mode) {
    if (size > INT_MAX)
        return EFBIG;
    vfs_context_t ctx = vfs_context_create(nullptr);
    if (ctx == nullptr)
        return ENOMEM;
    vnode_t vp = nullptr;
    const int fmode = FFLAGS(flags);
    int error = vnode_open(path, fmode, mode, 0, &vp, ctx);
    if (error == 0) {
        int residual = 0;
        error = vn_rdwr(UIO_WRITE, vp,
                        reinterpret_cast<caddr_t>(const_cast<void *>(bytes)),
                        static_cast<int>(size), 0, UIO_SYSSPACE, IO_SYNC,
                        vfs_context_ucred(ctx), &residual, vfs_context_proc(ctx));
        if (error == 0 && residual != 0)
            error = EIO;
        const int closeError = vnode_close(vp, fmode, ctx);
        if (error == 0)
            error = closeError;
    }
    vfs_context_rele(ctx);
    return error;
}

} // namespace rgfxgate
