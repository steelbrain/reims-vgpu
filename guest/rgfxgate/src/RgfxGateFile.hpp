// Narrow file helpers used only by the bundle installer and preference writer.
#pragma once
#include <IOKit/IOLib.h>
#include <sys/errno.h>
#include <sys/fcntl.h>
#include <sys/uio.h>
#include <sys/vnode.h>
#if defined(__arm64e__)
#include "RgfxGateVnop.hpp"
#endif

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

#if defined(__arm64e__)
// Write `bytes` to `path`, creating it with `mode` or truncating it. The x86
// writeFile below opens with vnode_open(O_CREAT|O_TRUNC), which the
// create/open/truncate MAC routes admit for the install's writes; arm64 installs
// none of those routes, so this one stays below the MAC layer as the directory
// creates do (RgfxGateVnop.hpp): an existing file is truncated with
// vnode_setattr, a missing one made with VNOP_CREATE through the op vector, and
// the bytes written with vn_rdwr. `flags` is unused: the semantics are fixed
// (create or truncate, then write). The install calls this only after
// rgfxVnopInstallReady has confirmed the op vector.
static inline int writeFile(const char *path, const void *bytes, size_t size,
                            int flags, int mode) {
    (void)flags;
    if (size > INT_MAX)
        return EFBIG;
    vfs_context_t ctx = vfs_context_create(nullptr);
    if (ctx == nullptr)
        return ENOMEM;
    vnode_t vp = nullptr;
    int error = 0;
    if (vnode_lookup(path, VNODE_LOOKUP_NOFOLLOW, &vp, ctx) == 0) {
        struct vnode_attr va;
        VATTR_INIT(&va);
        VATTR_SET(&va, va_data_size, 0);
        error = vnode_setattr(vp, &va, ctx);
    } else {
        const char *slash = nullptr;
        for (const char *p = path; *p; p++)
            if (*p == '/')
                slash = p;
        char parent[1024];
        const size_t plen = slash == nullptr ? 0 : static_cast<size_t>(slash - path);
        vnode_t dvp = nullptr;
        if (slash == nullptr || slash == path || plen >= sizeof(parent)) {
            error = EINVAL;
        } else {
            memcpy(parent, path, plen);
            parent[plen] = '\0';
            error = vnode_lookup(parent, 0, &dvp, ctx);
        }
        if (error == 0) {
            const char *leaf = slash + 1;
            struct componentname cn;
            bzero(&cn, sizeof(cn));
            cn.cn_nameiop = CREATE;
            cn.cn_flags = 0;
            cn.cn_pnbuf = const_cast<char *>(leaf);
            cn.cn_pnlen = static_cast<int>(strlen(leaf));
            cn.cn_nameptr = const_cast<char *>(leaf);
            cn.cn_namelen = static_cast<int>(strlen(leaf));
            struct vnode_attr va;
            VATTR_INIT(&va);
            VATTR_SET(&va, va_type, VREG);
            VATTR_SET(&va, va_mode, mode);
            VATTR_SET(&va, va_uid, 0);
            VATTR_SET(&va, va_gid, 0);
            error = rgfxVnopCreate(dvp, &vp, &cn, &va, ctx);
            vnode_put(dvp);
        }
    }
    if (error == 0 && vp != nullptr) {
        int residual = 0;
        error = vn_rdwr(UIO_WRITE, vp,
                        reinterpret_cast<caddr_t>(const_cast<void *>(bytes)),
                        static_cast<int>(size), 0, UIO_SYSSPACE, IO_SYNC,
                        vfs_context_ucred(ctx), &residual, vfs_context_proc(ctx));
        if (error == 0 && residual != 0)
            error = EIO;
    }
    if (vp != nullptr)
        vnode_put(vp);
    vfs_context_rele(ctx);
    return error;
}
#else
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

#endif // __arm64e__

} // namespace rgfxgate
