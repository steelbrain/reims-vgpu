/*
 * The C test harness of reims-vgpu-c: a C program written only against the committed
 * include/reims_vgpu.h, linked to the library's dynamic library, and run by tests/c_harness.rs.
 * It drives the API end to end on the library's platform, as a C host does: a device over a
 * memory map, its MMIO, a display, a frame encoded into a frame target and copied into memory,
 * suspend and resume, an IOSurface device, a video device, a scaler device where the host has
 * a GPU for it (its refusal elsewhere), the refusals, the files behind the map's ranges,
 * and each table's one destroy.
 *
 * Every rvg_ function the header declares is called at least once (tests/c_harness.rs checks
 * that). The device checks need a GPU: a Metal device, or with the vulkan feature a Vulkan one.
 *
 * Usage: harness             run every check; exit 0 on success
 *        harness die-null    a NULL device given to a function without a status (the process
 *                            aborts)
 */
/* glibc declares POSIX's nanosleep, mkstemp and ftruncate under -std=c11 only when asked;
 * macOS declares them regardless. */
#define _DEFAULT_SOURCE

#include "reims_vgpu.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static int failures;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "harness.c:%d: check failed: %s\n", __LINE__, #cond); \
            failures++;                                                          \
        }                                                                        \
    } while (0)

#define CHECK_STATUS(expr, want)                                                     \
    do {                                                                             \
        RvgStatus got_ = (expr);                                                     \
        if (got_ != (want)) {                                                        \
            fprintf(stderr, "harness.c:%d: %s: %u (%s), want %u\n", __LINE__, #expr, \
                    got_, rvg_status_message(got_), (unsigned)(want));               \
            failures++;                                                              \
        }                                                                            \
    } while (0)

/* Each table's destroys, and the thread of the last. */
static _Atomic int host_destroyed;
static _Atomic int display_destroyed;
static _Atomic int iosurface_destroyed;
static _Atomic int log_lines;
static _Atomic int display_calls_on_main;
static pthread_t main_thread;

static void on_host_destroy(void *ctx) {
    CHECK(ctx == &host_destroyed);
    host_destroyed++;
}

static void on_display_destroy(void *ctx) {
    CHECK(ctx == &display_destroyed);
    display_destroyed++;
}

static void on_iosurface_destroy(void *ctx) {
    CHECK(ctx == &iosurface_destroyed);
    iosurface_destroyed++;
}

static void on_raise(void *ctx, uint32_t vector) {
    (void)ctx;
    (void)vector;
}

static void *on_create_task(void *ctx, uint64_t size, void **base) {
    (void)ctx;
    (void)size;
    (void)base;
    return NULL;
}

static void *on_add_trace(void *ctx, RvgPhysicalMemoryRange range, RvgTraceRangeHandler *h) {
    (void)ctx;
    (void)range;
    (void)h;
    return NULL;
}

static void on_display_call(void) {
    if (pthread_equal(pthread_self(), main_thread)) {
        display_calls_on_main++;
    }
}

static void on_mode_change(void *ctx, uint16_t w, uint16_t h, uint32_t format) {
    (void)ctx;
    (void)w;
    (void)h;
    (void)format;
    on_display_call();
}

static void on_new_frame(void *ctx) {
    (void)ctx;
    on_display_call();
}

static void on_glyph(void *ctx, const RvgCursorGlyph *glyph) {
    (void)ctx;
    CHECK(glyph->struct_size >= sizeof(RvgCursorGlyph));
    on_display_call();
}

static void on_show(void *ctx, uint8_t show) {
    (void)ctx;
    (void)show;
    on_display_call();
}

static void on_log(void *ctx, RvgLogLevel level, const char *target, size_t target_length,
                   const char *message, size_t message_length) {
    (void)ctx;
    (void)level;
    (void)target;
    (void)target_length;
    (void)message;
    (void)message_length;
    log_lines++;
}

/* Waits up to 10 s for `*n` to reach `want`: a destroy that runs on the display thread. */
static int wait_for(_Atomic int *n, int want) {
    for (int i = 0; i < 10000 && *n < want; i++) {
        struct timespec ms = {0, 1000000};
        nanosleep(&ms, NULL);
    }
    return *n == want;
}

/* Guest RAM: 16 MiB at GPA 0x40000000. */
#define RAM_BASE 0x40000000ull
#define RAM_LENGTH (16ull << 20)
static void *ram;
static RvgMemoryRange ram_range;
static RvgMemoryMap ram_map;

static RvgHostCallbacks host_callbacks(void) {
    RvgHostCallbacks cb;
    memset(&cb, 0, sizeof cb);
    cb.struct_size = sizeof cb;
    cb.ctx = &host_destroyed;
    cb.destroy = on_host_destroy;
    cb.raise_interrupt = on_raise;
    return cb;
}

static RvgDeviceConfig device_config(const RvgHostCallbacks *cb) {
    RvgDeviceConfig c;
    memset(&c, 0, sizeof c);
    c.struct_size = sizeof c;
    c.display_port_count = 2;
    c.memory_map = &ram_map;
    c.callbacks = cb;
    return c;
}

static void check_status_and_log(void) {
    CHECK(strcmp(rvg_status_message(RVG_OK), "success") == 0);
    CHECK(strcmp(rvg_status_message(RVG_ERR_FRAME_STATE), "unknown status") != 0);
    CHECK(strcmp(rvg_status_message(RVG_ERR_FRAME_STATE + 1), "unknown status") == 0);
    rvg_log_set_handler(on_log, NULL, RVG_LOG_TRACE);
    rvg_log_set_handler(NULL, NULL, RVG_LOG_OFF);
    CHECK(RVG_MAX_DISPLAY_PORTS == 8 && RVG_DEVICE_MMIO_SIZE == 0x4000);
    CHECK(RVG_PCI_VENDOR_ID == 0x106b && RVG_PCI_DEVICE_ID == 0xeeee);
    CHECK(RVG_PCI_BAR_MMIO == 0 && RVG_PCI_MSI_VECTORS == 1);
    CHECK(RVG_RESUME_ERROR_NONE == UINT32_MAX);
}

/* The device's refusals, in the header's order, each with its table's one destroy before the
 * create returns, and none for a config the create cannot read. */
static void check_device_refusals(void) {
    RvgDevice *dev = (RvgDevice *)0x1;
    CHECK_STATUS(rvg_device_create(NULL, &dev), RVG_ERR_INVALID_ARGUMENT);
    CHECK(dev == (RvgDevice *)0x1);

    RvgHostCallbacks cb = host_callbacks();
    RvgDeviceConfig c = device_config(&cb);
    host_destroyed = 0;

    c.struct_size = 4;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_INVALID_ARGUMENT);
    c.struct_size = sizeof c;
    cb.struct_size = 4;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_INVALID_ARGUMENT);
    cb.struct_size = sizeof cb;
    CHECK(host_destroyed == 0);

    struct {
        RvgDeviceConfig c;
        uint64_t later;
    } big;
    memset(&big, 0, sizeof big);
    big.c = c;
    big.c.struct_size = sizeof big;
    big.later = 1;
    CHECK_STATUS(rvg_device_create(&big.c, &dev), RVG_ERR_UNSUPPORTED);
    CHECK(host_destroyed == 0);

    CHECK_STATUS(rvg_device_create(&c, NULL), RVG_ERR_INVALID_ARGUMENT);
    CHECK(host_destroyed == 1);

    c.display_port_count = 0;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_DISPLAY_PORT_COUNT);
    c.display_port_count = RVG_MAX_DISPLAY_PORTS + 1;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_DISPLAY_PORT_COUNT);
    c.display_port_count = 1;
    CHECK(host_destroyed == 3);

    c.metal_registry_id = UINT64_MAX;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_NO_METAL_DEVICE);
    c.metal_registry_id = 0;

    cb.create_task = on_create_task;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_MEMORY_CONFLICT);
    cb.create_task = NULL;

    RvgMemoryRange overlap[2] = {ram_range, ram_range};
    RvgMemoryMap bad = {overlap, 2};
    c.memory_map = &bad;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_MEMORY_MAP);
    RvgMemoryMap null_ranges = {NULL, 1};
    c.memory_map = &null_ranges;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_INVALID_ARGUMENT);
    c.memory_map = &ram_map;

    cb.add_trace_range = on_add_trace;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_INVALID_ARGUMENT);
    cb.add_trace_range = NULL;

    c.memory_map = NULL;
    c.process_isolation = true;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_ISOLATION_WITHOUT_MAP);
    c.process_isolation = false;
    c.memory_map = &ram_map;
    CHECK(host_destroyed == 9);
    CHECK(dev == (RvgDevice *)0x1);

    /* A longer config whose later bytes are zero is accepted; a table with no functions ends
     * at the device's release. */
    big.later = 0;
    RvgHostCallbacks only_destroy;
    memset(&only_destroy, 0, sizeof only_destroy);
    only_destroy.struct_size = sizeof only_destroy;
    only_destroy.ctx = &host_destroyed;
    only_destroy.destroy = on_host_destroy;
    big.c.callbacks = &only_destroy;
    CHECK_STATUS(rvg_device_create(&big.c, &dev), RVG_OK);
    CHECK(host_destroyed == 9);
    rvg_device_release(dev);
    CHECK(host_destroyed == 10);
}

/* The files behind the map's ranges: a device over guest RAM that is a shared mapping of a file,
 * named in memory_files, and the refusals of the files, each with its table's one destroy. A
 * host built before memory_files passes 48 bytes and is read without them. */
static void check_memory_files(void) {
    char path[] = "/tmp/rvg-harness-ram.XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    unlink(path);
    CHECK(ftruncate(fd, (off_t)RAM_LENGTH) == 0);
    void *shared = mmap(NULL, RAM_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    CHECK(shared != MAP_FAILED);
    RvgMemoryRange range = {RAM_BASE, RAM_LENGTH, shared};
    RvgMemoryMap map = {&range, 1};
    RvgMemoryRangeFile file = {fd, 0, 0};

    RvgHostCallbacks cb = host_callbacks();
    RvgDeviceConfig c = device_config(&cb);
    c.memory_map = &map;
    c.memory_files = &file;
    host_destroyed = 0;
    RvgDevice *dev = NULL;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_OK);
    /* The library holds its own duplicate: the host may close its descriptor. */
    rvg_device_mmio_write(dev, 0x1034, 0xf5);
    CHECK(rvg_device_mmio_read(dev, 0x1034) == 0xf5);
    rvg_device_release(dev);
    CHECK(host_destroyed == 1);

    dev = (RvgDevice *)0x1;
    c.memory_map = NULL;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_INVALID_ARGUMENT);
    c.memory_map = &map;
    file.fd = -2;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_INVALID_ARGUMENT);
    file.fd = fd;
    file.flags = 1;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_INVALID_ARGUMENT);
    file.flags = 0;
    file.offset = 0x10;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_INVALID_ARGUMENT);
    file.offset = 0;
    range.physical_length = RAM_LENGTH * 2;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_ERR_INVALID_ARGUMENT);
    range.physical_length = RAM_LENGTH;
    CHECK(host_destroyed == 6);
    CHECK(dev == (RvgDevice *)0x1);

    /* The first version: the bad entry past 48 bytes is not read. */
    file.fd = -2;
    c.struct_size = 48;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_OK);
    rvg_device_release(dev);
    CHECK(host_destroyed == 7);

    /* A range with no file. */
    file.fd = -1;
    c.struct_size = sizeof c;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_OK);
    rvg_device_release(dev);
    CHECK(host_destroyed == 8);

    munmap(shared, RAM_LENGTH);
    close(fd);
}

static RvgDisplayCallbacks display_callbacks(void) {
    RvgDisplayCallbacks cb;
    memset(&cb, 0, sizeof cb);
    cb.struct_size = sizeof cb;
    cb.ctx = &display_destroyed;
    cb.destroy = on_display_destroy;
    cb.mode_change = on_mode_change;
    cb.new_frame = on_new_frame;
    cb.cursor_glyph = on_glyph;
    cb.cursor_show = on_show;
    cb.cursor_move = on_new_frame;
    return cb;
}

/* With the display 0x5eed1 on port 0. */
static void check_display_refusals(RvgDevice *dev) {
    RvgDisplayCallbacks cb = display_callbacks();
    RvgDisplayConfig c;
    memset(&c, 0, sizeof c);
    c.struct_size = sizeof c;
    c.serial_number = 0x5eed2;
    c.port = 1;
    c.callbacks = &cb;
    RvgDisplay *d = (RvgDisplay *)0x1;
    CHECK_STATUS(rvg_display_create(dev, NULL, &d), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_display_create(NULL, &c, &d), RVG_ERR_INVALID_ARGUMENT);
    CHECK(display_destroyed == 1);
    c.port = 2;
    CHECK_STATUS(rvg_display_create(dev, &c, &d), RVG_ERR_PORT_OUT_OF_RANGE);
    CHECK(display_destroyed == 2);
    c.port = 1;
    c.serial_number = 0x5eed1;
    CHECK_STATUS(rvg_display_create(dev, &c, &d), RVG_ERR_SERIAL_IN_USE);
    c.serial_number = 0x5eed2;
    c.port = 0;
    CHECK_STATUS(rvg_display_create(dev, &c, &d), RVG_ERR_PORT_IN_USE);
    CHECK(display_destroyed == 4);
    CHECK(d == (RvgDisplay *)0x1);
    display_destroyed = 0;
}

/* A frame of the display, encoded into a target, committed, waited on and copied. */
static void check_frames(RvgDevice *dev, RvgDisplay *disp) {
    RvgFrameTargetConfig tc = {sizeof tc, 64, 32, RVG_PIXEL_FORMAT_BGRA8_UNORM};
    RvgFrameTarget *t = NULL;
    CHECK_STATUS(rvg_frame_target_create(dev, NULL, &t), RVG_ERR_INVALID_ARGUMENT);
    tc.width = 0;
    CHECK_STATUS(rvg_frame_target_create(dev, &tc, &t), RVG_ERR_INVALID_ARGUMENT);
    tc.width = 16385;
    CHECK_STATUS(rvg_frame_target_create(dev, &tc, &t), RVG_ERR_INVALID_ARGUMENT);
    tc.width = 64;
    tc.format = 73;
    CHECK_STATUS(rvg_frame_target_create(dev, &tc, &t), RVG_ERR_INVALID_ARGUMENT);
    tc.format = RVG_PIXEL_FORMAT_BGRA8_UNORM;
    CHECK_STATUS(rvg_frame_target_create(dev, &tc, &t), RVG_OK);
    CHECK(t != NULL);
    if (!t) {
        return;
    }
    CHECK(rvg_frame_target_retain(t) == t);
    rvg_frame_target_release(t);

    RvgFrame *f = NULL;
    CHECK_STATUS(rvg_display_encode_frame(disp, NULL, NULL, &f), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_display_encode_frame(disp, t, NULL, NULL), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_display_encode_frame(disp, t, NULL, &f), RVG_OK);
    CHECK_STATUS(rvg_frame_wait(f), RVG_ERR_FRAME_STATE);
    CHECK_STATUS(rvg_frame_commit(f), RVG_OK);
    CHECK_STATUS(rvg_frame_commit(f), RVG_ERR_FRAME_STATE);
    CHECK_STATUS(rvg_frame_wait(f), RVG_OK);
    CHECK_STATUS(rvg_frame_wait(f), RVG_OK);
    rvg_frame_release(f);

    size_t bpr = 64 * 4 + 16;
    size_t extent = bpr * 31 + 64 * 4;
    unsigned char *px = malloc(extent);
    memset(px, 0xa5, extent);
    CHECK_STATUS(rvg_frame_target_read_pixels(t, px, 64 * 4 - 1, extent),
                 RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_frame_target_read_pixels(t, px, bpr, extent - 1), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_frame_target_read_pixels(t, NULL, bpr, extent), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_frame_target_read_pixels(NULL, px, bpr, extent), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_frame_target_read_pixels(t, px, bpr, extent), RVG_OK);
    /* No guest frame yet: the whole target is cleared to one colour. Each row's padding is
     * untouched. */
    int uniform = 1;
    for (size_t y = 0; y < 32; y++) {
        for (size_t x = 0; x < 64 * 4; x++) {
            uniform &= px[y * bpr + x] == px[x % 4];
        }
        if (y < 31) {
            uniform &= px[y * bpr + 64 * 4] == 0xa5;
        }
    }
    CHECK(uniform);
    CHECK(!(px[0] == 0xa5 && px[1] == 0xa5 && px[2] == 0xa5 && px[3] == 0xa5));

    /* An update of pixels just read changes no row; one of a row written over restores that row
     * alone and flags it. The padding stays as it was. */
    unsigned char changed[32];
    memset(changed, 0x77, sizeof changed);
    CHECK_STATUS(rvg_frame_target_update_pixels(t, px, bpr, extent, changed, 31),
                 RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_frame_target_update_pixels(t, px, bpr, extent, NULL, 32),
                 RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_frame_target_update_pixels(t, NULL, bpr, extent, changed, 32),
                 RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_frame_target_update_pixels(NULL, px, bpr, extent, changed, 32),
                 RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_frame_target_update_pixels(t, px, bpr, extent - 1, changed, 32),
                 RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_frame_target_update_pixels(t, px, bpr, extent, changed, 32), RVG_OK);
    int none_changed = 1;
    for (size_t y = 0; y < 32; y++) {
        none_changed &= changed[y] == 0;
    }
    CHECK(none_changed);
    unsigned char first = px[5 * bpr];
    px[5 * bpr] = (unsigned char)(first ^ 0xff);
    CHECK_STATUS(rvg_frame_target_update_pixels(t, px, bpr, extent, changed, 32), RVG_OK);
    int only_row_5 = 1;
    for (size_t y = 0; y < 32; y++) {
        only_row_5 &= changed[y] == (y == 5);
    }
    CHECK(only_row_5);
    CHECK(px[5 * bpr] == first);
    CHECK(px[5 * bpr + 64 * 4] == 0xa5);

    /* An uncommitted frame over a region, released: never drawn. */
    RvgRegion r = {8, 8, 16, 16};
    CHECK_STATUS(rvg_display_encode_frame(disp, t, &r, &f), RVG_OK);
    rvg_frame_release(f);
    rvg_frame_release(NULL);
    rvg_frame_target_release(t);
    rvg_frame_target_release(NULL);
    free(px);
}

static void check_suspend_and_resume(RvgDevice *dev) {
    RvgResumeError e = 99;
    CHECK_STATUS(rvg_device_finish_suspend(NULL, NULL), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_device_will_resume(NULL, NULL, 0, &e), RVG_ERR_INVALID_ARGUMENT);
    CHECK(e == 99);
    rvg_device_will_suspend(dev);
    RvgBytes state = {NULL, 0};
    CHECK_STATUS(rvg_device_finish_suspend(dev, NULL), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_device_finish_suspend(dev, &state), RVG_OK);
    CHECK(state.data != NULL && state.length > 0);
    CHECK_STATUS(rvg_device_will_resume(dev, state.data, state.length, &e), RVG_OK);
    CHECK(e == RVG_RESUME_ERROR_NONE);
    rvg_bytes_free(state);
    RvgBytes none = {NULL, 0};
    rvg_bytes_free(none);
    rvg_device_did_resume(dev);
}

static void check_device(void) {
    RvgHostCallbacks cb = host_callbacks();
    RvgDeviceConfig c = device_config(&cb);
    RvgDevice *dev = NULL;
    host_destroyed = 0;
    CHECK_STATUS(rvg_device_create(&c, &dev), RVG_OK);
    CHECK(dev != NULL);
    if (!dev) {
        return;
    }
    /* MMIO: the port count register (1 until the guest's version offers multiple displays), a
     * staged FIFO length, and an offset below the window. */
    CHECK(rvg_device_mmio_read(dev, 0x122c) == 1);
    rvg_device_mmio_write(dev, 0x1004, 0x4000);
    (void)rvg_device_mmio_read(dev, 0);

    RvgDisplayCallbacks dcb = display_callbacks();
    RvgDisplayConfig dc;
    memset(&dc, 0, sizeof dc);
    dc.struct_size = sizeof dc;
    dc.serial_number = 0x5eed1;
    dc.port = 0;
    dc.name = "Harness";
    dc.name_length = 7;
    dc.width_millimeters = 400.0;
    dc.height_millimeters = 300.0;
    dc.callbacks = &dcb;
    RvgDisplay *disp = NULL;
    display_destroyed = 0;
    CHECK_STATUS(rvg_display_create(dev, &dc, &disp), RVG_OK);
    CHECK(disp != NULL);
    check_display_refusals(dev);
    if (disp) {
        RvgDisplayCoord pos = rvg_display_cursor_position(disp);
        CHECK(pos.x == 0xffff && pos.y == 0xffff);
        uint64_t guest = 1, host = 1;
        rvg_display_present_counts(disp, &guest, &host);
        CHECK(guest == 0 && host == 0);
        rvg_display_present_counts(disp, NULL, NULL);
        RvgDisplayMode modes[2] = {
            {1280, 720, 0, 60.0},
            {1920, 1080, RVG_DISPLAY_MODE_EDR | RVG_DISPLAY_MODE_10BPC, 60.0},
        };
        rvg_display_set_modes(disp, modes, 2);
        rvg_display_set_modes(disp, NULL, 0);
        check_frames(dev, disp);
        rvg_display_release(disp);
        /* The last closure may go on the display thread (the new-frame source's cancel). */
        CHECK(wait_for(&display_destroyed, 1));
    }
    rvg_display_release(NULL);

    check_suspend_and_resume(dev);
    rvg_device_pause(dev);
    rvg_device_unpause(dev);
    rvg_device_reset(dev);
    rvg_device_stop(dev);
    CHECK(host_destroyed == 0);
    rvg_device_release(dev);
    CHECK(host_destroyed == 1);
    rvg_device_release(NULL);
    CHECK(display_calls_on_main == 0);
}

static void check_iosurface(void) {
    RvgIOSurfaceCallbacks cb;
    memset(&cb, 0, sizeof cb);
    cb.struct_size = sizeof cb;
    cb.ctx = &iosurface_destroyed;
    cb.destroy = on_iosurface_destroy;
    cb.raise_interrupt = on_raise;
    RvgIOSurfaceDeviceConfig c = {sizeof c, NULL, &cb};
    RvgIOSurfaceDevice *d = NULL;
    iosurface_destroyed = 0;
    CHECK_STATUS(rvg_iosurface_device_create(NULL, &d), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_iosurface_device_create(&c, &d), RVG_ERR_NO_MEMORY_MAP);
    CHECK(iosurface_destroyed == 1);
    c.memory_map = &ram_map;
    CHECK_STATUS(rvg_iosurface_device_create(&c, &d), RVG_OK);
    CHECK(d != NULL);
    if (!d) {
        return;
    }
    (void)rvg_iosurface_device_mmio_read(d, 0x1000);
    CHECK(rvg_iosurface_device_mmio_read(d, 0x0) == UINT64_MAX);
    rvg_iosurface_device_mmio_write(d, 0x0, 0);
    rvg_iosurface_device_release(d);
    CHECK(iosurface_destroyed == 2);
    rvg_iosurface_device_release(NULL);
}

static _Atomic int video_destroyed;

static void on_video_destroy(void *ctx) {
    atomic_fetch_add((_Atomic int *)ctx, 1);
}

static int video_replies;

/* Counts the host session's replies to 'decs': a vtpv header of type 'decs', flags 4. */
static void on_send_to_guest(void *ctx, const uint8_t *message, size_t length) {
    (void)ctx;
    static const uint8_t reply[] = {'v', 't', 'p', 'v', 'd', 'e', 'c', 's', 0, 0, 0, 4};
    if (length >= 0x30 && memcmp(message + 4, reply, sizeof reply) == 0) {
        video_replies++;
    }
}

/* A video device over no IOSurface device: no session until its start, none after a reset; its
 * session answers the guest's 'decs' with one reply. */
static void check_video(void) {
    RvgVideoCallbacks cb;
    memset(&cb, 0, sizeof cb);
    cb.struct_size = sizeof cb;
    cb.ctx = &video_destroyed;
    cb.destroy = on_video_destroy;
    cb.send_to_guest = on_send_to_guest;
    RvgVideoDeviceConfig c = {sizeof c, NULL, &cb};
    RvgVideoDevice *d = NULL;
    /* The guest's 'decs': a vtpv header of type 'decs', flags 1 (a reply is expected), then
     * ',ver' as '=u32' 6. */
    uint8_t message[0x44] = {0, 0, 0, 0x30, 'v', 't', 'p', 'v', 'd', 'e', 'c', 's', 0, 0, 0, 1};
    static const uint8_t version[] = {0,   0,   0,   20,  'p', 'a', 'r', 'a', ',', 'v',
                                      'e', 'r', '=', 'u', '3', '2', 0,   0,   0,   6};
    memcpy(message + 0x30, version, sizeof version);
    video_destroyed = 0;
    video_replies = 0;
    CHECK_STATUS(rvg_video_device_create(NULL, &d), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_video_device_create(&c, &d), RVG_OK);
    CHECK(d != NULL);
    if (!d) {
        return;
    }
    CHECK(rvg_video_device_deliver(d, message, sizeof message) == RVG_VIDEO_INVALID_SESSION);
    rvg_video_device_start(d);
    CHECK(rvg_video_device_deliver(d, message, sizeof message) == 0);
    CHECK(video_replies == 1);
    /* A message past the bound is answered unread, as a host answers it before it copies. */
    uint8_t *large = calloc(1, (size_t)RVG_VIDEO_MAX_MESSAGE_SIZE + 1);
    CHECK(large != NULL);
    if (large) {
        memcpy(large, message, sizeof message);
        CHECK(rvg_video_device_deliver(d, large, (size_t)RVG_VIDEO_MAX_MESSAGE_SIZE + 1) ==
              RVG_VIDEO_ALLOCATION_FAILED);
        CHECK(video_replies == 1);
        free(large);
    }
    rvg_video_device_invalidate(d);
    rvg_video_device_reset(d);
    CHECK(rvg_video_device_deliver(d, message, sizeof message) == RVG_VIDEO_INVALID_SESSION);
    rvg_video_device_release(d);
    CHECK(video_destroyed == 1);
    rvg_video_device_release(NULL);
}

static _Atomic int scaler_destroyed;

/* The scaler's guest RAM: 0x1000 bytes at guest-physical SCALER_RAM. */
#define SCALER_RAM 0x80000000u
static uint8_t scaler_ram[0x1000];

static bool scaler_in_ram(uint64_t address, uint64_t length) {
    if (length == 0) {
        length = 1;
    }
    return address >= SCALER_RAM && length <= sizeof scaler_ram &&
           address - SCALER_RAM <= sizeof scaler_ram - length;
}

static bool on_scaler_is_ram(void *ctx, uint64_t address, uint64_t length) {
    (void)ctx;
    return scaler_in_ram(address, length);
}

static bool on_scaler_read(void *ctx, uint64_t address, uint8_t *bytes, size_t length) {
    (void)ctx;
    if (!scaler_in_ram(address, length)) {
        return false;
    }
    memcpy(bytes, scaler_ram + (address - SCALER_RAM), length);
    return true;
}

static bool on_scaler_write(void *ctx, uint64_t address, const uint8_t *bytes, size_t length) {
    (void)ctx;
    if (!scaler_in_ram(address, length)) {
        return false;
    }
    memcpy(scaler_ram + (address - SCALER_RAM), bytes, length);
    return true;
}

static void on_scaler_destroy(void *ctx) {
    atomic_fetch_add((_Atomic int *)ctx, 1);
}

/* A scaler device, made on every host: a request before its start is a needs-reset. Where the
 * host has a GPU for it, once started, the guest's capabilities request is answered in place,
 * the capabilities written into its RAM, and after a reset it needs-resets again; where it has
 * none, the device does not scale, its start fails and every request needs-resets. Its release
 * ends its table. */
static void check_scaler(void) {
    RvgScalerGuestMemory m;
    memset(&m, 0, sizeof m);
    m.struct_size = sizeof m;
    m.ctx = &scaler_destroyed;
    m.destroy = on_scaler_destroy;
    m.is_ram = on_scaler_is_ram;
    m.read = on_scaler_read;
    m.write = on_scaler_write;
    RvgScalerDeviceConfig c = {sizeof c, NULL, &m};
    RvgScalerDevice *d = NULL;
    /* The guest kext's type 1: its 0x288 bytes of capabilities at SCALER_RAM + 0x100. */
    uint8_t request[0x20] = {1};
    uint64_t address = SCALER_RAM + 0x100, length = 0x288;
    char reason[64] = "";
    memcpy(request + 8, &address, sizeof address);
    memcpy(request + 0x18, &length, sizeof length);
    scaler_destroyed = 0;
    CHECK_STATUS(rvg_scaler_device_create(NULL, &d), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_scaler_device_create(&c, &d), RVG_OK);
    CHECK(d != NULL);
    if (!d) {
        return;
    }
    CHECK(rvg_scaler_device_process(d, request, sizeof request, sizeof request, reason,
                                    sizeof reason) == RVG_SCALER_NEEDS_RESET);
    CHECK(reason[0] != 0);
    if (rvg_scaler_device_scales(d)) {
        CHECK(rvg_scaler_device_start(d));
        CHECK(rvg_scaler_device_process(d, request, sizeof request, sizeof request, NULL, 0) ==
              RVG_SCALER_REPLY);
        memcpy(&length, request + 0x18, sizeof length);
        CHECK(request[4] == 0 && length == 0x288);
        rvg_scaler_device_reset(d);
    } else {
        CHECK(!rvg_scaler_device_start(d));
    }
    CHECK(rvg_scaler_device_process(d, request, sizeof request, sizeof request, NULL, 0) ==
          RVG_SCALER_NEEDS_RESET);
    rvg_scaler_device_release(d);
    CHECK(scaler_destroyed == 1);
    rvg_scaler_device_release(NULL);
}

static void check_trace_handlers(void) {
    RvgPhysicalMemoryRange r = {0, 0};
    rvg_trace_range_handler_invoke(NULL, r);
    CHECK(rvg_trace_range_handler_retain(NULL) == NULL);
    rvg_trace_range_handler_release(NULL);
}

int main(int argc, char **argv) {
    main_thread = pthread_self();
    if (argc > 1 && strcmp(argv[1], "die-null") == 0) {
        rvg_device_mmio_write(NULL, 0, 0);
        puts("survived");
        return 0;
    }
    ram = mmap(NULL, RAM_LENGTH, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (ram == MAP_FAILED) {
        perror("mmap");
        return 2;
    }
    ram_range.physical_address = RAM_BASE;
    ram_range.physical_length = RAM_LENGTH;
    ram_range.host_address = ram;
    ram_map.ranges = &ram_range;
    ram_map.range_count = 1;

    check_status_and_log();
    check_device_refusals();
    check_memory_files();
    check_device();
    check_iosurface();
    check_video();
    check_scaler();
    check_trace_handlers();
    if (failures) {
        fprintf(stderr, "harness: %d failures\n", failures);
        return 1;
    }
    puts("harness: ok");
    return 0;
}
