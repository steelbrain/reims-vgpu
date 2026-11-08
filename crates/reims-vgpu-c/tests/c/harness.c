/*
 * The C test harness of reims-vgpu-c: a C program written only against the committed
 * include/reims_vgpu.h, linked to the library's dynamic library, and run by tests/c_harness.rs.
 * It drives the API end to end on the library's platform, as a C host does: a graphics device
 * with no window, its registers, region, doorbell, control ring, cursor and display read, its
 * refusals, the files behind its memory map's ranges, and each table's one destroy.
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

/* Waits up to 10 s for `*n` to reach `want`: a callback that runs on a library thread. */
static int wait_for(_Atomic int *n, int want) {
    for (int i = 0; i < 10000 && *n < want; i++) {
        struct timespec ms = {0, 1000000};
        nanosleep(&ms, NULL);
    }
    return *n == want;
}

static void check_status(void) {
    CHECK(strcmp(rvg_status_message(RVG_OK), "success") == 0);
    CHECK(strcmp(rvg_status_message(RVG_ERR_HOST_RESOURCES), "unknown status") != 0);
    CHECK(strcmp(rvg_status_message(RVG_ERR_HOST_RESOURCES + 1), "unknown status") == 0);
    CHECK(RVG_GRAPHICS_PCI_VENDOR_ID == 0x1234 && RVG_GRAPHICS_PCI_DEVICE_ID == 0x5247);
    CHECK(RVG_GRAPHICS_PCI_BAR_REGISTERS == 0 && RVG_GRAPHICS_PCI_BAR_REGION == 2);
    CHECK(RVG_GRAPHICS_MSIX_VECTORS == 1);
}

static _Atomic int graphics_destroyed;
static _Atomic int graphics_raised;

static void on_graphics_destroy(void *ctx) {
    atomic_fetch_add((_Atomic int *)ctx, 1);
}

static void on_graphics_raise(void *ctx, uint32_t vector) {
    (void)ctx;
    if (vector == 0) {
        graphics_raised++;
    }
}

static _Atomic int graphics_moved;

static void on_graphics_cursor_move(void *ctx) {
    (void)ctx;
    graphics_moved++;
}

/* A graphics device with no window, on every host: its registers, its region, its doorbell, a
 * GET_DISPLAYS the guest writes into the control ring answered with DISPLAYS and an interrupt,
 * a MOVE_CURSOR the host is told of and reads, a taking write, a read of its display, its reset,
 * and its release, which ends its table once its threads have ended. */
static void check_graphics(void) {
    RvgGraphicsCallbacks cb = {sizeof cb,
                               &graphics_destroyed,
                               on_graphics_destroy,
                               on_graphics_raise,
                               NULL,
                               NULL,
                               on_graphics_cursor_move};
    RvgGraphicsDeviceConfig c = {sizeof c, &cb, NULL, NULL, NULL};
    RvgGraphicsDevice *d = NULL;
    graphics_destroyed = 0;
    graphics_raised = 0;
    graphics_moved = 0;
    CHECK_STATUS(rvg_graphics_device_create(NULL, &d), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_OK);
    CHECK(d != NULL);
    if (!d) {
        return;
    }
    uint64_t size = 0;
    uint8_t *region = rvg_graphics_device_region(d, &size);
    CHECK(region != NULL && ((uintptr_t)region & 0xfff) == 0);
    CHECK(size >= RVG_GRAPHICS_ALLOCATIONS_OFFSET && (size & (size - 1)) == 0);
    CHECK(rvg_graphics_device_mmio_read(d, 0x000) == 0x58464752); /* MAGIC, "RGFX" */
    CHECK(rvg_graphics_device_mmio_read(d, 0x014) == (uint32_t)size);
    CHECK(rvg_graphics_device_mmio_read(d, 0x018) == (uint32_t)(size >> 32));
    CHECK(rvg_graphics_device_mmio_read(d, 0x7fc) == 0);

    int doorbell[2];
    CHECK(pipe(doorbell) == 0);
    CHECK_STATUS(rvg_graphics_device_set_doorbell(d, -1), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_graphics_device_set_doorbell(d, doorbell[0]), RVG_OK);
    CHECK_STATUS(rvg_graphics_device_set_doorbell(d, doorbell[0]), RVG_ERR_INVALID_ARGUMENT);
    close(doorbell[0]);

    /* The guest starts the device, arms the completion ring (at 0x10000, its armed word 0xc0
     * in), writes GET_DISPLAYS (type 1, 16 bytes, seqno 1) at the submission ring's data, 256
     * bytes in, publishes it and rings. */
    rvg_graphics_device_mmio_write_taking(d, 0x010, 1);
    atomic_store((_Atomic uint32_t *)(void *)(region + 0x10000 + 0xc0), 1);
    CHECK(rvg_graphics_device_mmio_read(d, 0x010) == 1);
    uint32_t command[4] = {1, 16, 1, 0};
    memcpy(region + 256, command, sizeof command);
    atomic_store((_Atomic uint64_t *)(void *)region, 16);
    CHECK(write(doorbell[1], "\1\0\0\0\0\0\0\0", 8) == 8);
    CHECK(wait_for(&graphics_raised, 1));
    /* The completion ring, at 0x10000: DISPLAYS (0x101) answering seqno 1. */
    uint64_t produced = atomic_load((_Atomic uint64_t *)(void *)(region + 0x10000));
    uint32_t answer[4];
    memcpy(answer, region + 0x10000 + 256, sizeof answer);
    CHECK(produced >= 16 && answer[0] == 0x101 && answer[2] == 1);
    /* MOVE_CURSOR (type 4, 32 bytes, seqno 2) to (300, 200), shown, after it: the host is told
     * on the display thread and reads where the hot spot is; a reset puts it at (0, 0). */
    CHECK(rvg_graphics_device_cursor_position(d).x == 0);
    uint32_t move[8] = {4, 32, 2, 0, 0, 300, 200, 1};
    memcpy(region + 256 + 16, move, sizeof move);
    atomic_store((_Atomic uint64_t *)(void *)region, 48);
    CHECK(write(doorbell[1], "\1\0\0\0\0\0\0\0", 8) == 8);
    CHECK(wait_for(&graphics_moved, 1));
    RvgDisplayCoord at = rvg_graphics_device_cursor_position(d);
    CHECK(at.x == 300 && at.y == 200);
    /* A ring no queue has is ignored. */
    rvg_graphics_device_mmio_write_taking(d, 0x020, 0x40000);
    CHECK(rvg_graphics_device_mmio_read(d, 0x010) == 1);
    /* The display read back: a NULL buffer asks its size, 1920 x 1080 with no window, and with
     * no scanout it shows black, B, G, R, A. */
    uint32_t width = 0, height = 0;
    CHECK_STATUS(rvg_graphics_device_read_display(d, NULL, 0, &width, &height),
                 RVG_ERR_INVALID_ARGUMENT);
    CHECK(width == 1920 && height == 1080);
    size_t length = (size_t)width * height * 4;
    uint8_t *pixels = malloc(length);
    CHECK(pixels != NULL);
    if (pixels) {
        CHECK_STATUS(rvg_graphics_device_read_display(d, pixels, length, NULL, NULL), RVG_OK);
        CHECK(pixels[0] == 0 && pixels[1] == 0 && pixels[2] == 0 && pixels[3] == 0xff);
        CHECK(pixels[length - 1] == 0xff);
        free(pixels);
    }

    rvg_graphics_device_reset(d);
    CHECK(rvg_graphics_device_mmio_read(d, 0x010) == 0);
    CHECK(wait_for(&graphics_moved, 2));
    at = rvg_graphics_device_cursor_position(d);
    CHECK(at.x == 0 && at.y == 0);
    rvg_graphics_device_release(d);
    close(doorbell[1]);
    CHECK(wait_for(&graphics_destroyed, 1));
    rvg_graphics_device_release(NULL);
}

/* The files behind the memory map's ranges: a device over guest RAM that is a shared mapping of
 * a file, named in memory_files, and the refusals of the files, each with its table's one
 * destroy. */
static _Atomic int files_destroyed;

static void check_memory_files(void) {
    const uint64_t length = 16ull << 20;
    char path[] = "/tmp/rvg-harness-ram.XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    unlink(path);
    CHECK(ftruncate(fd, (off_t)length) == 0);
    void *shared = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    CHECK(shared != MAP_FAILED);
    RvgMemoryRange range = {0x40000000ull, length, shared};
    RvgMemoryMap map = {&range, 1};
    RvgMemoryRangeFile file = {fd, 0, 0};

    RvgGraphicsCallbacks cb;
    memset(&cb, 0, sizeof cb);
    cb.struct_size = sizeof cb;
    cb.ctx = &files_destroyed;
    cb.destroy = on_graphics_destroy;
    RvgGraphicsDeviceConfig c = {sizeof c, &cb, NULL, &map, &file};
    files_destroyed = 0;
    RvgGraphicsDevice *d = NULL;
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_OK);
    /* The library holds its own duplicate: the host may close its descriptor. */
    rvg_graphics_device_release(d);
    CHECK(wait_for(&files_destroyed, 1));

    d = (RvgGraphicsDevice *)0x1;
    c.memory_map = NULL;
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_ERR_INVALID_ARGUMENT);
    c.memory_map = &map;
    file.fd = -2;
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_ERR_INVALID_ARGUMENT);
    file.fd = fd;
    file.flags = 1;
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_ERR_INVALID_ARGUMENT);
    file.flags = 0;
    file.offset = 0x10;
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_ERR_INVALID_ARGUMENT);
    file.offset = 0;
    range.physical_length = length * 2;
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_ERR_INVALID_ARGUMENT);
    range.physical_length = length;
    CHECK(files_destroyed == 6);
    CHECK(d == (RvgGraphicsDevice *)0x1);

    /* A range with no file. */
    file.fd = -1;
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_OK);
    rvg_graphics_device_release(d);
    CHECK(wait_for(&files_destroyed, 7));

    munmap(shared, length);
    close(fd);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "die-null") == 0) {
        rvg_graphics_device_reset(NULL);
        puts("survived");
        return 0;
    }
    check_status();
    check_graphics();
    check_memory_files();
    if (failures) {
        fprintf(stderr, "harness: %d failures\n", failures);
        return 1;
    }
    puts("harness: ok");
    return 0;
}
