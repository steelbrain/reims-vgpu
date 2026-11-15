/*
 * The C test harness of reims-vgpu-c: a C program written only against the committed
 * include/reims_vgpu.h, linked to the library's dynamic library, and run by tests/c_harness.rs.
 * It drives the API end to end on the library's platform, as a C host does: a graphics device
 * with no window, its registers, region, doorbell, control ring, cursor and display read, its
 * refusals, the files behind its memory map's ranges, and each table's one destroy.
 *
 * Every rvg_ function the header declares is called at least once (tests/c_harness.rs checks
 * that). The device checks need a GPU the Vulkan platform runs on (over MoltenVK on a Mac).
 * On Windows the host's doorbell is an event and the guest's RAM a pagefile-backed section, as
 * QEMU makes them there; elsewhere a pipe and a shared mapping of a file.
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
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#endif

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
#if defined(_WIN32)
        Sleep(1);
#else
        struct timespec ms = {0, 1000000};
        nanosleep(&ms, NULL);
#endif
    }
    return *n == want;
}

#if defined(_WIN32)
/* The host's side of the control ring's doorbell on Windows: QEMU's manual-reset event, which
 * the device holds a copy of. */
typedef HANDLE Doorbell;

/* Rings `bell` as QEMU does for the guest's write to the doorbell register: sets the event. */
static int ring(Doorbell bell) {
    return SetEvent(bell) != 0;
}
#else
/* The host's side of the control ring's doorbell: a pipe's write side, whose read side the
 * device holds a copy of. */
typedef int Doorbell;

/* Rings `bell` as a host does for the guest's write to the doorbell register: 8 bytes, as an
 * eventfd takes them. */
static int ring(Doorbell bell) {
    return write(bell, "\1\0\0\0\0\0\0\0", 8) == 8;
}
#endif

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
 * a MOVE_CURSOR the host is told of and reads, a taking write, its doorbell pages (offered, and
 * a write at a page no process holds), a read of its display, its reset, and its release, which
 * ends its table once its threads have ended. */
static void check_graphics(void) {
    RvgGraphicsCallbacks cb = {sizeof cb,
                               &graphics_destroyed,
                               on_graphics_destroy,
                               on_graphics_raise,
                               NULL,
                               NULL,
                               on_graphics_cursor_move,
                               NULL,
                               NULL};
    RvgGraphicsDeviceConfig c = {sizeof c, &cb, NULL, NULL, NULL, RVG_GUEST_ARCH_X86_64, 2};
    RvgGraphicsDevice *d = NULL;
    graphics_destroyed = 0;
    graphics_raised = 0;
    graphics_moved = 0;
    CHECK_STATUS(rvg_graphics_device_create(NULL, &d), RVG_ERR_INVALID_ARGUMENT);
    /* doorbell_pages is 0 or 1; the refusal ends its table, with its one destroy. */
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_ERR_INVALID_ARGUMENT);
    CHECK(d == NULL && graphics_destroyed == 1);
    c.doorbell_pages = 1;
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
    /* FEATURES_HI offers DOORBELL_PAGES, bit 46, where the host maps BAR 4. */
    CHECK((rvg_graphics_device_mmio_read(d, 0x00c) >> 14) & 1);

#if defined(_WIN32)
    /* NULL names no handle; the device takes a copy of the event, and refuses a second. */
    Doorbell bell = CreateEventW(NULL, TRUE, FALSE, NULL);
    CHECK(bell != NULL);
    CHECK_STATUS(rvg_graphics_device_set_doorbell(d, NULL), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_graphics_device_set_doorbell(d, bell), RVG_OK);
    CHECK_STATUS(rvg_graphics_device_set_doorbell(d, bell), RVG_ERR_INVALID_ARGUMENT);
#else
    int doorbell[2];
    CHECK(pipe(doorbell) == 0);
    CHECK_STATUS(rvg_graphics_device_set_doorbell(d, -1), RVG_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(rvg_graphics_device_set_doorbell(d, doorbell[0]), RVG_OK);
    CHECK_STATUS(rvg_graphics_device_set_doorbell(d, doorbell[0]), RVG_ERR_INVALID_ARGUMENT);
    close(doorbell[0]);
    Doorbell bell = doorbell[1];
#endif

    /* The guest starts the device, arms the completion ring (at 0x10000, its armed word 0xc0
     * in), writes GET_DISPLAYS (type 1, 16 bytes, seqno 1) at the submission ring's data, 256
     * bytes in, publishes it and rings. */
    rvg_graphics_device_mmio_write_taking(d, 0x010, 1);
    atomic_store((_Atomic uint32_t *)(void *)(region + 0x10000 + 0xc0), 1);
    CHECK(rvg_graphics_device_mmio_read(d, 0x010) == 1);
    uint32_t command[4] = {1, 16, 1, 0};
    memcpy(region + 256, command, sizeof command);
    atomic_store((_Atomic uint64_t *)(void *)region, 16);
    CHECK(ring(bell));
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
    CHECK(ring(bell));
    CHECK(wait_for(&graphics_moved, 1));
    RvgDisplayCoord at = rvg_graphics_device_cursor_position(d);
    CHECK(at.x == 300 && at.y == 200);
    /* A ring no queue has is ignored, and a write at a doorbell page no process holds is
     * dropped. */
    rvg_graphics_device_mmio_write_taking(d, 0x020, 0x40000);
    rvg_graphics_device_doorbell_write(d, 0, 0x40000);
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
#if defined(_WIN32)
    CloseHandle(bell);
#else
    close(bell);
#endif
    CHECK(wait_for(&graphics_destroyed, 2));
    rvg_graphics_device_release(NULL);
}

#if defined(_WIN32)
/* The file behind a range on Windows, as the header's RvgMemoryRangeFile names it: a section's
 * handle; NULL for none. */
typedef HANDLE RamFile;
#define FILE_OF(entry) ((entry).handle)
#define NO_FILE NULL

/* `length` bytes of guest RAM as QEMU makes it on Windows, in `*file`: a view of a new
 * pagefile-backed section; NULL where Windows refuses. */
static void *ram_map(uint64_t length, RamFile *file) {
    *file = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE | SEC_COMMIT,
                               (DWORD)(length >> 32), (DWORD)length, NULL);
    if (*file == NULL) {
        return NULL;
    }
    return MapViewOfFile(*file, FILE_MAP_ALL_ACCESS, 0, 0, (SIZE_T)length);
}

static void ram_unmap(void *shared, uint64_t length, RamFile file) {
    (void)length;
    UnmapViewOfFile(shared);
    CloseHandle(file);
}

/* A handle of the process that names no section: an event, which the refusal closes. */
static RamFile not_a_file(void) {
    return CreateEventW(NULL, TRUE, FALSE, NULL);
}

static void not_a_file_close(RamFile file) {
    CloseHandle(file);
}
#else
/* The file behind a range, as the header's RvgMemoryRangeFile names it: a descriptor; -1 for
 * none. */
typedef int RamFile;
#define FILE_OF(entry) ((entry).fd)
#define NO_FILE (-1)

/* `length` bytes of guest RAM as a host maps it, in `*file`: a shared mapping of a new file;
 * NULL where the host refuses. */
static void *ram_map(uint64_t length, RamFile *file) {
    char path[] = "/tmp/rvg-harness-ram.XXXXXX";
    *file = mkstemp(path);
    if (*file < 0) {
        return NULL;
    }
    unlink(path);
    if (ftruncate(*file, (off_t)length) != 0) {
        return NULL;
    }
    void *shared = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, *file, 0);
    return shared == MAP_FAILED ? NULL : shared;
}

static void ram_unmap(void *shared, uint64_t length, RamFile file) {
    munmap(shared, length);
    close(file);
}

/* A descriptor the library refuses: one below -1. */
static RamFile not_a_file(void) {
    return -2;
}

static void not_a_file_close(RamFile file) {
    (void)file;
}
#endif

/* The files behind the memory map's ranges: a device over guest RAM that is a shared mapping of
 * a file (on Windows a view of a section), named in memory_files, and the refusals of the files,
 * each with its table's one destroy. */
static _Atomic int files_destroyed;

static void check_memory_files(void) {
    const uint64_t length = 16ull << 20;
    RamFile ram_file;
    void *shared = ram_map(length, &ram_file);
    CHECK(shared != NULL);
    if (!shared) {
        return;
    }
    RvgMemoryRange range = {0x40000000ull, length, shared};
    RvgMemoryMap map = {&range, 1};
    RvgMemoryRangeFile file = {ram_file, 0, 0};

    RvgGraphicsCallbacks cb;
    memset(&cb, 0, sizeof cb);
    cb.struct_size = sizeof cb;
    cb.ctx = &files_destroyed;
    cb.destroy = on_graphics_destroy;
    RvgGraphicsDeviceConfig c = {sizeof c, &cb, NULL, &map, &file, RVG_GUEST_ARCH_X86_64, 0};
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
    RamFile refused = not_a_file();
    FILE_OF(file) = refused;
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_ERR_INVALID_ARGUMENT);
    not_a_file_close(refused);
    FILE_OF(file) = ram_file;
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
    FILE_OF(file) = NO_FILE;
    CHECK_STATUS(rvg_graphics_device_create(&c, &d), RVG_OK);
    rvg_graphics_device_release(d);
    CHECK(wait_for(&files_destroyed, 7));

    ram_unmap(shared, length, ram_file);
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
