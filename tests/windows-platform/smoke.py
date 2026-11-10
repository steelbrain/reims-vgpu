"""Run the upstream authored protocol-245 fill workload through the real Windows DLL.

Only the harness's allocation, guest page geometry and sleep API are platform-specific.
The FIFO commands and expected guest RAM bytes are unchanged. No native window opens.
"""
from pathlib import Path
import os
import subprocess
import argparse
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--library", required=True, type=Path, help="Directory containing the DLL and GNU import library")
parser.add_argument("--output", required=True, type=Path, help="Separate temporary build and log directory")
args = parser.parse_args()
SOURCE = Path(__file__).resolve().parents[2]
LIBRARY = args.library.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
compiler = shutil.which("gcc")
if compiler is None:
    raise SystemExit("Run with the matching MinGW GCC directory on PATH")
text = (SOURCE / "xtask/src/smoke_guest.m").read_text(encoding="utf-8")
# The upstream harness is arm64-only. The x86-64 wire ABI uses 4 KiB guest
# pages (guest/records.rs::APV_PAGE_SHIFT=12), including its one-page FIFO.
# Keep the same disjoint page layout and workload; 16 KiB here addresses the
# wrong GPA and would also overlap the task root after changing PAGE alone.
text = text.replace('#define PAGE 0x4000ull', '#define PAGE 0x1000ull')
text = text.replace('#define FIFO_LENGTH 0x4000u', '#define FIFO_LENGTH 0x1000u')
text = text.replace('128 pages of 16 KiB.', '128 pages of 4 KiB (x86-64).')
text = text.replace('#import <Foundation/Foundation.h>\n', '#include <windows.h>\n#include <io.h>\n#include <fcntl.h>\n#include <assert.h>\n')
for header in ('dlfcn.h', 'mach/mach.h', 'mach/mach_vm.h', 'unistd.h'):
    text = text.replace(f'#include <{header}>\n', '')
text = text.replace('#include <reims_vgpu.h>', '#include <reims_vgpu.h>\n#define usleep(us) Sleep(((us) + 999) / 1000)')
start = text.index('static const char *image_of(')
end = text.index('// The guest: bring-up', start)
text = text[:start] + text[end:]
start = text.index('int main(void) {')
text = text[:start] + r'''
static void log_line(void *ctx, RvgLogLevel level, const char *target, size_t tn,
                     const char *message, size_t mn) {
    (void)ctx; (void)level;
    fprintf(stderr, "%.*s: %.*s\n", (int)tn, target, (int)mn, message);
}
/* EFI maps guest pages read-only before making its no-copy Metal buffer. Exercise
 * that real path and change the source afterwards to catch stale copied aliases. */
static atomic_uint efi_notifications;
static void efi_ready(void *ctx) {
    (void)ctx;
    atomic_fetch_add(&efi_notifications, 1);
}
static void efi_check(void) {
    RvgDisplay *display = NULL;
    RvgDisplayCallbacks callbacks = {.struct_size=sizeof(callbacks), .new_frame=efi_ready};
    RvgDisplayConfig dc = {.struct_size=sizeof(dc), .serial_number=123, .port=0, .callbacks=&callbacks};
    assert(rvg_display_create(device, &dc, &display) == RVG_OK);
    RvgDisplayMode mode = {.width=64, .height=32, .refresh_rate_hz=60};
    rvg_display_set_modes(display, &mode, 1);
    uint32_t *pixels = (uint32_t *)gpa(page(BP + 64));
    for (size_t i=0; i<64*32; i++) pixels[i] = 0xffff0000u;
    mmio_write(0x1200, 1);
    mmio_write(0x121c, 64 | (32 << 16));
    mmio_write(0x1218, 32);
    mmio_write(0x1228, 256);
    mmio_write(0x1210, BP + 64);
    mmio_write(0x1214, 8192);
    RvgFrameTarget *target = NULL;
    RvgFrameTargetConfig tc = {.struct_size=sizeof(tc), .width=64, .height=32,
                              .format=RVG_PIXEL_FORMAT_BGRA8_UNORM};
    assert(rvg_frame_target_create(device, &tc, &target) == RVG_OK);
    for (int round=0; round<2; round++) {
        uint64_t before=0, now=0;
        rvg_display_present_counts(display, &before, NULL);
        if (round) for (size_t i=0; i<64*32; i++) pixels[i] = 0xff0000ffu;
        atomic_thread_fence(memory_order_seq_cst);
        for (int n=0; n<1000 && now<=before; n++) {
            Sleep(1);
            rvg_display_present_counts(display, &now, NULL);
        }
        assert(now > before);
        assert(atomic_load(&efi_notifications) > 0);
        RvgFrame *frame = NULL;
        assert(rvg_display_encode_frame(display, target, NULL, &frame) == RVG_OK);
        assert(rvg_frame_commit(frame) == RVG_OK);
        assert(rvg_frame_wait(frame) == RVG_OK);
        rvg_frame_release(frame);
        uint32_t readback[64*32];
        assert(rvg_frame_target_read_pixels(target, readback, 256, sizeof(readback)) == RVG_OK);
        for (size_t i=0; i<64*32; i++) {
            if ((readback[i]&0xffffffu) != (round ? 0xffu : 0xff0000u)) {
                fprintf(stderr,"EFI round %d pixel %zu: 0x%x\n",round,i,readback[i]);
                abort();
            }
        }
        printf("efi_%s: ok\n",round ? "live_update" : "readonly_frame");
    }
    mmio_write(0x1200, 0);
    rvg_frame_target_release(target);
    rvg_display_release(display);
}
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    rvg_log_set_handler(log_line, NULL, RVG_LOG_INFO);
    /* The test file and the DLL see exactly the same pages, as WHPX and the GPU will. */
    HANDLE file = CreateFileW(L"smoke-ram.tmp", GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, CREATE_NEW,
        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
    assert(file != INVALID_HANDLE_VALUE);
    LARGE_INTEGER length = {.QuadPart = RAM_PAGES * PAGE};
    assert(SetFilePointerEx(file, length, NULL, FILE_BEGIN));
    assert(SetEndOfFile(file));
    HANDLE section = CreateFileMappingW(file, NULL, PAGE_READWRITE, 0, 0, NULL);
    assert(section);
    ram = MapViewOfFile(section, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, RAM_PAGES * PAGE);
    assert(ram);
    CloseHandle(section);
    int fd = _open_osfhandle((intptr_t)file, _O_RDWR | _O_BINARY);
    assert(fd >= 0);
    RvgMemoryRange range = {RAM_BASE, RAM_PAGES * PAGE, ram};
    RvgMemoryMap map = {&range, 1};
    RvgMemoryRangeFile memory_file = {.fd = fd};
    RvgHostCallbacks callbacks = {.struct_size = sizeof(callbacks), .raise_interrupt = raise_interrupt};
    RvgDeviceConfig config = {
        .struct_size = sizeof(config), .display_port_count = 1, .process_isolation = true,
        .memory_map = &map, .memory_files = &memory_file, .callbacks = &callbacks,
    };
    RvgStatus status = rvg_device_create(&config, &device);
    printf("device: %s\n", status == RVG_OK ? "created" : rvg_status_message(status));
    if (status == RVG_OK) { efi_check(); guest(); rvg_device_release(device); }
    assert(UnmapViewOfFile(ram));
    assert(_close(fd) == 0);
    printf("done: ok\n");
    return status == RVG_OK ? 0 : 1;
}
'''
c = OUT / "smoke.c"
c.write_text(text, encoding="utf-8")
exe = OUT / "smoke.exe"
environment = dict(os.environ, PATH=str(LIBRARY) + os.pathsep + os.environ['PATH'])
subprocess.run([compiler, '-std=c11', '-O2', '-Wall', '-Wextra', str(c),
    '-I' + str(SOURCE / 'crates/reims-vgpu-c/include'), '-L' + str(LIBRARY),
    '-lreims_vgpu_c', '-o', str(exe)], check=True, env=environment)
with (OUT / 'stdout.log').open('w') as out, (OUT / 'stderr.log').open('w') as err:
    result = subprocess.run([str(exe)], cwd=OUT, env=environment, stdout=out, stderr=err, timeout=120)
facts = (OUT / 'stdout.log').read_text()
print(facts)
expected = {'device':'created','version':'245','root_page':'ok','fifo_enabled':'1',
    'fault_offset':'0xffffffff','nop':'ok','read':'12','interrupt':'raised',
    'task_define':'ok','task_handshake':'accepted','task_root_word':'ok','task_length':'ok',
    'task_map':'ok','task_objects':'ok','task_exec':'ok','task_fill':'ok','faulted':'0',
    'efi_readonly_frame':'ok','efi_live_update':'ok'}
observed = dict(line.split(': ',1) for line in facts.splitlines() if ': ' in line)
failures = {k: observed.get(k) for k, value in expected.items() if observed.get(k) != value}
if result.returncode or failures:
    print((OUT / 'stderr.log').read_text()[-12000:])
    raise SystemExit(f'Windows protocol/GPU fill check failed: exit={result.returncode}, observations={failures}')
print('PASS: 17 original protocol/GPU writeback checks and 2 real EFI rendering checks matched')
