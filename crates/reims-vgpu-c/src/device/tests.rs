use reims_vgpu_core::guest::records::APV_PAGE_SHIFT;
use reims_vgpu_core::testkit::GuestRam;
use reims_vgpu_core::testkit::host::FakeHost;

use super::*;
use crate::status::*;
use crate::testkit::{HostCtx, Rec, Rig, config, host_table, t_destroy, test_platform};

/// A table with only a destroy, over `rec`.
fn only_destroy(rec: &Arc<Rec>) -> RvgHostCallbacks {
    RvgHostCallbacks {
        struct_size: crate::sized::size_u32::<RvgHostCallbacks>(),
        ctx: rec.ctx(),
        destroy: Some(t_destroy),
        create_task: None,
        destroy_task: None,
        map_memory: None,
        unmap_memory: None,
        read_memory: None,
        raise_interrupt: None,
        add_trace_range: None,
        remove_trace_range: None,
    }
}

fn create(c: &RvgDeviceConfig) -> (RvgStatus, *mut RvgDevice) {
    let mut d = std::ptr::null_mut();
    // SAFETY: a live config; `d` is writable.
    let s = unsafe { rvg_device_create(c, &mut d) };
    (s, d)
}

/// Each refusal in the header's order, with the table's one destroy before the create returns
/// on the caller's thread; a config or table the create cannot read runs none
#[test]
fn refusals_end_the_table() {
    let p = test_platform();
    let rec = Rec::new();
    let table = only_destroy(&rec);
    let mut c = config(1, &table);
    // SAFETY: NULL config.
    let s = unsafe { rvg_device_create(std::ptr::null(), &mut std::ptr::null_mut()) };
    assert_eq!(s, RVG_ERR_INVALID_ARGUMENT);
    c.struct_size = 8;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    c.struct_size = crate::sized::size_u32::<RvgDeviceConfig>();
    let mut small = table;
    small.struct_size = 8;
    c.callbacks = &small;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    c.callbacks = &table;
    assert!(rec.destroys().is_empty());

    // SAFETY: a live config; NULL out-pointer.
    let s = unsafe { rvg_device_create(&c, std::ptr::null_mut()) };
    assert_eq!(s, RVG_ERR_INVALID_ARGUMENT);
    c.display_port_count = 0;
    assert_eq!(create(&c).0, RVG_ERR_DISPLAY_PORT_COUNT);
    c.display_port_count = 1;
    c.metal_registry_id = 1;
    assert_eq!(create(&c).0, RVG_ERR_NO_METAL_DEVICE);
    c.metal_registry_id = reims_vgpu_core::testkit::platform::FAKE_REGISTRY_ID;
    c.process_isolation = true;
    assert_eq!(create(&c).0, RVG_ERR_ISOLATION_WITHOUT_MAP);
    c.process_isolation = false;
    let mut with_add = table;
    with_add.add_trace_range = Some(crate::testkit::t_add_trace);
    c.callbacks = &with_add;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    let null_ranges = RvgMemoryMap {
        ranges: std::ptr::null(),
        range_count: 1,
    };
    c.callbacks = &table;
    c.memory_map = &null_ranges;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    let me = std::thread::current().id();
    assert_eq!(rec.destroys(), vec![me; 6]);
    drop(p);
}

/// A task or memory callback beside a memory map is a conflict
#[test]
fn memory_conflict() {
    let p = test_platform();
    let host = FakeHost::new(&p, GuestRam::new());
    let ctx = Arc::new(HostCtx {
        rec: Rec::new(),
        blocks: host.blocks(),
    });
    let table = host_table(&ctx);
    let map = RvgMemoryMap {
        ranges: std::ptr::null(),
        range_count: 0,
    };
    let mut c = config(1, &table);
    c.memory_map = &map;
    assert_eq!(create(&c).0, RVG_ERR_MEMORY_CONFLICT);
    assert_eq!(ctx.rec.destroys().len(), 1);
}

/// The table's destroy runs once, after the device's end, on the releasing thread; a longer
/// config with zero past the library's size is accepted
#[test]
fn destroy_after_the_end() {
    let mut rig = Rig::new(1);
    assert!(rig.ctx.rec.destroys().is_empty());
    // SAFETY: the rig's live device.
    let d = unsafe { &*rig.device };
    for off in (0x1000..0x1240).step_by(4) {
        // SAFETY: as above.
        let v = unsafe { rvg_device_mmio_read(rig.device, off) };
        assert_eq!(v, d.device.core().mmio_read(off), "{off:#x}");
    }
    rig.release();
    assert_eq!(rig.ctx.rec.destroys(), vec![std::thread::current().id()]);

    #[repr(C)]
    struct Longer {
        c: RvgDeviceConfig,
        later: u64,
    }
    let rec = Rec::new();
    let table = only_destroy(&rec);
    let mut l = Longer {
        c: config(1, &table),
        later: 0,
    };
    l.c.struct_size = crate::sized::size_u32::<Longer>();
    let (s, d) = create(&l.c);
    assert_eq!(s, RVG_OK);
    assert!(rec.destroys().is_empty());
    // SAFETY: the device made above.
    unsafe { rvg_device_release(d) };
    assert_eq!(rec.destroys().len(), 1);
    l.later = 1;
    assert_eq!(create(&l.c).0, RVG_ERR_UNSUPPORTED);
    assert_eq!(rec.destroys().len(), 1);
}

/// Suspend gives bytes the device takes back; no state is refused with its reason; the
/// status functions refuse NULL
#[test]
fn suspend_and_resume() {
    let rig = Rig::new(1);
    let mut e = 99;
    // SAFETY: the rig's live device; `e` is writable.
    unsafe {
        assert_eq!(
            rvg_device_will_resume(rig.device, std::ptr::null(), 0, &mut e),
            RVG_ERR_RESUME
        );
        assert_eq!(e, RVG_RESUME_ERROR_INVALID_CONTENT);
        rvg_device_will_suspend(rig.device);
        let mut state = RvgBytes {
            data: std::ptr::null_mut(),
            length: 0,
        };
        assert_eq!(rvg_device_finish_suspend(rig.device, &mut state), RVG_OK);
        assert!(!state.data.is_null() && state.length > 0);
        assert_eq!(
            rvg_device_will_resume(rig.device, state.data, state.length, &mut e),
            RVG_OK
        );
        assert_eq!(e, RVG_RESUME_ERROR_NONE);
        assert_eq!(
            rvg_device_will_resume(rig.device, state.data, state.length, std::ptr::null_mut()),
            RVG_OK
        );
        rvg_bytes_free(state);
        rvg_bytes_free(RvgBytes {
            data: std::ptr::null_mut(),
            length: 0,
        });
        rvg_device_did_resume(rig.device);
        rvg_device_pause(rig.device);
        rvg_device_unpause(rig.device);
        rvg_device_reset(rig.device);
        rvg_device_stop(rig.device);
        let mut again = RvgBytes {
            data: std::ptr::null_mut(),
            length: 0,
        };
        assert_eq!(
            rvg_device_finish_suspend(std::ptr::null_mut(), &mut again),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_device_finish_suspend(rig.device, std::ptr::null_mut()),
            RVG_ERR_INVALID_ARGUMENT
        );
        e = 7;
        assert_eq!(
            rvg_device_will_resume(std::ptr::null_mut(), std::ptr::null(), 0, &mut e),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(e, 7);
        rvg_device_release(std::ptr::null_mut());
    }
}

/// The fake host's callbacks reach the host through the table: the device's order of calls
/// is its twin's, made on the Rust API over the same blocks
#[test]
fn host_calls_are_the_twins() {
    let rig = Rig::new(1);
    // SAFETY: the rig's live device: a reset, which stops and restarts the FIFOs.
    unsafe {
        rvg_device_mmio_write(rig.device, 0x1000, 0);
        rvg_device_reset(rig.device);
    }
    let ours = host_calls(&rig.p);
    let twin_p = std::sync::Arc::new(reims_vgpu_core::testkit::FakePlatform::new());
    let twin_host = FakeHost::new(&twin_p, GuestRam::with_range(0, 0x40_0000).unwrap());
    let b = twin_host.blocks();
    let c = reims_vgpu::DeviceConfig {
        callbacks: reims_vgpu::HostCallbacks {
            create_task: b.create_task,
            destroy_task: b.destroy_task,
            map_memory: b.map_memory,
            unmap_memory: b.unmap_memory,
            read_memory: b.read_memory,
            raise_interrupt: b.raise_interrupt,
            add_trace_range: b.add_trace_range,
            remove_trace_range: b.remove_trace_range,
        },
        ..reims_vgpu::DeviceConfig::default()
    };
    let twin = Device::with_platform(twin_p.clone(), c).unwrap();
    twin.mmio_write(0x1000, 0);
    twin.reset();
    assert_eq!(ours, host_calls(&twin_p));
}

/// The host-block calls `p`'s fake host saw, in order.
fn host_calls(p: &reims_vgpu_core::testkit::FakePlatform) -> Vec<&'static str> {
    p.call_log()
        .calls()
        .into_iter()
        .filter(|c| c.class == "HostBlocks")
        .map(|c| c.sel)
        .collect()
}

/// The device info pairs the guest reads in page `page`, where it asked for them, up to their
/// terminator: 8 bytes each, the key low and the value high.
fn device_info_pairs(rig: &Rig, page: u32) -> Vec<(u32, u32)> {
    // What the device wrote through its mapping of the page, in the guest's RAM.
    rig.p.fake_vm().sync();
    let at = u64::from(page) << APV_PAGE_SHIFT;
    let mut pairs = Vec::new();
    for i in 0..64 {
        let w = rig._host.ram().read_u64(at + 8 * i).unwrap();
        let pair = (
            u32::try_from(w & 0xffff_ffff).unwrap(),
            u32::try_from(w >> 32).unwrap(),
        );
        pairs.push(pair);
        if pair.0 == 0 {
            break;
        }
    }
    pairs
}

/// The feature level reaches the guest: the binary version it reads back after asking for
/// 0xf5, capped by levels 1 to 3; and the device info it reads, which those levels trim (key
/// 18's value, the highest key below 40, and keys 42 and 44); levels 0 and 4 read the same
#[test]
fn the_feature_level_reaches_the_guest() {
    let mut seen = Vec::new();
    let mut infos = Vec::new();
    for level in 0..=4 {
        let rig = Rig::with(1, |c| c.feature_level = level);
        // SAFETY: the rig's live device: the version, then a root node for the mapping.
        let version = unsafe {
            rvg_device_mmio_write(rig.device, 0x1034, 0xf5);
            let v = rvg_device_mmio_read(rig.device, 0x1034);
            rvg_device_mmio_write(rig.device, 0x101c, 1);
            v
        };
        // The guest's device info request (opcode 58) for every key, into guest page 0x20.
        // SAFETY: as above.
        let core = unsafe { &*rig.device }.device.core();
        core.get_device_info(45, 46, 0x20).unwrap();
        let pairs = device_info_pairs(&rig, 0x20);
        let value = |key| pairs.iter().find(|p| p.0 == key).map(|p| p.1);
        let below_40 = pairs.iter().map(|p| p.0).filter(|&k| k < 40).max();
        seen.push((
            version,
            value(18),
            below_40,
            value(42).is_some() && value(44).is_some(),
        ));
        infos.push(pairs);
    }
    assert_eq!(
        seen,
        [
            (0xf5, Some(0x20007), Some(37), true),
            (0x3e, Some(0x20002), Some(18), false),
            (0x71, Some(0x20006), Some(28), false),
            (0xd1, Some(0x20007), Some(37), false),
            (0xf5, Some(0x20007), Some(37), true),
        ]
    );
    assert_eq!(infos[0], infos[4]);
}

/// The config's first version ends at `callbacks`, 48 bytes; `memory_files` makes it 56, and
/// `process_isolation_follows_guest` 64
#[test]
fn a_first_version_device_config_is_48_bytes() {
    assert_eq!(<RvgDeviceConfig as HostStruct>::V1_SIZE, 48);
    assert_eq!(std::mem::offset_of!(RvgDeviceConfig, memory_files), 48);
    assert_eq!(
        std::mem::offset_of!(RvgDeviceConfig, process_isolation_follows_guest),
        56
    );
    assert_eq!(std::mem::offset_of!(RvgDeviceConfig, dirty_source), 64);
    assert_eq!(size_of::<RvgDeviceConfig>(), 72);
    assert_eq!(size_of::<RvgDirtySource>(), 56);
    assert_eq!(size_of::<RvgMemoryRangeFile>(), 16);
    assert_eq!(std::mem::offset_of!(RvgMemoryRangeFile, flags), 4);
    assert_eq!(std::mem::offset_of!(RvgMemoryRangeFile, offset), 8);
}

/// `len` bytes of the fake VM's memory, as one range at guest physical 0.
fn fake_range(p: &reims_vgpu_core::testkit::FakePlatform, len: u64) -> RvgMemoryRange {
    let host = reims_vgpu_core::platform::Platform::vm(p)
        .allocate_anywhere(len)
        .unwrap();
    RvgMemoryRange {
        physical_address: 0,
        physical_length: len,
        host_address: host.as_ptr(),
    }
}

/// A file of `len` bytes in the temporary directory, removed at its drop.
struct TempFile {
    path: std::path::PathBuf,
    file: std::fs::File,
}

impl TempFile {
    fn new(name: &str, len: u64) -> TempFile {
        let path =
            std::env::temp_dir().join(format!("rvg-memory-file-{name}-{}", std::process::id()));
        let file = std::fs::File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        file.set_len(len).unwrap();
        TempFile { path, file }
    }

    fn fd(&self) -> i32 {
        std::os::fd::AsRawFd::as_raw_fd(&self.file)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The descriptors open in this process (nextest runs each test in a process of its own).
fn open_fds() -> usize {
    std::fs::read_dir("/dev/fd").unwrap().count()
}

/// A host built before `memory_files` passes 48: its config is read without files, whatever
/// follows it in memory; read at its full size, the same bytes are refused, with the table's
/// one destroy
#[test]
fn a_48_byte_device_config_is_accepted() {
    let p = test_platform();
    let rec = Rec::new();
    let table = only_destroy(&rec);
    let range = fake_range(&p, 0x1_0000);
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    let bad = RvgMemoryRangeFile {
        fd: -5,
        flags: 1,
        offset: 1,
    };
    let mut c = config(1, &table);
    c.memory_map = &map;
    c.memory_files = &bad;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    assert_eq!(rec.destroys().len(), 1);
    c.struct_size = 48;
    let (s, d) = create(&c);
    assert_eq!(s, RVG_OK);
    // SAFETY: the device made above.
    unsafe { rvg_device_release(d) };
    assert_eq!(rec.destroys().len(), 2);
}

/// An isolated device over a memory map is made with isolation following the guest and
/// without, and from a host built before the field (56 bytes), which reads it as false
#[test]
fn isolation_following_the_guest_is_accepted() {
    let p = test_platform();
    let rec = Rec::new();
    let table = only_destroy(&rec);
    let range = fake_range(&p, 0x1_0000);
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    for (size, follows) in [(64, true), (64, false), (56, true)] {
        let mut c = config(1, &table);
        c.struct_size = size;
        c.memory_map = &map;
        c.process_isolation = true;
        c.process_isolation_follows_guest = follows;
        let (s, d) = create(&c);
        assert_eq!(s, RVG_OK, "{size} bytes, following {follows}");
        // SAFETY: the device made above.
        unsafe { rvg_device_release(d) };
    }
    assert_eq!(rec.destroys().len(), 3);
}

/// Each file refusal in the header's order, the first range's first, each with its line and no
/// descriptor duplicated
#[test]
fn memory_file_refusals() {
    let p = test_platform();
    let page = host_page_size().get();
    let file = TempFile::new("refusals", 0x1_0000);
    let range = fake_range(&p, 0x8000);
    let ranges = [range, range];
    let map = RvgMemoryMap {
        ranges: ranges.as_ptr(),
        range_count: 2,
    };
    let ok = RvgMemoryRangeFile {
        fd: file.fd(),
        flags: 0,
        offset: 0x8000,
    };
    let none = RvgMemoryRangeFile {
        fd: -1,
        flags: 0,
        offset: 1,
    };
    let entry = |fd: i32, flags: u32, offset: u64| RvgMemoryRangeFile { fd, flags, offset };
    let logs = reims_vgpu_core::testkit::logs::capture();
    let fds = open_fds();
    let refused = |files: [RvgMemoryRangeFile; 2]| -> String {
        // SAFETY: a live map of two ranges, and two entries.
        let r = unsafe { memory_map(&map, files.as_ptr()) };
        assert_eq!(r.err(), Some(RVG_ERR_INVALID_ARGUMENT));
        let lines = logs.take();
        assert_eq!(lines.len(), 1, "{lines:?}");
        lines[0].2.clone()
    };
    // Files without a map.
    // SAFETY: no map; the entry is live.
    let r = unsafe { memory_map(std::ptr::null(), &ok) };
    assert_eq!(r.err(), Some(RVG_ERR_INVALID_ARGUMENT));
    assert_eq!(
        logs.take()[0].2,
        "device not created: memory files without a memory map"
    );
    // Each check, on range 1 after a range 0 that passes.
    assert_eq!(
        refused([ok, entry(-2, 1, 0x10)]),
        "device not created: the file of memory range 1 has fd -2, below -1"
    );
    assert_eq!(
        refused([ok, entry(i32::MAX, 1, 0x10)]),
        "device not created: the file of memory range 1 has flags 0x1, not 0"
    );
    assert_eq!(
        refused([ok, entry(-1, 2, 0)]),
        "device not created: the file of memory range 1 has flags 0x2, not 0"
    );
    let unreadable = refused([ok, entry(i32::MAX, 0, 0x10)]);
    assert!(
        unreadable.starts_with(
            "device not created: the file of memory range 1 has fd 2147483647, which fstat \
             cannot read: "
        ),
        "{unreadable}"
    );
    assert_eq!(
        refused([ok, entry(file.fd(), 0, 0x8010)]),
        "device not created: the file of memory range 1 holds 0x10000 bytes, fewer than its \
         offset 0x8010 and the range's 0x8000"
    );
    assert_eq!(
        refused([ok, entry(file.fd(), 0, u64::MAX - 1)]),
        "device not created: the file of memory range 1 holds 0x10000 bytes, fewer than its \
         offset 0xfffffffffffffffe and the range's 0x8000"
    );
    assert_eq!(
        refused([ok, entry(file.fd(), 0, 0x10)]),
        format!(
            "device not created: the file of memory range 1 has offset 0x10, not a multiple of \
             the page size {page:#x}"
        )
    );
    // The first range's refusal comes first.
    assert_eq!(
        refused([entry(file.fd(), 0, 0x10), entry(-2, 0, 0)]),
        format!(
            "device not created: the file of memory range 0 has offset 0x10, not a multiple of \
             the page size {page:#x}"
        )
    );
    // No refusal duplicated a descriptor; a range without a file takes no offset check.
    assert_eq!(open_fds(), fds);
    // SAFETY: a live map of two ranges, and two entries.
    let m = unsafe { memory_map(&map, [none, none].as_ptr()) }.unwrap();
    assert!(m.unwrap().files().iter().all(Option::is_none));
    assert!(logs.take().is_empty());
}

/// The files the map takes: each a duplicate of the host's descriptor, close-on-exec, of the
/// same file, at the host's offset, in the ranges' order; none for -1 or NULL; the duplicates
/// close with the last map that holds them
#[test]
fn memory_files_are_duplicated() {
    let p = test_platform();
    let file = TempFile::new("duplicated", 0x2_0000);
    let range = fake_range(&p, 0x8000);
    let ranges = [range, range];
    let map = RvgMemoryMap {
        ranges: ranges.as_ptr(),
        range_count: 2,
    };
    let files = [
        RvgMemoryRangeFile {
            fd: -1,
            flags: 0,
            offset: 0,
        },
        RvgMemoryRangeFile {
            fd: file.fd(),
            flags: 0,
            offset: 0x1_0000,
        },
    ];
    let fds = open_fds();
    // SAFETY: a live map of two ranges, and two entries.
    let m = unsafe { memory_map(&map, files.as_ptr()) }
        .unwrap()
        .unwrap();
    assert_eq!(open_fds(), fds + 1);
    assert!(m.files()[0].is_none());
    let f = m.files()[1].as_ref().unwrap();
    assert_eq!(f.offset, 0x1_0000);
    let dup = std::os::fd::AsRawFd::as_raw_fd(&*f.fd);
    assert_ne!(dup, file.fd());
    // SAFETY: a query of a descriptor the map holds open.
    let flags = unsafe { libc::fcntl(dup, libc::F_GETFD) };
    assert_eq!(flags & libc::FD_CLOEXEC, libc::FD_CLOEXEC);
    let ino = |fd: i32| {
        let mut st = std::mem::MaybeUninit::<libc::stat>::zeroed();
        // SAFETY: `st` is a writable `struct stat`, and `fd` is open.
        assert_eq!(unsafe { libc::fstat(fd, st.as_mut_ptr()) }, 0);
        // SAFETY: filled by the call above.
        let st = unsafe { st.assume_init() };
        (st.st_dev, st.st_ino)
    };
    assert_eq!(ino(dup), ino(file.fd()));
    // NULL files: none.
    // SAFETY: a live map of two ranges.
    let bare = unsafe { memory_map(&map, std::ptr::null()) }
        .unwrap()
        .unwrap();
    assert!(bare.files().iter().all(Option::is_none));
    let copy = m.clone();
    drop(m);
    assert_eq!(open_fds(), fds + 1);
    drop(copy);
    assert_eq!(open_fds(), fds);
}

/// The fake host's slots: two, each passed to `rvg_dirty_slot`, and the host's writes, page 1 of
/// each take.
unsafe extern "C-unwind" fn t_slots(ctx: *mut c_void, slot_ctx: *mut c_void) -> bool {
    Rec::push(ctx, "for_each_slot".into());
    // SAFETY: the library's function over its own slot_ctx, inside this call.
    unsafe {
        rvg_dirty_slot(slot_ctx, 1 << 16 | 3, 0x10_0000, 0x4_0000, 0x8_0000);
        rvg_dirty_slot(slot_ctx, 4, 0x20_0000, 0x8_0000, 0);
    }
    true
}

unsafe extern "C-unwind" fn t_take(ctx: *mut c_void, offset: u64, length: u64, words: *mut u64) {
    Rec::push(ctx, format!("take {offset:#x} {length:#x}"));
    // SAFETY: the library's words, a bit a page of `length`.
    unsafe { *words |= 2 };
}

unsafe extern "C-unwind" fn t_flush(ctx: *mut c_void, timeout_ns: u64) -> bool {
    Rec::push(ctx, format!("flush {timeout_ns}"));
    true
}

unsafe extern "C-unwind" fn t_start(ctx: *mut c_void) {
    Rec::push(ctx, "start".into());
}

fn dirty_table(rec: &Arc<Rec>, fd: i32) -> RvgDirtySource {
    RvgDirtySource {
        struct_size: crate::sized::size_u32::<RvgDirtySource>(),
        kvm_vm_fd: fd,
        ctx: rec.ctx(),
        destroy: Some(t_destroy),
        start: Some(t_start),
        flush_vcpus: Some(t_flush),
        for_each_slot: Some(t_slots),
        take_host_writes: Some(t_take),
    }
}

/// A source's functions reach the host with its ctx, each slot's fields in order, and the
/// host's writes into the library's words.
#[test]
fn a_dirty_source_calls_the_host() {
    let rec = Rec::new();
    let d = dirty_table(&rec, 7);
    let table = Table::new(d.ctx, d.destroy);
    let source = dirty_source(&table, &d).unwrap();
    assert_eq!(source.kvm_vm_fd, 7);
    source.host.start();
    assert!(source.host.flush_vcpus(Duration::from_millis(4)));
    let mut slots = Vec::new();
    assert!(source.host.for_each_slot(&mut |s| slots.push(s)));
    assert_eq!(
        slots,
        [
            KvmSlot {
                slot: 1 << 16 | 3,
                guest_physical_address: 0x10_0000,
                length: 0x4_0000,
                host_offset: 0x8_0000,
            },
            KvmSlot {
                slot: 4,
                guest_physical_address: 0x20_0000,
                length: 0x8_0000,
                host_offset: 0,
            },
        ]
    );
    let mut words = [0u64; 2];
    source.host.take_host_writes(0x4_0000, 0x8_0000, &mut words);
    assert_eq!(words, [2, 0]);
    assert_eq!(
        rec.lines(),
        [
            "start",
            "flush 4000000",
            "for_each_slot",
            "take 0x40000 0x80000"
        ]
    );
    drop(source);
    assert!(rec.destroys().is_empty());
    drop(table);
    assert_eq!(rec.destroys().len(), 1);

    let d = RvgDirtySource {
        kvm_vm_fd: -1,
        ..dirty_table(&rec, 0)
    };
    assert!(dirty_source(&Table::new(d.ctx, d.destroy), &d).is_none());
    let d = RvgDirtySource {
        for_each_slot: None,
        ..dirty_table(&rec, 0)
    };
    assert!(dirty_source(&Table::new(d.ctx, d.destroy), &d).is_none());
}

/// A device over a memory map keeps its dirty source until its end; one without a map, a source
/// that is none and a refusal end it before the create returns, and a source the create cannot
/// read is refused, ending neither table.
#[test]
fn a_dirty_source_lives_with_its_device() {
    let p = test_platform();
    let rec = Rec::new();
    let src = Rec::new();
    let table = only_destroy(&rec);
    let range = fake_range(&p, 0x10_0000);
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    let d = dirty_table(&src, 9);
    let mut c = config(1, &table);
    c.memory_map = &map;
    c.dirty_source = &d;
    let (s, dev) = create(&c);
    assert_eq!(s, RVG_OK);
    assert!(src.destroys().is_empty());
    // SAFETY: the device made above.
    unsafe { rvg_device_release(dev) };
    assert_eq!(src.wait_destroys(1).len(), 1);
    assert!(src.lines().is_empty(), "no harvest was wanted");

    c.memory_map = std::ptr::null();
    let (s, dev) = create(&c);
    assert_eq!(s, RVG_OK);
    assert_eq!(
        src.destroys().len(),
        2,
        "no map: ended before the create returned"
    );
    // SAFETY: the device made above.
    unsafe { rvg_device_release(dev) };

    c.memory_map = &map;
    let none = RvgDirtySource { kvm_vm_fd: -1, ..d };
    c.dirty_source = &none;
    let (s, dev) = create(&c);
    assert_eq!(s, RVG_OK);
    assert_eq!(
        src.destroys().len(),
        3,
        "none: ended before the create returned"
    );
    // SAFETY: the device made above.
    unsafe { rvg_device_release(dev) };

    c.dirty_source = &d;
    c.display_port_count = 0;
    assert_eq!(create(&c).0, RVG_ERR_DISPLAY_PORT_COUNT);
    assert_eq!(src.destroys().len(), 4, "a refusal ends it");
    c.display_port_count = 1;

    let small = RvgDirtySource {
        struct_size: 4,
        ..d
    };
    c.dirty_source = &small;
    let before = rec.destroys().len();
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    assert_eq!(src.destroys().len(), 4);
    assert_eq!(rec.destroys().len(), before);

    c.dirty_source = &d;
    c.struct_size = 64;
    let (s, dev) = create(&c);
    assert_eq!(s, RVG_OK, "a third-version config has no source");
    assert_eq!(src.destroys().len(), 4);
    // SAFETY: the device made above.
    unsafe { rvg_device_release(dev) };
}
