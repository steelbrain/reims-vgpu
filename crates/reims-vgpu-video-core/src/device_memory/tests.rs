use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use super::*;
use crate::fourcc::FourCc;
use crate::runtime::{Runtime, StdRuntime};

/// A surface's memory of `length` bytes, with no dma-buf or view.
struct Memory {
    life: Mutex<Option<MemoryLife>>,
    identity: MemoryIdentity,
    length: usize,
}

impl Memory {
    fn new(length: usize) -> Self {
        let life = MemoryLife::new();
        Self {
            identity: life.identity(),
            life: Mutex::new(Some(life)),
            length,
        }
    }

    /// The guest unmaps it.
    fn unmap(&self) {
        self.life.lock().take();
    }
}

impl DeviceMemory for Memory {
    fn identity(&self) -> MemoryIdentity {
        self.identity.clone()
    }

    fn length(&self) -> usize {
        self.length
    }
}

/// Waits for `done`, five seconds at most.
fn eventually(done: impl Fn() -> bool) -> bool {
    let started = StdRuntime.now();
    while !done() {
        if StdRuntime.elapsed_since(started) > Duration::from_secs(5) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

#[test]
fn each_mapping_has_a_key_of_its_own_and_says_when_it_goes() {
    let (one, two) = (MemoryLife::new(), MemoryLife::new());
    let identity = one.identity();
    assert_ne!(identity.key(), two.identity().key());
    assert_eq!(identity.key(), one.identity().key());
    assert!(identity.is_live());
    let told = Arc::new(AtomicUsize::new(0));
    let counted = told.clone();
    identity.when_gone(move || {
        counted.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(told.load(Ordering::SeqCst), 0);
    drop(one);
    assert_eq!(told.load(Ordering::SeqCst), 1);
    assert!(!identity.is_live());
    // Told at once once it has gone.
    let counted = told.clone();
    identity.when_gone(move || {
        counted.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(told.load(Ordering::SeqCst), 2);
    // A mapping made later never takes a key that was used.
    assert!(MemoryLife::new().identity().key() > two.identity().key());
}

/// An import made for a mapping is kept, used again for the same mapping
/// and variant, and dropped on the reaper's thread once the mapping goes;
/// the next mapping over the same memory gets one of its own.
#[test]
fn an_import_lives_as_long_as_its_mapping() -> Result<(), ()> {
    struct Import(Arc<AtomicUsize>, Arc<Mutex<Option<String>>>);
    impl Drop for Import {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
            *self.1.lock() = std::thread::current().name().map(str::to_owned);
        }
    }
    let imports = DeviceImports::<Import>::new(&StdRuntime::shared(), "test", 8);
    let dropped = Arc::new(AtomicUsize::new(0));
    let thread = Arc::new(Mutex::new(None));
    let memory = Memory::new(64);
    let made = Arc::new(AtomicUsize::new(0));
    let get = |memory: &Memory, variant| {
        imports.get_or_make(
            &memory.identity(),
            variant,
            || {
                made.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ()>(Import(dropped.clone(), thread.clone()))
            },
            || (),
        )
    };
    let first = get(&memory, 0)?;
    let again = get(&memory, 0)?;
    assert!(Arc::ptr_eq(&first, &again));
    let other = get(&memory, 1)?;
    assert!(!Arc::ptr_eq(&first, &other));
    assert_eq!((made.load(Ordering::SeqCst), imports.len()), (2, 2));
    drop((first, again, other));
    memory.unmap();
    assert!(eventually(|| imports.is_empty()));
    assert!(eventually(|| dropped.load(Ordering::SeqCst) == 2));
    assert_eq!(thread.lock().as_deref(), Some("reims-vgpu-video.imports"));
    assert_eq!(imports.dropped(), 2);
    // A gone mapping gets none.
    assert_eq!(get(&memory, 0).err(), Some(()));
    let remapped = Memory::new(64);
    assert!(get(&remapped, 0).is_ok());
    assert_eq!(made.load(Ordering::SeqCst), 3);
    Ok(())
}

/// An import still in use when its mapping goes lives until its user is
/// done with it.
#[test]
fn an_import_in_use_outlives_its_entry() -> Result<(), ()> {
    let imports = DeviceImports::<Vec<u8>>::new(&StdRuntime::shared(), "test", 8);
    let memory = Memory::new(64);
    let import = imports.get_or_make(&memory.identity(), 0, || Ok::<_, ()>(vec![7; 4]), || ())?;
    memory.unmap();
    assert!(eventually(|| imports.is_empty()));
    assert_eq!(*import, vec![7; 4]);
    Ok(())
}

/// Past its capacity the import used longest ago goes; a make that fails
/// keeps nothing.
#[test]
fn imports_past_the_capacity_drop_the_oldest() -> Result<(), ()> {
    let imports = DeviceImports::<u32>::new(&StdRuntime::shared(), "test", 2);
    let memories: Vec<Memory> = (0..3).map(|_| Memory::new(64)).collect();
    let get = |at: usize| {
        imports.get_or_make(
            &memories[at].identity(),
            0,
            || Ok::<_, ()>(u32::try_from(at).unwrap_or(u32::MAX)),
            || (),
        )
    };
    assert_eq!(*get(0)?, 0);
    assert_eq!(*get(1)?, 1);
    // 0 is used again, so 1 is the oldest when 2 comes.
    assert_eq!(*get(0)?, 0);
    assert_eq!(*get(2)?, 2);
    assert_eq!(imports.len(), 2);
    assert_eq!(imports.dropped(), 1);
    let failed = imports.get_or_make(&memories[1].identity(), 0, || Err("no"), || "gone");
    assert_eq!(failed.err(), Some("no"));
    assert_eq!(imports.len(), 2);
    Ok(())
}

fn nv12(width: u32, height: u32, rows: [usize; 2], offsets: [usize; 2]) -> SurfaceLayout {
    SurfaceLayout {
        pixel_format: FourCc::new(*b"420v"),
        width,
        height,
        planes: vec![
            PlaneLayout {
                offset: offsets[0],
                bytes_per_row: rows[0],
                width,
                height,
            },
            PlaneLayout {
                offset: offsets[1],
                bytes_per_row: rows[1],
                width: width.div_ceil(2),
                height: height.div_ceil(2),
            },
        ],
        bytes_per_row: 0,
    }
}

#[test]
fn a_target_is_its_layout_checked_against_its_memory() -> Result<(), String> {
    let memory = Memory::new(64 * 48 + 4096 + 64 * 24);
    let layout = nv12(64, 48, [64, 64], [0, 4096]);
    let target = GuestTarget::new(&layout, &memory)?;
    assert_eq!(target.pixel_format(), PixelFormat::Nv12VideoRange);
    let [luma, chroma] = target.planes();
    assert_eq!((luma.offset, luma.bytes_per_row, luma.rows), (0, 64, 48));
    assert_eq!(
        (chroma.offset, chroma.rows, chroma.row_bytes),
        (4096, 24, 64)
    );
    assert_eq!(chroma.end_of(24, 64), Some(4096 + 64 * 24));
    assert_eq!(luma.end_of(0, 64), Some(0));
    assert_eq!(target.written_size(80, 40), (64, 40));

    // 16-bit samples: a row of 64 needs 128 bytes.
    let mut p010 = nv12(64, 48, [128, 128], [0, 8192]);
    p010.pixel_format = FourCc::new(*b"x420");
    let memory = Memory::new(8192 + 128 * 24);
    assert!(GuestTarget::new(&p010, &memory).is_ok());
    let short = nv12(64, 48, [128, 64], [0, 8192]);
    let mut short = short;
    short.pixel_format = FourCc::new(*b"x420");
    assert!(
        GuestTarget::new(&short, &memory)
            .err()
            .is_some_and(|e| e.contains("chroma"))
    );
    Ok(())
}

#[test]
fn a_target_past_its_memory_or_out_of_shape_is_refused() -> Result<(), String> {
    let memory = Memory::new(64 * 48 + 64 * 24);
    // The chroma plane's last row ends a byte past the memory.
    let past = nv12(64, 48, [64, 64], [0, 64 * 48 + 1]);
    assert!(
        GuestTarget::new(&past, &memory)
            .err()
            .is_some_and(|e| e.contains("past"))
    );
    // Overlapping planes.
    let over = nv12(64, 48, [64, 64], [0, 64 * 47]);
    assert!(
        GuestTarget::new(&over, &memory)
            .err()
            .is_some_and(|e| e.contains("inside"))
    );
    // Rows shorter than their samples, and planes shorter than the picture.
    let narrow = nv12(64, 48, [63, 64], [0, 64 * 48]);
    assert!(GuestTarget::new(&narrow, &memory).is_err());
    let mut low = nv12(64, 48, [64, 64], [0, 64 * 48]);
    low.planes[1].height = 23;
    assert!(GuestTarget::new(&low, &memory).is_err());
    // An odd width's chroma rows hold a pair for the last sample too.
    let odd = nv12(63, 48, [64, 62], [0, 64 * 48]);
    assert!(
        GuestTarget::new(&odd, &memory)
            .err()
            .is_some_and(|e| e.contains("chroma"))
    );
    // Formats and plane counts no device writes.
    let mut bgra = nv12(64, 48, [64, 64], [0, 64 * 48]);
    bgra.pixel_format = FourCc::new(*b"BGRA");
    assert!(GuestTarget::new(&bgra, &memory).is_err());
    let mut one = nv12(64, 48, [64, 64], [0, 64 * 48]);
    one.planes.truncate(1);
    assert!(GuestTarget::new(&one, &memory).is_err());
    // Sizes that overflow.
    let huge = nv12(64, u32::MAX, [usize::MAX / 2, 64], [0, 64]);
    assert!(GuestTarget::new(&huge, &memory).is_err());
    Ok(())
}
