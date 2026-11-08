//! Fake: the subprocess death and hang harness, `death_test!` and `hang_test!`.
//!
//! In-process capture (`testkit::capture`) turns a death into a panic, and the unwinding runs every
//! `Drop` on the way out: a lock guard releases, a blocked thread goes on, and nothing shows
//! what a real death leaves behind. Here a test runs twice:
//!
//! - **the parent** (the test as `cargo test` runs it) re-executes the current test binary,
//!   filtered to this one test, with the marker [`ENV_CHILD`] set to the test's name and
//!   [`ENV_FILE`] naming a scratch file. It waits for the child, reads what the child left in
//!   the file and on stderr, and checks it: a [`Report`];
//! - **the child** turns capture off (`testkit::capture::set_capture_enabled(false)`), so `fatal`
//!   writes its stderr line and calls `abort()` as production does: no panic, no `Drop`. It
//!   then runs the body.
//!
//! The scratch file is mapped `MAP_SHARED` in the child and written in place, so what the
//! child wrote before it died survives its `SIGABRT`:
//! - an append-only trace of the ops the body and its fakes record ([`trace`], [`Child::op`]),
//!   and a death marker a signal handler appends on the dying thread. Each record fills a
//!   fixed slot, so a record a thread had reserved but not written when the child ended hides
//!   none after it;
//! - guest RAM and raw regions the body shares ([`Child::share_ram`], [`Child::share_region`]):
//!   their pages are remapped in place onto the file, at the same addresses.
//!
//! `Died { id, op, thread }` comes from the child's stderr death line (the id), the marker's
//! thread, and the trace's last op before the marker. When deaths race, the harness does not pair a
//! line with a thread ([`Died::raced`]); the lines other aborts print count as deaths
//! ([`parse_abort_lines`]). The parent asserts the outcome, that the body neither returned nor
//! panicked before it (a death test fails when `f` returns first, and fails with `f`'s panic when
//! `f` panics first), and that no thread recorded an op after the marker ("nothing after"): one
//! blocked behind a death never records. A test with a thread that records freely beside the death,
//! which as in production runs on for a moment while the process is torn down, judges only the
//! dying thread ([`After::DyingThread`]).
//!
//! The handlers chain to the actions they replace, on the alternate signal stack: std's
//! stack-overflow handler still reports an overflow and aborts, as it does without the harness.
//!
//! No product change: the child uses the capture's public switch, and this module is
//! test-only (`cfg(any(test, feature = "testkit"))`).

// Architecture 17.3, test harness: `mmap` over live pages, signal handlers and `_exit`.
#![allow(unsafe_code)]
// A test harness reports failure by panicking, and times its child in real time (17.4:
// test-only code, as `testkit::capture`).
#![allow(clippy::panic, clippy::disallowed_methods)]

use std::ffi::{OsString, c_char, c_int, c_void};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd as _;
use std::os::unix::fs::FileExt as _;
use std::os::unix::process::ExitStatusExt as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::num::{ToU64, ToUsize, Truncate};
use crate::raise::death::DeathSite;
use crate::sync::Mutex;
use crate::testkit::capture::DeathSlot;
use crate::testkit::capture::VirtualTime;
use crate::testkit::capture::set_capture_enabled;
use crate::testkit::capture::with_slot;
use crate::testkit::guest_ram::{GuestRam, RawRegion};

/// The environment marker: set to the full test name in the child that runs it.
pub const ENV_CHILD: &str = "REIMS_VGPU_DEATH_CHILD";
/// The scratch file the child maps `MAP_SHARED` (header, trace, shared RAM).
pub const ENV_FILE: &str = "REIMS_VGPU_DEATH_FILE";

/// How long the parent waits for the child, in real time, by default.
pub const CHILD_TIMEOUT: Duration = Duration::from_secs(60);
/// Real time a hang is given to settle before and after virtual time advances
/// ([`Child::hang`]).
pub const HANG_SETTLE: Duration = Duration::from_millis(200);
/// The child's exit status when [`Child::hang`] judged a hang.
pub const HANG_EXIT: i32 = 86;

// The scratch file: a header page, the trace (fixed-size slots), then the shared RAM.
const MAGIC: u64 = 0x4854_4145_4447_5652; // "RVGDEATH", little-endian
const H_MAGIC: usize = 0;
const H_STATE: usize = 8;
const H_OVERFLOW: usize = 12;
const H_BODY_TID: usize = 16;
/// The number of trace slots reserved so far.
const H_CURSOR: usize = 24;
const H_DIR_COUNT: usize = 32;
const DIR_OFF: usize = 64;
const DIR_ENTRY: usize = 64;
const DIR_TAG: usize = 32;
const TRACE_OFF: usize = 0x4000;
const DIR_MAX: usize = (TRACE_OFF - DIR_OFF) / DIR_ENTRY;
const SLOT: usize = 512;
const SLOTS: usize = 0x1_0000;
const TRACE_CAP: usize = SLOT * SLOTS;
const RAM_OFF: usize = TRACE_OFF + TRACE_CAP;

const STATE_SPAWNED: u32 = 0;
const STATE_STARTED: u32 = 1;
const STATE_RETURNED: u32 = 2;
const STATE_HUNG: u32 = 3;
const STATE_PANICKED: u32 = 4;

const DIR_PA: u32 = 1;
const DIR_TAGGED: u32 = 2;

// A trace record fills one slot: committed u32 (stored last), kind u8, name_len u8, text_len u16,
// aux u32, flags u32, tid u64, then the thread name and the text. The slots are fixed, so a slot
// reserved but never written (its thread stopped between the reservation and the commit when the
// child ended) hides no later record.
const REC_HEAD: usize = 24;
const NAME_MAX: usize = 64;
/// The longest op text a trace record keeps whole. A longer one is cut to this length and the
/// [`Report`] says so ([`Report::truncated`]); `death_test!` and `hang_test!` then fail.
pub const TEXT_MAX: usize = SLOT - REC_HEAD - NAME_MAX;
const F_TRUNCATED: u32 = 1;
const K_OP: u8 = 1;
const K_DEATH: u8 = 2;
const K_HUNG: u8 = 3;
const K_RETURNED: u8 = 4;
const K_ADVANCED: u8 = 5;
const K_PANICKED: u8 = 6;

mod sys;

// ---------------------------------------------------------------------------------------------
// The child.

/// The header and trace mapping, once this process is a child (null otherwise). The signal
/// handler reads it, so it is a plain atomic.
static SHARED: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());

struct ChildState {
    file: File,
    alloc: Mutex<Alloc>,
}

#[derive(Default)]
struct Alloc {
    /// The next free offset for shared RAM in the file.
    next: u64,
    /// Every span remapped so far: `(start, end, file offset)`.
    spans: Vec<(usize, usize, u64)>,
    /// Directory entries written.
    dir: usize,
}

static CHILD: OnceLock<ChildState> = OnceLock::new();

/// Whether this process is a child the harness started.
pub fn in_child() -> bool {
    !SHARED.load(Ordering::Acquire).is_null()
}

/// The calling thread's system id and name, without allocating (the signal handler calls it).
fn this_thread() -> (u64, [u8; NAME_MAX], usize) {
    let tid = sys::current_thread_id();
    let mut name = [0u8; NAME_MAX];
    // SAFETY: `pthread_self()` is the calling thread; the call writes within the buffer given.
    // For the calling thread it takes no lock (async-signal-safe in practice).
    unsafe {
        sys::pthread_getname_np(
            sys::pthread_self(),
            name.as_mut_ptr().cast::<c_char>(),
            NAME_MAX,
        );
    }
    let len = name.iter().position(|b| *b == 0).unwrap_or(NAME_MAX);
    (tid, name, len)
}

/// A header field as an atomic. `off` is a constant inside the header page.
fn header_u32(base: *mut u8, off: usize) -> &'static AtomicU32 {
    // SAFETY: `base` maps the header for the rest of the process's life (never unmapped), `off`
    // is 4-aligned inside it, and every access to the field is atomic.
    unsafe { AtomicU32::from_ptr(base.wrapping_add(off).cast::<u32>()) }
}

/// As [`header_u32`], for a 64-bit field.
fn header_u64(base: *mut u8, off: usize) -> &'static AtomicU64 {
    // SAFETY: as `header_u32`; `off` is 8-aligned.
    unsafe { AtomicU64::from_ptr(base.wrapping_add(off).cast::<u64>()) }
}

/// Copies `bytes` into `buf` at `at`, if they fit.
fn put(buf: &mut [u8], at: usize, bytes: &[u8]) {
    if let Some(dst) = buf.get_mut(at..at.saturating_add(bytes.len())) {
        dst.copy_from_slice(bytes);
    }
}

/// Appends one record to the trace. Async-signal-safe: atomics and copies into the mapping.
fn append(kind: u8, aux: u32, thread: (u64, &[u8]), text: &[u8]) {
    let base = SHARED.load(Ordering::Acquire);
    if base.is_null() {
        return;
    }
    let (tid, name) = thread;
    let name = name.get(..name.len().min(NAME_MAX)).unwrap_or_default();
    let flags = if text.len() > TEXT_MAX {
        F_TRUNCATED
    } else {
        0
    };
    let text = text.get(..text.len().min(TEXT_MAX)).unwrap_or_default();
    let slot = header_u64(base, H_CURSOR).fetch_add(1, Ordering::AcqRel);
    let slot = slot.to_usize();
    if slot >= SLOTS {
        header_u32(base, H_OVERFLOW).store(1, Ordering::Release);
        return;
    }
    let rec = base
        .wrapping_add(TRACE_OFF)
        .wrapping_add(slot.wrapping_mul(SLOT));
    let mut head = [0u8; REC_HEAD];
    put(&mut head, 4, &[kind, name.len().to_u64().low_u8()]);
    put(&mut head, 6, &text.len().to_u64().low_u16().to_ne_bytes());
    put(&mut head, 8, &aux.to_ne_bytes());
    put(&mut head, 12, &flags.to_ne_bytes());
    put(&mut head, 16, &tid.to_ne_bytes());
    // SAFETY: slot `slot` lies inside the trace area (checked above) and was reserved for this
    // record alone by the `fetch_add`; name and text fit in it (`NAME_MAX`, `TEXT_MAX`); the
    // sources are distinct buffers. The committed word at the slot's start is stored last.
    unsafe {
        std::ptr::copy_nonoverlapping(
            head.as_ptr().wrapping_add(4),
            rec.wrapping_add(4),
            REC_HEAD.saturating_sub(4),
        );
        std::ptr::copy_nonoverlapping(name.as_ptr(), rec.wrapping_add(REC_HEAD), name.len());
        std::ptr::copy_nonoverlapping(
            text.as_ptr(),
            rec.wrapping_add(REC_HEAD).wrapping_add(name.len()),
            text.len(),
        );
        AtomicU32::from_ptr(rec.cast::<u32>()).store(1, Ordering::Release);
    }
}

/// Appends a record for the calling thread.
fn append_here(kind: u8, aux: u32, text: &[u8]) {
    let (tid, name, len) = this_thread();
    append(kind, aux, (tid, name.get(..len).unwrap_or_default()), text);
}

/// Records one op in the child's trace, on the calling thread. Outside a child it does
/// nothing, so fakes and host blocks may call it unconditionally.
pub fn trace(op: &str) {
    if in_child() {
        append_here(K_OP, 0, op.as_bytes());
    }
}

/// The actions the harness's handler replaced, by signal: the handler (`SIG_DFL` is 0) and its
/// flags. Stored before the handler is installed; the handler only reads them.
static PREV_HANDLER: [AtomicUsize; sys::NSIG] =
    [const { AtomicUsize::new(sys::SIG_DFL) }; sys::NSIG];
static PREV_FLAGS: [AtomicU32; sys::NSIG] = [const { AtomicU32::new(0) }; sys::NSIG];

/// The handler for the signals that end a child: it marks the death on the dying thread and
/// leaves the signal to do what it would have done without the harness.
///
/// - With no earlier handler (`SIG_DFL`), it appends the death marker, restores the default
///   action and re-raises: the process dies of the signal when the handler returns.
/// - With one (std installs its stack-overflow handler for `SIGSEGV` and `SIGBUS`, on an
///   alternate stack), it calls that handler first, as the signal would have. std's reports an
///   overflow and aborts, so the death is marked by the `SIGABRT` it raises; for any other fault
///   it restores the default action and returns. If it returns, the handler appends the marker
///   and returns, and the faulting instruction runs again and ends the process.
extern "C" fn on_fatal_signal(sig: c_int, info: *mut c_void, ctx: *mut c_void) {
    let i = usize::try_from(sig).unwrap_or(0);
    let prev = PREV_HANDLER
        .get(i)
        .map_or(sys::SIG_DFL, |h| h.load(Ordering::Acquire));
    let flags = PREV_FLAGS.get(i).map_or(0, |f| f.load(Ordering::Acquire));
    if prev == sys::SIG_DFL || prev == sys::SIG_IGN {
        append_here(K_DEATH, sig.cast_unsigned(), b"");
        let _ = set_action(sig, sys::Handler { raw: sys::SIG_DFL }, 0);
        // `raise` pends (the signal is blocked while its handler runs) and is delivered, with
        // the default action, when the handler returns.
        let _ = sys::raise(sig);
        return;
    }
    let chained = sys::Handler { raw: prev };
    // SAFETY: `prev` is the handler `sigaction` reported for `sig` (neither `SIG_DFL` nor
    // `SIG_IGN`), called the way its flags say the kernel would have called it.
    unsafe {
        if flags & sys::SA_SIGINFO.cast_unsigned() != 0 {
            (chained.action)(sig, info, ctx);
        } else {
            (chained.handler)(sig);
        }
    }
    append_here(K_DEATH, sig.cast_unsigned(), b"");
}

/// Sets `sig`'s action to `handler` (on the alternate stack, with `flags`), and returns the
/// action it replaced.
fn set_action(sig: c_int, handler: sys::Handler, flags: c_int) -> Option<sys::SigAction> {
    let act = sys::SigAction::new(handler, flags | sys::SA_ONSTACK);
    let mut old = sys::SigAction::new(sys::Handler { raw: sys::SIG_DFL }, 0);
    // SAFETY: both structs are the C library's `struct sigaction`, and live across the call.
    let r = unsafe { sys::sigaction(sig, &raw const act, &raw mut old) };
    (r == 0).then_some(old)
}

/// Installs [`on_fatal_signal`] for `sig`, keeping the action it replaces for the handler to
/// chain to. A signal the process ignores is left ignored.
fn arm(sig: c_int) {
    let mut cur = sys::SigAction::new(sys::Handler { raw: sys::SIG_DFL }, 0);
    // SAFETY: a query: `act` is null and `old` is a live `struct sigaction`.
    if unsafe { sys::sigaction(sig, std::ptr::null(), &raw mut cur) } != 0 {
        return;
    }
    // SAFETY: every bit pattern of the union is a valid `usize`.
    let raw = unsafe { cur.handler.raw };
    if raw == sys::SIG_IGN {
        return;
    }
    let i = usize::try_from(sig).unwrap_or(0);
    if let (Some(h), Some(f)) = (PREV_HANDLER.get(i), PREV_FLAGS.get(i)) {
        h.store(raw, Ordering::Release);
        f.store(cur.flags.cast_unsigned(), Ordering::Release);
    }
    let _ = set_action(
        sig,
        sys::Handler {
            action: on_fatal_signal,
        },
        sys::SA_SIGINFO,
    );
}

fn page_size() -> usize {
    usize::try_from(sys::getpagesize()).unwrap_or(0x4000)
}

/// The child's side of [`run`]: [`enter`], run `body`, [`Child::finish`]. A body that panics
/// unwinds past `child`, whose `Drop` records it.
fn child_main(body: impl FnOnce(&Child)) {
    let child = enter();
    body(&child);
    child.finish();
}

/// Makes this process a child of [`spawn_command`]: maps the scratch file [`ENV_FILE`] names, turns
/// the capture off (a death aborts as in production), gives the calling thread an alternate signal
/// stack its nested handlers fit on (`sys::widen_alternate_stack`), arms the death marker's
/// signal handlers, and records the calling thread as the body's. A child that is not a libtest test (a tool's binary)
/// calls it itself, holds the [`Child`] on that thread while its body runs, and calls
/// [`Child::finish`] if the body returns: a body that panics unwinds past the `Child`, whose `Drop`
/// records it (the test then fails with `f`'s panic).
pub fn enter() -> Child {
    let Some(path) = std::env::var_os(ENV_FILE) else {
        panic!("death harness: {ENV_FILE} is not set");
    };
    let file = match OpenOptions::new().read(true).write(true).open(&path) {
        Ok(f) => f,
        Err(e) => panic!("death harness: cannot open {}: {e}", path.display()),
    };
    // SAFETY: a fresh shared mapping of the file's first `RAM_OFF` bytes (the parent sized the
    // file), at an address the kernel picks. It is never unmapped.
    let p = unsafe {
        sys::mmap(
            std::ptr::null_mut(),
            RAM_OFF,
            sys::PROT_READ | sys::PROT_WRITE,
            sys::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if p.addr() == usize::MAX || p.is_null() {
        panic!("death harness: cannot map {}", path.display());
    }
    let base = p.cast::<u8>();
    if header_u64(base, H_MAGIC).load(Ordering::Acquire) != MAGIC {
        panic!("death harness: {} is not a harness file", path.display());
    }
    if CHILD
        .set(ChildState {
            file,
            alloc: Mutex::new(Alloc {
                next: RAM_OFF.to_u64(),
                ..Alloc::default()
            }),
        })
        .is_err()
    {
        panic!("death harness: one body per child");
    }
    SHARED.store(base, Ordering::Release);
    set_capture_enabled(false);
    sys::widen_alternate_stack();
    for sig in [
        sys::SIGABRT,
        sys::SIGTRAP,
        sys::SIGILL,
        sys::SIGSEGV,
        sys::SIGBUS,
    ] {
        arm(sig);
    }
    let (tid, _, _) = this_thread();
    header_u64(base, H_BODY_TID).store(tid, Ordering::Release);
    header_u32(base, H_STATE).store(STATE_STARTED, Ordering::Release);
    Child { _p: () }
}

/// The body's handle on the child: its trace and its shared memory. Dropped while its thread
/// unwinds, it records that the body panicked ([`RecordKind::Panicked`]).
#[derive(Debug)]
pub struct Child {
    _p: (),
}

impl Drop for Child {
    /// A body that panics unwinds past its `Child`: its outcome is `Panicked` once it has unwound,
    /// and a death test then fails with `f`'s panic unless a death came first. The record goes
    /// down after the body's own locals dropped.
    fn drop(&mut self) {
        if std::thread::panicking() {
            append_here(K_PANICKED, 0, b"");
            let base = SHARED.load(Ordering::Acquire);
            if !base.is_null() {
                header_u32(base, H_STATE).store(STATE_PANICKED, Ordering::Release);
            }
        }
    }
}

impl Child {
    /// Records that the body returned: the parent reports [`Outcome::Returned`] if the child
    /// then exits 0.
    pub fn finish(self) {
        append_here(K_RETURNED, 0, b"");
        let base = SHARED.load(Ordering::Acquire);
        header_u32(base, H_STATE).store(STATE_RETURNED, Ordering::Release);
        // The body returned: nothing for `Drop` to record, even if a caller finishes while
        // unwinding.
        std::mem::forget(self);
    }

    /// Records one op on the calling thread ([`trace`]).
    pub fn op(&self, op: &str) {
        trace(op);
    }

    /// Makes every range of `ram` shared with the parent, in place: the parent reads it by
    /// guest-physical address ([`Report::ram`]) after the child ends. Call it before any other
    /// thread touches the ranges' pages; ranges added later are not shared.
    pub fn share_ram(&self, ram: &GuestRam) {
        for r in ram.ranges() {
            self.share(
                r.region.addr().addr(),
                r.region.len(),
                DIR_PA,
                r.physical_address,
                "",
            );
        }
    }

    /// Makes `region` shared with the parent under `tag` ([`Report::region`]), in place.
    pub fn share_region(&self, tag: &str, region: &RawRegion) {
        self.share(region.addr().addr(), region.len(), DIR_TAGGED, 0, tag);
    }

    /// Remaps the pages under `addr..addr+len` onto the file, keeping their bytes, and writes
    /// a directory entry for the parent.
    fn share(&self, addr: usize, len: u64, kind: u32, key: u64, tag: &str) {
        let (Some(st), false) = (CHILD.get(), SHARED.load(Ordering::Acquire).is_null()) else {
            panic!("death harness: share outside a child");
        };
        let base = SHARED.load(Ordering::Acquire);
        if tag.len() > DIR_TAG {
            panic!("death harness: tag {tag:?} is longer than {DIR_TAG} bytes");
        }
        let pg = page_size();
        let ulen = len.to_usize();
        let start = addr & !(pg.wrapping_sub(1));
        let Some(end) = addr
            .checked_add(ulen)
            .and_then(|e| e.checked_add(pg.wrapping_sub(1)))
            .map(|e| e & !(pg.wrapping_sub(1)))
        else {
            panic!("death harness: region at {addr:#x} overflows");
        };
        let mut a = st.alloc.lock();
        let map_off = if len == 0 {
            0
        } else if let Some(&(_, _, off)) = a.spans.iter().find(|s| s.0 == start && s.1 == end) {
            off
        } else {
            if a.spans.iter().any(|s| s.0 < end && start < s.1) {
                panic!("death harness: {addr:#x}+{len:#x} overlaps a region already shared");
            }
            let span = end.wrapping_sub(start);
            let off = a.next;
            a.next = off.saturating_add(span.to_u64());
            if let Err(e) = st.file.set_len(a.next) {
                panic!("death harness: cannot grow the file: {e}");
            }
            let first = std::ptr::with_exposed_provenance::<u8>(start);
            // SAFETY: `start..end` are whole pages that hold the region, so they are mapped;
            // the copy keeps the bytes of any neighbouring allocation in those pages too, which
            // the remap below must not lose.
            let now = unsafe { std::slice::from_raw_parts(first, span) };
            if let Err(e) = st.file.write_all_at(now, off) {
                panic!("death harness: cannot copy the region: {e}");
            }
            // SAFETY: replaces those same pages, at the same address, with a shared mapping of
            // the copy; every pointer into them stays valid and sees the same bytes.
            let p = unsafe {
                sys::mmap(
                    std::ptr::with_exposed_provenance_mut::<c_void>(start),
                    span,
                    sys::PROT_READ | sys::PROT_WRITE,
                    sys::MAP_SHARED | sys::MAP_FIXED,
                    st.file.as_raw_fd(),
                    off.cast_signed(),
                )
            };
            if p.addr() != start {
                panic!("death harness: cannot remap {start:#x}+{span:#x}");
            }
            a.spans.push((start, end, off));
            off
        };
        let first_byte = map_off.saturating_add(addr.wrapping_sub(start).to_u64());
        let i = a.dir;
        if i >= DIR_MAX {
            panic!("death harness: more than {DIR_MAX} shared regions");
        }
        a.dir = i.wrapping_add(1);
        let e = base.wrapping_add(DIR_OFF.saturating_add(i.saturating_mul(DIR_ENTRY)));
        let mut entry = [0u8; DIR_ENTRY];
        let fields: [(usize, &[u8]); 5] = [
            (0, &kind.to_ne_bytes()),
            (4, &tag.len().low_u32().to_ne_bytes()),
            (8, &key.to_ne_bytes()),
            (16, &len.to_ne_bytes()),
            (24, &first_byte.to_ne_bytes()),
        ];
        for (at, bytes) in fields.into_iter().chain([(32, tag.as_bytes())]) {
            if let Some(dst) = entry.get_mut(at..at.saturating_add(bytes.len())) {
                dst.copy_from_slice(bytes);
            }
        }
        // SAFETY: entry `i` lies inside the header's directory (`i < DIR_MAX`), and only this
        // thread writes it (under `alloc`); the parent reads it after the child ends.
        unsafe { std::ptr::copy_nonoverlapping(entry.as_ptr(), e, DIR_ENTRY) };
        header_u32(base, H_DIR_COUNT).store(a.dir.low_u32(), Ordering::Release);
    }

    /// A hang test's child: runs `f` on its own thread (`hang-body`), lets it settle for
    /// [`HANG_SETTLE`], advances `clock` by `d`, and lets it settle again. If `f` is still blocked,
    /// the hang is judged: a hang marker is recorded for `f`'s thread and the child exits at once
    /// with [`HANG_EXIT`], leaving `f` blocked. If `f` returned, this returns (and a panic in `f`
    /// resumes here), so the parent sees the body return.
    ///
    /// `f` runs in a fresh `DeathSlot`.
    pub fn hang(&self, clock: &dyn VirtualTime, d: Duration, f: impl FnOnce() + Send + 'static) {
        let tid = std::sync::Arc::new(AtomicU64::new(0));
        let tid2 = tid.clone();
        let slot = DeathSlot::new();
        let spawned = std::thread::Builder::new()
            .name("hang-body".into())
            .spawn(move || {
                tid2.store(this_thread().0, Ordering::Release);
                with_slot(Some(slot), f)
            });
        let h = match spawned {
            Ok(h) => h,
            Err(e) => panic!("death harness: cannot spawn: {e}"),
        };
        let finished = |h: std::thread::JoinHandle<()>| {
            if let Err(p) = h.join() {
                std::panic::resume_unwind(p)
            }
        };
        std::thread::sleep(HANG_SETTLE);
        if h.is_finished() {
            return finished(h);
        }
        clock.advance(d);
        append_here(K_ADVANCED, 0, b"");
        std::thread::sleep(HANG_SETTLE);
        if h.is_finished() {
            return finished(h);
        }
        let base = SHARED.load(Ordering::Acquire);
        if !base.is_null() {
            append(K_HUNG, 0, (tid.load(Ordering::Acquire), b"hang-body"), b"");
            header_u32(base, H_STATE).store(STATE_HUNG, Ordering::Release);
        }
        sys::_exit(HANG_EXIT)
    }
}

// ---------------------------------------------------------------------------------------------
// The parent.

/// What the parent checks before the test's own check runs ([`Report::expect`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    /// The child died before the body returned or panicked, and nothing was recorded after the
    /// death (`death_test!`).
    Death,
    /// The child hung before the body returned or panicked, and nothing was recorded after the
    /// hang was judged (`hang_test!`).
    Hang,
    /// Any outcome: the check decides (the harness's own tests).
    Any,
}

/// Which threads "nothing after" judges ([`Options::after`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum After {
    /// Every thread (the default): no op is recorded after the death or hang marker. A thread
    /// blocked behind a death never records, so an op after the marker is an effect the death
    /// should have stopped: a lock released on the way to the abort, a host call, a stamp.
    #[default]
    EveryThread,
    /// Only the dying or hung thread, for a test with a thread that records freely beside the
    /// death (a ticker). As in production, such a thread runs on for a moment while the process
    /// is torn down, so its ops after the marker are a race, not an effect of the death.
    DyingThread,
}

/// How the parent runs the child and judges it.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Real time the child may run before the parent kills it ([`Outcome::TimedOut`]).
    pub timeout: Duration,
    /// Which threads "nothing after" judges.
    pub after: After,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            timeout: CHILD_TIMEOUT,
            after: After::EveryThread,
        }
    }
}

/// A thread of the child, as its trace records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadTag {
    /// The system thread id (`pthread_threadid_np`), 0 if unknown.
    pub tid: u64,
    /// The thread's name, empty if it has none.
    pub name: String,
    /// Whether it is the thread that ran the body.
    pub body: bool,
}

impl ThreadTag {
    /// The thread's name, if it has one.
    pub fn name(&self) -> Option<&str> {
        (!self.name.is_empty()).then_some(self.name.as_str())
    }
}

/// A death's identity, from the child's stderr death line (`raise::death::fatal`'s diagnostic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeathId {
    /// The `DeathKind`, as `Debug` prints it.
    pub kind: String,
    /// The site's file (or selector).
    pub file: String,
    /// The site's line.
    pub line: u32,
}

impl DeathId {
    /// Whether this is `site`.
    pub fn is(&self, site: DeathSite) -> bool {
        self.kind == format!("{:?}", site.kind) && self.file == site.file && self.line == site.line
    }
}

/// How the child died.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Died {
    /// The site: the child's death line, when it printed one and no other death raced it.
    /// `None` for a death with no death line (a crash, a bare `abort()`, another abort:
    /// [`Died::abort_lines`]), and when deaths raced ([`Died::raced`]): which line a thread
    /// printed is not recorded, so the harness never pairs one death's site with another's
    /// thread.
    pub id: Option<DeathId>,
    /// Whether more than one death raced, so the harness cannot tell which death line is the
    /// marked thread's and pairs none: the child printed more than one line a death prints (a
    /// death line, or the line of another abort: [`Died::abort_lines`]), marked more than one
    /// thread dying, or printed one while the marked thread died of a signal other than
    /// `SIGABRT` (every abort that prints a line raises it). Then `id` is `None`, `ids` and
    /// `abort_lines` hold every line and `thread` is the first thread marked. One race it
    /// cannot see: a thread that printed its line but was not marked yet, beside an abort that
    /// prints no line the harness knows on another thread: a bare `abort()` (the code has
    /// none: it aborts only in `fatal` and the D9 panic hooks, which print) or one the
    /// platform makes with a line of its own.
    pub raced: bool,
    /// Every death line the child printed, in order.
    pub ids: Vec<DeathId>,
    /// Every line another abort printed on its way ([`parse_abort_lines`]), in order: each is a
    /// death with no site, which [`Died::raced`] counts with `ids`.
    pub abort_lines: Vec<String>,
    /// The trace's last op before the death marker, on any thread (before the trace's end if
    /// no marker was recorded; `None` if there is no op). It is the op that preceded the death
    /// when every thread that records is sequenced with it, as in a driven program. It is not
    /// the dying thread's own last op: a pooled worker runs many queues' blocks, so the op a
    /// worker recorded last may be an earlier, unrelated block's, and which worker runs a
    /// block is a race. A thread recording freely beside the death makes this a race too; there,
    /// read one thread's last op with [`Report::last_op_before_end`].
    pub op: Option<String>,
    /// The thread that died: the one the first death marker names, which the signal hit
    /// (unknown, tid 0, with no death marker).
    pub thread: ThreadTag,
    /// The signal that ended the child.
    pub signal: i32,
}

/// Where the child hung.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hung {
    /// The last op recorded before the hang was judged, on any thread (every thread working
    /// for `f` has settled by then).
    pub op: Option<String>,
    /// The blocked thread (`hang-body`).
    pub thread: ThreadTag,
}

/// How the child ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The body returned.
    Returned,
    /// The child died.
    Died(Died),
    /// [`Child::hang`] judged a hang.
    Hung(Hung),
    /// The child exited otherwise: a test failure (101) or a harness error.
    Failed {
        /// The exit code.
        code: Option<i32>,
    },
    /// The child ran no test: the filter matched nothing.
    NotRun,
    /// The child outlived [`Options::timeout`] and was killed.
    TimedOut,
}

/// The kind of a trace record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    /// An op the body or a fake recorded.
    Op,
    /// The death marker, with the signal.
    Death(i32),
    /// The hang marker.
    Hung,
    /// The body returned.
    Returned,
    /// [`Child::hang`] advanced virtual time.
    Advanced,
    /// The body panicked: its thread unwound past the [`Child`].
    Panicked,
}

/// One trace record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// What it is.
    pub kind: RecordKind,
    /// The op's text (empty for markers).
    pub text: String,
    /// The thread that recorded it.
    pub thread: ThreadTag,
}

#[derive(Debug, Clone)]
struct Entry {
    kind: u32,
    key: u64,
    tag: String,
    len: u64,
    off: u64,
}

/// What the parent learned from one child.
#[derive(Debug)]
pub struct Report {
    name: String,
    after: After,
    outcome: Outcome,
    records: Vec<Record>,
    torn: bool,
    truncated: bool,
    overflowed: bool,
    dir: Vec<Entry>,
    file: File,
    stderr: String,
}

impl Report {
    /// How the child ended.
    pub fn outcome(&self) -> &Outcome {
        &self.outcome
    }

    /// Every record, in the order they were appended.
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// The ops' texts, in order.
    pub fn ops(&self) -> Vec<&str> {
        self.records
            .iter()
            .filter(|r| r.kind == RecordKind::Op)
            .map(|r| r.text.as_str())
            .collect()
    }

    /// The child's stderr and stdout.
    pub fn stderr(&self) -> &str {
        &self.stderr
    }

    /// Whether a record was reserved but never written: its thread was between reserving and
    /// committing it when the child ended. Later records are read all the same.
    pub fn torn(&self) -> bool {
        self.torn
    }

    /// Whether an op's text was longer than [`TEXT_MAX`] and was cut.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// Whether the trace ran out of slots (later records, markers included, were lost).
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// The position of the first death or hang marker.
    fn end_marker(&self) -> Option<usize> {
        self.records
            .iter()
            .position(|r| matches!(r.kind, RecordKind::Death(_) | RecordKind::Hung))
    }

    /// The ops recorded after the first death or hang marker by `on`'s threads.
    fn ops_after_end(&self, on: impl Fn(&ThreadTag) -> bool) -> Vec<&Record> {
        match self.end_marker() {
            Some(i) => self
                .records
                .iter()
                .skip(i.saturating_add(1))
                .filter(|r| r.kind == RecordKind::Op && on(&r.thread))
                .collect(),
            None => Vec::new(),
        }
    }

    /// The ops the dying or hung thread (the one the first death or hang marker names)
    /// recorded after that marker: none, in a sound run. `death_test!` and `hang_test!` assert
    /// it under [`After::DyingThread`].
    pub fn after_end(&self) -> Vec<&Record> {
        let Some(tid) = self
            .end_marker()
            .and_then(|i| self.records.get(i))
            .map(|m| m.thread.tid)
        else {
            return Vec::new();
        };
        self.ops_after_end(|t| t.tid == tid)
    }

    /// The ops any thread recorded after the first death or hang marker: none, in a sound run
    /// with no thread recording freely. `death_test!` and `hang_test!` assert it by default
    /// ([`After::EveryThread`]). A thread blocked behind the death never appears here; one
    /// running freely may, as in production it runs on for a moment after a death.
    pub fn after_end_any(&self) -> Vec<&Record> {
        self.ops_after_end(|_| true)
    }

    /// Whether a `kind` record comes before the first death or hang marker (anywhere when
    /// there is none).
    fn before_end(&self, kind: RecordKind) -> bool {
        let end = self.end_marker().unwrap_or(self.records.len());
        self.records.iter().take(end).any(|r| r.kind == kind)
    }

    /// Whether the body returned ([`Child::finish`]) before the first death or hang marker, or at
    /// all when there is none. A death test fails when `f` returns first.
    pub fn returned_before_end(&self) -> bool {
        self.before_end(RecordKind::Returned)
    }

    /// Whether the body panicked (its thread unwound past the [`Child`]) before the first death or
    /// hang marker, or at all when there is none. Death and hang tests fail with `f`'s panic when
    /// `f` panics first.
    pub fn panicked_before_end(&self) -> bool {
        self.before_end(RecordKind::Panicked)
    }

    /// The last op recorded by a thread `on` accepts before the first death or hang marker
    /// (before the trace's end if there is none): for a death on one thread while another
    /// drives the test (`|t| t.body` gives the body's last op).
    pub fn last_op_before_end(&self, on: impl Fn(&ThreadTag) -> bool) -> Option<&str> {
        let end = self.end_marker().unwrap_or(self.records.len());
        self.records
            .iter()
            .take(end)
            .rev()
            .find(|r| r.kind == RecordKind::Op && on(&r.thread))
            .map(|r| r.text.as_str())
    }

    fn fail(&self, what: &str) -> ! {
        panic!(
            "{}: {what}; outcome {:?}; ops {:?}\n--- child output ---\n{}",
            self.name,
            self.outcome,
            self.ops(),
            self.stderr
        )
    }

    /// The death, or a failure naming what happened instead.
    pub fn died(&self) -> &Died {
        match &self.outcome {
            Outcome::Died(d) => d,
            _ => self.fail("expected a death"),
        }
    }

    /// The hang, or a failure naming what happened instead.
    pub fn hung(&self) -> &Hung {
        match &self.outcome {
            Outcome::Hung(h) => h,
            _ => self.fail("expected a hang"),
        }
    }

    /// Asserts a death at `site` and returns it.
    pub fn assert_died(&self, site: DeathSite) -> &Died {
        let d = self.died();
        if d.raced {
            self.fail(&format!(
                "expected a death at {site:?}, but deaths raced (lines {:?}, other aborts' \
                 lines {:?}, first marked thread {:?}): the harness pairs no site with a thread",
                d.ids, d.abort_lines, d.thread
            ));
        }
        if !d.id.as_ref().is_some_and(|id| id.is(site)) {
            self.fail(&format!("expected a death at {site:?}, got {:?}", d.id));
        }
        d
    }

    /// Asserts a death at `site` whose last op is `op`, and returns it.
    pub fn assert_died_at(&self, site: DeathSite, op: &str) -> &Died {
        let d = self.assert_died(site);
        if d.op.as_deref() != Some(op) {
            self.fail(&format!("expected the death after {op:?}"));
        }
        d
    }

    /// Asserts a hang whose last op is `op`, and returns it.
    pub fn assert_hung_at(&self, op: &str) -> &Hung {
        let h = self.hung();
        if h.op.as_deref() != Some(op) {
            self.fail(&format!("expected the hang after {op:?}"));
        }
        h
    }

    /// The gate `death_test!` and `hang_test!` run before the test's check (and a tool that
    /// starts its own child with [`spawn_command`] may run): the expected outcome, reached
    /// before the body returned or panicked; no op after the death or hang marker on the
    /// threads [`Options::after`] names; no overflowed trace and no cut op. `Expect::Any` checks
    /// nothing.
    pub fn expect(&self, e: Expect) {
        let what = match e {
            Expect::Death => {
                let _ = self.died();
                "death"
            }
            Expect::Hang => {
                let _ = self.hung();
                "hang"
            }
            Expect::Any => return,
        };
        if self.returned_before_end() {
            self.fail(&format!(
                "expected a {what}, but the body returned before it"
            ));
        }
        if self.panicked_before_end() {
            self.fail(&format!(
                "expected a {what}, but the body panicked before it"
            ));
        }
        let late = match self.after {
            After::EveryThread => self.after_end_any(),
            After::DyingThread => self.after_end(),
        };
        if !late.is_empty() {
            self.fail(&format!("recorded after the end: {late:?}"));
        }
        if self.overflowed {
            self.fail("the trace overflowed");
        }
        if self.truncated {
            self.fail(&format!(
                "an op was longer than {TEXT_MAX} bytes and was cut"
            ));
        }
    }

    fn read_at(&self, off: u64, len: u64) -> Option<Vec<u8>> {
        let mut v = vec![0u8; usize::try_from(len).ok()?];
        self.file.read_exact_at(&mut v, off).ok()?;
        Some(v)
    }

    /// Guest RAM at guest-physical `pa..pa+len`, inside one range the child shared
    /// ([`Child::share_ram`]), as the child left it.
    pub fn ram(&self, pa: u64, len: u64) -> Option<Vec<u8>> {
        let e = self.dir.iter().find(|e| {
            e.kind == DIR_PA
                && pa >= e.key
                && pa
                    .checked_add(len)
                    .is_some_and(|end| end <= e.key.saturating_add(e.len))
        })?;
        self.read_at(e.off.saturating_add(pa.wrapping_sub(e.key)), len)
    }

    /// A little-endian `u32` of guest RAM.
    pub fn ram_u32(&self, pa: u64) -> Option<u32> {
        let b = self.ram(pa, 4)?;
        Some(u32::from_le_bytes(b.try_into().ok()?))
    }

    /// A little-endian `u64` of guest RAM.
    pub fn ram_u64(&self, pa: u64) -> Option<u64> {
        let b = self.ram(pa, 8)?;
        Some(u64::from_le_bytes(b.try_into().ok()?))
    }

    /// The whole region the child shared under `tag` ([`Child::share_region`]).
    pub fn region(&self, tag: &str) -> Option<Vec<u8>> {
        let e = self
            .dir
            .iter()
            .find(|e| e.kind == DIR_TAGGED && e.tag == tag)?;
        self.read_at(e.off, e.len)
    }
}

/// The test's name as libtest filters it: the module path without the crate, then the name.
pub fn test_name(module_path: &str, name: &str) -> String {
    match module_path.split_once("::") {
        Some((_, rest)) => format!("{rest}::{name}"),
        None => name.to_owned(),
    }
}

/// Parses `raise::death::fatal`'s stderr line in a testkit build,
/// `reims-vgpu: fatal: <what died> [site <Kind> <file>:<line>]`: the first one in `out`, wherever
/// it starts on its line.
pub fn parse_death_line(out: &str) -> Option<DeathId> {
    parse_death_lines(out).into_iter().next()
}

/// Every `raise::death::fatal` stderr line in `out`, in order ([`parse_death_line`]).
pub fn parse_death_lines(out: &str) -> Vec<DeathId> {
    out.lines()
        .filter_map(|l| {
            // libtest's own progress text may precede it on the same line.
            let rest = l.get(l.find(DEATH_PREFIX)?.saturating_add(DEATH_PREFIX.len())..)?;
            let (_, site) = rest.rsplit_once(" [site ")?;
            let (kind, at) = site.strip_suffix(']')?.split_once(' ')?;
            let (file, line) = at.rsplit_once(':')?;
            Some(DeathId {
                kind: kind.to_owned(),
                file: file.to_owned(),
                line: line.parse().ok()?,
            })
        })
        .collect()
}

/// What `raise::death::fatal`'s line starts with.
const DEATH_PREFIX: &str = "reims-vgpu: fatal: ";

/// Every line in `out` that an abort other than `fatal`'s prints on its way, wherever it starts
/// on its line:
/// - the D9 panic hook's: `reims-vgpu: panic: <panic>` (`hook_message`,
///   crates/reims-vgpu/src/singleton.rs), written just before its `abort()`;
/// - std's: `fatal runtime error: …` (its runtime aborts, a stack overflow's among them),
///   `… panic. aborting.` (a panic that cannot unwind, a panic while panicking) and
///   `memory allocation of … failed` (the default allocation-error hook).
///
/// `fatal`'s own line (`reims-vgpu: fatal: …`, [`parse_death_lines`]) is not one.
pub fn parse_abort_lines(out: &str) -> Vec<String> {
    const HOOK: &str = "reims-vgpu: ";
    out.lines()
        .filter(|l| {
            l.match_indices(HOOK)
                .any(|(i, _)| l.get(i..).is_some_and(|s| !s.starts_with(DEATH_PREFIX)))
                || l.contains("fatal runtime error: ")
                || l.contains("panic. aborting.")
                || (l.contains("memory allocation of ") && l.contains(" failed"))
        })
        .map(str::to_owned)
        .collect()
}

fn ne_u32(b: &[u8], at: usize) -> u32 {
    b.get(at..at.saturating_add(4))
        .and_then(|s| s.try_into().ok())
        .map_or(0, u32::from_ne_bytes)
}

fn ne_u64(b: &[u8], at: usize) -> u64 {
    b.get(at..at.saturating_add(8))
        .and_then(|s| s.try_into().ok())
        .map_or(0, u64::from_ne_bytes)
}

fn text(b: &[u8], at: usize, len: usize) -> String {
    String::from_utf8_lossy(b.get(at..at.saturating_add(len)).unwrap_or_default()).into_owned()
}

/// What the trace held.
struct Trace {
    records: Vec<Record>,
    /// A slot was reserved but never committed.
    torn: bool,
    /// A record's text was cut.
    truncated: bool,
}

/// Reads the trace's slots in the order they were reserved.
fn parse_trace(t: &[u8], body_tid: u64) -> Trace {
    let mut out = Trace {
        records: Vec::new(),
        torn: false,
        truncated: false,
    };
    for rec in t.chunks(SLOT) {
        if ne_u32(rec, 0) != 1 {
            out.torn = true;
            continue;
        }
        let byte = |i: usize| rec.get(i).copied().unwrap_or(0);
        let kind = match byte(4) {
            K_OP => RecordKind::Op,
            K_DEATH => RecordKind::Death(ne_u32(rec, 8).cast_signed()),
            K_HUNG => RecordKind::Hung,
            K_RETURNED => RecordKind::Returned,
            K_ADVANCED => RecordKind::Advanced,
            K_PANICKED => RecordKind::Panicked,
            _ => {
                out.torn = true;
                continue;
            }
        };
        let name_len = usize::from(byte(5));
        let text_len = usize::from(u16::from_ne_bytes([byte(6), byte(7)]));
        out.truncated |= ne_u32(rec, 12) & F_TRUNCATED != 0;
        let tid = ne_u64(rec, 16);
        out.records.push(Record {
            kind,
            text: text(rec, REC_HEAD.saturating_add(name_len), text_len),
            thread: ThreadTag {
                tid,
                name: text(rec, REC_HEAD, name_len),
                body: tid != 0 && tid == body_tid,
            },
        });
    }
    out
}

static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

/// A scratch path for one child.
fn scratch(ext: &str) -> PathBuf {
    let n = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("reims-vgpu-death-{}-{n}.{ext}", std::process::id()))
}

/// Runs the current test binary as the child for the test `name` (as libtest filters it), with
/// [`ENV_CHILD`] set, and reads what it left.
pub fn spawn(name: &str, opts: Options) -> Report {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => panic!("death harness: no current executable: {e}"),
    };
    let mut cmd = Command::new(exe);
    cmd.args([
        name,
        "--exact",
        "--nocapture",
        "--include-ignored",
        "--test-threads=1",
    ])
    .env(ENV_CHILD, name);
    spawn_command(cmd, name, opts)
}

/// Runs `cmd` as a child with [`ENV_FILE`] set (it must call [`enter`], as a libtest child does
/// through [`run`]), waits for it up to `opts.timeout`, and reads what it left. `label` names it
/// in failures.
pub fn spawn_command(mut cmd: Command, label: &str, opts: Options) -> Report {
    let name = label;
    let path = scratch("map");
    let out_path = scratch("out");
    let fail = |what: &str, e: std::io::Error| -> ! { panic!("death harness: {what}: {e}") };
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap_or_else(|e| fail("cannot create the scratch file", e));
    if let Err(e) = file
        .set_len(RAM_OFF.to_u64())
        .and_then(|()| file.write_all_at(&MAGIC.to_ne_bytes(), 0))
    {
        fail("cannot size the scratch file", e)
    }
    let out = File::create(&out_path).unwrap_or_else(|e| fail("cannot create the output file", e));
    let out2 = out
        .try_clone()
        .unwrap_or_else(|e| fail("cannot share the output file", e));
    let mut child = cmd
        .env(ENV_FILE, OsString::from(&path))
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(out2))
        .spawn()
        .unwrap_or_else(|e| fail("cannot start the child", e));
    let started = Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {}
            Err(e) => fail("cannot wait for the child", e),
        }
        if started.elapsed() >= opts.timeout {
            timed_out = true;
            let _ = child.kill();
            break child
                .wait()
                .unwrap_or_else(|e| fail("cannot reap the child", e));
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    let stderr = std::fs::read(&out_path)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_file(&path);

    let mut head = vec![0u8; TRACE_OFF];
    if let Err(e) = file.read_exact_at(&mut head, 0) {
        fail("cannot read the header", e)
    }
    let state = ne_u32(&head, H_STATE);
    let body_tid = ne_u64(&head, H_BODY_TID);
    let used = ne_u64(&head, H_CURSOR)
        .to_usize()
        .min(SLOTS)
        .saturating_mul(SLOT);
    let mut t = vec![0u8; used];
    if let Err(e) = file.read_exact_at(&mut t, TRACE_OFF.to_u64()) {
        fail("cannot read the trace", e)
    }
    let Trace {
        records,
        torn,
        truncated,
    } = parse_trace(&t, body_tid);
    let dir = (0..ne_u32(&head, H_DIR_COUNT).to_usize().min(DIR_MAX))
        .map(|i| {
            let e = DIR_OFF.saturating_add(i.saturating_mul(DIR_ENTRY));
            let tag_len = ne_u32(&head, e.saturating_add(4)).to_usize().min(DIR_TAG);
            Entry {
                kind: ne_u32(&head, e),
                key: ne_u64(&head, e.saturating_add(8)),
                len: ne_u64(&head, e.saturating_add(16)),
                off: ne_u64(&head, e.saturating_add(24)),
                tag: text(&head, e.saturating_add(32), tag_len),
            }
        })
        .collect();

    // The last op before `end`, on any thread.
    let last_op_before = |end: usize| {
        records
            .iter()
            .take(end)
            .rev()
            .find(|r| r.kind == RecordKind::Op)
            .map(|r| r.text.clone())
    };
    let unknown = ThreadTag {
        tid: 0,
        name: String::new(),
        body: false,
    };
    let outcome = if timed_out {
        Outcome::TimedOut
    } else if let Some(signal) = status.signal() {
        let marker = records
            .iter()
            .enumerate()
            .find(|(_, r)| matches!(r.kind, RecordKind::Death(_)));
        let ids = parse_death_lines(&stderr);
        let abort_lines = parse_abort_lines(&stderr);
        // Pair the one death line with the one thread marked dying, and only then: which line a
        // thread printed is not recorded, so two deaths at once could pair one's site with the
        // other's thread. `fatal` prints its line, then aborts (SIGABRT), as do the D9 panic
        // hooks and std's aborts, whose lines count as deaths too.
        let mut marked: Vec<u64> = Vec::new();
        for r in records
            .iter()
            .filter(|r| matches!(r.kind, RecordKind::Death(_)))
        {
            if !marked.contains(&r.thread.tid) {
                marked.push(r.thread.tid);
            }
        }
        let first_signal = marker.and_then(|(_, m)| match m.kind {
            RecordKind::Death(s) => Some(s),
            _ => None,
        });
        let lines = ids.len().saturating_add(abort_lines.len());
        let raced = lines > 1
            || marked.len() > 1
            || (lines == 1 && first_signal.is_some_and(|s| s != sys::SIGABRT));
        Outcome::Died(Died {
            id: if raced { None } else { ids.first().cloned() },
            raced,
            ids,
            abort_lines,
            op: last_op_before(marker.map_or(records.len(), |(i, _)| i)),
            thread: marker.map_or(unknown, |(_, m)| m.thread.clone()),
            signal,
        })
    } else {
        match (status.code(), state) {
            (Some(0), STATE_RETURNED) => Outcome::Returned,
            (Some(0), STATE_SPAWNED) => Outcome::NotRun,
            (Some(HANG_EXIT), STATE_HUNG) => {
                let end = records.iter().position(|r| r.kind == RecordKind::Hung);
                Outcome::Hung(Hung {
                    op: last_op_before(end.unwrap_or(records.len())),
                    thread: end
                        .and_then(|i| records.get(i))
                        .map_or(unknown, |r| r.thread.clone()),
                })
            }
            (code, _) => Outcome::Failed { code },
        }
    };
    Report {
        name: name.to_owned(),
        after: opts.after,
        outcome,
        records,
        torn,
        truncated,
        overflowed: ne_u32(&head, H_OVERFLOW) != 0,
        dir,
        file,
        stderr,
    }
}

/// The body of every `death_test!` and `hang_test!`. In the child named by [`ENV_CHILD`] it
/// runs `body`; everywhere else it runs that child, checks `expect` and then runs `check` on
/// the [`Report`].
pub fn run(
    expect: Expect,
    opts: Options,
    module_path: &str,
    name: &str,
    body: impl FnOnce(&Child),
    check: impl FnOnce(&Report),
) {
    let full = test_name(module_path, name);
    if std::env::var_os(ENV_CHILD).is_some_and(|v| v == full.as_str()) {
        return child_main(body);
    }
    let report = spawn(&full, opts);
    if report.outcome == Outcome::NotRun {
        report.fail("the child ran no test (is the name its libtest path?)");
    }
    report.expect(expect);
    check(&report);
}

/// A test whose body dies: `death_test!(name, |t| body, |r| check)`. The body runs in a child
/// process ([`run`]) where a death aborts as in production. The parent
/// asserts a death before the body returned, with nothing recorded after it on any thread, then
/// runs `check` on the [`Report`].
///
/// `death_test!(name, options = Options { .. }, |t| body, |r| check)` runs it with other
/// [`Options`]: a test with a thread that records freely beside
/// the death judges "nothing after" on the dying thread alone
/// ([`After::DyingThread`]).
#[macro_export]
macro_rules! death_test {
    ($(#[$meta:meta])* $name:ident, options = $opts:expr, $body:expr, $check:expr $(,)?) => {
        $(#[$meta])*
        #[test]
        fn $name() {
            $crate::testkit::death::run(
                $crate::testkit::death::Expect::Death,
                $opts,
                module_path!(),
                stringify!($name),
                $body,
                $check,
            )
        }
    };
    ($(#[$meta:meta])* $name:ident, $body:expr, $check:expr $(,)?) => {
        $crate::death_test!(
            $(#[$meta])* $name,
            options = $crate::testkit::death::Options::default(),
            $body,
            $check
        );
    };
}

/// A test whose body hangs: `hang_test!(name, |t| { …; t.hang(&clock, d, f) }, |r| check)`.
/// As [`death_test!`], but the parent asserts a hang
/// ([`Child::hang`](crate::testkit::death::Child::hang)) with nothing recorded after it. It takes
/// `options = …` as `death_test!` does.
#[macro_export]
macro_rules! hang_test {
    ($(#[$meta:meta])* $name:ident, options = $opts:expr, $body:expr, $check:expr $(,)?) => {
        $(#[$meta])*
        #[test]
        fn $name() {
            $crate::testkit::death::run(
                $crate::testkit::death::Expect::Hang,
                $opts,
                module_path!(),
                stringify!($name),
                $body,
                $check,
            )
        }
    };
    ($(#[$meta:meta])* $name:ident, $body:expr, $check:expr $(,)?) => {
        $crate::hang_test!(
            $(#[$meta])* $name,
            options = $crate::testkit::death::Options::default(),
            $body,
            $check
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::os::{Clock as _, Deadline, MonoTime};
    use crate::raise::death::DeathKind;
    use crate::raise::death::DeathSite;
    use crate::testkit::capture::capture_enabled;
    use crate::testkit::clock::VirtualClock;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    const SITE: DeathSite = DeathSite {
        kind: DeathKind::Abort,
        file: "site_t",
        line: 0,
    };
    const WRITER_SITE: DeathSite = DeathSite {
        kind: DeathKind::NullDeref,
        file: "site_w",
        line: 0,
    };

    /// Records "dropped" when dropped: a death must never run it.
    struct Noisy;
    impl Drop for Noisy {
        fn drop(&mut self) {
            trace("dropped");
        }
    }

    fn named(name: &str, f: impl FnOnce() + Send + 'static) {
        let _ = std::thread::Builder::new()
            .name(name.into())
            .spawn(f)
            .unwrap();
    }

    fn park_forever() -> ! {
        loop {
            std::thread::park();
        }
    }

    death_test!(
        /// A death after a side effect shows the effect, and runs no `Drop`
        a_death_after_a_side_effect_shows_the_effect,
        |t| {
            let ram = GuestRam::with_range(0x1_0000_0000, 0x4000).unwrap();
            t.share_ram(&ram);
            let _noisy = Noisy;
            t.op(&format!("capture={}", capture_enabled()));
            ram.write_u32(0x1_0000_0040, 0xdead_beef);
            t.op("effect");
            crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t")
        },
        |r| {
            let d = r.assert_died_at(SITE, "effect");
            assert!(d.thread.body, "died on the body's thread: {d:?}");
            assert!(!d.raced && d.ids.len() == 1, "{d:?}");
            assert_eq!(d.signal, 6);
            assert_eq!(r.ops(), ["capture=false", "effect"]);
            assert_eq!(r.ram_u32(0x1_0000_0040), Some(0xdead_beef));
        }
    );

    death_test!(
        /// A death under a held lock never lets a blocked second thread record
        a_death_under_a_held_lock_never_lets_a_blocked_thread_record,
        |t| {
            let lock = Arc::new(Mutex::new(()));
            let held = lock.lock();
            let waiting = Arc::new(AtomicBool::new(false));
            let (l2, w2) = (lock.clone(), waiting.clone());
            named("second", move || {
                trace("second: waiting");
                w2.store(true, Ordering::Release);
                let _g = l2.lock();
                trace("second: locked");
            });
            while !waiting.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            // Let the second thread block on the lock.
            std::thread::sleep(Duration::from_millis(50));
            t.op("first: dying");
            let _held = held;
            crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t")
        },
        |r| {
            let d = r.assert_died_at(SITE, "first: dying");
            assert!(d.thread.body);
            assert_eq!(r.ops(), ["second: waiting", "first: dying"]);
            assert!(r.after_end_any().is_empty(), "{:?}", r.after_end_any());
            let second = &r.records()[0].thread;
            assert_eq!(second.name(), Some("second"));
            assert!(!second.body);
        }
    );

    death_test!(
        /// Post-death RAM is visible, on every shared range and region, with the bytes written
        /// before the share kept
        post_death_ram_is_visible,
        |t| {
            let ram = GuestRam::with_range(0x1_0000_0000, 0x8000).unwrap();
            ram.add_range(0x2_0000_0000, 0x4000).unwrap();
            ram.write_u32(0x1_0000_0000, 0x1111);
            let root = RawRegion::new(16).unwrap();
            root.write_u32(12, 0x2222);
            t.share_ram(&ram);
            t.share_ram(&ram); // idempotent
            t.share_region("root", &root);
            named("writer", move || {
                ram.write_u64(0x2_0000_0008, 0x0123_4567_89ab_cdef);
                assert!(root.store_16_atomic(0, 5, 6));
                trace("wrote");
                crate::testkit::capture::die(crate::raise::death::DeathKind::NullDeref, "site_w")
            });
            park_forever()
        },
        |r| {
            let d = r.assert_died_at(WRITER_SITE, "wrote");
            assert_eq!(d.thread.name(), Some("writer"));
            assert!(!d.thread.body);
            assert_eq!(r.ram_u32(0x1_0000_0000), Some(0x1111));
            assert_eq!(r.ram_u64(0x2_0000_0008), Some(0x0123_4567_89ab_cdef));
            assert_eq!(r.ram(0x2_0000_3ffc, 8), None, "no read across the end");
            let root = r.region("root").unwrap();
            assert_eq!(root.len(), 16);
            assert_eq!(&root[..8], &5u64.to_le_bytes());
            assert_eq!(&root[8..], &6u64.to_le_bytes());
        }
    );

    hang_test!(
        /// A hang is judged at the last op, on the blocked thread, after virtual time advanced
        a_hang_is_judged_at_the_last_op,
        |t| {
            let clock = Arc::new(VirtualClock::new());
            let c2 = clock.clone();
            t.hang(&*clock, Duration::from_secs(1), move || {
                trace("sleeping an hour");
                c2.sleep_until(Deadline::Mono(MonoTime::from_nanos(3_600_000_000_000)));
                trace("woke");
            });
        },
        |r| {
            let h = r.assert_hung_at("sleeping an hour");
            assert_eq!(h.thread.name(), Some("hang-body"));
            assert!(r.records().iter().any(|x| x.kind == RecordKind::Advanced));
        }
    );

    /// A child the harness's own tests drive with `Expect::Any`.
    fn any(name: &str, timeout: Duration, body: impl FnOnce(&Child), check: impl FnOnce(&Report)) {
        run(
            Expect::Any,
            Options {
                timeout,
                ..Options::default()
            },
            module_path!(),
            name,
            body,
            check,
        );
    }

    /// A blocked wait that virtual time ends is not a hang: the body returns
    #[test]
    fn a_wait_that_time_ends_returns() {
        any(
            "a_wait_that_time_ends_returns",
            CHILD_TIMEOUT,
            |t| {
                let clock = Arc::new(VirtualClock::new());
                let c2 = clock.clone();
                t.hang(&*clock, Duration::from_secs(7200), move || {
                    c2.sleep_until(Deadline::Mono(MonoTime::from_nanos(3_600_000_000_000)));
                    trace("woke");
                });
            },
            |r| {
                assert_eq!(r.outcome(), &Outcome::Returned, "{}", r.stderr());
                assert_eq!(r.ops(), ["woke"]);
                assert!(r.after_end().is_empty());
            },
        );
    }

    /// A body that returns, a body that panics and a body that blocks are told apart
    #[test]
    fn outcomes_other_than_death() {
        any(
            "outcomes_other_than_death",
            CHILD_TIMEOUT,
            |t| t.op("fine"),
            |r| {
                assert_eq!(r.outcome(), &Outcome::Returned, "{}", r.stderr());
                assert_eq!(r.ops(), ["fine"]);
                assert!(r.returned_before_end() && !r.panicked_before_end());
            },
        );
    }

    /// A panicking body fails the child (libtest's 101), and the child records that it panicked
    /// once it unwound past `run`'s `Child`
    #[test]
    fn a_panicking_body_fails() {
        any(
            "a_panicking_body_fails",
            CHILD_TIMEOUT,
            |t| {
                t.op("checking");
                panic!("boom")
            },
            |r| {
                assert_eq!(r.outcome(), &Outcome::Failed { code: Some(101) });
                assert!(r.stderr().contains("boom"));
                let kinds: Vec<_> = r.records().iter().map(|x| x.kind).collect();
                assert_eq!(kinds, [RecordKind::Op, RecordKind::Panicked]);
                assert!(r.panicked_before_end() && !r.returned_before_end());
            },
        );
    }

    /// A body that blocks with nothing to judge it is killed at the timeout
    #[test]
    fn a_blocked_body_times_out() {
        any(
            "a_blocked_body_times_out",
            Duration::from_millis(500),
            |t| {
                t.op("blocking");
                park_forever()
            },
            |r| {
                assert_eq!(r.outcome(), &Outcome::TimedOut);
            },
        );
    }

    /// A bare abort (no death line) is a death with no id, on its thread
    #[test]
    fn a_bare_abort_has_no_id() {
        any(
            "a_bare_abort_has_no_id",
            CHILD_TIMEOUT,
            |t| {
                t.op("aborting");
                std::process::abort()
            },
            |r| {
                let d = r.died();
                assert_eq!(d.id, None);
                assert_eq!(d.op.as_deref(), Some("aborting"));
                assert!(d.thread.body);
            },
        );
    }

    /// A test the filter does not reach is reported, not passed
    #[test]
    #[should_panic(expected = "ran no test")]
    fn a_wrong_name_is_not_run() {
        run(
            Expect::Death,
            Options::default(),
            module_path!(),
            "no_such_test",
            |_| {},
            |_| {},
        );
    }

    /// The death line and the test path parse
    #[test]
    fn names_parse() {
        let out = "noise\ntest x ... reims-vgpu: fatal: device.x [site NilBlock device:742]\n\
                   reims-vgpu: fatal: uncaught raise a at src/a.rs:3: why [site Abort x:1]\n";
        let id = parse_death_line(out).unwrap();
        assert!(id.is(DeathSite {
            kind: DeathKind::NilBlock,
            file: "device",
            line: 742
        }));
        assert_eq!(parse_death_line("nothing"), None);
        let all = parse_death_lines(out);
        assert_eq!(all.len(), 2);
        assert_eq!((all[1].file.as_str(), all[1].line), ("x", 1));
        assert!(
            parse_abort_lines(out).is_empty(),
            "fatal's lines are not other aborts'"
        );
        // The lines other aborts print.
        let aborts = "test x ... reims-vgpu: fatal: site_a [site Abort site_a:1]\n\
            test y ... reims-vgpu: panicked at crates/x.rs:1:1:\n\
            fake: unscripted call\n\
            fatal runtime error: stack overflow, aborting\n\
            thread caused non-unwinding panic. aborting.\n\
            memory allocation of 64 bytes failed\n";
        assert_eq!(parse_death_lines(aborts).len(), 1);
        assert_eq!(
            parse_abort_lines(aborts),
            [
                "test y ... reims-vgpu: panicked at crates/x.rs:1:1:",
                "fatal runtime error: stack overflow, aborting",
                "thread caused non-unwinding panic. aborting.",
                "memory allocation of 64 bytes failed",
            ]
        );
        assert_eq!(
            test_name("reims_vgpu_core::testkit::death::tests", "x"),
            "testkit::death::tests::x"
        );
        assert_eq!(test_name("machines_guest", "x"), "x");
    }

    /// Outside a child, `trace` does nothing
    #[test]
    fn trace_outside_a_child_is_a_no_op() {
        assert!(!in_child());
        trace("ignored");
    }

    /// The options of a test with a thread that records freely beside the death.
    const DYING_THREAD: Options = Options {
        timeout: CHILD_TIMEOUT,
        after: After::DyingThread,
    };

    /// An op the dying thread records after its death marker fails a death test, even judging the
    /// dying thread alone ("nothing after")
    #[test]
    #[should_panic(expected = "recorded after the end")]
    fn an_op_after_the_death_fails() {
        run(
            Expect::Death,
            DYING_THREAD,
            module_path!(),
            "an_op_after_the_death_fails",
            |t| {
                t.op("before");
                // A marker, then an op after it on the same thread: what a dying thread that
                // went on would leave.
                append_here(K_DEATH, 6, b"");
                t.op("after");
                std::process::abort()
            },
            |_| {},
        );
    }

    death_test!(
        /// Judging the dying thread alone, an op another thread records after the death marker
        /// neither fails the death nor becomes its op
        another_threads_op_after_the_death_is_not_the_deaths,
        options = DYING_THREAD,
        |t| {
            let (go, done) = (
                Arc::new(AtomicBool::new(false)),
                Arc::new(AtomicBool::new(false)),
            );
            let (go2, done2) = (go.clone(), done.clone());
            named("other", move || {
                while !go2.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                trace("late");
                done2.store(true, Ordering::Release);
                park_forever()
            });
            t.op("dying");
            // The dying thread's marker is down; the other thread records before the kill.
            append_here(K_DEATH, 6, b"");
            go.store(true, Ordering::Release);
            while !done.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            std::process::abort()
        },
        |r| {
            let d = r.died();
            assert_eq!(d.op.as_deref(), Some("dying"));
            assert!(d.thread.body);
            assert!(r.after_end().is_empty());
            let late: Vec<_> = r.after_end_any().iter().map(|x| x.text.as_str()).collect();
            assert_eq!(late, ["late"]);
            assert_eq!(r.last_op_before_end(|t| !t.body), None);
        }
    );

    death_test!(
        /// Judging the dying thread alone, a thread recording freely while another dies neither
        /// fails the death nor hides the dying thread's op (`Died.op`, the trace's last op, is a
        /// race here)
        a_free_running_recorder_does_not_fail_the_death,
        options = DYING_THREAD,
        |t| {
            let ticks = Arc::new(AtomicUsize::new(0));
            let t2 = ticks.clone();
            named("ticker", move || {
                // Bounded: a body the host keeps off the CPU must not let it fill the trace.
                for _ in 0..SLOTS / 4 {
                    trace("tick");
                    t2.fetch_add(1, Ordering::Relaxed);
                }
                park_forever()
            });
            while ticks.load(Ordering::Relaxed) < 100 {
                std::hint::spin_loop();
            }
            t.op("dying");
            crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t")
        },
        |r| {
            let d = r.assert_died(SITE);
            assert!(d.thread.body);
            let tid = d.thread.tid;
            assert_eq!(r.last_op_before_end(|t| t.tid == tid), Some("dying"));
            assert!(matches!(d.op.as_deref(), Some("tick" | "dying")), "{d:?}");
            assert_eq!(r.last_op_before_end(|t| !t.body), Some("tick"));
            assert!(
                r.after_end_any()
                    .iter()
                    .all(|x| x.thread.name() == Some("ticker"))
            );
        }
    );

    /// Waits, yielding, until `done`.
    fn wait_until(done: impl Fn() -> bool) {
        while !done() {
            std::thread::yield_now();
        }
    }

    death_test!(
        /// A death in a queue block that records nothing, on a worker an earlier block used, is at
        /// the op that preceded it, not at the worker's stale op
        a_death_on_a_reused_worker_is_at_the_op_before_it,
        |t| {
            let q = crate::rt::queue::SerialQueue::new(
                &crate::testkit::threads::FakeThreads::default(),
                "A",
            );
            let done = Arc::new(AtomicBool::new(false));
            let d2 = done.clone();
            q.exec_async(move || {
                trace("A: block done");
                d2.store(true, Ordering::Release);
            });
            // The worker ran the first block and waits for the next.
            wait_until(|| done.load(Ordering::Acquire));
            t.op("body: submit C");
            q.exec_async(|| {
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t")
            });
            park_forever()
        },
        |r| {
            let d = r.assert_died_at(SITE, "body: submit C");
            assert!(!d.thread.body, "{d:?}");
            // The reused worker's own last op is A's finished block.
            let tid = d.thread.tid;
            assert_eq!(
                r.last_op_before_end(|t| t.tid == tid),
                Some("A: block done")
            );
            assert_eq!(r.ops(), ["A: block done", "body: submit C"]);
        }
    );

    death_test!(
        /// A dying block on one of two workers that each ran a block is at the op before it
        a_death_on_either_of_two_workers_is_at_the_op_before_it,
        |t| {
            let threads = crate::testkit::threads::FakeThreads::default();
            let (qa, qb) = (
                crate::rt::queue::SerialQueue::new(&threads, "A"),
                crate::rt::queue::SerialQueue::new(&threads, "B"),
            );
            let go = Arc::new(AtomicBool::new(false));
            let started = Arc::new(AtomicUsize::new(0));
            let finished = Arc::new(AtomicUsize::new(0));
            for (q, op) in [(&qa, "A: block"), (&qb, "B: block")] {
                let (go, started, finished) = (go.clone(), started.clone(), finished.clone());
                q.exec_async(move || {
                    trace(op);
                    let _ = started.fetch_add(1, Ordering::AcqRel);
                    // Both blocks run at once, on two workers.
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    let _ = finished.fetch_add(1, Ordering::AcqRel);
                });
            }
            wait_until(|| started.load(Ordering::Acquire) == 2);
            go.store(true, Ordering::Release);
            wait_until(|| finished.load(Ordering::Acquire) == 2);
            t.op("body: submit C");
            qa.exec_async(|| {
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t")
            });
            park_forever()
        },
        |r| {
            let d = r.assert_died_at(SITE, "body: submit C");
            assert!(!d.thread.body, "{d:?}");
            let tid = d.thread.tid;
            let stale = r.last_op_before_end(|t| t.tid == tid);
            assert!(matches!(stale, Some("A: block" | "B: block")), "{stale:?}");
        }
    );

    /// A slot reserved but never written (a thread stopped between its reservation and its commit)
    /// hides no later record, the death marker included
    #[test]
    fn a_slot_reserved_but_never_written_hides_nothing() {
        any(
            "a_slot_reserved_but_never_written_hides_nothing",
            CHILD_TIMEOUT,
            |t| {
                t.op("before");
                // What a preempted appender leaves: a reservation with nothing written in it.
                let base = SHARED.load(Ordering::Acquire);
                let _ = header_u64(base, H_CURSOR).fetch_add(1, Ordering::AcqRel);
                t.op("after");
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t")
            },
            |r| {
                assert!(r.torn());
                assert_eq!(r.ops(), ["before", "after"]);
                let d = r.assert_died_at(SITE, "after");
                assert!(d.thread.body, "{d:?}");
            },
        );
    }

    /// An op longer than `TEXT_MAX` is cut and flagged
    #[test]
    fn a_long_op_is_cut_and_flagged() {
        any(
            "a_long_op_is_cut_and_flagged",
            CHILD_TIMEOUT,
            |t| {
                t.op(&"x".repeat(TEXT_MAX));
                t.op(&"y".repeat(TEXT_MAX.wrapping_add(1)));
            },
            |r| {
                assert_eq!(r.outcome(), &Outcome::Returned, "{}", r.stderr());
                assert!(r.truncated());
                let ops = r.ops();
                assert_eq!(ops.len(), 2);
                assert_eq!(ops[0], "x".repeat(TEXT_MAX));
                assert_eq!(ops[1], "y".repeat(TEXT_MAX));
            },
        );
    }

    /// Recurses until the stack overflows.
    #[inline(never)]
    fn recurse(n: u64) -> u64 {
        let a = [n; 64];
        std::hint::black_box(&a);
        if std::hint::black_box(n) == u64::MAX {
            return 0;
        }
        recurse(std::hint::black_box(n.wrapping_add(1))).wrapping_add(a[0])
    }

    /// The control for `a_stack_overflow_dies_as_without_the_harness`: without the harness, a
    /// stack overflow in this test binary is std's death, its diagnostic and `SIGABRT`.
    #[test]
    fn a_stack_overflow_without_the_harness_aborts() {
        const ENV: &str = "REIMS_VGPU_DEATH_OVERFLOW_CONTROL";
        if std::env::var_os(ENV).is_some() {
            let _ = recurse(0);
            return;
        }
        let out = Command::new(std::env::current_exe().unwrap())
            .args([
                "testkit::death::tests::a_stack_overflow_without_the_harness_aborts",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(ENV, "1")
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.signal(), Some(6), "{err}");
        assert!(err.contains("overflowed its stack"), "{err}");
    }

    death_test!(
        /// Under the harness a stack overflow dies as it does without it (std's diagnostic and
        /// `SIGABRT`), marked on the overflowing thread
        a_stack_overflow_dies_as_without_the_harness,
        |t| {
            t.op("recursing");
            let _ = recurse(0);
        },
        |r| {
            let d = r.died();
            assert_eq!(d.signal, 6, "{}", r.stderr());
            assert!(d.thread.body, "{d:?}");
            assert_eq!(d.op.as_deref(), Some("recursing"));
            assert_eq!(d.id, None);
            assert!(
                r.stderr().contains("overflowed its stack"),
                "{}",
                r.stderr()
            );
            // std's line is one death's, and one death alone did not race.
            assert!(!d.raced, "{d:?}");
            assert_eq!(d.abort_lines.len(), 1, "{d:?}");
            assert!(
                d.abort_lines[0].contains("fatal runtime error: stack overflow"),
                "{d:?}"
            );
        }
    );

    /// A death after the body returned fails a death test, as it must when `f` returns first. The
    /// body returns (`finish`, as `run`'s child records it), then a thread it started dies while
    /// the process is still up.
    #[test]
    #[should_panic(expected = "expected a death, but the body returned before it")]
    fn a_death_after_the_body_returned_fails() {
        let full = test_name(module_path!(), "a_death_after_the_body_returned_fails");
        if std::env::var_os(ENV_CHILD).is_some_and(|v| v == full.as_str()) {
            let child = enter();
            child.op("body: returning");
            child.finish();
            named("late", || {
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t")
            });
            park_forever()
        }
        let r = spawn(&full, Options::default());
        // It did die, at the site, after the body returned.
        let d = r.died();
        assert!(d.id.as_ref().is_some_and(|id| id.is(SITE)), "{d:?}");
        assert_eq!(d.thread.name(), Some("late"));
        assert!(r.returned_before_end());
        r.expect(Expect::Death);
    }

    /// The control for `a_death_deferred_past_the_call_fails`: a death test fails when the death
    /// comes after `f` returned ("returned without dying"). The late thread dies only once
    /// `expect_death` has judged.
    #[test]
    fn expect_death_fails_when_the_death_comes_after_f_returns() {
        let go = Arc::new(AtomicBool::new(false));
        let g2 = go.clone();
        let judged = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::testkit::capture::expect_death(SITE, move || {
                let slot = crate::testkit::capture::current_slot();
                named("late", move || {
                    crate::testkit::capture::with_slot(slot, || {
                        wait_until(|| g2.load(Ordering::Acquire));
                        crate::testkit::capture::die(
                            crate::raise::death::DeathKind::Abort,
                            "site_t",
                        )
                    })
                });
            });
        }));
        go.store(true, Ordering::Release);
        let p = judged.unwrap_err();
        let msg = p.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(msg.contains("returned without dying"), "{msg}");
    }

    death_test!(
        /// A planted wrong variant that defers a death inside a call onto a queue block, run once
        /// the body has returned, fails a death test whether the child exits first or the block
        /// dies first
        #[should_panic(expected = "expected a death")]
        a_death_deferred_past_the_call_fails,
        |t| {
            let q = crate::rt::queue::SerialQueue::new(
                &crate::testkit::threads::FakeThreads::default(),
                "Q",
            );
            t.op("body: calling");
            trace("api: called");
            q.exec_async(|| {
                let base = SHARED.load(Ordering::Acquire);
                wait_until(|| header_u32(base, H_STATE).load(Ordering::Acquire) == STATE_RETURNED);
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t")
            });
            // The call returned: the body is done.
            std::mem::forget(q);
        },
        |_| {}
    );

    /// A thread blocked behind the death that goes on after the death marker fails a death test by
    /// default: a planted wrong death path releases the held lock on its way to the abort. The
    /// marker is written first, as the handler writes it before the kill lands, so the order is
    /// fixed.
    #[test]
    #[should_panic(expected = "recorded after the end")]
    fn a_lock_released_on_the_way_to_the_abort_fails() {
        run(
            Expect::Death,
            Options::default(),
            module_path!(),
            "a_lock_released_on_the_way_to_the_abort_fails",
            |t| {
                let lock = Arc::new(Mutex::new(()));
                let held = lock.lock();
                let (waiting, locked) = (
                    Arc::new(AtomicBool::new(false)),
                    Arc::new(AtomicBool::new(false)),
                );
                let (l2, w2, k2) = (lock.clone(), waiting.clone(), locked.clone());
                named("second", move || {
                    trace("second: waiting");
                    w2.store(true, Ordering::Release);
                    let _g = l2.lock();
                    trace("second: locked");
                    k2.store(true, Ordering::Release);
                    park_forever()
                });
                wait_until(|| waiting.load(Ordering::Acquire));
                t.op("first: dying");
                append_here(K_DEATH, 6, b"");
                drop(held);
                wait_until(|| locked.load(Ordering::Acquire));
                std::process::abort()
            },
            |_| {},
        );
    }

    const A_SITE: DeathSite = DeathSite {
        kind: DeathKind::Abort,
        file: "site_a",
        line: 0,
    };
    const B_SITE: DeathSite = DeathSite {
        kind: DeathKind::Abort,
        file: "site_b",
        line: 0,
    };

    /// On a thread named `name`: what `fatal` leaves between its line and its abort, a thread
    /// that printed `file:line`'s death line and was not marked yet. Returns once it printed.
    fn printed_not_marked(name: &str, file: &'static str, line: u32) {
        printed_line_not_marked(
            name,
            format!("reims-vgpu: fatal: {file} [site Abort {file}:{line}]"),
        );
    }

    /// On a thread named `name`: what an abort leaves between its line and its `abort()`, a
    /// thread that printed `text` and was not marked yet. Returns once it printed.
    fn printed_line_not_marked(name: &str, text: String) {
        let printed = Arc::new(AtomicBool::new(false));
        let p2 = printed.clone();
        named(name, move || {
            use std::io::Write as _;
            let _ = writeln!(std::io::stderr(), "{text}");
            p2.store(true, Ordering::Release);
            park_forever()
        });
        wait_until(|| printed.load(Ordering::Acquire));
    }

    /// The ids' files and lines.
    fn sites(d: &Died) -> Vec<(&str, u32)> {
        d.ids.iter().map(|i| (i.file.as_str(), i.line)).collect()
    }

    /// Asserts that `assert_died(site)` fails on `r`.
    fn assert_died_fails(r: &Report, site: DeathSite) {
        let paired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = r.assert_died(site);
        }));
        assert!(
            paired.is_err(),
            "assert_died({site:?}) passed a raced death"
        );
    }

    death_test!(
        /// Two deaths' lines are not paired with the one thread marked: `a` printed its line and
        /// `b` died at its own site
        two_death_lines_are_not_paired,
        |_| {
            printed_not_marked("a", "site_a", 0);
            named("b", || {
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_b")
            });
            park_forever()
        },
        |r| {
            let d = r.died();
            assert!(d.raced, "{d:?}");
            assert_eq!(d.id, None);
            assert_eq!(d.thread.name(), Some("b"));
            assert_eq!(sites(d), [("site_a", 0), ("site_b", 0)]);
            assert_died_fails(r, A_SITE);
            assert_died_fails(r, B_SITE);
        }
    );

    /// `__builtin_trap`'s death on arm64: `SIGTRAP` on the calling thread, which ends the process
    /// and prints nothing.
    fn trap() -> ! {
        sys::raise(sys::SIGTRAP);
        panic!("SIGTRAP returned")
    }

    death_test!(
        /// A death line is not paired with a thread that died of another signal than `fatal`'s
        /// `SIGABRT`: `a` printed its line and `b` trapped
        a_death_line_beside_a_trap_is_not_paired,
        |_| {
            printed_not_marked("a", "site_a", 0);
            named("b", || trap());
            park_forever()
        },
        |r| {
            let d = r.died();
            assert!(d.raced, "{d:?}");
            assert_eq!(d.id, None);
            assert_eq!(d.signal, 5);
            assert_eq!(d.thread.name(), Some("b"));
            assert_eq!(sites(d), [("site_a", 0)]);
            assert_died_fails(r, A_SITE);
        }
    );

    death_test!(
        /// A death line is not paired when two threads were marked dying: `a` was marked (its
        /// signal had not ended the process yet) and `b` died at its site
        two_marked_threads_are_not_paired,
        |_| {
            let marked = Arc::new(AtomicBool::new(false));
            let m2 = marked.clone();
            named("a", move || {
                append_here(K_DEATH, 6, b"");
                m2.store(true, Ordering::Release);
                park_forever()
            });
            wait_until(|| marked.load(Ordering::Acquire));
            named("b", || {
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_b")
            });
            park_forever()
        },
        |r| {
            let d = r.died();
            assert!(d.raced, "{d:?}");
            assert_eq!(d.id, None);
            assert_eq!(d.thread.name(), Some("a"));
            assert_eq!(sites(d), [("site_b", 0)]);
            assert_died_fails(r, B_SITE);
        }
    );

    death_test!(
        /// A death test whose body panics fails (the outcome check)
        #[should_panic(expected = "expected a death")]
        a_death_test_whose_body_panics_fails,
        |_| panic!("boom"),
        |_| {}
    );

    hang_test!(
        /// A hang test whose body dies fails (the outcome check)
        #[should_panic(expected = "expected a hang")]
        a_hang_test_whose_body_dies_fails,
        |_| crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t"),
        |_| {}
    );

    // A body that panics before the death.

    /// A death after the body panicked fails a death test with `f`'s panic, as it must when `f`
    /// panics first. The body unwinds past its `Child` (as `run`'s does), then a thread it started
    /// dies while the process is still up, so the order is fixed.
    #[test]
    #[should_panic(expected = "expected a death, but the body panicked before it")]
    fn a_death_after_the_body_panicked_fails() {
        let full = test_name(module_path!(), "a_death_after_the_body_panicked_fails");
        if std::env::var_os(ENV_CHILD).is_some_and(|v| v == full.as_str()) {
            let unwound = std::panic::catch_unwind(|| {
                let child = enter();
                child.op("body: checking");
                panic!("the body's own check failed")
            });
            assert!(unwound.is_err());
            named("late", || {
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t")
            });
            park_forever()
        }
        let r = spawn(&full, Options::default());
        // It did die, at the site, after the body panicked.
        let d = r.died();
        assert!(
            !d.raced && d.id.as_ref().is_some_and(|id| id.is(SITE)),
            "{d:?}"
        );
        assert_eq!(d.thread.name(), Some("late"));
        assert_eq!(d.op.as_deref(), Some("body: checking"));
        assert!(r.panicked_before_end() && !r.returned_before_end());
        assert!(r.stderr().contains("the body's own check failed"));
        r.expect(Expect::Death);
    }

    /// The control for `a_death_after_the_body_panicked_fails`: a death test fails with `f`'s panic
    /// when `f` panics before the death. The late thread dies only once `expect_death` has judged.
    #[test]
    fn expect_death_fails_with_fs_panic_when_f_panics_first() {
        let go = Arc::new(AtomicBool::new(false));
        let g2 = go.clone();
        let judged = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::testkit::capture::expect_death(SITE, move || {
                let slot = crate::testkit::capture::current_slot();
                named("late", move || {
                    crate::testkit::capture::with_slot(slot, || {
                        wait_until(|| g2.load(Ordering::Acquire));
                        crate::testkit::capture::die(
                            crate::raise::death::DeathKind::Abort,
                            "site_t",
                        )
                    })
                });
                panic!("the body's own check failed");
            });
        }));
        go.store(true, Ordering::Release);
        let p = judged.unwrap_err();
        let msg = p
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| p.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert!(msg.contains("the body's own check failed"), "{msg}");
    }

    death_test!(
        /// A death test whose body's own check fails before a death fails, whether the child exits
        /// first (101) or the death lands after the body panicked
        #[should_panic(expected = "expected a death")]
        a_death_test_whose_body_check_fails_before_the_death_fails,
        |t| {
            named("late", || {
                let base = SHARED.load(Ordering::Acquire);
                wait_until(|| header_u32(base, H_STATE).load(Ordering::Acquire) == STATE_PANICKED);
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "site_t")
            });
            t.op("body: checking");
            assert_eq!(
                std::hint::black_box(1 + 1),
                3,
                "the body's own check failed"
            );
        },
        |r| {
            let _ = r.assert_died_at(SITE, "body: checking");
        }
    );

    /// `Child::hang` resumes `f`'s panic on the body's thread, which records that the body panicked
    #[test]
    fn a_hang_whose_f_panics_records_that_the_body_panicked() {
        any(
            "a_hang_whose_f_panics_records_that_the_body_panicked",
            CHILD_TIMEOUT,
            |t| {
                let clock = Arc::new(VirtualClock::new());
                t.hang(&*clock, Duration::from_secs(1), || panic!("f failed"));
            },
            |r| {
                assert_eq!(r.outcome(), &Outcome::Failed { code: Some(101) });
                assert!(r.stderr().contains("f failed"), "{}", r.stderr());
                let last = r.records().last().map(|x| x.kind);
                assert_eq!(last, Some(RecordKind::Panicked));
                assert!(r.panicked_before_end());
            },
        );
    }

    // The lines other aborts print are deaths too.

    /// In the child: a panic hook of the D9 hook's shape (crates/reims-vgpu/src/singleton.rs,
    /// `install_panic_hook`). On the thread named `b` it writes `<prefix>: <info>` and aborts;
    /// any other panic goes to the previous hook.
    fn install_d9_like_hook(prefix: &'static str) {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if std::thread::current().name() == Some("b") {
                use std::io::Write as _;
                let _ = writeln!(std::io::stderr(), "{prefix}: {info}");
                std::process::abort();
            }
            previous(info);
        }));
    }

    /// Checks a child where `a` printed its death line and was never marked, while `b` died in
    /// another abort that printed `line`.
    fn check_a_beside_another_abort(r: &Report, line: &str) {
        let d = r.died();
        assert!(d.raced, "{d:?}");
        assert_eq!(d.id, None);
        assert_eq!(d.signal, 6);
        assert_eq!(d.thread.name(), Some("b"));
        assert_eq!(sites(d), [("site_a", 0)]);
        assert_eq!(d.abort_lines.len(), 1, "{d:?}");
        assert!(d.abort_lines[0].contains(line), "{d:?}");
        assert_died_fails(r, A_SITE);
    }

    death_test!(
        /// A death line is not paired with a thread that died in the library's D9 panic hook,
        /// which prints a line of its own before its `abort()`: `a` printed its line and `b`
        /// panicked
        a_death_line_beside_a_panic_hook_abort_is_not_paired,
        |_| {
            install_d9_like_hook("reims-vgpu");
            printed_not_marked("a", "site_a", 0);
            named("b", || panic!("fake: unscripted call"));
            park_forever()
        },
        |r| check_a_beside_another_abort(r, "reims-vgpu: panicked at ")
    );

    /// A panic that cannot unwind past its `extern "C"` frame: std aborts the process.
    extern "C" fn a_panic_in_a_trampoline() {
        panic!("fake: unscripted call");
    }

    death_test!(
        /// A death line is not paired with a thread std aborted, with a line of its own, for a
        /// panic that could not unwind: `a` printed its line and `b` panicked inside an
        /// `extern "C"` function
        a_death_line_beside_a_std_abort_is_not_paired,
        |_| {
            printed_not_marked("a", "site_a", 0);
            named("b", a_panic_in_a_trampoline);
            park_forever()
        },
        |r| check_a_beside_another_abort(r, "panic. aborting.")
    );

    death_test!(
        /// A death line is not paired with a thread std's allocation-error hook aborted, with a
        /// line of its own
        a_death_line_beside_an_allocation_failure_is_not_paired,
        |_| {
            printed_not_marked("a", "site_a", 0);
            named("b", || {
                std::alloc::handle_alloc_error(std::alloc::Layout::new::<[u64; 8]>())
            });
            park_forever()
        },
        |r| check_a_beside_another_abort(r, "memory allocation of 64 bytes failed")
    );

    death_test!(
        /// Another abort's line beside a thread that died of another signal than `SIGABRT` is two
        /// deaths: `a` printed a D9 panic hook's line and `b` trapped
        an_abort_line_beside_a_trap_is_raced,
        |_| {
            printed_line_not_marked(
                "a",
                "reims-vgpu: panicked at crates/x.rs:1:1:\nfake: unscripted call".to_owned(),
            );
            named("b", || trap());
            park_forever()
        },
        |r| {
            let d = r.died();
            assert!(d.raced, "{d:?}");
            assert_eq!((d.id.as_ref(), d.ids.len()), (None, 0), "{d:?}");
            assert_eq!(d.abort_lines.len(), 1, "{d:?}");
            assert_eq!(d.signal, 5);
            assert_eq!(d.thread.name(), Some("b"));
        }
    );

    death_test!(
        /// A D9 panic hook's abort alone is one death with no site: nothing raced
        a_panic_hook_abort_alone_is_one_death,
        |t| {
            install_d9_like_hook("reims-vgpu");
            t.op("body: calling");
            named("b", || panic!("fake: unscripted call"));
            park_forever()
        },
        |r| {
            let d = r.died();
            assert!(!d.raced, "{d:?}");
            assert_eq!((d.id.as_ref(), d.ids.len()), (None, 0), "{d:?}");
            assert_eq!(d.abort_lines.len(), 1, "{d:?}");
            assert!(
                d.abort_lines[0].contains("reims-vgpu: panicked at "),
                "{d:?}"
            );
            assert_eq!(d.thread.name(), Some("b"));
            assert_eq!(d.op.as_deref(), Some("body: calling"));
        }
    );
}
