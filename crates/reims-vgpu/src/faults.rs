//! On Windows, the lines a fault writes on stderr: what was raised, at which module and offset,
//! and what called it.
//!
//! A fault in a Windows process (an access violation, an illegal instruction, a stack overflow,
//! the rest of [`FAULTS`]) ends it with the exception's code as its exit status, and where
//! Windows Error Reporting writes no report, nothing says where it happened (QEMU's access
//! violations on the UHD Graphics 630 host, 2026-11-16, left no event in the Application or System
//! logs). [`install`] adds a vectored exception handler, once per process, which writes for each
//! fault, before any frame's handler sees it:
//!
//! - `reims-vgpu: exception <code> at <module>+0x<offset> (<what>, a read of 0x<address>), thread
//!   <id>; the process ends unless a handler takes it`: the faulting instruction's module and its
//!   offset from the module's base, and for an access violation the access and its address;
//! - `reims-vgpu: exception <code> at <module>+0x<offset>: called from <module>+0x<offset>, …`: the
//!   return addresses of up to [`CALLERS`] frames above it, unwound from the fault's context by
//!   the modules' unwind tables (`RtlVirtualUnwind`);
//! - `reims-vgpu: exception <code> at <module>+0x<offset>: on thread <id> "<name>"`, where the
//!   thread has a name.
//!
//! The handler takes nothing: it answers `EXCEPTION_CONTINUE_SEARCH`, and the process goes on as
//! it would have without it, ending where no handler takes the exception. It is the last of the
//! process's vectored handlers when it is added, and the Vulkan crate's VM handler
//! (`reims-vgpu-vulkan/src/os/windows/vm.rs`), added at the front of the list, takes its own
//! faults before it. A fault a frame's handler takes (a driver's probe
//! under `__try`) writes its lines too, so each site, a code at an address, writes them once per
//! process ([`first_at`]). A stack overflow writes the first line alone, as little stack is left.
//!
//! Each line is one `WriteFile` to stderr, from a buffer on the faulting thread's stack, with no
//! lock and no allocation of the library's: the module's name is the loader's
//! (`GetModuleFileNameW`, under the loader lock), and the thread's name, read last, is allocated
//! by the system. A fault inside the handler (a stack the unwind cannot read) is left to the
//! process's other handlers, after the lines already written ([`Inside`]).
//!
//! The lines are diagnostics: the contract leaves crash messages free.

use std::ffi::c_void;
use std::fmt::Write as _;
use std::sync::Once;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use windows_sys::Win32::Foundation::{INVALID_HANDLE_VALUE, LocalFree};
use windows_sys::Win32::Storage::FileSystem::WriteFile;
use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, CONTEXT, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    EXCEPTION_RECORD, RtlLookupFunctionEntry, RtlPcToFileHeader, RtlVirtualUnwind,
    UNW_FLAG_NHANDLER,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW;
use windows_sys::Win32::System::Threading::{
    GetCurrentThread, GetCurrentThreadId, GetThreadDescription,
};

/// The exceptions that write lines, by code: those that end a process when no handler takes
/// them, each with what it says.
pub(crate) const FAULTS: [(u32, &str); 12] = [
    (0xC000_0005, "an access violation"),
    (0xC000_0006, "a page the system could not read in"),
    (0xC000_001D, "an illegal instruction"),
    (0xC000_0096, "a privileged instruction"),
    (0xC000_0094, "an integer division by zero"),
    (0xC000_0095, "an integer overflow"),
    (STACK_OVERFLOW, "a stack overflow"),
    (0xC000_008C, "an array index out of its bounds"),
    (0x8000_0002, "a misaligned access"),
    (0xC000_0374, "a corrupted heap"),
    (0xC000_0420, "a failed assertion"),
    (0x8000_0003, "a breakpoint"),
];

/// `STATUS_STACK_OVERFLOW`, whose handler writes its first line alone.
const STACK_OVERFLOW: u32 = 0xC000_00FD;

/// `STATUS_ACCESS_VIOLATION` and `STATUS_IN_PAGE_ERROR`, whose records name the access and its
/// address.
const ACCESS_FAULTS: [u32; 2] = [0xC000_0005, 0xC000_0006];

/// How many callers the second line names at most.
pub(crate) const CALLERS: usize = 24;

/// How many sites [`SEEN`] holds; past them, every fault writes its lines.
const SITES: usize = 64;

/// A line's bytes at most, its newline included: a longer line is cut.
const LINE: usize = 1536;

/// The sites that wrote their lines ([`site_key`]), 0 where a slot is free.
static SEEN: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];

/// The threads inside the handler, by id, 0 where a slot is free ([`Inside`]).
static INSIDE: [AtomicU32; 8] = [const { AtomicU32::new(0) }; 8];

/// Adds the handler (module docs), once per process.
pub(crate) fn install() {
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        // SAFETY: the handler reads the exception's record and context, the loader's tables and
        // the faulting thread's stack, and writes to stderr; it is never removed.
        unsafe { AddVectoredExceptionHandler(0, Some(on_exception)) };
    });
}

/// What the exception of `code` is, where it writes lines ([`FAULTS`]).
fn fault_name(code: u32) -> Option<&'static str> {
    FAULTS
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, what)| *what)
}

/// A site's key in [`SEEN`]: the code in the top half, over the address, never 0 (every code of
/// [`FAULTS`] has its top bit set, and a user-mode address is below 2^47).
fn site_key(code: u32, at: usize) -> u64 {
    (u64::from(code) << 32) ^ u64::try_from(at).unwrap_or(u64::MAX)
}

/// Whether `key` is new to `seen`, which then holds it; true too where `seen` is full.
fn first_at(seen: &[AtomicU64], key: u64) -> bool {
    for slot in seen {
        match slot.compare_exchange(0, key, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(held) if held == key => return false,
            Err(_) => {}
        }
    }
    true
}

/// A thread inside the handler, holding its slot in [`INSIDE`] until dropped: a fault the
/// handler itself meets finds its thread there, and goes to the process's other handlers.
struct Inside(Option<&'static AtomicU32>);

impl Inside {
    /// The thread `tid` entering the handler; `None` where it is inside already. Where every
    /// slot is held, the thread enters without one.
    fn enter(tid: u32) -> Option<Inside> {
        if INSIDE
            .iter()
            .any(|slot| slot.load(Ordering::Acquire) == tid)
        {
            return None;
        }
        let slot = INSIDE.iter().find(|slot| {
            slot.compare_exchange(0, tid, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        });
        Some(Inside(slot))
    }
}

impl Drop for Inside {
    fn drop(&mut self) {
        if let Some(slot) = self.0 {
            slot.store(0, Ordering::Release);
        }
    }
}

/// The handler (module docs). Takes nothing: every exception goes on. Its own frame is small,
/// so a stack overflow's line fits in what is left of the stack; the rest is [`write_more`]'s.
unsafe extern "system" fn on_exception(pointers: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: Windows hands the handler live exception pointers for the call.
    let (record, context) = unsafe { ((*pointers).ExceptionRecord, (*pointers).ContextRecord) };
    if record.is_null() || context.is_null() {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: as above: the record lives for the call.
    let record = unsafe { &*record };
    let code = record.ExceptionCode.cast_unsigned();
    let Some(what) = fault_name(code) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    // SAFETY: no precondition.
    let tid = unsafe { GetCurrentThreadId() };
    let Some(_inside) = Inside::enter(tid) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    let at = record.ExceptionAddress.addr();
    if !first_at(&SEEN, site_key(code, at)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    write_first(code, what, record, tid);
    if code != STACK_OVERFLOW {
        // SAFETY: as above: the context lives for the call.
        write_more(code, at, tid, unsafe { &*context });
    }
    EXCEPTION_CONTINUE_SEARCH
}

/// The first line (module docs).
#[inline(never)]
fn write_first(code: u32, what: &str, record: &EXCEPTION_RECORD, tid: u32) {
    let mut line = Line::new();
    let _ = write!(line, "reims-vgpu: exception 0x{code:08x} at ");
    write_site(&mut line, record.ExceptionAddress.addr());
    let _ = write!(line, " ({what}");
    write_access(&mut line, code, record);
    let _ = write!(
        line,
        "), thread {tid}; the process ends unless a handler takes it"
    );
    line.write_out();
}

/// The callers' line, and the thread's name's where it has one (module docs), for the fault of
/// `code` at `at` in `context`.
#[inline(never)]
fn write_more(code: u32, at: usize, tid: u32, context: &CONTEXT) {
    let mut line = Line::new();
    let _ = write!(line, "reims-vgpu: exception 0x{code:08x} at ");
    write_site(&mut line, at);
    let _ = write!(line, ": called from");
    write_callers(&mut line, *context);
    line.write_out();
    let mut name = [0u16; 128];
    let len = thread_name(&mut name);
    if len > 0 {
        let mut line = Line::new();
        let _ = write!(line, "reims-vgpu: exception 0x{code:08x} at ");
        write_site(&mut line, at);
        let _ = write!(line, ": on thread {tid} \"");
        write_wide(&mut line, name.get(..len).unwrap_or_default());
        let _ = write!(line, "\"");
        line.write_out();
    }
}

/// For an access violation or a page that could not be read in, `, a read of 0x<address>` (or a
/// write, or an execution), from its record's parameters.
fn write_access(line: &mut Line, code: u32, record: &EXCEPTION_RECORD) {
    if !ACCESS_FAULTS.contains(&code) || record.NumberParameters < 2 {
        return;
    }
    let access = match record.ExceptionInformation[0] {
        0 => "a read",
        1 => "a write",
        8 => "an execution",
        _ => "an access",
    };
    let _ = write!(
        line,
        ", {access} of 0x{:016x}",
        record.ExceptionInformation[1]
    );
}

/// Writes `at` as `<module>+0x<offset>`, the module's file name and the offset from its base, or
/// as `0x<address> (in no module)`.
fn write_site(line: &mut Line, at: usize) {
    let mut base: *mut c_void = std::ptr::null_mut();
    // SAFETY: a query of the loader's tables into a live out-parameter.
    let found = unsafe { RtlPcToFileHeader(std::ptr::without_provenance(at), &mut base) };
    if found.is_null() || base.is_null() {
        let _ = write!(line, "0x{at:016x} (in no module)");
        return;
    }
    let mut path = [0u16; 260];
    let capacity = u32::try_from(path.len()).unwrap_or(0);
    // SAFETY: `base` is a loaded module's (the query above), and `path` a live buffer of
    // `capacity` units.
    let len = unsafe { GetModuleFileNameW(base, path.as_mut_ptr(), capacity) };
    let path = path
        .get(..usize::try_from(len).unwrap_or(0))
        .unwrap_or_default();
    let file = path
        .iter()
        .rposition(|&unit| unit == u16::from(b'\\') || unit == u16::from(b'/'))
        .map_or(path, |sep| {
            path.get(sep.saturating_add(1)..).unwrap_or_default()
        });
    if file.is_empty() {
        let _ = write!(line, "0x{:x}", base.addr());
    } else {
        write_wide(line, file);
    }
    let _ = write!(line, "+0x{:x}", at.wrapping_sub(base.addr()));
}

/// Appends ` <site>,` for each caller of the frame `context` is at, up to [`CALLERS`], unwound by
/// the modules' unwind tables. The faulting frame alone may have none (a leaf function, or a call
/// to an address no module holds), where its return address is at the top of its stack.
fn write_callers(line: &mut Line, mut context: CONTEXT) {
    let mut named = 0usize;
    while named < CALLERS {
        let pc = context.Rip;
        let mut base = 0u64;
        // SAFETY: a query of the loader's function tables into a live out-parameter.
        let entry = unsafe { RtlLookupFunctionEntry(pc, &mut base, std::ptr::null_mut()) };
        if entry.is_null() {
            if named > 0 {
                break;
            }
            let Ok(sp) = usize::try_from(context.Rsp) else {
                break;
            };
            // SAFETY: none can be given: the faulting frame's stack pointer, read where its
            // thread's stack should be. A read that faults is left to the process's other
            // handlers ([`Inside`]), after the first line.
            context.Rip = unsafe { std::ptr::with_exposed_provenance::<u64>(sp).read_unaligned() };
            context.Rsp = context.Rsp.wrapping_add(8);
        } else {
            let mut data: *mut c_void = std::ptr::null_mut();
            let mut frame = 0u64;
            // SAFETY: `entry` is the function entry of `pc` in the module at `base` (the query
            // above), and `context` a live copy the unwind rewrites to the caller's. A frame the
            // unwind cannot read faults, as above.
            unsafe {
                RtlVirtualUnwind(
                    UNW_FLAG_NHANDLER,
                    base,
                    pc,
                    entry,
                    &mut context,
                    &mut data,
                    &mut frame,
                    std::ptr::null_mut(),
                )
            };
        }
        let Ok(caller) = usize::try_from(context.Rip) else {
            break;
        };
        if caller == 0 {
            break;
        }
        let _ = write!(line, "{}", if named == 0 { " " } else { ", " });
        write_site(line, caller);
        named = named.saturating_add(1);
    }
    if named == 0 {
        let _ = write!(line, " nothing the unwind could name");
    }
}

/// The calling thread's name (`GetThreadDescription`) in `name`, cut at its end; its length.
fn thread_name(name: &mut [u16]) -> usize {
    let mut wide: *mut u16 = std::ptr::null_mut();
    // SAFETY: the calling thread's pseudo-handle, and a live out-parameter.
    let result = unsafe { GetThreadDescription(GetCurrentThread(), &mut wide) };
    if result < 0 || wide.is_null() {
        return 0;
    }
    let mut len = 0usize;
    for slot in name.iter_mut() {
        // SAFETY: the system's NUL-terminated string, read up to its NUL, or `name`'s length.
        let unit = unsafe { wide.add(len).read() };
        if unit == 0 {
            break;
        }
        *slot = unit;
        len = len.saturating_add(1);
    }
    // SAFETY: the string the system allocated for the call, freed once.
    unsafe { LocalFree(wide.cast()) };
    len
}

/// Appends UTF-16 `units`, an unpaired surrogate as U+FFFD.
fn write_wide(line: &mut Line, units: &[u16]) {
    for c in char::decode_utf16(units.iter().copied()) {
        let _ = line.write_char(c.unwrap_or(char::REPLACEMENT_CHARACTER));
    }
}

/// One line, on the faulting thread's stack, cut at [`LINE`] bytes.
pub(crate) struct Line {
    buf: [u8; LINE],
    len: usize,
}

impl Line {
    /// An empty line.
    pub(crate) fn new() -> Line {
        Line {
            buf: [0; LINE],
            len: 0,
        }
    }

    /// The line so far.
    #[cfg(test)]
    pub(crate) fn as_str(&self) -> &str {
        std::str::from_utf8(self.buf.get(..self.len).unwrap_or_default()).unwrap_or_default()
    }

    /// Ends the line and writes it to stderr in one `WriteFile`.
    fn write_out(mut self) {
        let end = self.len.min(LINE.saturating_sub(1));
        if let Some(byte) = self.buf.get_mut(end) {
            *byte = b'\n';
        }
        let bytes = self.buf.get(..end.saturating_add(1)).unwrap_or_default();
        // SAFETY: no precondition.
        let stderr = unsafe { GetStdHandle(STD_ERROR_HANDLE) };
        if stderr.is_null() || stderr == INVALID_HANDLE_VALUE {
            return;
        }
        let mut written = 0u32;
        // SAFETY: a live buffer of the length given, and a live out-parameter; no overlapped I/O.
        unsafe {
            WriteFile(
                stderr,
                bytes.as_ptr(),
                u32::try_from(bytes.len()).unwrap_or(0),
                &mut written,
                std::ptr::null_mut(),
            )
        };
    }
}

impl std::fmt::Write for Line {
    /// Appends what fits, cut at a character's boundary; the newline's byte stays free.
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let room = LINE.saturating_sub(1).saturating_sub(self.len);
        let mut take = s.len().min(room);
        while !s.is_char_boundary(take) {
            take = take.saturating_sub(1);
        }
        let src = s.as_bytes().get(..take).unwrap_or_default();
        if let Some(dst) = self.buf.get_mut(self.len..self.len.saturating_add(take)) {
            dst.copy_from_slice(src);
            self.len = self.len.saturating_add(take);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
