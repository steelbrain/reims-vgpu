use std::process::Command;
use std::sync::atomic::AtomicU64;

use windows_sys::Win32::System::Diagnostics::Debug::{
    SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX, SetErrorMode,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;

use super::*;

/// A store of one byte at `addr`, which faults where no page maps it.
#[inline(never)]
fn store_at(addr: usize) {
    // SAFETY: none: the store faults on purpose, at an address no page maps, and the child ends
    // there.
    unsafe {
        std::arch::asm!("mov byte ptr [{addr}], 1", addr = in(reg) addr, options(nostack));
    }
}

/// The offset of `f` from the base of this test binary.
fn offset_of(f: usize) -> usize {
    // SAFETY: the process's own executable's handle; no precondition.
    let base = unsafe { GetModuleHandleW(std::ptr::null()) };
    f - base.addr()
}

/// A fault writes its lines and the process ends as it would have: the faulting instruction's
/// module and offset (`store_at`'s, in this binary), the access and its address, then callers
/// that name this binary. In a child, which faults with the handler installed.
#[test]
fn a_fault_writes_its_module_offset_and_callers() {
    const TEST: &str = "faults::tests::a_fault_writes_its_module_offset_and_callers";
    if std::env::var_os("RVG_FAULT_CHILD").is_some() {
        // SAFETY: no precondition; the child writes no error report of the system's.
        unsafe { SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX) };
        install();
        install(); // once only: a second install adds no second handler
        store_at(0x18);
        unreachable!("the store faults");
    }
    let exe = std::env::current_exe().unwrap();
    let out = Command::new(&exe)
        .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
        .env("RVG_FAULT_CHILD", "1")
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(0xC000_0005_u32.cast_signed()),
        "{out:?}"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    let file = exe.file_name().unwrap().to_string_lossy();
    let head = format!("reims-vgpu: exception 0xc0000005 at {file}+0x");
    let firsts: Vec<&str> = stderr
        .lines()
        .filter(|l| {
            l.starts_with(&head) && l.ends_with("; the process ends unless a handler takes it")
        })
        .collect();
    assert_eq!(firsts.len(), 1, "{stderr}");
    let first = firsts[0];
    assert!(
        first.contains(" (an access violation, a write of 0x0000000000000018), thread "),
        "{first}"
    );
    let rest = &first[head.len()..];
    let hex = &rest[..rest.find(' ').unwrap()];
    let at = usize::from_str_radix(hex, 16).unwrap();
    let store = offset_of(store_at as fn(usize) as usize);
    assert!(
        (store..store + 64).contains(&at),
        "0x{at:x} vs store_at at 0x{store:x}"
    );
    let callers = stderr
        .lines()
        .find(|l| l.starts_with(&head) && l.contains(": called from "))
        .unwrap_or_else(|| panic!("no callers' line: {stderr}"));
    let called_from = &callers[callers.find(": called from ").unwrap()..];
    assert!(called_from.contains(&format!(" {file}+0x")), "{callers}");
    // libtest runs the test on a thread it names for the test.
    assert!(
        stderr.lines().any(|l| l.starts_with(&head)
            && l.contains(": on thread ")
            && l.ends_with(&format!("\"{TEST}\""))),
        "{stderr}"
    );
}

/// Each site writes its lines once: a key is new once, and a full table lets every key write.
#[test]
fn a_site_writes_once() {
    let seen: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    let (a, b, c) = (
        site_key(0xC000_0005, 0x1000),
        site_key(0xC000_001D, 0x1000),
        site_key(0xC000_0005, 0x2000),
    );
    assert_ne!(a, 0);
    assert_ne!(a, b);
    assert!(first_at(&seen, a));
    assert!(!first_at(&seen, a));
    assert!(first_at(&seen, b));
    assert!(first_at(&seen, c));
    assert!(first_at(&seen, c), "full: every key writes");
    assert!(!first_at(&seen, b));
}

/// A long line is cut at a character's boundary, with room left for its newline.
#[test]
fn a_long_line_is_cut_at_a_character() {
    let mut line = Line::new();
    let _ = write!(line, "{}", "a".repeat(LINE - 2));
    let _ = write!(line, "é");
    assert_eq!(line.as_str().len(), LINE - 2);
    let _ = write!(line, "b");
    assert_eq!(line.as_str().len(), LINE - 1);
    let _ = write!(line, "c");
    assert_eq!(line.as_str().len(), LINE - 1);
}
