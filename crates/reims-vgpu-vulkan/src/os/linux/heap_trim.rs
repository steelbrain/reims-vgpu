//! Handing back to Linux the heap that a heavy pipeline make leaves free ([`make_ended`]). This
//! runs only where the process's allocator is glibc's (`target_env = "gnu"`). musl's allocator,
//! macOS's and Windows' differ and are out of scope: there `crate::os`'s `make_ended` does nothing.
//!
//! **Why.** A pipeline's make lowers its functions (rt's translation) in the process, on the thread
//! that makes it, and then the driver compiles them. Past rt's stage bound the compile helper does
//! the compile, but the lowering still runs here. A heavy kernel's translation peaks at hundreds
//! of MiB live and frees nearly all of it as the make returns. glibc, though, hands back to Linux
//! only the free top of a heap past its trim threshold. The free chunks below the top stay
//! resident until a later make reuses them, so the process sits at its largest make's level.
//!
//! Measured on Linux, glibc 2.44, with a kernel of the modelling app:
//! - the kernel peaked at 250 MiB live and fell to 14 MiB live after its make;
//! - the process still held 158 MiB resident, about 140 MiB of it free below the heap's top;
//! - turning tcache and fastbins off changed nothing, so this is not small allocations' churn;
//! - across a launch, the process held up to about 450 MiB this way.
//!
//! `malloc_trim(0)` hands back the free pages of every arena, below the top too. It took that
//! make's 158 MiB to 32 MiB, and a larger kernel's 307 MiB to 32 MiB.
//!
//! **Why not the allocator's settings.** `M_TRIM_THRESHOLD` and `M_TOP_PAD` decide only how much
//! of a heap's free top glibc keeps. The memory here lies below the top, which no setting hands
//! back. `M_MMAP_THRESHOLD` would give every block past it a mapping of its own, unmapped as it is
//! freed. That costs a system call each way and a page fault on each page, at every make, on the
//! make's own path, which the guest waits for. Blocks below that threshold would still be left
//! free in the heap. A trim after the make costs the make nothing and hands back the most.
//!
//! **When.** After a compile-pool make (a pipeline's create, a library's, or a command buffer's
//! dispatch made ahead) whose wall time reached [`HEAVY`]. Every make has a wall time, while a
//! fill's weight exists only where a fill ran. A make held for its stage is made again on the pool
//! once its fill ends, with its lowering, so that second make is timed too.
//!
//! **Where.** On a thread of its own, `reims-vgpu.heap-trim`, started at the first heavy make. No
//! guest waits on that thread. The make's thread only asks ([`Trimmer::ask`]): it takes a lock and
//! wakes the trim thread. What is left is the trim's locks. glibc's trim takes each arena's lock in
//! turn while it walks that arena's free chunks. A thread the guest waits on (a make's, or the
//! ring's) that allocates from the arena being walked waits until that walk ends, at most the
//! trim's own time. A cold launch of the modelling app on a 780M (Linux, glibc) ran 13 trims, the
//! longest 5.04 ms; whether any such thread waited on one was not measured. Each trim's time is
//! logged.
//!
//! **Coalesced.** One trim runs at a time. All the makes that end while it runs ask, together, for
//! one more trim, which runs after it. A burst of heavy makes therefore trims at most one trim
//! after another.
//!
//! **Logged.** Each trim's line gives:
//! - the process's resident bytes before and after (`/proc/self/statm`);
//! - the private part of those bytes, which is resident less shared (the heap's, and not the
//!   guest's RAM, which is shared);
//! - the trim's time and its thread.
//!
//! The first trim's line is at info, and later ones are at debug, counted (`core::logging`'s once
//! per kind). The counts line carries the trims (`heap-trims`), their time in microseconds
//! (`heap-trim-us`) and the resident KiB they handed back (`heap-trim-returned-kib`).

use std::sync::Once;
use std::time::Duration;

use crate::core::platform::os::{Clock as _, Threads as _};
use crate::core::sync::{Condvar, Mutex};

use super::clock::LinuxClock;
use super::threads::LinuxThreads;

/// The wall time at which a make counts as heavy, so that a trim follows it. Measured in cold
/// launches of the modelling app on a Radeon 780M, an RTX 3090 and Iris Xe: 15 to 16 makes per
/// launch take 250 ms or more, the heavy kernels' among them, and the next longest took at most
/// 210 ms. Below this time are the many light makes, whose freed memory later makes reuse.
const HEAVY: Duration = Duration::from_millis(250);

/// The trim thread's name.
const THREAD: &str = "reims-vgpu.heap-trim";

/// The trim thread's stack, the same as a timer thread's: the trim and its line need little.
const STACK: usize = 256 * 1024;

/// The trim line's kind, which makes the line once per kind (`core::logging`).
const KIND: &str = "heap trimmed";

/// The counts line's code for the trims run.
const TRIMS: &str = "heap-trims";
/// The counts line's code for the trims' time, in microseconds.
const TRIM_US: &str = "heap-trim-us";
/// The counts line's code for the resident KiB the trims handed back.
const RETURNED_KIB: &str = "heap-trim-returned-kib";

/// The process's trims (the module docs).
static TRIMMER: Trimmer = Trimmer::new();

/// Starts the trim thread at the first heavy make.
static STARTED: Once = Once::new();

/// A compile-pool make ended after `took` of wall time. Where that reaches [`HEAVY`], a trim of
/// the heap follows on the trim thread, coalesced with the others asked for (the module docs).
pub fn make_ended(took: Duration) {
    if !heavy(took) {
        return;
    }
    STARTED.call_once(|| LinuxThreads::new().spawn(THREAD, STACK, Box::new(serve)));
    TRIMMER.ask();
}

/// Whether a make of `took` is heavy, so that a trim follows it.
fn heavy(took: Duration) -> bool {
    took >= HEAVY
}

/// The trim thread's body: one trim after another, each when asked.
fn serve() {
    loop {
        TRIMMER.serve_one(trim);
    }
}

/// The trims asked for, and the wake of the thread that runs them.
struct Trimmer {
    /// How many heavy makes ended since the trim thread last took its asks.
    asked: Mutex<u32>,
    /// Wakes the trim thread when a make asks.
    woken: Condvar,
}

impl Trimmer {
    /// No trim asked for.
    const fn new() -> Trimmer {
        Trimmer {
            asked: Mutex::new(0),
            woken: Condvar::new(),
        }
    }

    /// A heavy make ended: one more ask, and the trim thread woken where it waits.
    fn ask(&self) {
        {
            let mut asked = self.asked.lock();
            *asked = asked.saturating_add(1);
        }
        self.woken.notify_one();
    }

    /// Waits for an ask, then takes every ask made since the last take and runs `trim` once with
    /// their count. Makes that end during the trim ask for the next one.
    fn serve_one(&self, trim: impl FnOnce(u32)) {
        let asks = {
            let mut asked = self.asked.lock();
            while *asked == 0 {
                self.woken.wait(&mut asked);
            }
            std::mem::take(&mut *asked)
        };
        trim(asks);
    }
}

/// Trims the heap for the `asks` heavy makes that asked, then logs the trim and counts it (the
/// module docs).
fn trim(asks: u32) {
    let clock = LinuxClock::new();
    let before = Resident::now();
    let started = clock.now_mono();
    // SAFETY: `malloc_trim` takes no pointer of ours. glibc walks its own arenas, each under its
    // lock, and hands back only pages no allocation holds.
    let _ = unsafe { libc::malloc_trim(0) };
    let took = clock.now_mono().saturating_duration_since(started);
    let after = Resident::now();
    let returned = match (before, after) {
        (Some(before), Some(after)) => before.total.saturating_sub(after.total),
        _ => 0,
    };
    crate::answer::count_kind(TRIMS);
    crate::answer::count_kind_by(TRIM_US, u64::try_from(took.as_micros()).unwrap_or(u64::MAX));
    crate::answer::count_kind_by(RETURNED_KIB, returned / 1024);
    let mib = |bytes: u64| bytes as f64 / f64::from(1u32 << 20);
    let shown = |r: Option<Resident>| match r {
        Some(r) => format!(
            "{:.1} MiB ({:.1} MiB private)",
            mib(r.total),
            mib(r.private)
        ),
        None => "unknown".to_owned(),
    };
    log::info!(
        kind:% = KIND;
        "heap trimmed after {asks} heavy pipeline makes: resident {} before, {} after; the trim \
         took {:.2} ms on {}",
        shown(before),
        shown(after),
        took.as_secs_f64() * 1e3,
        std::thread::current().name().unwrap_or("an unnamed thread"),
    );
}

/// The process's resident bytes, as `/proc/self/statm` gives them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Resident {
    /// Every resident page's bytes.
    total: u64,
    /// The private pages' bytes: resident less shared. Shared covers file and shared-memory pages,
    /// the guest's RAM among them.
    private: u64,
}

impl Resident {
    /// The process's resident bytes now. `None` where `/proc` does not answer.
    fn now() -> Option<Resident> {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        Resident::of_statm(&statm, page_size())
    }

    /// The resident bytes in `statm`, a `/proc/<pid>/statm` line whose counts are pages of `page`
    /// bytes: its second field is the resident pages, and its third the shared ones among them.
    fn of_statm(statm: &str, page: u64) -> Option<Resident> {
        let mut pages = statm
            .split_ascii_whitespace()
            .skip(1)
            .map(str::parse::<u64>);
        let resident = pages.next()?.ok()?;
        let shared = pages.next()?.ok()?;
        Some(Resident {
            total: resident.saturating_mul(page),
            private: resident.saturating_sub(shared).saturating_mul(page),
        })
    }
}

/// The host's page size, in which `/proc` counts pages.
fn page_size() -> u64 {
    // SAFETY: `sysconf` reads a constant of the system's.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page).unwrap_or(4096)
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;

    /// A make of `HEAVY` or more is heavy, and a lighter one is not
    #[test]
    fn a_make_is_heavy_from_heavy_on() {
        assert!(!heavy(Duration::ZERO));
        assert!(!heavy(HEAVY - Duration::from_nanos(1)));
        assert!(heavy(HEAVY));
        assert!(heavy(Duration::from_secs(5)));
    }

    /// Makes that end while a trim runs ask, together, for one more trim, and no further trim
    /// follows it
    #[test]
    fn makes_that_end_during_a_trim_ask_for_one_more() {
        let trimmer = Trimmer::new();
        trimmer.ask();
        let mut trims = Vec::new();
        trimmer.serve_one(|asks| {
            trims.push(asks);
            for _ in 0..3 {
                trimmer.ask();
            }
        });
        trimmer.serve_one(|asks| trims.push(asks));
        assert_eq!(trims, [1, 3]);
        assert_eq!(*trimmer.asked.lock(), 0, "nothing more asked");
    }

    /// The trim thread waits until a make asks, and then trims once
    #[test]
    fn a_trim_waits_for_a_heavy_make_to_ask() {
        let trimmer = &Trimmer::new();
        let (trimmed, trims) = mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(move || trimmer.serve_one(|asks| trimmed.send(asks).unwrap()));
            assert!(
                trims.recv_timeout(Duration::from_millis(50)).is_err(),
                "no trim before an ask"
            );
            trimmer.ask();
            assert_eq!(trims.recv_timeout(Duration::from_secs(10)), Ok(1));
        });
    }

    /// A statm line's resident and private bytes are its resident pages, and those less its shared
    /// ones, in bytes. A line without them answers nothing
    #[test]
    fn resident_bytes_are_statms_pages() {
        assert_eq!(
            Resident::of_statm("1000 300 100 5 0 400 0\n", 4096),
            Some(Resident {
                total: 300 * 4096,
                private: 200 * 4096
            })
        );
        assert_eq!(Resident::of_statm("1000 300", 4096), None);
        assert_eq!(Resident::of_statm("1000 x 100", 4096), None);
        assert!(Resident::now().is_some_and(|r| r.total > 0 && r.private <= r.total));
    }
}
