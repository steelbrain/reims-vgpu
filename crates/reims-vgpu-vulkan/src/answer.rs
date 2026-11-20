//! The answer layer: the counts a boot logs of what the platform did otherwise than Metal's
//! fastest path or answered as when memory runs out, and the deaths that name a selector where
//! the platform cannot do what Metal does.
//!
//! - **Counts by code** ([`count_kind`], [`count_kind_by`], [`count_made`]): kinds the library
//!   counts outside rt's own metrics, each under a code of its own, which the counts line
//!   carries ([`degradation_counts`], [`write_counts`]): `reims-vgpu: degradation counts: <code>
//!   <n>, ...`, written at a device's reset and at the process's exit
//!   ([`crate::exit_gate`]). It is the line a boot's degradations are read from
//!   (docs/conventions.md). A kind's line is its caller's to log, once per kind.
//! - **Out of host memory** (`out_of_host_memory`, on Linux and Windows): a map, a file or
//!   memory the host refused for want of mappings, files, memory or commit charge where Metal's
//!   call needs none is answered as Metal's call answers when memory runs out, counted, and logged
//!   at each power of two; in strict mode the process ends there.
//! - **Strict mode** ([`strict`], `REIMS_VGPU_STRICT=1`, read once per process): out of host
//!   memory is a death rather than an answer. This crate's tests run strict (the tests'
//!   `relaxed` runs a call under the default).
//! - **Deaths that name a selector** ([`metal_reads_near_null`], [`unrepresentable`]): where Metal
//!   ends the process, or the platform has no form for a raw value the guest sends. Every death
//!   here is a platform death (`DeathId::Platform`): it ends the process, or, inside a refusal
//!   scope, refuses the op that met it.

use std::cell::Cell;

use crate::core::raise::death::{DeathId, DeathKind, fatal_with_detail};

/// Whether this process dies where it could answer as when memory runs out
/// (`out_of_host_memory`, on Linux and Windows). `REIMS_VGPU_STRICT` set to `1`, read once per process, turns it on;
/// by default the process answers and goes on. A test of this crate runs strict, but where the
/// tests' `relaxed` runs it on its thread.
pub fn strict() -> bool {
    #[cfg(test)]
    if let Some(strict) = TEST_STRICT.with(Cell::get) {
        return strict;
    }
    strict_process()
}

/// [`strict`] as the process is set, whatever a test's thread runs under (the tests' `relaxed`).
pub(crate) fn strict_process() -> bool {
    static STRICT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        cfg!(test) || std::env::var_os("REIMS_VGPU_STRICT").is_some_and(|v| v == "1")
    });
    *STRICT
}

#[cfg(test)]
thread_local! {
    /// The strictness a test runs its thread's calls under ([`relaxed`]), in place of the
    /// process's.
    static TEST_STRICT: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Runs `call` with the calls and notices on this thread under the default, not strict: a test's
/// of what the process does without `REIMS_VGPU_STRICT`.
#[cfg(test)]
pub(crate) fn relaxed<T>(call: impl FnOnce() -> T) -> T {
    /// Puts back the outer strictness on drop, an unwinding call's included.
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_STRICT.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(TEST_STRICT.with(|s| s.replace(Some(false))));
    call()
}

/// The line of the last death [`dies`] made in this process, which a test reads: the test
/// capture takes a death before its line is written.
#[cfg(test)]
pub(crate) static LAST_LINE: crate::core::sync::Mutex<Option<String>> =
    crate::core::sync::Mutex::new(None);

/// The death at `sel` of [`unrepresentable`], its line `line`.
fn dies(sel: &'static str, line: &str) -> ! {
    #[cfg(test)]
    {
        *LAST_LINE.lock() = Some(line.to_owned());
    }
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: sel,
        },
        line,
    )
}

/// The death of a send at which Metal reads `what` at a NULL or near-NULL address, in the zero
/// region every arm64 macOS executable maps (`__PAGEZERO`, below core's `ZERO_REGION_END`), where
/// the read ends the process with a segmentation fault.
pub fn metal_reads_near_null(sel: &'static str, what: &str) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::NullDeref,
            call: sel,
        },
        &format!("Metal reads {what} in the zero region (a segmentation fault)"),
    )
}

/// The death of a send whose raw value the platform's types cannot hold: `what` names it (a pixel
/// format no `MTLPixelFormat` is, an option bit the runtime does not define). Metal itself aborts
/// for some of these (a pixel format it refuses); for the others the runtime has no form.
pub fn unrepresentable(sel: &'static str, what: &str) -> ! {
    dies(sel, &format!("the Vulkan platform cannot hold {what}"))
}

/// How many times this process answered a call as when memory runs out (`out_of_host_memory`,
/// on Linux and Windows).
static OUT_OF_HOST_MEMORY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A map, a file or memory that this platform's host refused at `call` for want of mappings
/// (Linux's `vm.max_map_count`), of open files, of memory or of commit charge (Windows's), where
/// Metal's call needs none: in strict
/// mode the process ends there, naming `detail`, as it did before; by default the caller answers
/// as Metal's call does when memory runs out (`KERN_NO_SPACE`, no memory, a nil object), and the
/// answer is counted ([`out_of_host_memory_count`], [`degradation_counts`]) and logged at the
/// first, and at each power of two, at debug after the first (`core::logging`'s once per kind).
/// `detail` is asked only for a line.
#[cfg(any(target_os = "linux", windows))]
pub(crate) fn out_of_host_memory(call: &'static str, detail: impl FnOnce() -> String) {
    use std::sync::atomic::Ordering;
    if strict() {
        fatal_with_detail(
            DeathId::Platform {
                kind: DeathKind::Abort,
                call,
            },
            &detail(),
        );
    }
    let count = OUT_OF_HOST_MEMORY.fetch_add(1, Ordering::Relaxed) + 1;
    if count.is_power_of_two() {
        log::warn!(
            target: "reims_vgpu_vulkan::os",
            "{count} calls answered as when memory runs out, the latest at {call}: {}; \
             REIMS_VGPU_STRICT=1 makes each a death",
            detail()
        );
    }
}

/// How many times this process answered a call as when memory runs out (`out_of_host_memory`,
/// on Linux and Windows; none on macOS).
pub fn out_of_host_memory_count() -> u64 {
    OUT_OF_HOST_MEMORY.load(std::sync::atomic::Ordering::Relaxed)
}

/// The kinds counted by code ([`count_kind`], [`count_kind_by`]), each with how many times:
/// `reims-vgpu`'s, as its reims-graphics device tells it of its completion rings
/// (`control-ring-full`, `control-ring-post-waited`, `vblank-dropped`, `queue-ring-full`,
/// `queue-ring-post-waited`), as its decoder keeps a buffer dealt to two threads whole
/// (`two-ranges-kept-whole`), as its command buffers' dispatch pre-pass hands their dispatches'
/// shapes to its compile threads (`dispatch-prepass-handed`), leaves those past a buffer's most to
/// its walk (`dispatch-prepass-past-most`), drops those no thread took by the walk's end
/// (`dispatch-prepass-dropped`) and the walk's end waits for those being made
/// (`dispatch-prepass-waited`, its nanoseconds waited, `dispatch-prepass-wait-ns`, and the longest
/// wait, `dispatch-prepass-wait-max-ns`), as it makes a no-copy buffer over pages their process
/// maps read-only (`read-only-pages-buffers`, and their bytes' KiB, `read-only-pages-kib`), and as
/// it makes a no-copy buffer (`no-copy-buffers`) and the guest notes that one's identity escaped,
/// by the act that noted it (`buffer-escaped-gpu-address`, `-argument-encoder`, `-resource-id`,
/// `-residency-set`), one over a copy of its pages counted too (`buffer-escaped-read-only-pages`),
/// with each command buffer begun of a process that holds such a copy (`escaped-copy-commits`),
/// and as its VA-API video on the rt platform on Windows lands a picture by the CPU's copy, the
/// GPU's copy declined (`rt-video-gpu-copy-declined`), and draws a picture into a cache's planes
/// over Direct3D 11 textures there (`video-pictures-drawn`, and their crops' KiB,
/// `video-drawn-kib`), and as such a cache's surface uses a spare made ahead
/// (`video-drawn-spares-used`) or makes its own (`video-drawn-spares-missed`), waits for the
/// spare being made for it (`video-drawn-spares-waited`, its nanoseconds waited,
/// `video-drawn-spares-wait-ns`, and the longest wait, `video-drawn-spares-wait-max-ns`), with
/// the spares made (`video-drawn-spares-made`) and dropped untaken (`video-drawn-spares-dropped`),
/// and as
/// its video session's picture does not land after its command buffer
/// completed at its decode, the session's next decode failing in its stead
/// (`video-landing-failure-carried`); the rt platform's surfaces over Direct3D 11 textures on
/// Windows, made (`rt-d3d11-surfaces`) and refused (`rt-refused-d3d11-surface`); and the objects
/// made, live at once and imported by the kind of make ([`MadeCodes`], [`count_made`]), and, on a
/// platform that keeps a pin only in its known blocks, the pins not kept by the kind of object
/// (`rt-gpu-address-unpinned-<kind>`) and the textures placed at the guest's own pick
/// (`rt-texture-address-guest-pick`); and its contexts' Private takes by cause, those that
/// waited behind their context's earlier take (`private-take-queued`) and those their worker
/// took from the source (`private-take-sourced`, with the source's time in microseconds,
/// `private-take-sourced-us`), those of them longer than the guest's 16 MiB block, an object's
/// own span (`private-take-span`, with the source's time, `private-take-span-us`), with the
/// longest a take held the thread that carried it out (`private-take-lane-us-most`); and the
/// `Managed` buffers' creates at their CPU copies' addresses that the platform made at others,
/// where they are kept (`managed-at-copy-unkept`).
static COUNTED: crate::core::sync::Mutex<Vec<(&'static str, u64)>> =
    crate::core::sync::Mutex::new(Vec::new());

/// Counts one more of the kind of code `code`, kept outside the runtime's and this crate's own
/// tables, which [`degradation_counts`] gives with the rest. Its line is the caller's to log, once
/// per kind (docs/conventions.md).
pub fn count_kind(code: &'static str) {
    count_kind_by(code, 1);
}

/// Counts `n` more of the kind of code `code` ([`count_kind`]): a kind whose count is an amount,
/// such as bytes, rather than how many times it was met. None is nothing.
pub fn count_kind_by(code: &'static str, n: u64) {
    if n == 0 {
        return;
    }
    let mut counted = COUNTED.lock();
    match counted.iter_mut().find(|(c, _)| *c == code) {
        Some((_, k)) => *k = k.saturating_add(n),
        None => counted.push((code, n)),
    }
}

/// The codes of a kind of thing made, counted by [`count_made`]: how many were made, and the most
/// live at once, which [`count_gone`] lowers the live count of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MadeCodes {
    /// The code counted once per thing made.
    pub made: &'static str,
    /// The code whose count is the most of them live at once.
    pub live_most: &'static str,
}

/// How many of each kind of thing ([`MadeCodes`]) are live, and the most at once, by its `made`
/// code.
static LIVE: crate::core::sync::Mutex<Vec<(&'static str, u64, u64)>> =
    crate::core::sync::Mutex::new(Vec::new());

/// Counts one more made of the kind of `codes` (its `made` code), and one more live, a new most
/// counted as its `live_most` code's rise to it ([`count_kind_by`]), so the counts line carries
/// the most at once; [`count_gone`] counts it gone.
pub fn count_made(codes: MadeCodes) {
    count_kind(codes.made);
    let rise = {
        let mut live = LIVE.lock();
        let at = match live.iter().position(|&(code, _, _)| code == codes.made) {
            Some(at) => at,
            None => {
                live.push((codes.made, 0, 0));
                live.len() - 1
            }
        };
        let (_, now, most) = &mut live[at];
        *now += 1;
        let rise = now.saturating_sub(*most);
        *most = (*most).max(*now);
        rise
    };
    count_kind_by(codes.live_most, rise);
}

/// Counts one made of the kind of `codes` ([`count_made`]) gone: one fewer live.
pub fn count_gone(codes: MadeCodes) {
    let mut live = LIVE.lock();
    if let Some((_, now, _)) = live.iter_mut().find(|(code, _, _)| *code == codes.made) {
        *now = now.saturating_sub(1);
    }
}

/// How many of the kind of `codes` are live now ([`count_made`], [`count_gone`]).
pub fn live_now(codes: MadeCodes) -> u64 {
    let live = LIVE.lock();
    (live.iter())
        .find(|(code, _, _)| *code == codes.made)
        .map_or(0, |&(_, now, _)| now)
}

thread_local! {
    /// The kind of the make running on the thread ([`making_as`]).
    static MAKING: Cell<Option<MadeCodes>> = const { Cell::new(None) };
}

/// Runs `make` as a make of the kind of `codes` on this thread: what the platform imports of the
/// host's memory for it there is counted under those codes ([`making`]), so the counts say which
/// of the caller's makes imported memory. Where a make of another kind runs already on the thread,
/// that one names it: the outermost caller knows best what it makes.
pub fn making_as<T>(codes: MadeCodes, make: impl FnOnce() -> T) -> T {
    /// Puts back the kind that ran before, however `make` returns.
    struct Restore(Option<MadeCodes>);
    impl Drop for Restore {
        fn drop(&mut self) {
            MAKING.with(|m| m.set(self.0));
        }
    }
    let was = MAKING.with(Cell::get);
    let _restore = Restore(was);
    if was.is_none() {
        MAKING.with(|m| m.set(Some(codes)));
    }
    make()
}

/// The kind of the make running on this thread ([`making_as`]), where one runs.
pub fn making() -> Option<MadeCodes> {
    MAKING.with(Cell::get)
}

thread_local! {
    /// The code a refusal of the platform's on the thread is counted under too, while a make
    /// runs that names one ([`refusing_as`]).
    static REFUSING: Cell<Option<&'static str>> = const { Cell::new(None) };
}

/// Runs `make` with each refusal of the platform's on this thread counted under `code` too
/// ([`refusing`]), so the counts say which of the caller's makes the platform refused. Where a
/// make that names another code runs already on the thread, that one names it, as in
/// [`making_as`].
pub fn refusing_as<T>(code: &'static str, make: impl FnOnce() -> T) -> T {
    /// Puts back the code of the make that ran before, however `make` returns.
    struct Restore(Option<&'static str>);
    impl Drop for Restore {
        fn drop(&mut self) {
            REFUSING.with(|r| r.set(self.0));
        }
    }
    let was = REFUSING.with(Cell::get);
    let _restore = Restore(was);
    if was.is_none() {
        REFUSING.with(|r| r.set(Some(code)));
    }
    make()
}

/// The code a refusal of the platform's on this thread is counted under too ([`refusing_as`]),
/// where a make that names one runs.
pub fn refusing() -> Option<&'static str> {
    REFUSING.with(Cell::get)
}

/// The kinds counted by code ([`count_kind`], [`count_kind_by`]), each with how many times, in
/// the order each was first counted: the recorder's stats file writes them.
pub fn counted_kinds() -> Vec<(&'static str, u64)> {
    COUNTED.lock().clone()
}

/// Each kind this process counted, with how many times, by its code: the calls answered as when
/// memory runs out (`out-of-host-memory`, [`out_of_host_memory_count`]) and the kinds counted by
/// code ([`count_kind`]). Sorted by code, a code met twice summed.
pub fn degradation_counts() -> Vec<(String, u64)> {
    let mut all: Vec<(String, u64)> =
        std::iter::once(("out-of-host-memory".to_owned(), out_of_host_memory_count()))
            .chain(COUNTED.lock().iter().map(|&(c, n)| (c.to_owned(), n)))
            .filter(|&(_, n)| n > 0)
            .collect();
    all.sort();
    all.dedup_by(|later, kept| {
        let same = later.0 == kept.0;
        if same {
            kept.1 += later.1;
        }
        same
    });
    all
}

/// The line of `counts` ([`degradation_counts`]), with its newline:
/// `reims-vgpu: degradation counts: <code> <n>, <code> <n>, …`; `None` for none.
fn counts_line(counts: &[(String, u64)]) -> Option<String> {
    if counts.is_empty() {
        return None;
    }
    let counts: Vec<String> = counts.iter().map(|(c, n)| format!("{c} {n}")).collect();
    Some(format!(
        "reims-vgpu: degradation counts: {}\n",
        counts.join(", ")
    ))
}

/// The line [`write_counts`] last wrote.
static COUNTS_WRITTEN: crate::core::sync::Mutex<String> =
    crate::core::sync::Mutex::new(String::new());

/// Writes, on stderr, the line of the process's [`degradation_counts`] since it began, where any
/// and where they changed since this last wrote them: `reims-vgpu: degradation counts: <code>
/// <n>, …` ([`counts_line`]), the line the codes of a boot's degradations are read from; and the
/// log's count of the lines it wrote at debug after their kind's first
/// ([`crate::core::logging::write_repeats`]). A device's reset and the process's exit write them
/// (`reims-vgpu`'s reset, [`crate::exit_gate`]). The rt platform's over-copy amounts are folded
/// into the counts first (`rt_platform::device::fold_metrics`).
pub fn write_counts() {
    crate::core::logging::write_repeats();
    {
        crate::rt_platform::device::fold_metrics();
        crate::rt_platform::diag::write();
    }
    let Some(line) = counts_line(&degradation_counts()) else {
        return;
    };
    let mut written = COUNTS_WRITTEN.lock();
    if *written == line {
        return;
    }
    // One write, so the line never interleaves with another thread's.
    let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
    *written = line;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::testkit::capture::catch_death;

    /// A kind counted outside this crate by its code ([`count_kind`]) is in the degradation
    /// counts with how many times it was counted, as the reims-graphics device's control
    /// completion ring's kinds reach the counts' line
    #[test]
    fn a_kind_counted_by_code_is_in_the_degradation_counts() {
        let code = "test-counted-by-code";
        let counted = || (degradation_counts().into_iter()).find(|(c, _)| c == code);
        assert_eq!(counted(), None);
        count_kind(code);
        count_kind(code);
        assert_eq!(counted(), Some((code.to_owned(), 2)));
    }

    /// A kind counted by code is among the kinds counted ([`counted_kinds`]) with how many times
    /// it was counted, as in the degradation counts
    #[test]
    fn a_kind_counted_by_code_is_among_the_counted_kinds() {
        let code = "test-counted-kinds";
        count_kind_by(code, 4);
        count_kind(code);
        assert!(counted_kinds().contains(&(code, 5)));
    }

    /// A kind counted by an amount ([`count_kind_by`]) is in the degradation counts with the sum
    /// of its amounts, a kind counted once a time added to it alike, and an amount of none counts
    /// nothing, so a kind met with none is not in them
    #[test]
    fn a_kind_counted_by_an_amount_sums_its_amounts() {
        let (code, none) = ("test-counted-by-amount", "test-counted-by-none");
        let counted = |code: &str| (degradation_counts().into_iter()).find(|(c, _)| c == code);
        count_kind_by(code, 6);
        count_kind_by(code, 3);
        count_kind(code);
        assert_eq!(counted(code), Some((code.to_owned(), 10)));
        count_kind_by(none, 0);
        assert_eq!(counted(none), None);
    }

    /// A test of this crate runs strict; [`relaxed`] runs its call under the default, on its
    /// thread alone, and puts strict mode back after, an unwinding call's included
    #[test]
    fn tests_run_strict_but_where_relaxed() {
        assert!(strict());
        assert!(!relaxed(strict));
        assert!(strict());
        let unwound = catch_death(|| relaxed(|| dies("sel:", "a death inside")));
        assert!(unwound.is_err() && strict());
        let elsewhere = relaxed(|| std::thread::scope(|s| s.spawn(strict).join().unwrap()));
        assert!(elsewhere);
    }

    /// A value the platform cannot hold dies naming the selector
    #[test]
    fn unrepresentable_dies_at_the_selector() {
        let site =
            catch_death(|| unrepresentable("setPixelFormat:", "pixel format 1")).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "setPixelFormat:")
        );
    }

    /// A make's kind names what the platform makes for it on its thread, the outermost where two
    /// nest, and none outside one; each kind's things made are counted, and the most live at once.
    #[test]
    fn a_make_s_kind_counts_what_it_makes_and_the_most_live() {
        const OUTER: MadeCodes = MadeCodes {
            made: "test-made-outer",
            live_most: "test-made-outer-live-most",
        };
        const INNER: MadeCodes = MadeCodes {
            made: "test-made-inner",
            live_most: "test-made-inner-live-most",
        };
        assert_eq!(making(), None);
        let seen = making_as(OUTER, || (making(), making_as(INNER, making)));
        assert_eq!(seen, (Some(OUTER), Some(OUTER)));
        assert_eq!(making_as(INNER, making), Some(INNER));
        assert_eq!(making(), None);
        assert_eq!(refusing(), None);
        let seen = refusing_as("test-outer", || {
            (refusing(), refusing_as("test-inner", refusing))
        });
        assert_eq!(seen, (Some("test-outer"), Some("test-outer")));
        assert_eq!(refusing_as("test-inner", refusing), Some("test-inner"));
        assert_eq!(refusing(), None);
        let counted = |code: &str| {
            (counted_kinds().into_iter())
                .find(|&(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        count_made(INNER);
        count_made(INNER);
        count_gone(INNER);
        count_made(INNER);
        assert_eq!(counted(INNER.made), 3);
        assert_eq!(counted(INNER.live_most), 2);
        assert_eq!(live_now(INNER), 2);
        assert_eq!(counted(OUTER.made), 0);
    }
}
