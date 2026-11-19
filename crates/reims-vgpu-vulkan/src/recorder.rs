//! The call recorder: with `REIMS_VGPU_RECORD` naming a directory, the process writes there a
//! replay stream of every call core makes of the Vulkan platform, in metal2vulkan's replay format
//! version 2 ([`mod@format`]), so a player can run the device's work on any Vulkan node with no VM.
//! It is a diagnostic the host switches on, like `REIMS_VGPU_TRACE`: with the variable unset, each
//! hooked method costs one relaxed load and a branch, and nothing else changes.
//!
//! - **What is recorded.** Each method of the platform's handles that core calls (the
//!   `impl <Trait><VulkanPlatform>` methods of `gpu`) opens with [`rec_call!`] and ends with
//!   [`rec_done!`]: one call record per call core made, the outermost call alone where the
//!   platform calls itself (a thread's depth, [`enter`]). Getters and queries are not hooked, but
//!   for `gpuResourceID`, whose answer the program saw ([`resource_id`]).
//!   The selector is Metal's, as the trait's `sel:` line names it, or an `m2v.` name for what
//!   Metal has no call for; an encoder's command is its enum's selector. The receiver and every
//!   object an argument names are the objects' stream ids, which the recorder gives each object
//!   as a call makes it (`State::register`) and never gives again, a texture of the platform's
//!   own (the window's cursor glyph, `VulkanDevice::new_own_texture`) as any other
//!   `newTextureWithDescriptor:`, of a descriptor whose `m2v.textureDescriptor` map holds
//!   [`OUTSIDE`] (`outside: true`) so a player places it outside the guest's texture indices; an
//!   object's last reference going (`Obj`'s drop) is its `release` record.
//! - **The device's identity.** Right after the record that made a device, an
//!   [`IDENTITY`] record on it answers what decides how a player makes its resources: the
//!   device's `name`, which the guest's device info gives, and `hasUnifiedMemory`, by which a
//!   `Managed` buffer is one copy or two ([`Identity`]). Each later segment starts with the
//!   identity record of each device then live.
//! - **Indirect command buffers' commands.** What the CPU encodes into an indirect command buffer
//!   lives in the runtime's records, which no memory write carries, so each send of commands to
//!   one is a call record on the buffer: `indirectRenderCommandAtIndex:` or
//!   `indirectComputeCommandAtIndex:`, `[index, commands]`, the commands in the order the
//!   platform carried them out, each a map of one pair, `MTLIndirectRenderCommand`'s or
//!   `MTLIndirectComputeCommand`'s selector and the list of its arguments
//!   ([`typed::icb_command`]). A command sent on its own (`IndirectRenderCommand::encode`) is a
//!   list of one; the commands of one call (`encode_render_command`) are those it took, read as
//!   it took them ([`icb_commands`]).
//! - **Arguments.** The hottest commands and the pass descriptor's sets are read from their fields
//!   ([`typed`]), integers, booleans and ranges as themselves, and host memory, bytes, strings and
//!   events as such ([`spec`]); the rest is formatted as its `Debug` text, its objects as their
//!   stream ids, which the writer reads into values ([`debug_value`]). Text that names objects by
//!   pointer (`ObjId(..)`) has them named by their stream ids at once, while the pointers are the
//!   objects' ([`resolve_ptrs`]).
//! - **Memory.** Before each commit, the recorder waits for the earlier command buffers that can
//!   run, then writes `m2v.write` records of every 4 KiB of host memory the command buffer can
//!   reach that changed since the stream last held it ([`before_commit`]), a commit inside
//!   `m2v.split` or `m2v.commitPart` too. It can reach what its encoders' calls name, what a
//!   pass's descriptor named, and what the commands of an indirect command buffer it executes
//!   named, through copies of them too. Memory its GPU work finds by address alone, through an
//!   argument buffer or an address in a buffer, it reaches only where the guest names it
//!   (`useResource:usage:`, `useHeap:`), as Metal's residency asks. A buffer in a heap over the
//!   guest's memory is written as its block's bytes. A copy out of the guest's RAM carries the
//!   bytes it reads as its arguments ([`guest_ram_source`]), and a buffer made as a copy of host
//!   memory the bytes it starts with ([`contents`]). A no-copy buffer's bytes arrive as writes;
//!   on a device that keeps two copies of a `Managed` buffer, a `Managed` one's record also
//!   carries the bytes its create copied into the device's copy ([`no_copy_contents`]), which
//!   only a `didModifyRange:` changes afterwards.
//!
//!   What a blit or a pass writes, the player's own run writes again, so it is taken as the
//!   stream's without a write: the ranges a blit writes (a buffer copy's or fill's range, a
//!   guest-RAM copy's runs; all of a texture, and of a buffer a copy from a texture writes) and
//!   all of each attachment of a pass, each 4 KiB chunk wholly inside what one command buffer
//!   wrote. What else its GPU work could have changed is compared with the stream's at the
//!   completion, as at a commit, and each chunk that differs is written there
//!   ([`State::complete`]): the chunks a blit wrote only part of, and those its kernels and
//!   render stages could store to, which are what the calls of its compute and render encoders
//!   reach (the recorder knows no pipeline's arguments: every buffer, texture and heap they bind
//!   or use, what an indirect command buffer they execute binds, but a resource a
//!   `useResources:` names for reading alone), in host memory the GPU's work writes in place
//!   (not a `Managed` buffer's where the device keeps two copies of it). So the stream holds the
//!   GPU's bytes there from the completion on, and a CPU write after it differs from them, one
//!   that puts back the bytes from before the command buffer ran included (a guest that zeroes
//!   the counters its kernels add to). Such a write follows the record of the call that
//!   committed the command buffer, held while that call runs ([`release_completions`]), so a
//!   player, which waits before each write for the command buffers that can complete, puts it in
//!   after its own run of the command buffer. Both are done before the guest can learn of the
//!   completion, on every route by which it does ([`completion_seen`]): in the first of the
//!   command buffer's completion-seen handlers to run, before any of them or its completed
//!   handlers tell the guest, and before `waitUntilCompleted` returns; and where the platform
//!   answers its status final, runs one of its completed handlers or returns from a wait for it
//!   first (the runtime sets the status before it runs the handlers, and refuses a handler once
//!   the command buffer is enqueued). One whose status read final first is taken when the next
//!   commit reaps it ([`State::absorb_reaped`]). Until it is taken, the recorder holds the
//!   runtime's command buffer ([`Completion::raw`]), and reads its status and waits for it there:
//!   core lets go of the guest's at its commit (its release record follows the commit's) while
//!   the GPU still runs it. The work before a shared event signal a command buffer encodes is
//!   done once the event reaches the signal's value, which the guest can learn before the
//!   command buffer completes: where the platform learns an event's value (a listener's
//!   notification, a CPU wait that saw it reached, a read of it), before it can tell the guest,
//!   each command buffer to complete that signals the event to that value or below has what it
//!   could have written compared as at a completion, and its completion compares all of it
//!   ([`event_reached`]); so has each that signals an event a command buffer that completed
//!   waited for. The header says so ([`format::FLAG_COMPLETION_BYTES_RECORDED`]).
//!
//!   One case still loses a CPU write from the writes before a commit (an announce, below, still
//!   writes it): a write made while the command buffer runs, into the very bytes a blit or a
//!   pass of it writes, is taken with the GPU's bytes (the guest raced its own GPU work); one into
//!   what its kernels could store to is in the completion's write. The hashing runs on the thread
//!   that saw the completion, the runtime's for a handler, and holds back other completions while
//!   it runs: `<pid>.stats.txt` counts its time and chunks (`completion_*`), and the signals seen
//!   reached before their command buffer's completion (`signals_heard`).
//!
//!   An announce, a `didModifyRange:` of a `Managed` buffer, also writes each chunk of its range
//!   whose bytes differ from those the stream's last write of it recorded, or that no write
//!   recorded ([`before_modify`]): so at every announce the stream's last recorded bytes of the
//!   range are the guest's, those the GPU wrote and the guest read back included, however late
//!   their completion was taken. The header says so ([`format::FLAG_ANNOUNCED_BYTES_RECORDED`]),
//!   for a player that pushes the recorded bytes at an announce.
//! - **Private buffers around dispatches.** Around each recorded dispatch of a compute pipeline
//!   made with linked functions, a blit copies each Private buffer range it binds into a Shared
//!   buffer of the recorder's own, before the dispatch and after it, and at the command buffer's
//!   completion an `m2v.privateBytes` record names the dispatch and holds both bytes as blobs
//!   ([`private`]).
//! - **A no-copy buffer's escape.** Where the guest notes that a no-copy buffer's identity left
//!   its Metal with no message naming the buffer (the reims-graphics device's `BUFFER_ESCAPED`,
//!   once in the buffer's life), a [`BUFFER_ESCAPED`] record on the buffer, `[kind]`, the act
//!   that noted it, is queued at the note ([`buffer_escaped`]): after the `m2v.write` records of
//!   each commit before it, and before those of each commit after. From there on the GPU may reach
//!   the buffer by an address in memory the stream writes, where a player stops, typed.
//! - **Another API's writes.** The acquire of what another API wrote into a buffer's exported
//!   memory (`external_write_done`: the host's video landing a picture in a surface's device
//!   cache) is an [`EXTERNAL_WRITE_DONE`] record on the buffer, with no arguments, queued as the
//!   call returns, as any call's: the point from which the device's reads of the buffer see those
//!   writes, which the stream does not hold.
//! - **The calling thread captures; the writer shapes.** Each call's record is queued, owned,
//!   under one lock as the call returns, so the stream's order keeps each thread's, with what
//!   only the call's moment has: its objects' ids, the bytes it consumes and the memory a commit
//!   reaches. A thread of the recorder's own (`reims-vgpu.recorder`) takes the queue every
//!   [`BUSY`] while calls come ([`DRAIN`] while none do) and, in its order, reads the text, shapes
//!   and tracks each call and encodes it. A commit or a copy out of the guest's RAM first takes
//!   what is queued on its own thread, since its capture needs every earlier call tracked.
//! - **The file.** The writer writes the stream out every [`FLUSH`], into segments
//!   `<pid>.NNNN.m2vr` of at most [`SEGMENT`] bytes, each starting with the header, a later one
//!   with the devices' identities after it ([`State::write_out`]). Libraries go
//!   once each into the `blobs` directory beside the stream's, by the hex of their SHA-256, and
//!   so do the bytes of `m2v.privateBytes`.
//!   `<pid>.stats.txt` says what was written, every few seconds, and what the process counted by
//!   code (a `counted <code> <n>` line for each kind, [`crate::answer::counted_kinds`]).
//! - **Its build.** `<pid>.stats.txt` names the commit the library was built from
//!   (`recorder_commit`, [`COMMIT`]), the device's own display library by the hex of its SHA-256
//!   and its length (`display_library`, [`METALLIB`]), and each change to what a stream holds
//!   that the recorder carries (a `recorder_fix` line for each of [`FIXES`]), so a recording says
//!   which recorder made it. The stream records the display library's make as
//!   `newLibraryWithSource:options:error:` with no arguments ([`crate::gpu::shaders`]), and a
//!   player makes it from a copy of those bytes kept beside the stream. A change to what the
//!   recorder writes adds its name to [`FIXES`] in its own commit.
//!
//! Recordings hold the guest's data: they stay on the host that made them and are never
//! committed.

pub mod debug_value;
pub mod format;
pub mod private;
pub mod typed;

#[cfg(test)]
mod memory_tests;

use std::any::Any;
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use crate::core::host::HostPtr;
use crate::core::platform::Handle as _;
use crate::core::platform::resource::Texture as _;
use crate::core::platform::types::{CommandBufferStatus, NsRange};
use crate::core::sync::Mutex;
use crate::gpu::command::{CommandBufferObject, VulkanCommandBuffer};
use crate::gpu::resource::{BufferObject, TextureObject, Under, VulkanBuffer, VulkanTexture};
use crate::gpu::shaders::METALLIB;
use crate::platform::Obj;
use format::{Call, Hash, Stream, Value};

/// The environment variable naming the directory streams are written to.
pub const ENV_VAR: &str = "REIMS_VGPU_RECORD";

/// How long a record may wait in the buffer.
const FLUSH: Duration = Duration::from_millis(50);

/// How long a call may wait in the queue before the writer shapes it, while none came.
const DRAIN: Duration = Duration::from_millis(1);

/// How long, while calls come.
const BUSY: Duration = Duration::from_micros(50);

/// How often the counts are written.
const STATS: Duration = Duration::from_secs(5);

/// The commit the library was built from, which `cargo xtask dist` names for its Cargo steps in
/// `RVG_BUILD_COMMIT` (`git rev-parse HEAD`, with `-dirty` where the tree differs from it), else
/// `unknown`. The stats file writes it as `recorder_commit`.
pub const COMMIT: &str = match option_env!("RVG_BUILD_COMMIT") {
    Some(commit) => commit,
    None => "unknown",
};

/// Each change to what a stream holds since format version 2, by name, in the order they
/// landed, with the commit that made it. The stats file writes a `recorder_fix <name>` line for
/// each.
pub const FIXES: &[&str] = &[
    // 82b8c2e2a: the guest RAM a copy out of it reads
    "guest-ram-copy-reads",
    // bec103226: no record of a copy the platform did not make
    "no-unmade-ram-copy",
    // 967d30b73: whether m2v.commitPart cut its command buffer
    "commit-part-cut",
    // d04d02436: an object property set to nil as object 0
    "nil-property-object-0",
    // 82be400ae: a buffer copied from host memory's first bytes
    "host-copy-start-bytes",
    // 0d42c7cb8: each gpuResourceID the platform answers
    "gpu-resource-ids",
    // 00d59542b: Private buffers around linked-kernel dispatches
    "private-around-linked-kernels",
    // b27a603cb: commands a CPU encodes into an ICB
    "icb-cpu-commands",
    // 0ba37884a: each device's identity at each segment's start
    "device-identity",
    // 285f0bdd6: a no-copy Managed buffer's create-time bytes
    "no-copy-managed-create-bytes",
    // 7fc1cc427: memory before split and cut commits
    "split-capture",
    // 7cf445307: CPU writes into bytes the GPU work wrote
    "cpu-writes-over-gpu-writes",
    // cb7db3a59: what indirect commands bind
    "icb-reach",
    // 248751c3c, 642ef0ff9: the platform's own textures, marked
    "own-textures",
    // 0061da1f6: the guest's bytes at each Managed announce
    "announce-bytes",
    // e3898c754: what kernels changed, at their completion
    "kernel-writes-at-completion",
    // 38af6b525: each completion before the guest can learn of it
    "completion-before-guest-learns",
    // c6ab041d6: each no-copy buffer's escape, at the guest's first note
    "buffer-escaped",
    // the commit that adds this name: each acquire of another API's writes into a buffer
    "external-write-done",
];

/// The stats file's lines that name the recorder's build: its commit ([`COMMIT`]), the display
/// library the device makes ([`METALLIB`], by the hex of its SHA-256, as the `blobs` directory
/// names a library, and its length), then one for each of [`FIXES`].
fn write_build(out: &mut String) {
    let _ = writeln!(out, "recorder_commit {COMMIT}");
    let _ = writeln!(
        out,
        "display_library {} {}",
        hex(&format::sha256(METALLIB)),
        METALLIB.len()
    );
    for fix in FIXES {
        let _ = writeln!(out, "recorder_fix {fix}");
    }
}

/// The stats file's lines of the kinds counted by code
/// ([`crate::answer::counted_kinds`]): `counted <code> <n>` for each, as the degradation counts'
/// line counts it since the process began, the reims-graphics device's escapes of no-copy buffers
/// among them.
fn write_counted(out: &mut String) {
    for (code, n) in crate::answer::counted_kinds() {
        let _ = writeln!(out, "counted {code} {n}");
    }
}

/// The most bytes a segment holds before the next starts.
const SEGMENT: u64 = 1 << 30;

/// The unit memory is compared and written in.
const CHUNK: u64 = 4096;

/// How long a capture waits, at most, for the earlier command buffers that can run.
const WAIT_BOUND: Duration = Duration::from_secs(2);

/// Inline data shorter than this stays bytes; longer is a blob.
const BLOB_MIN: usize = 64;

/// Whether the process records.
static ON: AtomicBool = AtomicBool::new(false);

/// The recorder's state, made once where the variable is set.
static STATE: OnceLock<State> = OnceLock::new();

thread_local! {
    /// How many hooked calls this thread is inside.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// Whether this thread formats an argument for the stream: an `Obj` then prints as its id.
    static FORMATTING: Cell<bool> = const { Cell::new(false) };
    /// This thread's reading and queueing time not yet in the counts, and its calls since.
    static TIMES: Cell<(u64, u64, u32)> = const { Cell::new((0, 0, 0)) };
    /// The command buffers this thread committed inside its outermost call, whose writes at
    /// their completions wait for that call's record ([`release_completions`]).
    static COMMITTED: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// How many calls a thread's times wait for before they reach the counts ([`add_times`]).
const TIMES_EVERY: u32 = 256;

/// Adds a calling thread's time reading a call (`read`) and queueing it (`finish`, which ends the
/// call) to the counts once every [`TIMES_EVERY`] calls: a shared counter written on every call
/// would move its cache line between the threads, the writer's among them, on every call.
fn add_times(state: &State, read: u64, finish: u64, ends: bool) {
    TIMES.with(|t| {
        let (r, f, n) = t.get();
        let (r, f, n) = (r + read, f + finish, n + u32::from(ends));
        if n < TIMES_EVERY {
            t.set((r, f, n));
            return;
        }
        t.set((0, 0, 0));
        state.stats.read_ns.fetch_add(r, Ordering::Relaxed);
        state.stats.finish_ns.fetch_add(f, Ordering::Relaxed);
    });
}

/// The hasher of the recorder's maps, whose keys are pointers and stream ids: a multiply and a
/// shift, where the default's SipHash costs a calling thread more on each lookup.
#[derive(Default, Clone, Copy)]
struct KeyHasher(u64);

impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u64(u64::from(*b));
        }
    }

    fn write_u64(&mut self, n: u64) {
        let h = (self.0 ^ n).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = h ^ (h >> 32);
    }
}

/// A map by pointer or stream id.
type KeyMap<V> = HashMap<u64, V, std::hash::BuildHasherDefault<KeyHasher>>;

/// A set of stream ids.
type KeySet = HashSet<u64, std::hash::BuildHasherDefault<KeyHasher>>;

/// Whether the process records: one relaxed load.
#[inline]
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Starts recording where `REIMS_VGPU_RECORD` names a directory, once per process: called as the
/// process's platform is made, before any object.
pub fn start_from_env() {
    let Some(dir) = std::env::var_os(ENV_VAR) else {
        return;
    };
    let dir = PathBuf::from(dir);
    let mut started = false;
    STATE.get_or_init(|| {
        started = true;
        State::new(dir)
    });
    if started
        && let Some(state) = STATE.get()
        && state.open()
    {
        ON.store(true, Ordering::Release);
        log::info!(
            target: "reims_vgpu_vulkan::recorder",
            "recording the platform's calls to {}",
            state.base.display()
        );
    }
}

fn state() -> Option<&'static State> {
    STATE.get()
}

#[allow(clippy::disallowed_methods)] // the recorder's own timing, a diagnostic
fn now_ns() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    let start = *START.get_or_init(std::time::Instant::now);
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// One hooked call this thread is inside, while it lives.
#[derive(Debug)]
pub struct Token {
    outer: bool,
}

impl Drop for Token {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// The call a hooked method is: `None` where the process does not record.
#[inline]
pub fn enter() -> Option<Token> {
    if !on() {
        return None;
    }
    Some(enter_recording())
}

fn enter_recording() -> Token {
    let depth = DEPTH.with(|d| {
        let v = d.get();
        d.set(v + 1);
        v
    });
    Token { outer: depth == 0 }
}

/// A recorded call between its start and its end: what [`rec_call!`] read of it, where it is
/// the thread's outermost.
#[derive(Debug)]
pub struct Active {
    token: Token,
    pending: Option<Call>,
    /// An encoder command's `Debug` text, which the writer reads into its arguments.
    text: Option<String>,
    /// The snapshot of the dispatch it is, whose record names its position ([`private`]).
    snap: Option<u64>,
}

/// Opens a call: its selector, receiver and arguments, read only where it is the outermost.
pub fn begin<S: Into<Cow<'static, str>>>(
    token: Token,
    read: impl FnOnce() -> (S, u64, Vec<Value>),
) -> Active {
    let pending = token.outer.then(|| {
        let t0 = now_ns();
        let (selector, receiver, args) = read();
        if let Some(state) = state() {
            add_times(state, now_ns() - t0, 0, false);
        }
        Call {
            selector: selector.into().into_owned(),
            receiver,
            args,
            ..Call::default()
        }
    });
    Active {
        token,
        pending,
        text: None,
        snap: None,
    }
}

/// Opens an encoder's command, `cmd` on the encoder `receiver` names, read only where it is the
/// outermost.
pub fn begin_cmd<C: typed::Command + std::fmt::Debug>(
    token: Token,
    receiver: impl FnOnce() -> u64,
    cmd: &C,
) -> Active {
    if !token.outer {
        return Active {
            token,
            pending: None,
            text: None,
            snap: None,
        };
    }
    let t0 = now_ns();
    let receiver = receiver();
    let (args, text) = cmd_call(cmd);
    if let Some(state) = state() {
        add_times(state, now_ns() - t0, 0, false);
    }
    Active {
        token,
        pending: Some(Call {
            selector: cmd.sel().to_owned(),
            receiver,
            args,
            ..Call::default()
        }),
        text,
        snap: None,
    }
}

/// A command's arguments: read from its fields where it is one [`typed`] reads; else its text,
/// left for the writer, its objects named by pointer named by their ids now ([`resolve_ptrs`]).
fn cmd_call<C: typed::Command + std::fmt::Debug>(cmd: &C) -> (Vec<Value>, Option<String>) {
    let mut cx = typed::Cx::new(&id_at);
    if let Some(args) = cmd.typed(&mut cx) {
        count_unrecorded(cx.unrecorded);
        return (args, None);
    }
    (Vec::new(), Some(stream_text(debug_text(cmd))))
}

/// `text`, its objects named by pointer named by their stream ids ([`resolve_ptrs`]).
fn stream_text(text: String) -> String {
    if text.contains("ObjId(") {
        resolve_ptrs(&text, &id_at)
    } else {
        text
    }
}

/// `text` with each object it names by pointer (`ObjId(p)`) named by its stream id instead,
/// `ObjRef(id)`, or `Unrecorded(0)` where `ids` has none, as the reader would read it now:
/// the writer reads the same values from it later, when the pointers may be other objects'.
/// Quoted strings are left as they are.
fn resolve_ptrs(text: &str, ids: &dyn Fn(u64) -> Option<u64>) -> String {
    const TAG: &[u8] = b"ObjId(";
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let (mut at, mut copied, mut quoted) = (0, 0, false);
    while let Some(&b) = bytes.get(at) {
        if quoted {
            match b {
                b'\\' => at += 1,
                b'"' => quoted = false,
                _ => {}
            }
            at += 1;
            continue;
        }
        if b == b'"' {
            quoted = true;
            at += 1;
            continue;
        }
        let named = bytes.get(at..).is_some_and(|r| r.starts_with(TAG))
            && !at
                .checked_sub(1)
                .and_then(|p| bytes.get(p))
                .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_');
        if named {
            let digits = at + TAG.len();
            let end = digits
                + bytes
                    .get(digits..)
                    .map_or(0, |r| r.iter().take_while(|c| c.is_ascii_digit()).count());
            if bytes.get(end) == Some(&b')')
                && let Some(ptr) = text.get(digits..end).and_then(|d| d.parse::<u64>().ok())
            {
                out.push_str(text.get(copied..at).unwrap_or_default());
                match ids(ptr) {
                    Some(id) => {
                        let _ = write!(out, "ObjRef({id})");
                    }
                    None => out.push_str("Unrecorded(0)"),
                }
                at = end + 1;
                copied = at;
                continue;
            }
        }
        at += 1;
    }
    out.push_str(text.get(copied..).unwrap_or_default());
    out
}

fn count_unrecorded(n: u64) {
    if n > 0
        && let Some(state) = state()
    {
        state.stats.unrecorded_refs.fetch_add(n, Ordering::Relaxed);
    }
}

impl Active {
    /// Whether this call is recorded.
    pub fn recorded(&self) -> bool {
        self.token.outer && self.pending.is_some()
    }
}

/// Opens a recorded call: `rec_call!(selector, receiver, arguments…)`, `None` where the process
/// does not record. The arguments are read only where the call is the thread's outermost.
macro_rules! rec_call {
    (@cmd $recv:expr, $cmd:expr) => {
        match $crate::recorder::enter() {
            None => None,
            Some(token) => Some($crate::recorder::begin_cmd(
                token,
                || $crate::recorder::Receiver::stream_id($recv),
                &$cmd,
            )),
        }
    };
    ($sel:expr, $recv:expr $(, $arg:expr)* $(,)?) => {
        match $crate::recorder::enter() {
            None => None,
            Some(token) => Some($crate::recorder::begin(token, || {
                (
                    $sel,
                    $crate::recorder::Receiver::stream_id($recv),
                    vec![$($crate::recorder::rec_val!($arg)),*],
                )
            })),
        }
    };
}
pub(crate) use rec_call;

/// One argument's value ([`spec`]).
macro_rules! rec_val {
    ($e:expr) => {{
        #[allow(unused_imports)]
        use $crate::recorder::spec::{
            ViaBytes as _, ViaCommands as _, ViaDebug as _, ViaEvent as _, ViaHost as _,
            ViaNil as _, ViaScalar as _, ViaSet as _, ViaStr as _,
        };
        (&&&$crate::recorder::spec::V(&$e)).rec_value()
    }};
}
pub(crate) use rec_val;

/// Closes a recorded call with what it answered: `rec_done!(active, answer)`.
macro_rules! rec_done {
    ($active:expr, $made:expr) => {
        if let Some(active) = $active {
            #[allow(unused_imports)]
            use $crate::recorder::spec::{ViaMade as _, ViaNoMade as _};
            $crate::recorder::finish(active, (&$crate::recorder::spec::MadeSpec(&$made)).made());
        }
    };
}
pub(crate) use rec_done;

/// Runs a hooked method's body: a closure, so its `return`s and `?`s end the body alone.
#[inline(always)]
pub fn run<R>(body: impl FnOnce() -> R) -> R {
    body()
}

/// An encoder command's arguments, read from its `Debug` text: its variant's fields, in order.
pub fn cmd_args<T: std::fmt::Debug>(cmd: &T) -> Vec<Value> {
    cmd_args_text(&debug_text(cmd))
}

/// `x`'s `Debug` text, its objects as their stream ids.
fn debug_text<T: std::fmt::Debug + ?Sized>(x: &T) -> String {
    formatting(|| format!("{x:?}"))
}

/// The value `text`, an argument's `Debug` text, reads as, what it did not read counted.
fn parse_text(text: &str) -> Value {
    let (v, notes) = debug_value::parse(text, &id_at);
    if let Some(state) = state() {
        state
            .stats
            .unrecorded_refs
            .fetch_add(notes.unrecorded, Ordering::Relaxed);
        if notes.unread {
            state.stats.unread_args.fetch_add(1, Ordering::Relaxed);
        }
    }
    v
}

/// An encoder command's arguments from its `Debug` text.
fn cmd_args_text(text: &str) -> Vec<Value> {
    cmd_args_of(parse_text(text))
}

/// An encoder command's arguments from the value its text reads as.
fn cmd_args_of(v: Value) -> Vec<Value> {
    match v {
        Value::Map(fields) => match fields.len() {
            // A tuple variant reads as one pair of its name and its value(s).
            1 if fields
                .first()
                .is_some_and(|(k, _)| k.starts_with(|c: char| c.is_ascii_uppercase())) =>
            {
                match fields.into_iter().next().map(|(_, v)| v) {
                    Some(Value::List(items)) => items,
                    Some(v) => vec![v],
                    None => vec![],
                }
            }
            _ => fields.into_iter().map(|(_, v)| v).collect(),
        },
        Value::Str(_) => vec![],
        other => vec![other],
    }
}

/// The values of arguments by kind, by autoref specialisation: host memory, bytes, strings and
/// events are read as such, integers, booleans, ranges and the pass descriptor's sets from their
/// fields ([`typed`]), anything else that prints itself from its `Debug` text, and the
/// rest (blocks and handlers) is nil.
pub mod spec {
    use super::{Value, typed};
    use crate::core::platform::event::EventRef;
    use crate::core::platform::host_range::HostRange;
    use crate::core::platform::types::NsRange;
    use crate::core::platform::values::{
        ColorAttachmentSet, DepthAttachmentSet, PassSet, StencilAttachmentSet,
    };
    use crate::platform::VulkanPlatform;

    /// Integers, booleans and ranges, as their text reads.
    pub trait ViaScalar {
        /// The value.
        fn rec_value(&self) -> Value;
    }
    impl ViaScalar for &&V<'_, u64> {
        fn rec_value(&self) -> Value {
            typed::uint(*self.0)
        }
    }
    impl ViaScalar for &&V<'_, u32> {
        fn rec_value(&self) -> Value {
            typed::uint(u64::from(*self.0))
        }
    }
    impl ViaScalar for &&V<'_, i64> {
        fn rec_value(&self) -> Value {
            typed::sint(*self.0)
        }
    }
    impl ViaScalar for &&V<'_, bool> {
        fn rec_value(&self) -> Value {
            Value::Bool(*self.0)
        }
    }
    impl ViaScalar for &&V<'_, NsRange> {
        fn rec_value(&self) -> Value {
            typed::ns_range(*self.0)
        }
    }

    /// The pass descriptor's sets, from their fields.
    pub trait ViaSet {
        /// The value.
        fn rec_value(&self) -> Value;
    }
    impl ViaSet for &&V<'_, &PassSet<'_, VulkanPlatform>> {
        fn rec_value(&self) -> Value {
            with_cx(|cx| typed::pass_set(self.0, cx))
        }
    }
    impl ViaSet for &&V<'_, &ColorAttachmentSet<'_, VulkanPlatform>> {
        fn rec_value(&self) -> Value {
            with_cx(|cx| typed::color_attachment(self.0, cx))
        }
    }
    impl ViaSet for &&V<'_, &DepthAttachmentSet<'_, VulkanPlatform>> {
        fn rec_value(&self) -> Value {
            with_cx(|cx| typed::depth_attachment(self.0, cx))
        }
    }
    impl ViaSet for &&V<'_, &StencilAttachmentSet<'_, VulkanPlatform>> {
        fn rec_value(&self) -> Value {
            with_cx(|cx| typed::stencil_attachment(self.0, cx))
        }
    }

    /// An indirect command buffer's commands, each read from its fields.
    pub trait ViaCommands {
        /// The value.
        fn rec_value(&self) -> Value;
    }
    impl<C: typed::Command> ViaCommands for &&V<'_, super::IcbCommands<'_, C>> {
        fn rec_value(&self) -> Value {
            with_cx(|cx| Value::List(self.0.0.iter().map(|c| typed::icb_command(c, cx)).collect()))
        }
    }

    fn with_cx(read: impl FnOnce(&mut typed::Cx<'_>) -> Value) -> Value {
        let mut cx = typed::Cx::new(&super::id_at);
        let v = read(&mut cx);
        super::count_unrecorded(cx.unrecorded);
        v
    }

    /// An argument, borrowed.
    pub struct V<'a, T: ?Sized>(
        /// The argument.
        pub &'a T,
    );

    /// Host memory: its address and length.
    pub trait ViaHost {
        /// The value.
        fn rec_value(&self) -> Value;
    }
    impl ViaHost for &&V<'_, HostRange<'_>> {
        fn rec_value(&self) -> Value {
            Value::Host(self.0.ptr().addr(), self.0.len())
        }
    }

    /// Bytes.
    pub trait ViaBytes {
        /// The value.
        fn rec_value(&self) -> Value;
    }
    impl ViaBytes for &&V<'_, &[u8]> {
        fn rec_value(&self) -> Value {
            Value::Bytes(self.0.to_vec())
        }
    }

    /// A string.
    pub trait ViaStr {
        /// The value.
        fn rec_value(&self) -> Value;
    }
    impl ViaStr for &&V<'_, &str> {
        fn rec_value(&self) -> Value {
            Value::Str((*self.0).to_owned())
        }
    }

    /// An event, by its object.
    pub trait ViaEvent {
        /// The value.
        fn rec_value(&self) -> Value;
    }
    impl ViaEvent for &&V<'_, Option<EventRef<'_, VulkanPlatform>>> {
        fn rec_value(&self) -> Value {
            match self.0 {
                Some(EventRef::Event(e)) => super::object_value(*e),
                Some(EventRef::Shared(e)) => super::object_value(*e),
                None => Value::Nil,
            }
        }
    }

    /// Anything that prints itself.
    pub trait ViaDebug {
        /// The value.
        fn rec_value(&self) -> Value;
    }
    impl<T: std::fmt::Debug + ?Sized> ViaDebug for &V<'_, T> {
        fn rec_value(&self) -> Value {
            debug_value_of(self.0)
        }
    }

    /// The rest: nil.
    pub trait ViaNil {
        /// The value.
        fn rec_value(&self) -> Value;
    }
    impl<T: ?Sized> ViaNil for V<'_, T> {
        fn rec_value(&self) -> Value {
            Value::Nil
        }
    }

    /// `x`'s value from its `Debug` text, objects as their stream ids: the text, which the
    /// writer reads ([`Value::Debug`]), objects it names by pointer named by their ids now.
    pub fn debug_value_of<T: std::fmt::Debug + ?Sized>(x: &T) -> Value {
        Value::Debug(super::stream_text(super::debug_text(x)))
    }

    /// An answer that names an object the call made, or none.
    pub struct MadeSpec<'a, R>(
        /// The answer.
        pub &'a R,
    );

    /// An answer that holds a made object.
    pub trait ViaMade {
        /// The object.
        fn made(&self) -> Option<super::Made>;
    }
    impl<R: super::AsMade> ViaMade for MadeSpec<'_, R> {
        fn made(&self) -> Option<super::Made> {
            self.0.as_made()
        }
    }

    /// Any other answer.
    pub trait ViaNoMade {
        /// None.
        fn made(&self) -> Option<super::Made>;
    }
    impl<R> ViaNoMade for &MadeSpec<'_, R> {
        fn made(&self) -> Option<super::Made> {
            None
        }
    }
}

/// Commands sent to an indirect command buffer, as a call's argument: the list of each one read
/// from its fields ([`typed::icb_command`]).
pub struct IcbCommands<'a, C>(
    /// The commands, in the order sent.
    pub &'a [C],
);

/// Sends the commands `cmds` hands out to an indirect command buffer through `send`, in the call
/// `active` records (`encode_render_command`, `encode_compute_command`): each read from its
/// fields as `send` takes it, so the call's last argument lists those it took, in that order,
/// which are those the platform carried out, the one whose send raised the last. Out of line and
/// over one iterator type, so the platform's own path, which records nothing, keeps its shape.
#[cold]
#[inline(never)]
pub fn icb_commands<C: typed::Command, R>(
    mut active: Active,
    cmds: impl Iterator<Item = C>,
    send: impl FnOnce(&mut dyn Iterator<Item = C>) -> R,
) -> R {
    let answer = take_commands(&mut active, &id_at, cmds, send);
    finish(active, None);
    answer
}

/// [`icb_commands`]'s sends and reads, the stream ids of objects by pointer from `ids`.
fn take_commands<C: typed::Command, R>(
    active: &mut Active,
    ids: &dyn Fn(u64) -> Option<u64>,
    cmds: impl Iterator<Item = C>,
    send: impl FnOnce(&mut dyn Iterator<Item = C>) -> R,
) -> R {
    let mut taken = Vec::new();
    let mut cx = typed::Cx::new(ids);
    let answer = send(&mut cmds.inspect(|c| taken.push(typed::icb_command(c, &mut cx))));
    count_unrecorded(cx.unrecorded);
    if let Some(call) = active.pending.as_mut() {
        call.args.push(Value::List(taken));
    }
    answer
}

/// An object a call made: its pointer, a weak reference and its type, and a device's identity.
pub struct Made {
    ptr: u64,
    weak: Weak<dyn Any + Send + Sync>,
    type_name: &'static str,
    identity: Option<Identity>,
}

impl Made {
    fn of<T: Send + Sync + 'static>(o: &Obj<T>) -> Made {
        let weak: Weak<dyn Any + Send + Sync> = Obj::downgrade(o);
        Made {
            ptr: o.id().0,
            weak,
            type_name: std::any::type_name::<T>(),
            identity: (o as &dyn Any)
                .downcast_ref::<crate::gpu::device::VulkanDevice>()
                .map(Identity::of),
        }
    }
}

/// The selector of a device's identity record ([`Identity`]).
pub const IDENTITY: &str = "m2v.deviceIdentity";

/// What a device answered the program that decides how a player makes its resources, read as
/// the stream first names the device: its `name`, which the guest's device info gives, and
/// whether its GPU shares the CPU's memory (`hasUnifiedMemory`), by which a `Managed` buffer is
/// one copy or two (docs/graphics-kext/design.md, "Managed storage"). Its record, an
/// [`IDENTITY`] call on the device with the two as its answers, follows the record that made
/// the device, and starts each later segment once for each device then live
/// ([`State::write_out`]).
#[derive(Debug, Clone, PartialEq)]
struct Identity {
    name: String,
    unified: bool,
}

impl Identity {
    fn of(device: &crate::gpu::device::VulkanDevice) -> Identity {
        let props = device.raw.properties();
        Identity {
            name: props.name.clone(),
            unified: props.has_unified_memory,
        }
    }

    /// Its record, on the device of stream id `device`.
    fn record(&self, device: u64) -> Call {
        Call {
            selector: IDENTITY.to_owned(),
            receiver: device,
            answers: vec![
                ("name".to_owned(), Value::Str(self.name.clone())),
                ("hasUnifiedMemory".to_owned(), Value::Bool(self.unified)),
            ],
            ..Call::default()
        }
    }
}

/// An answer that holds the object a call made.
pub trait AsMade {
    /// The object made, where there is one.
    fn as_made(&self) -> Option<Made>;
}

impl<T: Send + Sync + 'static> AsMade for Obj<T> {
    fn as_made(&self) -> Option<Made> {
        Some(Made::of(self))
    }
}

impl<A: AsMade> AsMade for Option<A> {
    fn as_made(&self) -> Option<Made> {
        self.as_ref().and_then(AsMade::as_made)
    }
}

impl<A: AsMade, E> AsMade for Result<A, E> {
    fn as_made(&self) -> Option<Made> {
        self.as_ref().ok().and_then(AsMade::as_made)
    }
}

impl AsMade for crate::core::platform::values::HeapMade<crate::platform::VulkanPlatform> {
    fn as_made(&self) -> Option<Made> {
        self.heap.as_made()
    }
}

/// A receiver's stream id.
pub trait Receiver {
    /// Its id; 0 for none.
    fn stream_id(&self) -> u64;
}

impl<T: Send + Sync + 'static> Receiver for Obj<T> {
    fn stream_id(&self) -> u64 {
        receiver_id(self)
    }
}

impl Receiver for crate::platform::VulkanPlatform {
    fn stream_id(&self) -> u64 {
        0
    }
}

/// An indirect command buffer a command lent from it names: the buffer's object's id, its value's
/// address as an `Obj`'s is.
impl Receiver for crate::gpu::icb::IcbObject {
    fn stream_id(&self) -> u64 {
        use crate::core::num::ToU64 as _;
        let ptr = std::ptr::from_ref(self).cast::<u8>().addr().to_u64();
        id_at(ptr).unwrap_or_else(|| {
            if let Some(state) = state() {
                (state.stats.unrecorded_receivers).fetch_add(1, Ordering::Relaxed);
            }
            0
        })
    }
}

/// Runs `f` with this thread formatting for the stream.
fn formatting<R>(f: impl FnOnce() -> R) -> R {
    let was = FORMATTING.with(|c| c.replace(true));
    let r = f();
    FORMATTING.with(|c| c.set(was));
    r
}

/// Whether this thread formats for the stream, where `Obj`'s `Debug` prints its id.
pub fn is_formatting() -> bool {
    FORMATTING.with(Cell::get)
}

/// `Obj`'s `Debug` while formatting for the stream: `ObjRef(id)`, or `Unrecorded("type")` for an
/// object the stream never made.
pub fn fmt_obj(ptr: u64, type_name: &str, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match id_at(ptr) {
        Some(id) => write!(f, "ObjRef({id})"),
        None => write!(f, "Unrecorded({type_name:?})"),
    }
}

/// The stream id of the object at `ptr`: under a lock of its own, which the calling threads
/// alone take, never the writer's.
fn id_at(ptr: u64) -> Option<u64> {
    state()?.ptrs.lock().get(&ptr).copied()
}

fn object_value<T: Send + Sync + 'static>(o: &Obj<T>) -> Value {
    match id_at(o.id().0) {
        Some(id) => Value::Object(id),
        None => {
            if let Some(s) = state() {
                s.stats.unrecorded_refs.fetch_add(1, Ordering::Relaxed);
            }
            Value::Nil
        }
    }
}

/// The receiver's id; a device the stream has not seen is adopted, with its
/// `MTLCreateSystemDefaultDevice` record and its identity's.
fn receiver_id<T: Send + Sync + 'static>(o: &Obj<T>) -> u64 {
    let Some(state) = state() else { return 0 };
    if let Some(id) = id_at(o.id().0) {
        return id;
    }
    if std::any::type_name::<T>() == std::any::type_name::<crate::gpu::device::DeviceObject>() {
        let mut made = Made::of(o);
        let identity = made.identity.take();
        let id = state.register(made);
        let adopted = Item::Plain(Call {
            selector: "MTLCreateSystemDefaultDevice".to_owned(),
            result: id,
            ..Call::default()
        });
        state.push_made(adopted, id, identity);
        return id;
    }
    state
        .stats
        .unrecorded_receivers
        .fetch_add(1, Ordering::Relaxed);
    0
}

/// Gives a texture the stream never saw made (a window's drawable) a `newTextureWithDescriptor:`
/// record of its shape, before its first use.
pub fn adopt_texture(t: &VulkanTexture) {
    if !on() {
        return;
    }
    let Some(state) = state() else { return };
    if id_at(t.id().0).is_some() {
        return;
    }
    let _quiet = enter_recording();
    let desc = Value::Map(vec![
        ("textureType".to_owned(), Value::Uint(t.texture_type())),
        ("pixelFormat".to_owned(), Value::Uint(t.pixel_format())),
        ("width".to_owned(), Value::Uint(t.width())),
        ("height".to_owned(), Value::Uint(t.height())),
        ("usage".to_owned(), Value::Uint(t.usage())),
        ("storageMode".to_owned(), Value::Uint(2)),
    ]);
    let id = state.register(Made::of(t));
    state.push_plain(Call {
        selector: "newTextureWithDescriptor:".to_owned(),
        result: id,
        args: vec![desc],
        answers: vec![("m2v.drawable".to_owned(), Value::Bool(true))],
        ..Call::default()
    });
}

/// The key of an `m2v.textureDescriptor` record's map that marks a descriptor of a texture of the
/// platform's own (the window's cursor glyph), `true`: its texture takes no resource ID and no
/// range of Metal's space, so a player places it outside the guest's texture indices. Only such
/// a descriptor carries it ([`outside_descriptor`]).
pub const OUTSIDE: &str = "outside";

/// Records the making of `descriptor`, of `desc`, for a texture of the platform's own
/// (`VulkanDevice::new_own_texture`): an `m2v.textureDescriptor` on the platform, as its
/// `texture_descriptor` records one, whose map ends with [`OUTSIDE`] `true`. The texture's
/// `newTextureWithDescriptor:` names it as any other's names its descriptor.
pub fn outside_descriptor(
    descriptor: &crate::gpu::desc::VulkanTextureDescriptor,
    desc: &crate::core::platform::values::TextureDesc,
) {
    let Some(token) = enter() else { return };
    let active = begin(token, || {
        let mut fields = parse_text(&debug_text(desc));
        if let Value::Map(pairs) = &mut fields {
            pairs.push((OUTSIDE.to_owned(), Value::Bool(true)));
        }
        ("m2v.textureDescriptor", 0, vec![fields])
    });
    finish(active, Some(Made::of(descriptor)));
}

/// The guest's frame ends here, showing `t`: the texture over the surface a present names, as
/// the device's display shows it (reims-vgpu's `Presented::show`), with a window or without one.
/// A window's own draws of it into its drawables are ordinary command buffers in the stream.
/// `t` is a texture of any platform; only the Vulkan platform's is recorded.
pub fn present(t: &dyn Any) {
    if !on() {
        return;
    }
    let Some(t) = t.downcast_ref::<VulkanTexture>() else {
        return;
    };
    adopt_texture(t);
    let Some(state) = state() else { return };
    let arg = id_at(t.id().0).map_or(Value::Nil, Value::Object);
    state.push_plain(Call {
        selector: "m2v.present".to_owned(),
        args: vec![arg],
        ..Call::default()
    });
    state.stats.presents.fetch_add(1, Ordering::Relaxed);
}

/// The selector of a no-copy buffer's escape record ([`buffer_escaped`]).
pub const BUFFER_ESCAPED: &str = "m2v.bufferEscaped";

/// The guest noted that the identity of the no-copy buffer `b` left its Metal with no message
/// naming the buffer, by the act `kind`, an `RGFX_BUFFER_ESCAPED_*` value (0 the app's read of its
/// `gpuAddress`, 1 an argument encoder's write, 2 the app's read of a `gpuResourceID`, 3 a
/// residency set's add): from here on the GPU may reach the buffer through memory alone, by an
/// address in bytes the stream writes. Its record is a [`BUFFER_ESCAPED`] call on the buffer,
/// `[kind]`, of all of the buffer, which a player stops at, typed, where a later write holds an
/// address in the buffer. The reims-graphics device calls it once in the buffer's life, at the
/// guest's first note (reims-vgpu's `decode::escape`).
///
/// The record is queued here, on the calling thread, so it lies in the stream's order: after the
/// `m2v.write` records of each commit made before the note, which the commit queued before its
/// call returned ([`before_commit`]), and before those of each commit made after it. The guest's
/// device carries a command buffer out only after the notes its process posted before it.
///
/// `b` is a buffer of any platform; only the Vulkan platform's is recorded. The device makes each
/// no-copy buffer by a recorded call (`newBufferWithDescriptor:`), so the buffer has its stream
/// id: one with none is counted as an unrecorded receiver, with no record. The record is counted
/// by its selector in `<pid>.stats.txt`.
pub fn buffer_escaped(b: &dyn Any, kind: u64) {
    if !on() {
        return;
    }
    let Some(b) = b.downcast_ref::<VulkanBuffer>() else {
        return;
    };
    let Some(state) = state() else { return };
    let Some(id) = id_at(b.id().0) else {
        (state.stats.unrecorded_receivers).fetch_add(1, Ordering::Relaxed);
        return;
    };
    state.push_plain(Call {
        selector: BUFFER_ESCAPED.to_owned(),
        receiver: id,
        args: vec![Value::Uint(kind)],
        ..Call::default()
    });
    state.count_selector(BUFFER_ESCAPED);
}

/// The selector of the acquire of what another API wrote into a buffer's exported memory, the
/// Vulkan platform's `external_write_done`: a call on the buffer, with no arguments, of all of the
/// buffer, recorded as the call returns, as any call core makes. The host's video makes it once a
/// picture is in a surface's device cache (the hardware's decode into the cache's memory, or its
/// copy or video processing there, waited for), on the session's landing thread, right before the
/// session's signal (`setSignaledValue:`) that lets the decode's command buffer read the picture
/// past its wait, and while no command buffer that uses the buffer is in flight. So the record
/// follows the hardware's writes, which the stream does not hold, and precedes every command that
/// reads them: a player refills what lies over the buffer there. A refused acquire (a buffer not
/// made exportable) is recorded all the same, as any call's refusal.
pub const EXTERNAL_WRITE_DONE: &str = "m2v.externalWriteDone";

/// An object's last reference went, at `ptr`: its `release` record, after every record queued
/// before it. Its pointer is no longer the object's at once, and memory reached through it is
/// no longer read.
pub fn released(ptr: u64) {
    let Some(state) = state() else { return };
    let Some(id) = state.ptrs.lock().remove(&ptr) else {
        return;
    };
    if let Some(info) = state.objects.lock().info.get_mut(&id) {
        info.dead = true;
    }
    state.baselines.lock().remove(&id);
    state.push(Item::Release(id));
}

/// Records the call `active` opened, with the object it made: queued, for the writer to shape;
/// then, where it is the thread's outermost, what the completions of the command buffers it
/// committed wrote ([`release_completions`]).
pub fn finish(active: Active, made: Option<Made>) {
    let Active {
        token,
        pending,
        text,
        snap,
    } = active;
    let Some(state) = state() else { return };
    let Some(mut call) = pending else {
        if token.outer {
            release_completions(state);
        }
        return;
    };
    let t0 = now_ns();
    let mut identity = None;
    if let Some(mut made) = made {
        identity = made.identity.take();
        call.result = state.register(made);
    }
    state.capture_now(&mut call);
    let result = call.result;
    let item = Item::Call {
        call,
        text,
        shape: true,
        snap,
    };
    state.push_made(item, result, identity);
    release_completions(state);
    add_times(state, 0, now_ns() - t0, true);
    drop(token);
}

/// Records each of `draws` as its two commands, where the encoder's run of draws is the outermost
/// call; the commands it sends itself are then inside it.
pub fn draws<T: Send + Sync + 'static>(
    token: &Option<Token>,
    encoder: &Obj<T>,
    draws: &[crate::core::platform::render::VertexDraw<'_, crate::platform::VulkanPlatform>],
) {
    if !token.as_ref().is_some_and(|t| t.outer) {
        return;
    }
    let Some(state) = state() else { return };
    let t0 = now_ns();
    let receiver = receiver_id(encoder);
    for d in draws {
        for cmd in [d.vertex_buffer(), d.primitives()] {
            let (args, text) = cmd_call(&cmd);
            let call = Call {
                selector: cmd.selector().to_owned(),
                receiver,
                args,
                ..Call::default()
            };
            state.push(Item::Call {
                call,
                text,
                shape: false,
                snap: None,
            });
        }
    }
    add_times(state, now_ns() - t0, 0, false);
}

/// Before a commit: shapes what is queued, waits for the earlier command buffers that can run,
/// then writes the memory the command buffer reaches that changed. A commit inside another call
/// (`m2v.split`, `m2v.commitPart`), which records no `commit` of its own, writes it too, before
/// that call's record. The memory the command buffer's GPU work writes, and the memory its
/// kernels and render stages could store to, are kept for its completion ([`completion_seen`]),
/// which a completion-seen handler of the recorder's added now runs: the first taken as the
/// stream's, the second written where it changed, after the record of the call that committed it
/// ([`release_completions`]). So are the shared event signals it encodes, for the guest's
/// learning of one before it completes ([`event_reached`]), and the events it waits for, which its
/// completion tells reached; and the runtime's command buffer, held until the completion is taken
/// ([`Completion::raw`]).
pub fn before_commit(active: &Option<Active>, cb: &VulkanCommandBuffer) {
    if active.is_none() {
        return;
    }
    let Some(state) = state() else { return };
    let Some(id) = id_at(cb.id().0) else { return };
    state.drain(true);
    let t0 = now_ns();
    let mut reaped = state.reap();
    state.wait_for_earlier();
    reaped.extend(state.reap());
    let t1 = now_ns();
    state.absorb_reaped(&reaped);
    state.stats.wait_ns.fetch_add(t1 - t0, Ordering::Relaxed);
    let (reach, writes, stores, signals, waits): (Vec<u64>, _, Vec<u64>, Vec<_>, Vec<_>) = {
        let sync = state.sync.lock();
        (sync.cbs.get(&id))
            .map(|c| {
                let reach = c.reach.iter().copied().collect();
                (
                    reach,
                    c.gpu_writes.clone(),
                    c.stores.iter().copied().collect(),
                    c.signals.clone(),
                    c.waits.clone(),
                )
            })
            .unwrap_or_default()
    };
    let ranges = state.ranges_of(&reach);
    state.capture(&ranges);
    let written = state.written_spans(&writes);
    let stored = state.stored_spans(&stores);
    if !written.is_empty() || !stored.is_empty() || !waits.is_empty() {
        let completion = Completion {
            written,
            stored,
            signals: signals.into_iter().map(|(e, v)| (e, v, false)).collect(),
            waits,
            heard: false,
            raw: cb.raw.clone(),
        };
        {
            let mut sync = state.sync.lock();
            sync.absorbs.insert(id, completion);
            sync.unqueued.insert(id);
        }
        COMMITTED.with(|c| c.borrow_mut().push(id));
        // Run where the runtime sees the command buffer complete, after any of core's added
        // before (whose hook takes the bytes first), before its completed handlers and before
        // `waitUntilCompleted` returns. Refused, the bytes are taken where the platform first
        // tells of the completion, or once it is reaped.
        let seen: metal2vulkan_runtime::Handler =
            Box::new(move |_: &metal2vulkan_runtime::CommandBuffer| completion_seen(id));
        let _ = cb.raw.add_completion_seen_handler(seen);
    }
    let t2 = now_ns();
    (state.stats.capture_ns).fetch_add(t2 - t1, Ordering::Relaxed);
    state.stats.commits.fetch_add(1, Ordering::Relaxed);
    let took = t2 - t0;
    (state.stats.commit_max_ns).fetch_max(took, Ordering::Relaxed);
    for (over, count) in [
        (1_000_000, &state.stats.commits_over_1ms),
        (4_000_000, &state.stats.commits_over_4ms),
        (8_000_000, &state.stats.commits_over_8ms),
    ] {
        if took > over {
            count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The stream id of `cb` while the process records, which a completion-seen handler of core's
/// passes to [`completion_seen`] before its own work; `None`, at the cost of one relaxed load,
/// where it does not.
pub fn completion_key(cb: &VulkanCommandBuffer) -> Option<u64> {
    if !on() {
        return None;
    }
    id_at(cb.id().0)
}

/// The command buffer of stream id `id` seen complete, before the platform can tell the guest
/// ([`State::complete`]): by the first of its completion-seen handlers to run, before it or any
/// later handler tells the guest; and by the platform's answer of its status final, a completed
/// handler of it and a return from a wait for it, before each tells the guest, for a command
/// buffer whose status reads final before its handlers run or whose hook the runtime refused.
/// The bytes its blits and passes wrote that fill whole chunks become the stream's, and each
/// chunk its kernels and render stages could store to, or its blits wrote part of, is written
/// where it differs from the stream's, so a CPU write the guest makes once told reaches the
/// stream, one that puts back the bytes the stream held before the command buffer ran included.
/// Called again, for a command buffer whose writes were taken already
/// ([`State::absorb_reaped`]), or for one with none to take, it does nothing. A handler runs it
/// on the runtime's thread, which it holds while it hashes; it takes the recorder's locks, which
/// no thread holds while it waits for the device, and lets go of the runtime's command buffer
/// once it holds none ([`Completion::raw`]).
pub fn completion_seen(id: u64) {
    let Some(state) = state() else { return };
    let mut sync = state.sync.lock();
    let taken = sync.absorbs.remove(&id);
    if let Some(completion) = &taken {
        state.complete(&mut sync, id, completion);
    }
    drop(sync);
    drop(taken);
}

/// The stream id of the shared event `event` while the process records, which the platform
/// passes to [`event_reached`] where it learns the event's value; `None`, at the cost of one
/// relaxed load, where it does not.
pub fn event_key(event: &crate::gpu::event::VulkanSharedEvent) -> Option<u64> {
    if !on() {
        return None;
    }
    id_at(event.id().0)
}

/// The shared event of stream id `event` seen at `value` or above, where the platform learns
/// it (a listener's notification as it runs, a CPU wait that saw it reached, a read of the
/// value), before the guest can: each command buffer whose completion is still to be taken and
/// that signals the event to `value` or below, a signal not seen reached yet, ran its work
/// before that signal, so what it could have written is compared with the stream's as at a
/// completion ([`State::hear`]). So a CPU write the guest makes once told reaches the stream, one
/// that puts back the bytes the stream held before the command buffer ran included.
pub fn event_reached(event: u64, value: u64) {
    let Some(state) = state() else { return };
    let mut sync = state.sync.lock();
    state.reached(&mut sync, event, value);
}

/// After the record of a thread's outermost call is queued: the writes at the completions of
/// the command buffers it committed ([`before_commit`]) that came before that record are queued
/// now, after it, and those to come are queued as they come. So a completion's writes, the bytes
/// of the command buffer's own GPU work among them, follow its commit in the stream, though the
/// runtime may see it complete before the call that committed it returns.
fn release_completions(state: &State) {
    let ids = COMMITTED.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if ids.is_empty() {
        return;
    }
    let mut sync = state.sync.lock();
    for id in ids {
        sync.unqueued.remove(&id);
        if let Some(writes) = sync.held.remove(&id) {
            writes.into_iter().for_each(|w| state.push(Item::Write(w)));
        }
    }
}

/// The call `active` opened encoded nothing, so the stream holds no record of it: a copy between
/// a buffer and the guest's RAM that the platform did not make (it answered false, or a refusal).
pub fn unmade(active: &mut Option<Active>) {
    if let Some(a) = active {
        a.pending = None;
    }
}

/// Adds the answer `key` to the call `active` records: what the call answered that the program
/// sees, `value`, read only where the call is recorded.
pub fn answer(active: &mut Option<Active>, key: &'static str, value: impl FnOnce() -> Value) {
    if let Some(a) = active
        && a.token.outer
        && let Some(call) = a.pending.as_mut()
    {
        call.answers.push((key.to_owned(), value()));
    }
}

/// `object`'s `gpuResourceID` as `read` answers it, recorded as a call of Metal's getter on it
/// with its answer `gpuResourceID`: the ID the program saw, which a replay gives again where its
/// runtime can. A getter that raises answers nothing.
pub fn resource_id<T: Send + Sync + 'static, E>(
    object: &Obj<T>,
    read: impl FnOnce() -> Result<u64, E>,
) -> Result<u64, E> {
    let mut active = rec_call!("gpuResourceID", object);
    let id = run(read);
    if let Ok(id) = &id {
        answer(&mut active, "gpuResourceID", || Value::Uint(*id));
    }
    rec_done!(active, ());
    id
}

/// [`resource_id`] of a getter that never raises.
pub fn plain_resource_id<T: Send + Sync + 'static>(
    object: &Obj<T>,
    read: impl FnOnce() -> u64,
) -> u64 {
    let Ok(id) = resource_id(object, || Ok::<u64, std::convert::Infallible>(read()));
    id
}

/// The bytes a buffer made as a copy of host memory starts with (`newBufferWithDescriptor:`'s
/// `contents`, not no-copy), read now, while the call holds them: its descriptor's `contents`
/// becomes them, a blob where they are long. Memory a no-copy buffer is made over stays nil
/// ([`no_copy_contents`] but for a `Managed` one on a device that keeps two copies): its bytes
/// arrive as writes.
pub fn contents(active: &mut Option<Active>, bytes: crate::core::platform::HostRange<'_>) {
    let Some(a) = active else { return };
    if !a.recorded() || bytes.ptr().addr() == 0 {
        return;
    }
    let data = read_range(bytes, bytes.len());
    if let Some(call) = a.pending.as_mut() {
        put_contents(call, data);
    }
}

/// The first `n` bytes of `bytes`, read now.
fn read_range(bytes: crate::core::platform::HostRange<'_>, n: u64) -> Vec<u8> {
    let n = usize::try_from(n.min(bytes.len())).unwrap_or(0);
    let mut data = vec![0u8; n];
    // SAFETY: the range is the call's argument, which its caller keeps mapped for the call
    // (`HostRange`'s contract), the call is in progress, and `n` is at most its length.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.ptr().as_ptr::<u8>(), data.as_mut_ptr(), n);
    }
    data
}

/// Makes `data` the `contents` of the descriptor `call` takes first (`newBufferWithDescriptor:`).
fn put_contents(call: &mut Call, data: Vec<u8>) {
    let Some(desc) = call.args.first_mut() else {
        return;
    };
    if let Value::Debug(text) = desc {
        let read = parse_text(text);
        *desc = read;
    }
    if let Value::Map(fields) = desc
        && let Some((_, v)) = fields.iter_mut().find(|(k, _)| k == "contents")
    {
        *v = Value::Bytes(data);
    }
}

/// `MTLResourceOptions`' storage mode `Managed`, in its bits 4 to 7.
const MANAGED: u64 = 1;

/// How many bytes of a no-copy buffer's pages, from its first, its create copies into a copy of
/// the device's own: on a device that keeps two copies of a `Managed` buffer (`two_copies`, its
/// `hasUnifiedMemory` NO), a `Managed` one's (by `options`, the buffer's resource options) of
/// `length` bytes, but those from `zero_from` on, which the caller said read zero and which the
/// device's copy starts as zeros (`BufferDesc::zero_from`); else none: a `Shared` or `Private`
/// buffer, or any on a device that keeps one copy, lies in the pages alone.
fn create_copies(two_copies: bool, options: u64, length: u64, zero_from: Option<u64>) -> u64 {
    if !two_copies || (options >> 4) & 0xf != MANAGED {
        return 0;
    }
    zero_from.map_or(length, |z| z.min(length))
}

/// The bytes of a no-copy buffer's pages its create copies into the device's own copy, read now,
/// while the call holds them ([`create_copies`]: a `Managed` buffer of resource options `options`
/// on a device that keeps two copies, its bytes before `zero_from`): the descriptor's `contents`
/// of `newBufferWithDescriptor:`, or the first argument of
/// `newBufferWithBytesNoCopy:length:options:deallocator:`, a blob where they are long. A player
/// rebuilds the device's copy from them, which only a `didModifyRange:` changes afterwards
/// (docs/graphics-kext/dgpu-oracle.md, "A no-copy Managed buffer"). Elsewhere, and where the
/// create copies no byte, the record keeps nil there.
pub fn no_copy_contents(
    active: &mut Option<Active>,
    device: &crate::gpu::device::VulkanDevice,
    options: u64,
    bytes: crate::core::platform::HostRange<'_>,
    zero_from: Option<u64>,
) {
    if !active.as_ref().is_some_and(Active::recorded) {
        return;
    }
    let two_copies = !device.raw.properties().has_unified_memory;
    no_copy_contents_with(active, two_copies, options, bytes, zero_from);
}

/// [`no_copy_contents`] on a device that keeps two copies of a `Managed` buffer where
/// `two_copies`.
fn no_copy_contents_with(
    active: &mut Option<Active>,
    two_copies: bool,
    options: u64,
    bytes: crate::core::platform::HostRange<'_>,
    zero_from: Option<u64>,
) {
    let Some(a) = active else { return };
    if !a.recorded() || bytes.ptr().addr() == 0 {
        return;
    }
    let n = create_copies(two_copies, options, bytes.len(), zero_from);
    if n == 0 {
        return;
    }
    let data = read_range(bytes, n);
    let Some(call) = a.pending.as_mut() else {
        return;
    };
    if call.selector == NO_COPY_BYTES {
        if let Some(first) = call.args.first_mut() {
            *first = Value::Bytes(data);
        }
    } else {
        put_contents(call, data);
    }
}

/// The selector of a no-copy buffer over host memory.
const NO_COPY_BYTES: &str = "newBufferWithBytesNoCopy:length:options:deallocator:";

/// `x` as its `Debug` text reads, read now ([`debug_value`]): an enum value without data is its
/// name, one with data `{Name: value}`.
pub fn text_value<T: std::fmt::Debug>(x: &T) -> Value {
    parse_text(&debug_text(x))
}

/// The guest's RAM a copy into `destination` reads (`m2v.copyFromGuestRam`,
/// `m2v.copyFromGuestRamAt`), where the platform encoded the copy: once the earlier command
/// buffers that can run have finished, as before a commit ([`before_commit`]), each run's bytes,
/// read now, and the byte of `destination` they land from become the call's arguments,
/// `[destination, [[byte, bytes], …]]`, the bytes a blob where they are long.
pub fn guest_ram_source(
    active: &mut Option<Active>,
    destination: &VulkanBuffer,
    runs: impl Iterator<Item = (u64, std::ops::Range<u64>)>,
) {
    let Some(a) = active else { return };
    if !a.recorded() {
        return;
    }
    let Some(state) = state() else { return };
    state.drain(true);
    let t0 = now_ns();
    let mut reaped = state.reap();
    state.wait_for_earlier();
    reaped.extend(state.reap());
    state.absorb_reaped(&reaped);
    state
        .stats
        .wait_ns
        .fetch_add(now_ns() - t0, Ordering::Relaxed);
    let destination = id_at(destination.id().0).map_or(Value::Nil, Value::Object);
    // SAFETY: the platform copied the runs, so they lie in guest RAM a live lease holds
    // (`os::ram_over`), mapped while the lease lives.
    let (args, n) = unsafe { guest_ram_args(destination, runs) };
    state.stats.guest_ram_bytes.fetch_add(n, Ordering::Relaxed);
    if let Some(call) = a.pending.as_mut() {
        call.args = args;
    }
}

/// A copy's arguments from the guest's RAM it reads, `[destination, [[byte, bytes], …]]`, each
/// run's bytes read now, and how many bytes they are.
///
/// # Safety
///
/// Each run is host memory mapped for the call.
unsafe fn guest_ram_args(
    destination: Value,
    runs: impl Iterator<Item = (u64, std::ops::Range<u64>)>,
) -> (Vec<Value>, u64) {
    let mut pieces = Vec::new();
    let mut total = 0;
    for (byte, run) in runs {
        let n = usize::try_from(run.end.saturating_sub(run.start)).unwrap_or(0);
        let mut data = vec![0u8; n];
        if n > 0 {
            // SAFETY: the run is mapped (the caller's contract); the bytes are the guest's, which
            // its CPU may write meanwhile, as a copy of bytes tolerates.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    HostPtr::from_addr(usize::try_from(run.start).unwrap_or(0)).as_ptr::<u8>(),
                    data.as_mut_ptr(),
                    n,
                );
            }
        }
        total += n as u64;
        pieces.push(Value::List(vec![Value::Uint(byte), Value::Bytes(data)]));
    }
    (vec![destination, Value::List(pieces)], total)
}

/// Before `didModifyRange:` of a `Managed` buffer, an announce: the range's bytes, as a write of
/// each chunk whose bytes differ from its comparison point or from those the stream last
/// recorded for it, or for which it recorded none ([`State::announce`]). So at the announce the
/// stream's last recorded bytes of the range are the guest's, those the GPU wrote and the stream
/// took without a write ([`State::absorb`]) included, whenever it took them: what
/// [`format::FLAG_ANNOUNCED_BYTES_RECORDED`] promises a player.
pub fn before_modify(active: &Option<Active>, buffer: &VulkanBuffer, range: &NsRange) {
    let Some(a) = active else { return };
    if !a.recorded() {
        return;
    }
    let Some(state) = state() else { return };
    let Some(id) = id_at(buffer.id().0) else {
        return;
    };
    let ranges = state.ranges_of(&[id]);
    let wanted = range.location..range.location.saturating_add(range.length);
    let ranges: Vec<Span> = ranges
        .into_iter()
        .filter_map(|mut s| {
            let start = s.start.max(s.from.saturating_add(wanted.start));
            let end = s.end.min(s.from.saturating_add(wanted.end));
            (start < end).then(|| {
                s.start = start;
                s.end = end;
                s
            })
        })
        .collect();
    state.announce(&ranges);
}

/// What the recorder knows of one object.
#[derive(Default)]
struct Info {
    type_name: &'static str,
    weak: Option<Weak<dyn Any + Send + Sync>>,
    /// The command buffer of an encoder, the queue of a command buffer.
    parent: u64,
    /// A block's host memory: its address and length.
    block: Option<(usize, u64)>,
    /// Its GPU range, where the guest pinned it.
    pin: Option<(u64, u64)>,
    /// The objects a descriptor's calls named, or an indirect command buffer's commands.
    refs: KeySet,
    /// Whether its last reference went: its release is queued, and its memory no longer read.
    dead: bool,
}

/// What a calling thread hands the writer, in the stream's order.
enum Item {
    /// A hooked call: its text read (the command's where `text`), shaped where `shape`, tracked
    /// and written; where it is a dispatch snapshotted (`snap`), its position kept for the
    /// snapshot's records.
    Call {
        call: Call,
        text: Option<String>,
        shape: bool,
        snap: Option<u64>,
    },
    /// A dispatch's snapshot, read at its command buffer's completion: its `m2v.privateBytes`
    /// records ([`private`]).
    Private(private::Read),
    /// A record written as it is.
    Plain(Call),
    /// A device's identity record ([`Identity`]), written as it is and kept for each later
    /// segment's start.
    Identity(Call),
    /// An `m2v.write` of the chunks a capture found changed: its chunks scanned for addresses
    /// (`scan_addresses`), then written, counted by its selector.
    Write(Call),
    /// An object's release: its last record, after which the recorder forgets it.
    Release(u64),
}

#[derive(Default)]
struct Objects {
    info: KeyMap<Info>,
    /// Live pinned ranges: start, then end and id.
    pins: BTreeMap<u64, (u64, u64)>,
}

/// A command buffer's reach and its event waits and signals.
#[derive(Default)]
struct Cb {
    reach: KeySet,
    /// What its passes render into and its blits write, bytes only the GPU writes: each object
    /// with the byte ranges of it written, all of it where `None`.
    gpu_writes: KeyMap<Option<Vec<(u64, u64)>>>,
    /// What its kernels and render stages could store to: what the calls of its compute and
    /// render encoders reach, but for a `useResources:` whose usage holds no write.
    stores: KeySet,
    waits: Vec<(u64, u64)>,
    signals: Vec<(u64, u64)>,
}

impl Cb {
    /// Notes that its GPU work writes `bytes` of the object `id`, all of it where `None`.
    fn writes(&mut self, id: u64, bytes: Option<(u64, u64)>) {
        let held = self
            .gpu_writes
            .entry(id)
            .or_insert_with(|| Some(Vec::new()));
        match (held, bytes) {
            (Some(ranges), Some(range)) => ranges.push(range),
            (held, _) => *held = None,
        }
    }
}

/// A committed command buffer not yet seen complete.
struct Pending {
    /// The command buffer's stream id.
    id: u64,
    queue: u64,
    waits: Vec<(u64, u64)>,
    signals: Vec<(u64, u64)>,
    /// The command buffer, never upgraded by the writer ([`upgrade_cb`]).
    weak: Weak<dyn Any + Send + Sync>,
}

#[derive(Default)]
struct SyncState {
    cbs: KeyMap<Cb>,
    pending: Vec<Pending>,
    /// The highest value the CPU, or a completed command buffer, signalled each event to.
    signalled: KeyMap<u64>,
    /// What each committed command buffer's completion takes ([`Completion`]), by its stream
    /// id, until it is taken ([`completion_seen`]).
    absorbs: KeyMap<Completion>,
    /// The committed command buffers whose commit's record is not queued yet: the call that
    /// committed each is still running ([`release_completions`]).
    unqueued: KeySet,
    /// The writes of the completions of those command buffers, held until that record is.
    held: KeyMap<Vec<Call>>,
}

/// What a committed command buffer's completion does with host memory ([`State::complete`]),
/// and what the guest can learn of it before it completes.
struct Completion {
    /// What its blits and passes write, bytes only the GPU writes ([`Cb::gpu_writes`]): each
    /// chunk wholly inside taken as the stream's, each it writes part of compared.
    written: Vec<Span>,
    /// What its kernels and render stages could store to ([`Cb::stores`]), in CPU-visible
    /// memory the GPU's work writes in place: compared.
    stored: Vec<Span>,
    /// The events its encoders signal, each with the value and whether the recorder saw the
    /// event reach it ([`event_reached`]).
    signals: Vec<(u64, u64, bool)>,
    /// The events its encoders wait for, each with the value: reached once it completes.
    waits: Vec<(u64, u64)>,
    /// Whether a signal of it was seen reached ([`State::hear`]): what its blits and passes
    /// write was taken there, while its work after the signal may still write it, so its
    /// completion compares all of it.
    heard: bool,
    /// The runtime's command buffer, held until the completion is taken: core lets go of the
    /// guest's at its commit (its release record follows the commit's), and the recorder reads
    /// the status and waits here. It is no object of the stream's, and holding it changes nothing
    /// the guest observes: its object, its id and its release are core's, as before. It is
    /// dropped with no recorder lock held.
    raw: metal2vulkan_runtime::CommandBuffer,
}

/// One span of host memory to compare: `start..end` of the target's bytes, the target's first
/// byte at `base` in host memory and `len` bytes long; `from` is where the reached object starts
/// in the target.
#[derive(Debug, Clone, Copy)]
struct Span {
    target: u64,
    base: usize,
    len: u64,
    from: u64,
    start: u64,
    end: u64,
}

/// What the stream holds of one 4 KiB chunk of a target, as hashes of its bytes ([`hash64`]),
/// each 0 for none: bytes that hash to 0 are only written more often.
#[derive(Debug, Default, Clone, Copy)]
struct Chunk {
    /// The comparison point: the bytes the stream holds for it, from a write or taken as the
    /// GPU's at a completion ([`State::absorb`]).
    point: u64,
    /// The bytes the stream's last `m2v.write` of it recorded, 0 where none did.
    recorded: u64,
}

/// What a comparison of chunks with the stream's writes ([`State::compare`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compare {
    /// Before a commit: each chunk whose bytes differ from its comparison point.
    Commit,
    /// Before a `didModifyRange:`: each chunk whose bytes differ from its comparison point or
    /// from the bytes last recorded for it, or for which none were.
    Announce,
    /// At a completion, of what the GPU wrote: none; each chunk's bytes become its comparison
    /// point.
    Absorb,
    /// At a completion, of what the GPU could have written besides: each chunk whose bytes
    /// differ from its comparison point, as at a commit.
    Complete,
}

#[derive(Default)]
struct Stats {
    records: AtomicU64,
    bytes: AtomicU64,
    commits: AtomicU64,
    presents: AtomicU64,
    /// Bytes of the guest's RAM read for copies out of it ([`guest_ram_source`]).
    guest_ram_bytes: AtomicU64,
    wait_ns: AtomicU64,
    capture_ns: AtomicU64,
    /// Where a recording thread spends its time: reading a call's arguments ([`begin`]),
    /// queueing its record ([`finish`]), copying and hashing memory and queueing the changed
    /// chunks (`compare`), and shaping what is queued before a commit (`drain_ns`, with the
    /// wait for the writer's turn, over `drained_here` records); and the writer's: reading,
    /// shaping, tracking and encoding records wherever they ran (`process_ns`), and its file
    /// writes. A thread's `read_ns` and `finish_ns` reach the counts every [`TIMES_EVERY`] of
    /// its calls.
    read_ns: AtomicU64,
    finish_ns: AtomicU64,
    hash_ns: AtomicU64,
    write_ns: AtomicU64,
    drain_ns: AtomicU64,
    drained_here: AtomicU64,
    process_ns: AtomicU64,
    flush_ns: AtomicU64,
    /// How long the longest commit's wait and capture took, and how many took over 1, 4 and 8 ms.
    commit_max_ns: AtomicU64,
    commits_over_1ms: AtomicU64,
    commits_over_4ms: AtomicU64,
    commits_over_8ms: AtomicU64,
    /// Of the counts above, the completions' ([`State::complete`]): their time, the chunks they
    /// hashed, the writes they made, the chunks and bytes of those, and the writes held for the
    /// record of the call that committed their command buffer ([`release_completions`]).
    completion_ns: AtomicU64,
    completion_chunks_hashed: AtomicU64,
    completion_writes: AtomicU64,
    completion_chunks_written: AtomicU64,
    completion_write_bytes: AtomicU64,
    completion_writes_held: AtomicU64,
    /// The signals of command buffers seen reached before their completion, each a comparison
    /// ([`State::hear`]), whose time and writes the completions' counts hold.
    signals_heard: AtomicU64,
    waits: AtomicU64,
    held: AtomicU64,
    wait_timeouts: AtomicU64,
    chunks_hashed: AtomicU64,
    chunks_written: AtomicU64,
    chunks_refreshed: AtomicU64,
    write_bytes: AtomicU64,
    unrecorded_refs: AtomicU64,
    unrecorded_receivers: AtomicU64,
    unread_args: AtomicU64,
    /// 8-byte words in a live pinned range: at 8-aligned offsets, and at offsets 4 mod 8.
    address_words_aligned: AtomicU64,
    address_words_misaligned: AtomicU64,
    /// Private buffer ranges written around dispatches, and their bytes, before and after
    /// ([`private`]); and the dispatches or ranges not written: a dispatch over
    /// [`private::MOST`] bytes, a buffer the stream never made, a staging buffer or a handler the
    /// runtime refused, a command buffer that failed.
    private_ranges: AtomicU64,
    private_bytes: AtomicU64,
    private_skipped: AtomicU64,
}

/// The files a stream is written to: segments `<base>.NNNN.m2vr` of at most `limit` bytes past
/// one record, each starting with the header.
struct Segments {
    base: PathBuf,
    limit: u64,
    next: u32,
    file: Option<std::fs::File>,
    written: u64,
}

impl Segments {
    fn new(base: PathBuf, limit: u64) -> Segments {
        Segments {
            base,
            limit,
            next: 0,
            file: None,
            written: 0,
        }
    }

    /// Whether `bytes` more fill the segment they go to, so the bytes after them start the next.
    fn fills(&self, bytes: usize) -> bool {
        if bytes == 0 {
            return false;
        }
        let at = if self.starts() {
            format::HEADER.len() as u64
        } else {
            self.written
        };
        at.saturating_add(bytes as u64) >= self.limit
    }

    /// Whether the next bytes start a segment.
    fn starts(&self) -> bool {
        self.file.is_none() || self.written >= self.limit
    }

    /// Writes `bytes`, after a new segment's header where they start one.
    fn write(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if self.starts() {
            let path = PathBuf::from(format!("{}.{:04}.m2vr", self.base.display(), self.next));
            self.next += 1;
            self.file = std::fs::File::create(&path)
                .inspect_err(|e| {
                    log::warn!(target: "reims_vgpu_vulkan::recorder", "{}: {e}", path.display());
                })
                .ok();
            self.written = 0;
            if let Some(f) = self.file.as_mut() {
                let _ = f.write_all(&format::HEADER);
                self.written += format::HEADER.len() as u64;
            }
        }
        if let Some(f) = self.file.as_mut() {
            let _ = f.write_all(bytes);
            self.written += bytes.len() as u64;
        }
    }
}

struct State {
    base: PathBuf,
    blobs_dir: PathBuf,
    /// The records not yet shaped, in the stream's order.
    queue: Mutex<Vec<Item>>,
    /// Held while queued records are shaped, so they are shaped in order.
    shaping: Mutex<()>,
    stream: Mutex<Stream>,
    /// Each live object's stream id by its pointer ([`id_at`]).
    ptrs: Mutex<KeyMap<u64>>,
    /// The last stream id given.
    next: AtomicU64,
    objects: Mutex<Objects>,
    sync: Mutex<SyncState>,
    /// What the stream holds of each target's chunks, by the target's stream id.
    baselines: Mutex<KeyMap<Vec<Chunk>>>,
    selectors: Mutex<BTreeMap<String, u64>>,
    /// The last dispatch snapshot numbered ([`private`]).
    snaps: AtomicU64,
    /// Each snapshotted dispatch's position in the stream, by its snapshot, until its records.
    dispatched: Mutex<KeyMap<u64>>,
    /// The identity record of each device whose identity the stream holds and whose release it
    /// does not, by its stream id: what each later segment starts with. Taken under `stream`.
    identities: Mutex<BTreeMap<u64, Call>>,
    stats: Stats,
}

impl State {
    fn new(dir: PathBuf) -> State {
        let pid = std::process::id();
        let blobs_dir = dir
            .parent()
            .map_or_else(|| dir.join("blobs"), |p| p.join("blobs"));
        State {
            base: dir.join(pid.to_string()),
            blobs_dir,
            queue: Mutex::new(Vec::new()),
            shaping: Mutex::new(()),
            stream: Mutex::new(Stream::new()),
            ptrs: Mutex::new(KeyMap::default()),
            next: AtomicU64::new(0),
            objects: Mutex::new(Objects::default()),
            sync: Mutex::new(SyncState::default()),
            baselines: Mutex::new(KeyMap::default()),
            selectors: Mutex::new(BTreeMap::new()),
            snaps: AtomicU64::new(0),
            dispatched: Mutex::new(KeyMap::default()),
            identities: Mutex::new(BTreeMap::new()),
            stats: Stats::default(),
        }
    }

    /// Makes the directories and starts the writer's thread; false where either is refused.
    fn open(&'static self) -> bool {
        let made = self
            .base
            .parent()
            .map(std::fs::create_dir_all)
            .transpose()
            .and_then(|_| std::fs::create_dir_all(&self.blobs_dir));
        if let Err(e) = made {
            log::warn!(target: "reims_vgpu_vulkan::recorder", "{ENV_VAR}: {e}");
            return false;
        }
        std::thread::Builder::new()
            .name("reims-vgpu.recorder".to_owned())
            .spawn(move || self.write_loop())
            .inspect_err(|e| log::warn!(target: "reims_vgpu_vulkan::recorder", "{ENV_VAR}: {e}"))
            .is_ok()
    }

    fn write_loop(&self) {
        let mut segments = Segments::new(self.base.clone(), SEGMENT);
        let flush = u64::try_from(FLUSH.as_nanos()).unwrap_or(u64::MAX);
        let stats = u64::try_from(STATS.as_nanos()).unwrap_or(u64::MAX);
        let mut flushed = now_ns();
        let mut counted = flushed;
        loop {
            // While calls come, the queue is taken every `BUSY`, so a commit finds little left
            // to shape on its own thread.
            let busy = self.drain(false) > 0;
            std::thread::sleep(if busy { BUSY } else { DRAIN });
            let now = now_ns();
            if now - flushed < flush {
                continue;
            }
            flushed = now;
            let t0 = now_ns();
            self.write_out(&mut segments);
            (self.stats.flush_ns).fetch_add(now_ns() - t0, Ordering::Relaxed);
            if now - counted >= stats {
                counted = now;
                self.write_stats();
            }
        }
    }

    /// Queues a record, in the stream's order.
    fn push(&self, item: Item) {
        self.queue.lock().push(item);
    }

    /// Writes the stream's bytes since the last write out to its segments. Where they fill the
    /// segment, the identity record of each device then live is written at once, under the
    /// stream's lock, so the next bytes, which start the next segment, start with them: names
    /// carry across segments, so their records use the names the stream defined before. Under
    /// `shaping`, so no queued record is half processed meanwhile (a dispatch's position is read
    /// after its call is written).
    fn write_out(&self, segments: &mut Segments) {
        let bytes = {
            let _shaping = self.shaping.lock();
            let mut stream = self.stream.lock();
            let bytes = stream.take();
            if segments.fills(bytes.len()) {
                for call in self.identities.lock().values() {
                    stream.call(call);
                }
            }
            bytes
        };
        segments.write(&bytes);
    }

    /// Queues `item`, which made the object of stream id `id`, and right after it, where the
    /// object is a device, its identity record ([`Identity`]): under one lock, so no other
    /// thread's record comes between.
    fn push_made(&self, item: Item, id: u64, identity: Option<Identity>) {
        let mut queue = self.queue.lock();
        queue.push(item);
        if let Some(identity) = identity {
            queue.push(Item::Identity(identity.record(id)));
        }
    }

    /// Queues a record written as it is.
    fn push_plain(&self, call: Call) {
        self.push(Item::Plain(call));
    }

    /// Shapes and writes every queued record, in order: on the writer's thread, or on a calling
    /// thread (`here`) whose capture needs them tracked. How many it wrote.
    fn drain(&self, here: bool) -> u64 {
        let t0 = now_ns();
        let _shaping = self.shaping.lock();
        let items = std::mem::take(&mut *self.queue.lock());
        let n = items.len() as u64;
        if n > 0 {
            let t1 = now_ns();
            let (mut bytes, mut records) = (0, 0);
            for item in items {
                let (b, r) = self.process(item);
                bytes += b as u64;
                records += r;
            }
            self.stats.records.fetch_add(records, Ordering::Relaxed);
            self.stats.bytes.fetch_add(bytes, Ordering::Relaxed);
            self.stats
                .process_ns
                .fetch_add(now_ns() - t1, Ordering::Relaxed);
        }
        if here {
            self.stats
                .drain_ns
                .fetch_add(now_ns() - t0, Ordering::Relaxed);
            self.stats.drained_here.fetch_add(n, Ordering::Relaxed);
        }
        n
    }

    /// Shapes and writes one item: the length of its call records, and how many.
    fn process(&self, item: Item) -> (usize, u64) {
        match item {
            Item::Call {
                mut call,
                text,
                shape,
                snap,
            } => {
                if let Some(text) = text {
                    call.args = cmd_args_text(&text);
                }
                for v in &mut call.args {
                    if let Value::Debug(text) = v {
                        *v = parse_text(text);
                    }
                }
                if shape {
                    self.shape(&mut call);
                }
                self.track(&call);
                let n = self.emit(call);
                if let Some(snap) = snap {
                    // Items are shaped one at a time (`shaping`), so the last call written is this.
                    let at = self.stream.lock().calls().saturating_sub(1);
                    self.dispatched.lock().insert(snap, at);
                }
                (n, 1)
            }
            Item::Plain(call) => (self.emit(call), 1),
            Item::Identity(call) => {
                // Kept as it is written, under the stream's lock, so a segment that starts after
                // it starts with it ([`State::write_out`]).
                let mut stream = self.stream.lock();
                let n = stream.call(&call);
                self.identities.lock().insert(call.receiver, call);
                (n, 1)
            }
            Item::Write(call) => {
                for v in &call.args {
                    if let Value::Bytes(data) = v {
                        self.scan_addresses(data);
                    }
                }
                self.count_selector(&call.selector);
                (self.emit(call), 1)
            }
            Item::Private(read) => self.private_records(read),
            Item::Release(id) => {
                {
                    let mut objects = self.objects.lock();
                    if let Some(info) = objects.info.remove(&id)
                        && let Some((start, _)) = info.pin
                    {
                        objects.pins.remove(&start);
                    }
                }
                self.sync.lock().cbs.remove(&id);
                self.identities.lock().remove(&id);
                let n = self.emit(Call {
                    selector: "release".to_owned(),
                    receiver: id,
                    ..Call::default()
                });
                (n, 1)
            }
        }
    }

    /// What a call's record must take at its moment, before it is queued: the bytes of the host
    /// memory it consumes (`replaceRegion:`), and a block's memory, which captures read.
    fn capture_now(&self, call: &mut Call) {
        if call.selector.starts_with("replaceRegion:") {
            for v in &mut call.args {
                read_host(v, &call.selector);
            }
        }
        if call.selector == "m2v.newBlockWithLength:storageMode:"
            && let Some(Value::Host(p, l)) = call.args.first()
            && let Some(info) = self.objects.lock().info.get_mut(&call.result)
        {
            info.block = Some((*p, *l));
        }
    }

    fn write_stats(&self) {
        let s = &self.stats;
        let mut out = String::new();
        write_build(&mut out);
        let line = |out: &mut String, name: &str, v: &AtomicU64| {
            let _ = writeln!(out, "{name} {}", v.load(Ordering::Relaxed));
        };
        line(&mut out, "records", &s.records);
        line(&mut out, "bytes", &s.bytes);
        line(&mut out, "commits", &s.commits);
        line(&mut out, "presents", &s.presents);
        line(&mut out, "guest_ram_bytes", &s.guest_ram_bytes);
        line(&mut out, "wait_ns", &s.wait_ns);
        line(&mut out, "capture_ns", &s.capture_ns);
        line(&mut out, "read_ns", &s.read_ns);
        line(&mut out, "finish_ns", &s.finish_ns);
        line(&mut out, "hash_ns", &s.hash_ns);
        line(&mut out, "write_ns", &s.write_ns);
        line(&mut out, "drain_ns", &s.drain_ns);
        line(&mut out, "drained_here", &s.drained_here);
        line(&mut out, "process_ns", &s.process_ns);
        line(&mut out, "flush_ns", &s.flush_ns);
        line(&mut out, "commit_max_ns", &s.commit_max_ns);
        line(&mut out, "commits_over_1ms", &s.commits_over_1ms);
        line(&mut out, "commits_over_4ms", &s.commits_over_4ms);
        line(&mut out, "commits_over_8ms", &s.commits_over_8ms);
        line(&mut out, "completion_ns", &s.completion_ns);
        line(
            &mut out,
            "completion_chunks_hashed",
            &s.completion_chunks_hashed,
        );
        line(&mut out, "completion_writes", &s.completion_writes);
        line(
            &mut out,
            "completion_chunks_written",
            &s.completion_chunks_written,
        );
        line(
            &mut out,
            "completion_write_bytes",
            &s.completion_write_bytes,
        );
        line(
            &mut out,
            "completion_writes_held",
            &s.completion_writes_held,
        );
        line(&mut out, "signals_heard", &s.signals_heard);
        line(&mut out, "waits", &s.waits);
        line(&mut out, "held", &s.held);
        line(&mut out, "wait_timeouts", &s.wait_timeouts);
        line(&mut out, "chunks_hashed", &s.chunks_hashed);
        line(&mut out, "chunks_written", &s.chunks_written);
        line(&mut out, "chunks_refreshed", &s.chunks_refreshed);
        line(&mut out, "write_bytes", &s.write_bytes);
        line(&mut out, "unrecorded_refs", &s.unrecorded_refs);
        line(&mut out, "unrecorded_receivers", &s.unrecorded_receivers);
        line(&mut out, "unread_args", &s.unread_args);
        line(&mut out, "address_words_aligned", &s.address_words_aligned);
        line(
            &mut out,
            "address_words_misaligned",
            &s.address_words_misaligned,
        );
        line(&mut out, "private_ranges", &s.private_ranges);
        line(&mut out, "private_bytes", &s.private_bytes);
        line(&mut out, "private_skipped", &s.private_skipped);
        let _ = writeln!(out, "live_objects {}", self.objects.lock().info.len());
        write_counted(&mut out);
        for (sel, n) in self.selectors.lock().iter() {
            let _ = writeln!(out, "selector {n} {sel}");
        }
        let path = PathBuf::from(format!("{}.stats.txt", self.base.display()));
        let _ = std::fs::write(path, out);
    }

    /// Gives the object `made` its stream id: its information first, then its pointer's id, so
    /// a call that finds the id finds the object.
    fn register(&self, made: Made) -> u64 {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.objects.lock().info.insert(
            id,
            Info {
                type_name: made.type_name,
                weak: Some(made.weak),
                ..Info::default()
            },
        );
        let stale = self.ptrs.lock().insert(made.ptr, id);
        if let Some(old) = stale {
            if let Some(info) = self.objects.lock().info.get_mut(&old) {
                info.dead = true;
            }
            // The object at this address went with no release seen: its release comes first.
            self.baselines.lock().remove(&old);
            self.push(Item::Release(old));
        }
        id
    }

    fn type_of(&self, id: u64) -> &'static str {
        self.objects
            .lock()
            .info
            .get(&id)
            .map_or("", |i| i.type_name)
    }

    /// Puts a call in the stream's form: host memory, blocks, pins and data.
    fn shape(&self, call: &mut Call) {
        let block_of = |objects: &Objects, id: u64| objects.info.get(&id).and_then(|i| i.block);
        match call.selector.as_str() {
            "m2v.newBlockWithLength:storageMode:" => {
                // new_heap_over_host_memory(bytes, desc)
                let host = match call.args.first() {
                    Some(Value::Host(p, l)) => Some((*p, *l)),
                    _ => None,
                };
                let desc = call.args.get(1).cloned().unwrap_or(Value::Nil);
                let (size, pin) = desc_size_pin(&desc, "size");
                let len = host.map_or(size, |(_, l)| l);
                call.args = vec![Value::Uint(len), Value::Uint(0)];
                let mut objects = self.objects.lock();
                if let Some(info) = objects.info.get_mut(&call.result) {
                    info.block = host;
                }
                if let Some(pin) = pin {
                    self.pin_locked(&mut objects, call, pin, len);
                }
            }
            "newHeapWithDescriptor:" => {
                let (size, pin) = desc_size_pin(call.args.first().unwrap_or(&Value::Nil), "size");
                if let Some(pin) = pin {
                    let mut objects = self.objects.lock();
                    self.pin_locked(&mut objects, call, pin, size);
                }
            }
            "newBufferWithDescriptor:" => {
                let (len, pin) = desc_size_pin(call.args.first().unwrap_or(&Value::Nil), "length");
                if let Some(pin) = pin {
                    let mut objects = self.objects.lock();
                    self.pin_locked(&mut objects, call, pin, len);
                }
            }
            NO_COPY_BYTES => {
                // new_buffer_over_host_memory(bytes, options, deallocator): the bytes nil, but
                // those the create copied into the device's own copy ([`no_copy_contents`]).
                let mut args = std::mem::take(&mut call.args).into_iter();
                let (bytes, len) = match args.next() {
                    Some(Value::Host(_, l)) => (Value::Nil, l),
                    Some(Value::Bytes(data)) => {
                        let len = data.len() as u64;
                        (Value::Bytes(data), len)
                    }
                    _ => (Value::Nil, 0),
                };
                let options = args.next().unwrap_or(Value::Nil);
                call.args = vec![bytes, Value::Uint(len), options, Value::Nil];
            }
            "newBufferWithLength:options:offset:" => {
                // Heap::new_buffer_of_length(length, options, offset)
                let mut objects = self.objects.lock();
                let heap = call.receiver;
                let pin = objects.info.get(&heap).and_then(|i| i.pin);
                let (length, offset) = (uint(call.args.first()), uint(call.args.get(2)));
                if block_of(&objects, heap).is_some() {
                    let options = call.args.get(1).cloned().unwrap_or(Value::Nil);
                    call.selector = "m2v.newBufferInBlock:offset:length:options:".to_owned();
                    call.args = vec![
                        Value::Object(heap),
                        Value::Uint(offset),
                        Value::Uint(length),
                        options,
                    ];
                }
                if let Some((start, _)) = pin {
                    self.pin_locked(&mut objects, call, start + offset, length);
                }
            }
            "newLibraryWithData:error:" => {
                if let Some(Value::Bytes(data)) = call.args.first() {
                    call.args = vec![Value::Blob(self.blob_file(data))];
                }
            }
            _ => {}
        }
        for v in &mut call.args {
            read_host(v, &call.selector);
        }
    }

    /// Writes `data` once into the blob directory, as the hex of its SHA-256: its hash.
    fn blob_file(&self, data: &[u8]) -> Hash {
        let hash = format::sha256(data);
        let path = self.blobs_dir.join(hex(&hash));
        if !path.exists() {
            let tmp = (self.blobs_dir).join(format!("{}.{}", hex(&hash), std::process::id()));
            if std::fs::write(&tmp, data).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
        hash
    }

    fn pin_locked(&self, objects: &mut Objects, call: &mut Call, start: u64, len: u64) {
        call.answers
            .push(("gpuAddress".to_owned(), Value::Address(start)));
        call.answers
            .push(("gpuLength".to_owned(), Value::Uint(len)));
        if call.result != 0 && len > 0 {
            objects.pins.insert(start, (start + len, call.result));
            if let Some(info) = objects.info.get_mut(&call.result) {
                info.pin = Some((start, len));
            }
        }
    }

    /// Keeps what a call tells of command buffers: their encoders, reach, events and commits.
    fn track(&self, call: &Call) {
        self.count_selector(&call.selector);
        let recv_type = self.type_of(call.receiver);
        let cb_type = std::any::type_name::<CommandBufferObject>();
        let mut named = Vec::new();
        for a in &call.args {
            a.each_object(&mut |id| named.push(id));
        }
        // An object a command buffer or its encoder makes is the command buffer's; a command
        // buffer is its queue's, and so is the next part a command buffer makes.
        if call.result != 0 {
            let mut objects = self.objects.lock();
            let parent = match call.selector.as_str() {
                "m2v.split" | "m2v.part" => objects
                    .info
                    .get(&call.receiver)
                    .map_or(call.receiver, |i| i.parent),
                _ => call.receiver,
            };
            if let Some(info) = objects.info.get_mut(&call.result) {
                info.parent = parent;
            }
        }
        let cb = if recv_type == cb_type {
            call.receiver
        } else {
            let objects = self.objects.lock();
            let parent = objects.info.get(&call.receiver).map_or(0, |i| i.parent);
            let parent_is_cb = objects
                .info
                .get(&parent)
                .is_some_and(|i| i.type_name == cb_type);
            if parent_is_cb { parent } else { 0 }
        };
        if cb == 0 {
            // A descriptor keeps what its calls name, for the encoder made of it.
            // So does an indirect command buffer what its commands name, for the command buffers
            // that execute it, whose calls name only the buffer.
            let icb = recv_type == std::any::type_name::<crate::gpu::icb::IcbObject>();
            if (recv_type.contains("Descriptor") || icb) && !named.is_empty() {
                let mut objects = self.objects.lock();
                if let Some(info) = objects.info.get_mut(&call.receiver) {
                    info.refs.extend(named.iter().copied());
                }
            }
            if call.selector == "setSignaledValue:"
                && let Some(Value::Uint(v)) = call.args.first()
            {
                let mut sync = self.sync.lock();
                let e = sync.signalled.entry(call.receiver).or_insert(0);
                *e = (*e).max(*v);
            }
            return;
        }
        let mut reach: Vec<u64> = named.clone();
        let mut gpu_writes = Vec::new();
        {
            let mut objects = self.objects.lock();
            for id in &named {
                if let Some(info) = objects.info.get(id) {
                    reach.extend(info.refs.iter().copied());
                    // A pass's descriptor names what the pass renders into, all of each.
                    if call.selector.starts_with("renderCommandEncoder") {
                        gpu_writes.extend(info.refs.iter().map(|&r| (r, None)));
                    }
                }
            }
            // A copy of indirect commands copies what they name too.
            let copy = "copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:";
            if call.selector == copy
                && let [Value::Object(from), _, Value::Object(to), ..] = call.args.as_slice()
                && let Some(refs) = objects.info.get(from).map(|i| i.refs.clone())
                && let Some(info) = objects.info.get_mut(to)
            {
                info.refs.extend(refs);
            }
            // An indirect command buffer named here lets go of what was released since its
            // commands named it, which a capture skips (`ranges_of`), so its refs stay few.
            let icb = std::any::type_name::<crate::gpu::icb::IcbObject>();
            for id in &named {
                let Some(info) = objects.info.get(id).filter(|i| i.type_name == icb) else {
                    continue;
                };
                let gone: Vec<u64> = (info.refs.iter().copied())
                    .filter(|r| objects.info.get(r).is_none_or(|o| o.dead))
                    .collect();
                if let Some(info) = objects.info.get_mut(id) {
                    for r in &gone {
                        info.refs.remove(r);
                    }
                }
            }
        }
        if recv_type.contains("BlitCommandEncoder") {
            gpu_writes.extend(blit_writes(call, &named));
        }
        let stores = may_store(recv_type, call);
        let mut sync = self.sync.lock();
        let entry = sync.cbs.entry(cb).or_default();
        if stores {
            entry.stores.extend(reach.iter().copied());
        }
        entry.reach.extend(reach);
        for (id, bytes) in gpu_writes {
            entry.writes(id, bytes);
        }
        let event = |args: &[Value]| match (args.first(), args.get(1)) {
            (Some(Value::Object(e)), Some(Value::Uint(v))) => Some((*e, *v)),
            _ => None,
        };
        // A split that made the next part committed the command buffer, and so did a part's
        // commit that cut it ([`before_commit`]).
        let committed = match call.selector.as_str() {
            "commit" => true,
            "m2v.split" => call.result != 0,
            "m2v.commitPart" => (call.answers.iter())
                .any(|(k, v)| k == "cut" && matches!(v, Value::Str(s) if s == "Cut")),
            _ => false,
        };
        match call.selector.as_str() {
            "encodeSignalEvent:value:" | "encodeSignalEventScheduled:value:" => {
                if let Some(s) = event(&call.args) {
                    entry.signals.push(s);
                }
            }
            "encodeWaitForEvent:value:" => {
                if let Some(w) = event(&call.args) {
                    entry.waits.push(w);
                }
            }
            _ if committed => {
                let cb_info = sync.cbs.remove(&cb).unwrap_or_default();
                let objects = self.objects.lock();
                let info = objects.info.get(&cb);
                let queue = info.map_or(0, |i| i.parent);
                let weak = info
                    .filter(|i| i.type_name == cb_type)
                    .and_then(|i| i.weak.clone());
                drop(objects);
                if let Some(weak) = weak {
                    sync.pending.push(Pending {
                        id: cb,
                        queue,
                        waits: cb_info.waits,
                        signals: cb_info.signals,
                        weak,
                    });
                }
            }
            _ => {}
        }
    }

    /// Drops the committed command buffers that completed, their signals folded in, and those
    /// released with no completion to take: the stream ids of those dropped. One whose
    /// completion is to be taken is read through the runtime's command buffer the recorder holds
    /// for it ([`Completion::raw`]), so one the guest released stays until it completes.
    fn reap(&self) -> Vec<u64> {
        let mut held = Vec::new();
        let mut reaped = Vec::new();
        {
            let mut sync = self.sync.lock();
            let SyncState {
                pending,
                absorbs,
                signalled,
                ..
            } = &mut *sync;
            // The command buffers looked at are let go once the lock is (`let_go_cb` may record a
            // release).
            let mut done = Vec::new();
            pending.retain(|p| {
                let finished = match absorbs.get(&p.id) {
                    Some(completion) => raw_finished(&completion.raw),
                    None => match upgrade_cb(&p.weak) {
                        None => true,
                        Some(cb) => {
                            let finished = finished(&cb);
                            held.push(cb);
                            finished
                        }
                    },
                };
                if finished {
                    done.extend(p.signals.iter().copied());
                    reaped.push(p.id);
                }
                !finished
            });
            for (e, v) in done {
                let s = signalled.entry(e).or_insert(0);
                *s = (*s).max(v);
            }
        }
        held.into_iter().for_each(let_go_cb);
        reaped
    }

    /// Does the completion of each reaped command buffer ([`State::complete`]) where nothing
    /// took it yet ([`completion_seen`]): one whose status read final before its hook ran (a
    /// wait for earlier work returns there), one the guest released, or one whose hook the
    /// runtime refused. The sync lock is held throughout, so a hook that runs meanwhile, and
    /// what its command buffer's handlers tell the guest after it, wait for the bytes to be
    /// taken; the runtime's command buffers are dropped once it is let go.
    fn absorb_reaped(&self, reaped: &[u64]) {
        if reaped.is_empty() {
            return;
        }
        let mut sync = self.sync.lock();
        let mut taken = Vec::new();
        for &id in reaped {
            if let Some(completion) = sync.absorbs.remove(&id) {
                self.complete(&mut sync, id, &completion);
                taken.push(completion);
            }
        }
        drop(sync);
        drop(taken);
    }

    /// The completion of the command buffer of stream id `id`, the sync lock held (`sync`): what
    /// its blits and passes wrote that fills whole chunks becomes the stream's
    /// ([`State::absorb`]); then each chunk of what its kernels and render stages could store
    /// to, and of what its blits wrote part of, is written where its bytes differ from the
    /// stream's ([`Compare::Complete`]), so the stream's bytes there are the GPU's from here on
    /// and a CPU write after it, one that puts back the bytes before included, differs from
    /// them. Where a signal of it was seen reached first ([`State::hear`]), what its blits and
    /// passes wrote is compared too, since its work after the signal may have written it after
    /// the bytes were taken. Its writes are queued now, or, while the call that committed the
    /// command buffer has not queued its record, held until it does ([`release_completions`]).
    /// Each event it waited for reached its value, so the command buffers that signal it are
    /// heard ([`State::reached`]).
    fn complete(&self, sync: &mut SyncState, id: u64, completion: &Completion) {
        let t0 = now_ns();
        let mut spans = if completion.heard {
            completion.written.clone()
        } else {
            self.absorb(&completion.written)
        };
        spans.extend_from_slice(&completion.stored);
        let writes = self.compare(&spans, Compare::Complete);
        self.queue_completion(sync, id, writes);
        (self.stats.completion_ns).fetch_add(now_ns() - t0, Ordering::Relaxed);
        for &(event, value) in &completion.waits {
            self.reached(sync, event, value);
        }
    }

    /// The writes of the completion of the command buffer of stream id `id`, or of a signal of
    /// it heard, the sync lock held (`sync`): queued now, or, while the call that committed it has
    /// not queued its record, held until it does ([`release_completions`]).
    fn queue_completion(&self, sync: &mut SyncState, id: u64, writes: Vec<Call>) {
        let n = writes.len() as u64;
        if n == 0 {
            return;
        }
        self.stats.completion_writes.fetch_add(n, Ordering::Relaxed);
        if sync.unqueued.contains(&id) {
            (self.stats.completion_writes_held).fetch_add(n, Ordering::Relaxed);
            sync.held.entry(id).or_default().extend(writes);
        } else {
            writes.into_iter().for_each(|w| self.push(Item::Write(w)));
        }
    }

    /// The shared event of stream id `event` seen at `value`, the sync lock held (`sync`): each
    /// command buffer whose completion is still to be taken with a signal of it to `value` or
    /// below not seen reached yet has those signals marked reached and is heard
    /// ([`State::hear`]), in the order of their stream ids.
    fn reached(&self, sync: &mut SyncState, event: u64, value: u64) {
        let mut heard = Vec::new();
        for (&id, completion) in &mut sync.absorbs {
            let mut any = false;
            for signal in &mut completion.signals {
                if signal.0 == event && signal.1 <= value && !signal.2 {
                    signal.2 = true;
                    any = true;
                }
            }
            if any {
                heard.push(id);
            }
        }
        heard.sort_unstable();
        for id in heard {
            self.hear(sync, id);
        }
    }

    /// A signal of the command buffer of stream id `id` seen reached before its completion, the
    /// sync lock held (`sync`): its work before the signal ran, and the guest may learn so now.
    /// What it could have written is compared as at a completion ([`State::complete`]). Only the
    /// first time are the whole chunks its blits and passes write taken as the stream's
    /// ([`State::absorb`]): its work after the signal may still write them, so later signals and
    /// the completion compare all of it. So a CPU write the guest makes once told is in the
    /// stream, one that puts back the bytes the stream held before the command buffer ran
    /// included.
    fn hear(&self, sync: &mut SyncState, id: u64) {
        let t0 = now_ns();
        let Some(completion) = sync.absorbs.get_mut(&id) else {
            return;
        };
        let mut spans = if completion.heard {
            completion.written.clone()
        } else {
            self.absorb(&completion.written)
        };
        completion.heard = true;
        spans.extend_from_slice(&completion.stored);
        let writes = self.compare(&spans, Compare::Complete);
        self.stats.signals_heard.fetch_add(1, Ordering::Relaxed);
        self.queue_completion(sync, id, writes);
        (self.stats.completion_ns).fetch_add(now_ns() - t0, Ordering::Relaxed);
    }

    /// The host memory a command buffer's GPU work writes, `writes` ([`Cb::gpu_writes`]): each
    /// object's spans ([`State::ranges_of`]), cut to the bytes of it written.
    fn written_spans(&self, writes: &KeyMap<Option<Vec<(u64, u64)>>>) -> Vec<Span> {
        let mut out = Vec::new();
        for (&id, bytes) in writes {
            let spans = self.ranges_of(&[id]);
            let Some(ranges) = bytes else {
                out.extend(spans);
                continue;
            };
            for s in spans {
                for &(first, end) in ranges {
                    let start = s.start.max(s.from.saturating_add(first));
                    let end = s.end.min(s.from.saturating_add(end));
                    if start < end {
                        out.push(Span { start, end, ..s });
                    }
                }
            }
        }
        out
    }

    /// Takes as the stream's each 4 KiB of `spans`, what a command buffer's GPU work wrote, that
    /// lies wholly in them, spans of one target that meet joined; what lies in chunks they write
    /// only part of is the answer, for the completion to compare ([`State::complete`]): such a
    /// chunk keeps its comparison point, so a CPU write into the rest of it still reaches the
    /// stream.
    fn absorb(&self, spans: &[Span]) -> Vec<Span> {
        let mut spans = spans.to_vec();
        spans.sort_by_key(|s| (s.target, s.start));
        let mut joined: Vec<Span> = Vec::with_capacity(spans.len());
        for s in spans {
            match joined.last_mut() {
                Some(j) if j.target == s.target && s.start <= j.end => j.end = j.end.max(s.end),
                _ => joined.push(s),
            }
        }
        let (mut whole, mut parts) = (Vec::new(), Vec::new());
        for j in joined {
            let start = j.start.next_multiple_of(CHUNK);
            let end = if j.end >= j.len {
                j.len
            } else {
                j.end / CHUNK * CHUNK
            };
            if start >= end {
                parts.push(j);
                continue;
            }
            whole.push(Span { start, end, ..j });
            if j.start < start {
                parts.push(Span { end: start, ..j });
            }
            if end < j.end {
                parts.push(Span { start: end, ..j });
            }
        }
        self.compare(&whole, Compare::Absorb);
        parts
    }

    /// Waits, at most [`WAIT_BOUND`], for each committed command buffer that can run: one whose
    /// event waits a CPU signal, a completed command buffer or another that can run satisfies,
    /// and whose queue's earlier one can run. One whose completion is to be taken is waited for
    /// through the runtime's command buffer the recorder holds ([`Completion::raw`]), whether
    /// the guest released its own or not; a wait past the bound is counted (`wait_timeouts`).
    fn wait_for_earlier(&self) {
        let runnable: Vec<Watched> = {
            let sync = self.sync.lock();
            let n = sync.pending.len();
            let mut can = vec![false; n];
            loop {
                let mut changed = false;
                for i in 0..n {
                    if can[i] {
                        continue;
                    }
                    let p = &sync.pending[i];
                    let queue_ok = (0..i)
                        .filter(|&j| sync.pending[j].queue == p.queue)
                        .all(|j| can[j]);
                    let waits_ok = p.waits.iter().all(|&(e, v)| {
                        sync.signalled.get(&e).is_some_and(|&s| s >= v)
                            || (0..n).any(|j| {
                                can[j]
                                    && sync.pending[j]
                                        .signals
                                        .iter()
                                        .any(|&(se, sv)| se == e && sv >= v)
                            })
                    });
                    if queue_ok && waits_ok {
                        can[i] = true;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            let held = can.iter().filter(|c| !**c).count() as u64;
            self.stats.held.fetch_add(held, Ordering::Relaxed);
            sync.pending
                .iter()
                .zip(can)
                .filter(|(_, c)| *c)
                .map(|(p, _)| match sync.absorbs.get(&p.id) {
                    Some(completion) => Watched::Raw(completion.raw.clone()),
                    None => Watched::Weak(p.weak.clone()),
                })
                .collect()
        };
        let start = now_ns();
        let bound = u64::try_from(WAIT_BOUND.as_nanos()).unwrap_or(u64::MAX);
        for watched in &runnable {
            self.stats.waits.fetch_add(1, Ordering::Relaxed);
            loop {
                let done = match watched {
                    Watched::Raw(raw) => raw_finished(raw),
                    Watched::Weak(weak) => {
                        let Some(cb) = upgrade_cb(weak) else { break };
                        let done = finished(&cb);
                        let_go_cb(cb);
                        done
                    }
                };
                if done {
                    break;
                }
                if now_ns() - start > bound {
                    self.stats.wait_timeouts.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }

    /// The host memory each object of `ids` lies in, as spans of their targets.
    fn ranges_of(&self, ids: &[u64]) -> Vec<Span> {
        self.spans_of(ids, false)
    }

    /// The host memory of `ids`, what a command buffer's kernels and render stages could store
    /// to, that their stores reach in place: each object's spans ([`State::ranges_of`]) but a
    /// `Managed` buffer's where the device keeps two copies of it, whose CPU copy the GPU's work
    /// never writes (a blit's `synchronizeResource:` brings its bytes back, which the completion
    /// takes as a blit's) and whose CPU writes an announce records ([`before_modify`]). Where the
    /// device keeps one copy, a `Managed` buffer's bytes are the GPU's, as a `Shared` one's are.
    fn stored_spans(&self, ids: &[u64]) -> Vec<Span> {
        self.spans_of(ids, true)
    }

    /// The spans of `ids` ([`State::ranges_of`]), but, where `stored`, a buffer's of which the
    /// GPU writes a copy apart from the CPU's ([`gpu_copy_apart`]).
    fn spans_of(&self, ids: &[u64], stored: bool) -> Vec<Span> {
        let mut spans = Vec::new();
        let mut seen = HashSet::new();
        for &id in ids {
            if !seen.insert(id) {
                continue;
            }
            let (weak, block) = {
                let objects = self.objects.lock();
                let Some(info) = objects.info.get(&id).filter(|i| !i.dead) else {
                    continue;
                };
                (info.weak.clone(), info.block)
            };
            if let Some((base, len)) = block {
                spans.push(Span {
                    target: id,
                    base,
                    len,
                    from: 0,
                    start: 0,
                    end: len,
                });
                continue;
            }
            let Some(any) = weak.and_then(|w| w.upgrade()) else {
                continue;
            };
            if let Some(b) = any.downcast_ref::<BufferObject>() {
                if !(stored && gpu_copy_apart(b)) {
                    self.buffer_span(id, b, None, &mut spans);
                }
            } else if let Some(t) = any.downcast_ref::<TextureObject>() {
                match &t.under {
                    Some(Under::Buffer(buffer, _)) if stored && gpu_copy_apart(buffer) => {}
                    Some(Under::Buffer(buffer, bytes)) => {
                        if let Some(bid) = id_at(buffer.id().0) {
                            self.buffer_span(bid, buffer, Some(bytes.clone()), &mut spans);
                        }
                    }
                    Some(Under::Heap(heap, bytes)) => {
                        if let Some(hid) = id_at(heap.id().0)
                            && let Some((base, len)) = self.block_of(hid)
                        {
                            spans.push(Span {
                                target: hid,
                                base,
                                len,
                                from: 0,
                                start: bytes.start.min(len),
                                end: bytes.end.min(len),
                            });
                        }
                    }
                    None => {}
                }
            }
            let_go_any(any);
        }
        spans
    }

    fn block_of(&self, id: u64) -> Option<(usize, u64)> {
        let objects = self.objects.lock();
        objects
            .info
            .get(&id)
            .filter(|i| !i.dead)
            .and_then(|i| i.block)
    }

    fn buffer_span(
        &self,
        id: u64,
        b: &BufferObject,
        bytes: Option<std::ops::Range<u64>>,
        spans: &mut Vec<Span>,
    ) {
        let Some(ptr) = b.raw.contents() else { return };
        let ptr = ptr.as_ptr().addr();
        let length = b.raw.length();
        let bytes = bytes.unwrap_or(0..length);
        // A buffer in a block is written as the block's bytes.
        if let Some(heap) = &b.heap
            && let Some(hid) = id_at(heap.id().0)
            && let Some((base, len)) = self.block_of(hid)
            && ptr >= base
            && ((ptr - base) as u64) < len
        {
            let off = (ptr - base) as u64;
            spans.push(Span {
                target: hid,
                base,
                len,
                from: off,
                start: (off + bytes.start).min(len),
                end: (off + bytes.end).min(len),
            });
            return;
        }
        spans.push(Span {
            target: id,
            base: ptr,
            len: length,
            from: 0,
            start: bytes.start.min(length),
            end: bytes.end.min(length),
        });
    }

    /// Writes each 4 KiB of `spans` whose bytes changed since the stream last held them.
    fn capture(&self, spans: &[Span]) {
        let writes = self.compare(spans, Compare::Commit);
        writes.into_iter().for_each(|w| self.push(Item::Write(w)));
    }

    /// Writes each 4 KiB of `spans` whose bytes differ from the stream's or from those its last
    /// write of them recorded, or that no write recorded ([`before_modify`]).
    fn announce(&self, spans: &[Span]) {
        let writes = self.compare(spans, Compare::Announce);
        writes.into_iter().for_each(|w| self.push(Item::Write(w)));
    }

    /// Compares each 4 KiB of `spans` with the stream's: a chunk `how` writes is in the answer,
    /// one `m2v.write` a target, for the caller to queue, its bytes then both its comparison
    /// point and its recorded bytes; else, where `how` absorbs, its bytes are only taken as the
    /// stream's. A target released since its span was read is skipped: its release
    /// ([`released`]) marks it so before it takes the comparison points' lock, and the object
    /// goes only after that, so a target found live stays mapped while it is read.
    fn compare(&self, spans: &[Span], how: Compare) -> Vec<Call> {
        let t0 = now_ns();
        let mut writes: BTreeMap<u64, Vec<(u64, Vec<u8>)>> = BTreeMap::new();
        let (mut hashed, mut refreshed) = (0, 0);
        {
            let mut baselines = self.baselines.lock();
            let mut buf = vec![0u8; CHUNK as usize];
            for s in spans {
                if s.start >= s.end {
                    continue;
                }
                let objects = self.objects.lock();
                if objects.info.get(&s.target).is_none_or(|i| i.dead) {
                    continue;
                }
                drop(objects);
                let first = s.start / CHUNK;
                let last = (s.end - 1) / CHUNK;
                let chunks = s.len.div_ceil(CHUNK) as usize;
                let base = baselines.entry(s.target).or_default();
                if base.len() < chunks {
                    base.resize(chunks, Chunk::default());
                }
                for c in first..=last {
                    let off = c * CHUNK;
                    let n = CHUNK.min(s.len - off) as usize;
                    let Some(dst) = buf.get_mut(..n) else {
                        continue;
                    };
                    // SAFETY: `base..base + len` is host memory a live object of the stream lies
                    // over (a buffer's contents, a block's range), mapped while the object is
                    // live, as it was found above with the comparison points' lock held; the
                    // guest may write it meanwhile, which a copy of bytes tolerates.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            HostPtr::from_addr(s.base + off as usize).as_ptr::<u8>(),
                            dst.as_mut_ptr(),
                            n,
                        );
                    }
                    hashed += 1;
                    let h = hash64(dst);
                    let Some(slot) = base.get_mut(c as usize) else {
                        continue;
                    };
                    let write = match how {
                        Compare::Commit | Compare::Complete => slot.point != h,
                        Compare::Announce => {
                            slot.point != h || slot.recorded != h || slot.recorded == 0
                        }
                        Compare::Absorb => {
                            if slot.point != h {
                                slot.point = h;
                                refreshed += 1;
                            }
                            continue;
                        }
                    };
                    if !write {
                        continue;
                    }
                    *slot = Chunk {
                        point: h,
                        recorded: h,
                    };
                    writes
                        .entry(s.target)
                        .or_default()
                        .push((off, dst.to_vec()));
                }
            }
        }
        let t1 = now_ns();
        let s = &self.stats;
        s.hash_ns.fetch_add(t1 - t0, Ordering::Relaxed);
        s.chunks_hashed.fetch_add(hashed, Ordering::Relaxed);
        s.chunks_refreshed.fetch_add(refreshed, Ordering::Relaxed);
        let (mut written, mut bytes) = (0, 0);
        let mut calls = Vec::with_capacity(writes.len());
        for (target, chunks) in writes {
            let mut args = Vec::with_capacity(chunks.len() * 2);
            for (off, data) in chunks {
                written += 1;
                bytes += data.len() as u64;
                args.push(Value::Uint(off));
                // A chunk of `BLOB_MIN` bytes or more is written as a blob ([`blob_long`]).
                args.push(Value::Bytes(data));
            }
            calls.push(Call {
                selector: "m2v.write".to_owned(),
                receiver: target,
                args,
                ..Call::default()
            });
        }
        s.chunks_written.fetch_add(written, Ordering::Relaxed);
        s.write_bytes.fetch_add(bytes, Ordering::Relaxed);
        if how == Compare::Complete {
            (s.completion_chunks_hashed).fetch_add(hashed, Ordering::Relaxed);
            (s.completion_chunks_written).fetch_add(written, Ordering::Relaxed);
            (s.completion_write_bytes).fetch_add(bytes, Ordering::Relaxed);
        }
        s.write_ns.fetch_add(now_ns() - t1, Ordering::Relaxed);
        calls
    }

    /// Counts the 8-byte words of `data` that lie in a live pinned range: at 8-aligned offsets,
    /// and at offsets 4 mod 8.
    fn scan_addresses(&self, data: &[u8]) {
        const LOW: u64 = 1 << 46;
        const HIGH: u64 = 1 << 47;
        let objects = self.objects.lock();
        if objects.pins.is_empty() {
            return;
        }
        let mut at = 0;
        while at + 8 <= data.len() {
            let Some(word) = data.get(at..at + 8) else {
                break;
            };
            let v = u64::from_le_bytes(word.try_into().unwrap_or_default());
            if (LOW..HIGH).contains(&v)
                && objects
                    .pins
                    .range(..=v)
                    .next_back()
                    .is_some_and(|(_, &(end, _))| v < end)
            {
                if at.is_multiple_of(8) {
                    self.stats
                        .address_words_aligned
                        .fetch_add(1, Ordering::Relaxed);
                } else {
                    self.stats
                        .address_words_misaligned
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
            at += 4;
        }
    }

    /// Writes a record into the stream: the length of its call record.
    fn emit(&self, mut call: Call) -> usize {
        let mut stream = self.stream.lock();
        // Long inline data goes as a blob.
        for v in &mut call.args {
            blob_long(v, &mut stream);
        }
        stream.call(&call)
    }

    /// Counts a record by its selector.
    fn count_selector(&self, selector: &str) {
        let mut selectors = self.selectors.lock();
        match selectors.get_mut(selector) {
            Some(n) => *n += 1,
            None => {
                selectors.insert(selector.to_owned(), 1);
            }
        }
    }
}

/// The committed command buffer `weak` names, while it lives: upgraded on a calling thread
/// alone, so the writer never holds an object's last reference.
fn upgrade_cb(weak: &Weak<dyn Any + Send + Sync>) -> Option<Arc<CommandBufferObject>> {
    match weak.upgrade()?.downcast::<CommandBufferObject>() {
        Ok(cb) => Some(cb),
        Err(other) => {
            let_go_any(other);
            None
        }
    }
}

/// `MTLResourceUsageWrite`.
const USAGE_WRITE: u64 = 2;

/// Whether what `call`, on an object of type `recv_type`, reaches is memory a kernel or a render
/// stage could store to: a call of a compute or render encoder, but a `useResources:` whose
/// usage holds no write, which Metal's residency takes as the GPU's word that it only reads them.
/// The recorder knows no pipeline's arguments, so any buffer, texture or heap such a call binds
/// is taken as written, and so are the resources an argument buffer names, as `useResources:`
/// with a write and `useHeaps:` name them, and what an indirect command buffer's commands bind.
fn may_store(recv_type: &str, call: &Call) -> bool {
    let compute = std::any::type_name::<crate::gpu::predication::PredicatedEncoder>();
    let render = std::any::type_name::<crate::gpu::encode_render::RenderEncoderObject>();
    if recv_type != compute && recv_type != render {
        return false;
    }
    if !call.selector.starts_with("useResources:count:usage:") {
        return true;
    }
    match call.args.get(1) {
        Some(Value::Uint(usage)) => usage & USAGE_WRITE != 0,
        _ => true,
    }
}

/// What the blit `call`, naming `named`, writes: each object with the byte range of it written,
/// all of it where `None`. A copy between buffers and a fill write their ranges; a copy from the
/// guest's RAM each run's bytes from where it lands, and one into the guest's RAM no object; any
/// other blit its one object, or each after the first (its source), all of each, as does one of
/// those whose arguments do not read so.
fn blit_writes(call: &Call, named: &[u64]) -> Vec<(u64, Option<(u64, u64)>)> {
    let range = |at: u64, len: u64| Some((at, at.saturating_add(len)));
    let uint = |v: &Value| match v {
        Value::Uint(x) => Some(*x),
        _ => None,
    };
    let ranged = match (call.selector.as_str(), call.args.as_slice()) {
        (
            "fillBuffer:range:value:" | "fillBuffer:range:pattern4:",
            [Value::Object(b), Value::Map(r), ..],
        ) => {
            let field = |k: &str| r.iter().find(|(n, _)| n == k).and_then(|(_, v)| uint(v));
            (field("location").zip(field("length"))).map(|(at, len)| vec![(*b, range(at, len))])
        }
        (
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
            [_, _, dst, Value::Uint(at), Value::Uint(len)],
        ) => match dst {
            Value::Object(b) => Some(vec![(*b, range(*at, *len))]),
            _ => Some(Vec::new()),
        },
        (
            "m2v.copyFromGuestRam" | "m2v.copyFromGuestRamAt",
            [Value::Object(b), Value::List(runs)],
        ) => (runs.iter())
            .map(|run| match run {
                Value::List(r) => match r.as_slice() {
                    [Value::Uint(at), Value::Bytes(data)] => {
                        Some((*b, range(*at, data.len() as u64)))
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect(),
        ("m2v.copyToGuestRam", _) => Some(Vec::new()),
        _ => None,
    };
    ranged.unwrap_or_else(|| {
        let written = named.iter().skip(usize::from(named.len() > 1));
        written.map(|&id| (id, None)).collect()
    })
}

/// Whether the GPU's work writes a copy of `b` apart from the CPU's: a `Managed` buffer of a
/// device that keeps two copies of one.
fn gpu_copy_apart(b: &BufferObject) -> bool {
    b.raw.storage_mode().raw() == MANAGED && !b.device.raw.properties().has_unified_memory
}

/// Whether the command buffer completed or failed.
fn finished(cb: &CommandBufferObject) -> bool {
    raw_finished(&cb.raw)
}

/// Whether the runtime's command buffer completed or failed.
fn raw_finished(raw: &metal2vulkan_runtime::CommandBuffer) -> bool {
    let status = raw.status().raw();
    status == CommandBufferStatus::COMPLETED.raw() || status == CommandBufferStatus::ERROR.raw()
}

/// A committed command buffer a capture waits for ([`State::wait_for_earlier`]).
enum Watched {
    /// The runtime's, of one whose completion is to be taken ([`Completion::raw`]).
    Raw(metal2vulkan_runtime::CommandBuffer),
    /// The guest's, while it lives ([`upgrade_cb`]).
    Weak(Weak<dyn Any + Send + Sync>),
}

/// Drops a command buffer the recorder held for a moment: where that was its last reference, as
/// `Obj`'s drop does, its release record first.
fn let_go_cb(cb: Arc<CommandBufferObject>) {
    let ptr = Arc::as_ptr(&cb).cast::<u8>().addr() as u64;
    if let Some(inner) = Arc::into_inner(cb) {
        released(ptr);
        drop(inner);
    }
}

/// As [`let_go_cb`], for any object: where the recorder holds its last reference, the release
/// record comes first.
fn let_go_any(any: Arc<dyn Any + Send + Sync>) {
    if Arc::strong_count(&any) == 1 {
        released(Arc::as_ptr(&any).cast::<u8>().addr() as u64);
    }
    drop(any);
}

/// A descriptor's size and pinned address: `size_field` and `pinnedGpuAddress` of its map.
fn desc_size_pin(desc: &Value, size_field: &str) -> (u64, Option<u64>) {
    let Value::Map(fields) = desc else {
        return (0, None);
    };
    let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v);
    let size = match get(size_field) {
        Some(Value::Uint(s)) => *s,
        _ => 0,
    };
    let pin = match get("pinnedGpuAddress") {
        Some(Value::Uint(a)) if *a != 0 => Some(*a),
        _ => None,
    };
    (size, pin)
}

fn uint(v: Option<&Value>) -> u64 {
    match v {
        Some(Value::Uint(u)) => *u,
        _ => 0,
    }
}

/// Host memory an argument names: its bytes where the call consumes them (`replaceRegion:`),
/// else nil.
fn read_host(v: &mut Value, selector: &str) {
    match v {
        Value::Host(ptr, len) if selector.starts_with("replaceRegion:") => {
            let n = usize::try_from(*len).unwrap_or(0);
            let mut data = vec![0u8; n];
            if *ptr != 0 {
                // SAFETY: the range is the call's argument, which its caller keeps mapped for the
                // call (`HostRange`'s contract), and the call is in progress.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        HostPtr::from_addr(*ptr).as_ptr::<u8>(),
                        data.as_mut_ptr(),
                        n,
                    );
                }
            }
            *v = Value::Bytes(data);
        }
        Value::Host(..) => *v = Value::Nil,
        Value::List(items) => items.iter_mut().for_each(|i| read_host(i, selector)),
        Value::Map(pairs) => pairs.iter_mut().for_each(|(_, i)| read_host(i, selector)),
        _ => {}
    }
}

fn blob_long(v: &mut Value, stream: &mut Stream) {
    match v {
        Value::Bytes(data) if data.len() >= BLOB_MIN => {
            let hash = stream.blob(data);
            *v = Value::Blob(hash);
        }
        Value::List(items) => items.iter_mut().for_each(|i| blob_long(i, stream)),
        Value::Map(pairs) => pairs.iter_mut().for_each(|(_, i)| blob_long(i, stream)),
        _ => {}
    }
}

fn hex(h: &Hash) -> String {
    let mut s = String::with_capacity(64);
    for b in h {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// A 64-bit hash of `data`, never 0 (0 is a chunk the stream never held).
fn hash64(data: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut lanes = [K, K ^ 1, K ^ 2, K ^ 3];
    let (words, rest) = data.as_chunks::<8>();
    for (i, w) in words.iter().enumerate() {
        let v = u64::from_le_bytes(*w);
        if let Some(lane) = lanes.get_mut(i & 3) {
            *lane = (*lane ^ v).wrapping_mul(K).rotate_left(31);
        }
    }
    let mut h = data.len() as u64;
    for b in rest {
        h = (h ^ u64::from(*b)).wrapping_mul(K);
    }
    for l in lanes {
        h = (h ^ l).wrapping_mul(K).rotate_left(29);
    }
    h | 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stats file writes a `counted <code> <n>` line for each kind counted by code, with how
    /// many times it was counted.
    #[test]
    fn the_stats_write_the_kinds_counted_by_code() {
        crate::answer::count_kind_by("test-recorder-counted", 3);
        let mut out = String::new();
        write_counted(&mut out);
        assert!(
            out.lines().any(|l| l == "counted test-recorder-counted 3"),
            "{out}"
        );
    }

    /// The stats file names the recorder's build first: one `recorder_commit` line, the
    /// display library's `display_library` line (the SHA-256 of the bytes the device makes its
    /// library from, and their length), then a `recorder_fix` line for each fix, in their order,
    /// each name a single word given once.
    #[test]
    fn the_stats_name_the_recorders_build() {
        let mut out = String::new();
        write_build(&mut out);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines.first().copied(),
            Some(format!("recorder_commit {COMMIT}").as_str())
        );
        let display: Vec<&str> = lines[1]
            .strip_prefix("display_library ")
            .expect(&out)
            .split(' ')
            .collect();
        assert_eq!(
            display,
            [
                hex(&format::sha256(METALLIB)).as_str(),
                METALLIB.len().to_string().as_str()
            ],
            "{out}"
        );
        assert_eq!(display[0].len(), 64, "{out}");
        let fixes: Vec<&str> = lines[2..]
            .iter()
            .filter_map(|l| l.strip_prefix("recorder_fix "))
            .collect();
        assert_eq!(fixes, FIXES, "{out}");
        assert_eq!(lines.len(), 2 + FIXES.len(), "{out}");
        let names: HashSet<&str> = FIXES.iter().copied().collect();
        assert_eq!(names.len(), FIXES.len(), "a name given twice: {FIXES:?}");
        assert!(
            FIXES
                .iter()
                .chain([&COMMIT])
                .all(|n| !n.is_empty() && !n.contains(char::is_whitespace)),
            "{out}"
        );
    }

    /// A copy the platform did not make leaves its call nothing to record.
    #[test]
    fn an_unmade_copy_is_not_recorded() {
        let token = enter_recording();
        let mut active = Some(begin(token, || ("m2v.copyFromGuestRam", 3, Vec::new())));
        assert!(active.as_ref().is_some_and(Active::recorded));
        unmade(&mut active);
        assert!(!active.as_ref().is_some_and(Active::recorded));
    }

    /// A copy out of the guest's RAM carries the bytes it reads through the stream and back, each
    /// run with the byte it lands from: a long run as a blob, a short one inline.
    #[test]
    fn a_copy_from_guest_ram_carries_its_bytes() {
        let ram: Vec<u8> = (0..=255u8).cycle().take(8192).collect();
        let base = ram.as_ptr().addr() as u64;
        let runs = [(16, base + 4096..base + 8192), (0, base + 8..base + 24)];
        // SAFETY: the runs lie in `ram`, which lives through the call.
        let (args, n) = unsafe { guest_ram_args(Value::Object(7), runs.into_iter()) };
        assert_eq!(n, 4096 + 16);
        let mut call = Call {
            selector: "m2v.copyFromGuestRamAt".to_owned(),
            receiver: 3,
            args,
            ..Call::default()
        };
        let mut stream = Stream::new();
        stream.header();
        for v in &mut call.args {
            blob_long(v, &mut stream);
        }
        stream.call(&call);
        let read = format::read(&stream.take(), &mut Vec::new()).unwrap();
        let blobs: Vec<_> = read
            .records
            .iter()
            .filter_map(|r| match r {
                format::Record::Blob(hash, data) => Some((*hash, data.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(blobs.len(), 1);
        let (hash, data) = blobs[0].clone();
        assert_eq!(data, ram[4096..]);
        let Some(format::Record::Call(back)) = read.records.last() else {
            panic!("the call did not read back");
        };
        assert_eq!(
            back.args,
            vec![
                Value::Object(7),
                Value::List(vec![
                    Value::List(vec![Value::Uint(16), Value::Blob(hash)]),
                    Value::List(vec![Value::Uint(0), Value::Bytes(ram[8..24].to_vec())]),
                ]),
            ]
        );
    }

    /// The chunk hash tells changed bytes apart, and is never the "never held" 0
    #[test]
    fn the_chunk_hash_sees_one_changed_byte() {
        let a = vec![0u8; 4096];
        let mut b = a.clone();
        b[4095] = 1;
        assert_ne!(hash64(&a), hash64(&b));
        assert_ne!(hash64(&a), 0);
        assert_eq!(hash64(&a), hash64(&a.clone()));
    }

    /// A descriptor's size and pin read from its map; a 0 pin is none
    #[test]
    fn descriptor_size_and_pin() {
        let d = Value::Map(vec![
            ("size".to_owned(), Value::Uint(65536)),
            ("pinnedGpuAddress".to_owned(), Value::Uint(1 << 46)),
        ]);
        assert_eq!(desc_size_pin(&d, "size"), (65536, Some(1 << 46)));
        let d = Value::Map(vec![("pinnedGpuAddress".to_owned(), Value::Uint(0))]);
        assert_eq!(desc_size_pin(&d, "size"), (0, None));
    }

    /// An encoder command's arguments are its variant's fields in order; a tuple variant's its
    /// values; a unit variant has none
    #[test]
    fn command_arguments() {
        #[derive(Debug)]
        #[allow(dead_code)]
        enum Cmd {
            Draw {
                primitive_type: u64,
                vertex_start: u64,
                vertex_count: u64,
            },
            One(u64),
            End,
        }
        assert_eq!(
            cmd_args(&Cmd::Draw {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 6
            }),
            [Value::Uint(3), Value::Uint(0), Value::Uint(6)]
        );
        assert_eq!(cmd_args(&Cmd::One(7)), [Value::Uint(7)]);
        assert!(cmd_args(&Cmd::End).is_empty());
    }

    /// A call record of `args`, as the stream writes it.
    fn record(selector: &str, args: &[Value]) -> Vec<u8> {
        let mut s = Stream::new();
        s.call(&Call {
            selector: selector.to_owned(),
            receiver: 1,
            args: args.to_vec(),
            ..Call::default()
        });
        s.take()
    }

    /// The fake platform's `Debug` text of `text`'s objects as the Vulkan platform's prints while
    /// formatted for the stream: `ObjRef(id)` for an object the stream made, `Unrecorded(..)`
    /// for one it did not. `objs` holds each object's own text and its stream id.
    fn as_stream(text: &str, objs: &[(String, Option<u64>)]) -> String {
        let mut objs = objs.to_vec();
        objs.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));
        let mut out = text.to_owned();
        for (t, id) in objs {
            let by = id.map_or_else(
                || "Unrecorded(\"x\")".to_owned(),
                |id| format!("ObjRef({id})"),
            );
            out = out.replace(&t, &by);
        }
        out
    }

    /// Each command [`typed`] reads from its fields is the record its `Debug` text makes, byte
    /// for byte, with the same objects the stream never made counted: every typed command of each
    /// encoder, with objects made and not, empty and full lists, negative, extreme and
    /// non-finite numbers. A command it does not read is left to its text.
    #[test]
    fn typed_commands_are_the_records_their_text_makes() {
        use crate::core::platform::Handle as _;
        use crate::core::platform::blit::BlitCmd as B;
        use crate::core::platform::compute::ComputeCmd as Cc;
        use crate::core::platform::icb::{IcbComputeCmd as Ic, IcbRenderCmd as Ir};
        use crate::core::platform::render::RenderCmd as R;
        use crate::core::platform::types::{
            MtlOrigin, MtlRegion, MtlScissorRect, MtlSize, MtlViewport, NsRange,
        };
        use crate::core::testkit::objects::Model;
        use crate::core::testkit::{FakePlatform, Fk, kind};
        use std::fmt::Debug;
        use typed::Command as _;
        type F = FakePlatform;

        let p = FakePlatform::new();
        let w = p.world();
        let buf: Fk<kind::Buffer> = w.new_obj(Model::None);
        let lost: Fk<kind::Buffer> = w.new_obj(Model::None);
        let tex: Fk<kind::Texture> = w.new_obj(Model::None);
        let samp: Fk<kind::SamplerState> = w.new_obj(Model::None);
        let fence: Fk<kind::Fence> = w.new_obj(Model::None);
        let rps: Fk<kind::RenderPipeline> = w.new_obj(Model::None);
        let dss: Fk<kind::DepthStencilState> = w.new_obj(Model::None);
        let cps: Fk<kind::ComputePipeline> = w.new_obj(Model::None);
        let made: Vec<(u64, String, Option<u64>)> = vec![
            (buf.id().0, format!("{buf:?}"), Some(11)),
            (lost.id().0, format!("{lost:?}"), None),
            (tex.id().0, format!("{tex:?}"), Some(12)),
            (samp.id().0, format!("{samp:?}"), Some(13)),
            (fence.id().0, format!("{fence:?}"), Some(14)),
            (rps.id().0, format!("{rps:?}"), Some(15)),
            (dss.id().0, format!("{dss:?}"), Some(16)),
            (cps.id().0, format!("{cps:?}"), Some(170_000)),
        ];
        let ids = |p: u64| made.iter().find(|m| m.0 == p).and_then(|m| m.2);
        let objs: Vec<(String, Option<u64>)> = made.iter().map(|m| (m.1.clone(), m.2)).collect();
        /// A command's selector, its `Debug` text and its arguments read from its fields.
        struct Seen {
            sel: String,
            text: String,
            args: Option<Vec<Value>>,
        }
        fn of<C: typed::Command + Debug>(c: &C, cx: &mut typed::Cx<'_>) -> Seen {
            Seen {
                sel: c.sel().to_owned(),
                text: format!("{c:?}"),
                args: c.typed(cx),
            }
        }
        let mut checked = 0;
        let mut check = |cmd: &dyn Fn(&mut typed::Cx<'_>) -> Seen| {
            let mut cx = typed::Cx::new(&ids);
            let seen = cmd(&mut cx);
            let sel = seen.sel;
            let args = (seen.args).unwrap_or_else(|| panic!("{sel} is not typed"));
            let text = as_stream(&seen.text, &objs);
            let (v, notes) = debug_value::parse(&text, &ids);
            assert!(!notes.unread, "{text}");
            assert_eq!(cx.unrecorded, notes.unrecorded, "{text}");
            assert_eq!(record(&sel, &args), record(&sel, &cmd_args_of(v)), "{text}");
            checked += 1;
        };

        let bufs = [Some(&buf), None, Some(&lost)];
        let no_bufs: [Option<&Fk<kind::Buffer>>; 0] = [];
        let offs = [0, 16, u64::MAX];
        let samps = [Some(&samp), None];
        let texs = [Some(&tex), None];
        let size = |width, height, depth| MtlSize {
            width,
            height,
            depth,
        };
        let viewports = [
            MtlViewport {
                origin_x: 0.0,
                origin_y: -0.0,
                width: 1920.5,
                height: 1e-7,
                znear: f64::NAN,
                zfar: f64::INFINITY,
            },
            MtlViewport {
                origin_x: -f64::NAN,
                origin_y: f64::NEG_INFINITY,
                width: 1e300,
                height: 0.1,
                znear: -2.5,
                zfar: 1.0,
            },
        ];
        let render: Vec<R<'_, F>> = vec![
            R::DrawIndexedPrimitives {
                primitive_type: 3,
                index_count: 6,
                index_type: 1,
                index_buffer: &buf,
                index_buffer_offset: 64,
            },
            R::DrawIndexedPrimitives {
                primitive_type: 3,
                index_count: 6,
                index_type: 0,
                index_buffer: &lost,
                index_buffer_offset: 0,
            },
            R::DrawIndexedPrimitivesInstances {
                primitive_type: 4,
                index_count: 300,
                index_type: 1,
                index_buffer: &buf,
                index_buffer_offset: 8,
                instance_count: 2,
            },
            R::DrawIndexedPrimitivesBase {
                primitive_type: 3,
                index_count: 3,
                index_type: 1,
                index_buffer: &buf,
                index_buffer_offset: 0,
                instance_count: 1,
                base_vertex: -4,
                base_instance: 7,
            },
            R::DrawIndexedPrimitivesBase {
                primitive_type: 3,
                index_count: 3,
                index_type: 1,
                index_buffer: &buf,
                index_buffer_offset: 0,
                instance_count: 1,
                base_vertex: i64::MAX,
                base_instance: 0,
            },
            R::DrawPrimitives {
                primitive_type: 4,
                vertex_start: 0,
                vertex_count: 3,
            },
            R::DrawPrimitivesInstances {
                primitive_type: 4,
                vertex_start: 6,
                vertex_count: 3,
                instance_count: 9,
            },
            R::DrawPrimitivesBase {
                primitive_type: 4,
                vertex_start: 6,
                vertex_count: 3,
                instance_count: 9,
                base_instance: 2,
            },
            R::EndEncoding,
            R::MemoryBarrierWithScope {
                scope: 1,
                after: 2,
                before: 4,
            },
            R::SetCullMode { cull_mode: 2 },
            R::SetDepthStencilState {
                depth_stencil_state: Some(&dss),
            },
            R::SetDepthStencilState {
                depth_stencil_state: None,
            },
            R::SetFragmentBuffer {
                buffer: Some(&buf),
                offset: 16,
                index: 0,
            },
            R::SetFragmentBuffer {
                buffer: None,
                offset: 0,
                index: 30,
            },
            R::SetVertexBuffer {
                buffer: Some(&lost),
                offset: u64::MAX,
                index: 1,
            },
            R::SetFragmentBufferOffset {
                offset: 256,
                index: 1,
            },
            R::SetVertexBufferOffset {
                offset: 4096,
                index: 2,
            },
            R::SetFragmentBuffers {
                buffers: &bufs,
                offsets: &offs,
                range: NsRange::new(0, 3),
            },
            R::SetVertexBuffers {
                buffers: &no_bufs,
                offsets: &[],
                range: NsRange::new(4, 0),
            },
            R::SetFragmentSamplerStates {
                samplers: &samps,
                range: NsRange::new(0, 2),
            },
            R::SetVertexSamplerStates {
                samplers: &samps,
                range: NsRange::new(14, 2),
            },
            R::SetFragmentTexture {
                texture: Some(&tex),
                index: 3,
            },
            R::SetFragmentTexture {
                texture: None,
                index: 0,
            },
            R::SetFragmentTextures {
                textures: &texs,
                range: NsRange::new(1, 2),
            },
            R::SetVertexTextures {
                textures: &texs,
                range: NsRange::new(0, 2),
            },
            R::SetFrontFacingWinding {
                front_facing_winding: 1,
            },
            R::SetRenderPipelineState {
                pipeline_state: Some(&rps),
            },
            R::SetScissorRect {
                rect: MtlScissorRect {
                    x: 1,
                    y: 2,
                    width: 3,
                    height: 4,
                },
            },
            R::SetStencilFrontReferenceValue {
                front_reference_value: 0xff,
                back_reference_value: u32::MAX,
            },
            R::SetTriangleFillMode { fill_mode: 1 },
            R::SetViewport {
                viewport: viewports[0],
            },
            R::SetViewport {
                viewport: viewports[1],
            },
            R::UpdateFence {
                fence: &fence,
                stages: 1,
            },
            R::WaitForFence {
                fence: &fence,
                stages: 2,
            },
        ];
        for c in &render {
            check(&|cx| of(c, cx));
        }
        let compute: Vec<Cc<'_, F>> = vec![
            Cc::DispatchThreadgroups {
                threadgroups_per_grid: size(8, 4, 1),
                threads_per_threadgroup: size(32, 1, 1),
            },
            Cc::DispatchThreads {
                threads_per_grid: size(1920, 1080, 1),
                threads_per_threadgroup: size(16, 16, 1),
            },
            Cc::EndEncoding,
            Cc::MemoryBarrierWithScope { scope: 1 },
            Cc::SetBufferOffset {
                offset: 4,
                index: 2,
            },
            Cc::SetBuffers {
                buffers: &bufs,
                offsets: &offs,
                range: NsRange::new(0, 3),
            },
            Cc::SetComputePipelineState { state: Some(&cps) },
            Cc::SetComputePipelineState { state: None },
            Cc::SetSamplerStates {
                samplers: &samps,
                range: NsRange::new(0, 2),
            },
            Cc::SetStageInRegion {
                region: MtlRegion::new(MtlOrigin::new(1, 2, 3), size(4, 5, 6)),
            },
            Cc::SetTexture {
                texture: Some(&tex),
                index: 0,
            },
            Cc::SetTextures {
                textures: &texs,
                range: NsRange::new(0, 2),
            },
            Cc::SetThreadgroupMemoryLength {
                length: 1024,
                index: 0,
            },
            Cc::UpdateFence { fence: &fence },
            Cc::WaitForFence { fence: &fence },
        ];
        for c in &compute {
            check(&|cx| of(c, cx));
        }
        let blit: Vec<B<'_, F>> = vec![
            B::EndEncoding,
            B::UpdateFence { fence: &fence },
            B::WaitForFence { fence: &fence },
        ];
        for c in &blit {
            check(&|cx| of(c, cx));
        }
        let icb_render: Vec<Ir<'_, F>> = vec![
            Ir::ClearBarrier,
            Ir::DrawIndexedPatches {
                number_of_patch_control_points: 4,
                patch_start: 1,
                patch_count: 2,
                patch_index_buffer: Some(&buf),
                patch_index_buffer_offset: 8,
                control_point_index_buffer: None,
                control_point_index_buffer_offset: 16,
                instance_count: 3,
                base_instance: 4,
                buffer: Some(&lost),
                offset: 48,
                instance_stride: 12,
            },
            Ir::DrawIndexedPrimitives {
                primitive_type: 3,
                index_count: 6,
                index_type: 1,
                index_buffer: Some(&buf),
                index_buffer_offset: 64,
                instance_count: 1,
                base_vertex: -4,
                base_instance: 0,
            },
            Ir::DrawIndexedPrimitives {
                primitive_type: 3,
                index_count: 3,
                index_type: 0,
                index_buffer: None,
                index_buffer_offset: u64::MAX,
                instance_count: 2,
                base_vertex: i64::MAX,
                base_instance: 7,
            },
            Ir::DrawPatches {
                number_of_patch_control_points: 3,
                patch_start: 0,
                patch_count: 9,
                patch_index_buffer: None,
                patch_index_buffer_offset: 0,
                instance_count: 1,
                base_instance: 0,
                buffer: Some(&buf),
                offset: 4,
                instance_stride: 0,
            },
            Ir::DrawPrimitives {
                primitive_type: 4,
                vertex_start: 1,
                vertex_count: 6,
                instance_count: 2,
                base_instance: 7,
            },
            Ir::Reset,
            Ir::SetBarrier,
            Ir::SetFragmentBuffer {
                buffer: Some(&buf),
                offset: 16,
                index: 0,
            },
            Ir::SetRenderPipelineState {
                pipeline_state: Some(&rps),
            },
            Ir::SetRenderPipelineState {
                pipeline_state: None,
            },
            Ir::SetVertexBuffer {
                buffer: None,
                offset: 0,
                index: 30,
            },
            Ir::SetVertexBufferAttributeStride {
                buffer: Some(&lost),
                offset: 8,
                stride: 24,
                index: 1,
            },
        ];
        for c in &icb_render {
            check(&|cx| of(c, cx));
        }
        let icb_compute: Vec<Ic<'_, F>> = vec![
            Ic::ClearBarrier,
            Ic::ConcurrentDispatchThreadgroups {
                threadgroups_per_grid: size(8, 4, 1),
                threads_per_threadgroup: size(32, 1, 1),
            },
            Ic::ConcurrentDispatchThreads {
                threads_per_grid: size(u64::MAX, 1, 0),
                threads_per_threadgroup: size(16, 16, 1),
            },
            Ic::Reset,
            Ic::SetBarrier,
            Ic::SetComputePipelineState {
                pipeline_state: Some(&cps),
            },
            Ic::SetComputePipelineState {
                pipeline_state: None,
            },
            Ic::SetImageblockWidth {
                width: 16,
                height: 8,
            },
            Ic::SetKernelBuffer {
                buffer: Some(&buf),
                offset: 32,
                index: 2,
            },
            Ic::SetKernelBufferAttributeStride {
                buffer: Some(&lost),
                offset: 16,
                stride: 12,
                index: 0,
            },
            Ic::SetStageInRegion {
                region: MtlRegion::new(MtlOrigin::new(1, 2, 3), size(4, 5, 6)),
            },
            Ic::SetThreadgroupMemoryLength {
                length: 256,
                index: 30,
            },
        ];
        for c in &icb_compute {
            check(&|cx| of(c, cx));
        }
        assert_eq!(
            checked,
            render.len() + compute.len() + blit.len() + icb_render.len() + icb_compute.len()
        );
        let mut cx = typed::Cx::new(&ids);
        let untyped: R<'_, F> = R::SetLineWidth { line_width: 1.5 };
        assert!(
            untyped.typed(&mut cx).is_none(),
            "an f32 is read from its text"
        );
    }

    /// The pass descriptor's sets read from their fields are the values their `Debug` text,
    /// which names objects by pointer, reads as, but for an object set to nil (object 0): objects
    /// made and not, nil sets, set values and non-finite colours
    #[test]
    fn typed_sets_are_the_values_their_text_reads_as() {
        use crate::core::platform::Handle as _;
        use crate::core::platform::types::ClearColor;
        use crate::core::platform::values::{
            ColorAttachmentSet, DepthAttachmentSet, PassSet, StencilAttachmentSet,
        };
        use crate::core::testkit::objects::Model;
        use crate::core::testkit::{FakePlatform, Fk, kind};
        type F = FakePlatform;

        let p = FakePlatform::new();
        let w = p.world();
        let tex: Fk<kind::Texture> = w.new_obj(Model::None);
        let lost: Fk<kind::Texture> = w.new_obj(Model::None);
        let buf: Fk<kind::Buffer> = w.new_obj(Model::None);
        let made = [(tex.id().0, 21), (buf.id().0, 22)];
        let ids = |p: u64| made.iter().find(|m| m.0 == p).map(|m| m.1);
        // The text reads nil for an object set to nil (`Some(None)`) and for one left unset; the
        // set's value is object 0 for the first.
        let check = |read: &dyn Fn(&mut typed::Cx<'_>) -> Value, text: String| {
            let mut cx = typed::Cx::new(&ids);
            let v = read(&mut cx);
            let (mut want, notes) = debug_value::parse(&text, &ids);
            assert!(!notes.unread, "{text}");
            assert_eq!(cx.unrecorded, notes.unrecorded, "{text}");
            let objects = [
                ("visibilityResultBuffer", "visibility_result_buffer"),
                ("rasterizationRateMap", "rasterization_rate_map"),
                ("texture", "texture"),
                ("resolveTexture", "resolve_texture"),
            ];
            for (key, field) in objects {
                let nil = format!(" {field}: Some(None)");
                if (text.contains(&format!("{{{nil}")) || text.contains(&format!(",{nil}")))
                    && let Value::Map(fields) = &mut want
                    && let Some((_, f)) = fields.iter_mut().find(|(k, _)| k == key)
                {
                    *f = Value::Object(0);
                }
            }
            assert_eq!(record("s", &[v]), record("s", &[want]), "{text}");
        };
        let passes: [PassSet<'_, F>; 3] = [
            PassSet::default(),
            PassSet {
                visibility_result_buffer: Some(Some(&buf)),
                visibility_result_type: Some(1),
                render_target_width: Some(1920),
                render_target_height: Some(1080),
                tile_width: Some(32),
                ..PassSet::default()
            },
            PassSet {
                visibility_result_buffer: Some(None),
                rasterization_rate_map: Some(None),
                render_target_array_length: Some(0),
                default_raster_sample_count: Some(4),
                default_color_sample_count: Some(4),
                imageblock_sample_length: Some(16),
                threadgroup_memory_length: Some(1024),
                tile_height: Some(16),
                ..PassSet::default()
            },
        ];
        for s in &passes {
            check(&|cx| typed::pass_set(s, cx), format!("{s:?}"));
        }
        let colors: [ColorAttachmentSet<'_, F>; 3] = [
            ColorAttachmentSet::default(),
            ColorAttachmentSet {
                texture: Some(Some(&tex)),
                resolve_texture: Some(Some(&lost)),
                level: Some(1),
                slice: Some(2),
                depth_plane: Some(3),
                resolve_level: Some(4),
                resolve_slice: Some(5),
                resolve_depth_plane: Some(6),
                load_action: Some(2),
                store_action: Some(1),
                store_action_options: Some(0),
                clear_color: Some(ClearColor {
                    red: 0.5,
                    green: f64::NAN,
                    blue: -0.0,
                    alpha: f64::INFINITY,
                }),
            },
            ColorAttachmentSet {
                texture: Some(None),
                clear_color: Some(ClearColor::default()),
                ..ColorAttachmentSet::default()
            },
        ];
        for s in &colors {
            check(&|cx| typed::color_attachment(s, cx), format!("{s:?}"));
        }
        let depths: [DepthAttachmentSet<'_, F>; 2] = [
            DepthAttachmentSet::default(),
            DepthAttachmentSet {
                texture: Some(Some(&tex)),
                load_action: Some(2),
                clear_depth: Some(1.0),
                depth_resolve_filter: Some(1),
                resolve_texture: Some(None),
                ..DepthAttachmentSet::default()
            },
        ];
        for s in &depths {
            check(&|cx| typed::depth_attachment(s, cx), format!("{s:?}"));
        }
        let stencils: [StencilAttachmentSet<'_, F>; 2] = [
            StencilAttachmentSet::default(),
            StencilAttachmentSet {
                texture: Some(Some(&lost)),
                store_action: Some(0),
                clear_stencil: Some(u32::MAX),
                stencil_resolve_filter: Some(0),
                ..StencilAttachmentSet::default()
            },
        ];
        for s in &stencils {
            check(&|cx| typed::stencil_attachment(s, cx), format!("{s:?}"));
        }
    }

    /// Integers, booleans and ranges are the values their text reads as; any other argument is
    /// its text, read by the writer as it would have been read at once, but for text naming
    /// objects by pointer, read at once
    #[test]
    fn scalars_are_their_text_and_the_rest_waits_for_the_writer() {
        use crate::core::platform::ObjId;
        use crate::core::platform::types::NsRange;
        let read = |text: String| debug_value::parse(&text, &|_| None).0;
        let (a, b, c, d, e): (u64, u32, i64, i64, bool) = (u64::MAX, 7, -3, 9, true);
        let r = NsRange::new(4, 12);
        assert_eq!(rec_val!(a), read(format!("{a:?}")));
        assert_eq!(rec_val!(b), read(format!("{b:?}")));
        assert_eq!(rec_val!(c), read(format!("{c:?}")));
        assert_eq!(rec_val!(d), read(format!("{d:?}")));
        assert_eq!(rec_val!(e), read(format!("{e:?}")));
        assert_eq!(rec_val!(r), read(format!("{r:?}")));
        #[derive(Debug)]
        #[allow(dead_code)] // read through its `Debug` alone
        struct Desc {
            pixel_format: u64,
            label: &'static str,
        }
        let desc = Desc {
            pixel_format: 80,
            label: "a",
        };
        let Value::Debug(text) = rec_val!(desc) else {
            panic!("a struct waits for the writer");
        };
        assert_eq!(parse_text(&text), read(format!("{desc:?}")));
        let by_ptr = Some(ObjId(0x1000));
        let Value::Debug(text) = rec_val!(by_ptr) else {
            panic!("text naming an object by pointer waits for the writer");
        };
        assert_eq!(text, "Some(Unrecorded(0))", "no object at 0x1000");
        let (v, notes) = debug_value::parse(&text, &|_| None);
        assert_eq!((v, notes.unrecorded), (Value::Nil, 1));
    }

    /// Objects named by pointer are named by their stream ids, or as unrecorded, so the text
    /// reads later as it reads now; a quoted string and a longer name are left as they are.
    #[test]
    fn objects_by_pointer_are_named_by_their_ids() {
        let ids = |p: u64| (p == 4096).then_some(21);
        let text = "Desc { f: ObjId(4096), g: Some(ObjId(8192)), h: [ObjId(4096)], \
                    label: \"ObjId(4096) \\\" ObjId(4096)\", i: MyObjId(4096) }";
        let named = resolve_ptrs(text, &ids);
        assert_eq!(
            named,
            "Desc { f: ObjRef(21), g: Some(Unrecorded(0)), h: [ObjRef(21)], \
             label: \"ObjId(4096) \\\" ObjId(4096)\", i: MyObjId(4096) }"
        );
        let now = debug_value::parse(text, &ids);
        let later = debug_value::parse(&named, &|_| None);
        assert_eq!(later.0, now.0);
        assert_eq!(
            (later.1.unrecorded, later.1.unread),
            (now.1.unrecorded, now.1.unread)
        );
    }

    /// The writer shapes what calling threads queued in their order: a command's text and an
    /// argument's text read into values, a write's long chunk a blob before it, counted by its
    /// selector, and a release after the calls before it
    #[test]
    fn the_writer_shapes_queued_records_in_order() {
        let state = State::new(PathBuf::from("/nonexistent/recorder"));
        state.push(Item::Call {
            call: Call {
                selector: "setCullMode:".to_owned(),
                receiver: 5,
                ..Call::default()
            },
            text: Some("SetCullMode { cull_mode: 2 }".to_owned()),
            shape: true,
            snap: None,
        });
        state.push(Item::Call {
            call: Call {
                selector: "m2v.textureDescriptor".to_owned(),
                result: 6,
                args: vec![Value::Debug(
                    "TextureDesc { pixel_format: 80, width: Some(64) }".to_owned(),
                )],
                ..Call::default()
            },
            text: None,
            shape: true,
            snap: None,
        });
        state.push(Item::Write(Call {
            selector: "m2v.write".to_owned(),
            receiver: 6,
            args: vec![Value::Uint(4096), Value::Bytes(vec![7; 4096])],
            ..Call::default()
        }));
        state.push(Item::Release(6));
        assert_eq!(state.queue.lock().len(), 4);
        assert_eq!(state.drain(true), 4);
        assert_eq!(state.stats.records.load(Ordering::Relaxed), 4);
        assert!(state.queue.lock().is_empty());
        assert_eq!(state.stats.drained_here.load(Ordering::Relaxed), 4);
        let mut bytes = format::HEADER.to_vec();
        bytes.extend(state.stream.lock().take());
        let read = format::read(&bytes, &mut Vec::new()).unwrap();
        let calls: Vec<Call> = read
            .records
            .iter()
            .filter_map(|r| match r {
                format::Record::Call(c) => Some(c.clone()),
                _ => None,
            })
            .collect();
        let hash = format::sha256(&[7; 4096]);
        assert!(
            read.records
                .contains(&format::Record::Blob(hash, vec![7; 4096]))
        );
        assert_eq!(
            calls,
            [
                Call {
                    selector: "setCullMode:".to_owned(),
                    receiver: 5,
                    args: vec![Value::Uint(2)],
                    ..Call::default()
                },
                Call {
                    selector: "m2v.textureDescriptor".to_owned(),
                    result: 6,
                    args: vec![Value::Map(vec![
                        ("pixelFormat".to_owned(), Value::Uint(80)),
                        ("width".to_owned(), Value::Uint(64)),
                    ])],
                    ..Call::default()
                },
                Call {
                    selector: "m2v.write".to_owned(),
                    receiver: 6,
                    args: vec![Value::Uint(4096), Value::Blob(hash)],
                    ..Call::default()
                },
                Call {
                    selector: "release".to_owned(),
                    receiver: 6,
                    ..Call::default()
                },
            ]
        );
        let selectors = state.selectors.lock();
        assert_eq!(selectors.get("m2v.write"), Some(&1));
        assert_eq!(selectors.get("setCullMode:"), Some(&1));
    }

    /// Host memory a call consumes is read as bytes; any other is nil
    #[test]
    fn host_memory_is_read_only_where_consumed() {
        let data = [1u8, 2, 3, 4];
        let mut v = Value::List(vec![Value::Host(data.as_ptr().addr(), 4)]);
        read_host(&mut v, "replaceRegion:mipmapLevel:withBytes:bytesPerRow:");
        assert_eq!(v, Value::List(vec![Value::Bytes(vec![1, 2, 3, 4])]));
        let mut v = Value::Host(data.as_ptr().addr(), 4);
        read_host(
            &mut v,
            "newBufferWithBytesNoCopy:length:options:deallocator:",
        );
        assert_eq!(v, Value::Nil);
    }

    /// The call `active` opened, as it is queued.
    fn queued(active: Option<Active>) -> Call {
        active
            .and_then(|a| a.pending)
            .expect("the call is recorded")
    }

    /// `calls`, queued as calls are, shaped and written by the writer and read back by the crate's
    /// reader: the call records and the blobs.
    fn written(calls: Vec<Call>) -> (Vec<Call>, Vec<(Hash, Vec<u8>)>) {
        let state = State::new(PathBuf::from("/nonexistent/recorder"));
        for call in calls {
            state.push(Item::Call {
                call,
                text: None,
                shape: true,
                snap: None,
            });
        }
        state.drain(true);
        let mut bytes = format::HEADER.to_vec();
        bytes.extend(state.stream.lock().take());
        let read = format::read(&bytes, &mut Vec::new()).unwrap();
        let (mut calls, mut blobs) = (Vec::new(), Vec::new());
        for r in read.records {
            match r {
                format::Record::Call(c) => calls.push(c),
                format::Record::Blob(hash, data) => blobs.push((hash, data)),
                _ => {}
            }
        }
        (calls, blobs)
    }

    /// `m2v.commitPart` answers whether it cut its command buffer, `cut`, as an enum's value:
    /// `"Cut"`, `"Fails"` or `{Whole: reason}`, through the stream and back.
    #[test]
    fn a_commit_part_answers_its_cut() {
        use crate::core::platform::command::{KeptWhole, PartCut};
        let cuts = [
            PartCut::Cut,
            PartCut::Fails,
            PartCut::Whole(KeptWhole::Committed),
        ];
        let calls = cuts
            .iter()
            .map(|cut| {
                let mut active = Some(begin(enter_recording(), || {
                    ("m2v.commitPart", 3, Vec::new())
                }));
                answer(&mut active, "cut", || text_value(cut));
                queued(active)
            })
            .collect();
        let (calls, _) = written(calls);
        let answers: Vec<_> = calls.iter().map(|c| c.answers.clone()).collect();
        let cut = |v| vec![("cut".to_owned(), v)];
        assert_eq!(
            answers,
            [
                cut(Value::Str("Cut".to_owned())),
                cut(Value::Str("Fails".to_owned())),
                cut(Value::Map(vec![(
                    "Whole".to_owned(),
                    Value::Str("Committed".to_owned())
                )])),
            ]
        );
    }

    /// An attachment's texture set to nil is object 0, and one left unset nil, through the stream
    /// and back, so a player tells them apart.
    #[test]
    fn an_object_set_to_nil_is_object_0() {
        use crate::core::platform::values::ColorAttachmentSet;
        use crate::core::testkit::FakePlatform;
        let set: ColorAttachmentSet<'_, FakePlatform> = ColorAttachmentSet {
            texture: Some(None),
            level: Some(1),
            ..ColorAttachmentSet::default()
        };
        let none = |_: u64| None::<u64>;
        let mut cx = typed::Cx::new(&none);
        let call = Call {
            selector: "m2v.setColorAttachment".to_owned(),
            receiver: 4,
            args: vec![Value::Uint(0), typed::color_attachment(&set, &mut cx)],
            ..Call::default()
        };
        let (calls, _) = written(vec![call]);
        let Some(Value::Map(fields)) = calls[0].args.get(1) else {
            panic!("the set did not read back: {:?}", calls[0]);
        };
        let get = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        assert_eq!(get("texture"), Some(&Value::Object(0)));
        assert_eq!(get("resolveTexture"), Some(&Value::Nil));
        assert_eq!(get("level"), Some(&Value::Uint(1)));
    }

    /// A buffer made as a copy of host memory carries the bytes it starts with, its descriptor's
    /// `contents`, through the stream and back: a blob where they are long, inline where short.
    #[test]
    fn a_copied_buffer_carries_its_contents() {
        use crate::core::platform::HostRange;
        use crate::core::platform::values::BufferDesc;
        let long: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let short = [9u8; 8];
        let calls = [&long[..], &short[..]].map(|data| {
            let desc = BufferDesc {
                length: u64::try_from(data.len()).unwrap(),
                options: Vec::new(),
                pinned_gpu_address: None,
                contents: Some(HostRange::of_slice(data)),
                no_copy: false,
                deallocator: None,
                zero_from: None,
            };
            let mut active = Some(begin(enter_recording(), || {
                (
                    "newBufferWithDescriptor:",
                    0,
                    vec![crate::recorder::rec_val!(desc)],
                )
            }));
            contents(&mut active, HostRange::of_slice(data));
            queued(active)
        });
        let (calls, blobs) = written(calls.into());
        let hash = format::sha256(&long);
        assert_eq!(blobs, [(hash, long.clone())]);
        let contents_of = |c: &Call| match c.args.first() {
            Some(Value::Map(fields)) => fields
                .iter()
                .find(|(k, _)| k == "contents")
                .map(|(_, v)| v.clone()),
            _ => None,
        };
        assert_eq!(contents_of(&calls[0]), Some(Value::Blob(hash)));
        assert_eq!(contents_of(&calls[1]), Some(Value::Bytes(short.to_vec())));
    }

    /// A `gpuResourceID` getter's call is on its object, with the ID it answered under
    /// `gpuResourceID`, through the stream and back.
    #[test]
    fn a_resource_id_is_its_getters_answer() {
        let mut active = Some(begin(enter_recording(), || {
            ("gpuResourceID", 5, Vec::new())
        }));
        answer(&mut active, "gpuResourceID", || Value::Uint(7));
        let (calls, _) = written(vec![queued(active)]);
        assert_eq!(
            calls,
            [Call {
                selector: "gpuResourceID".to_owned(),
                receiver: 5,
                answers: vec![("gpuResourceID".to_owned(), Value::Uint(7))],
                ..Call::default()
            }]
        );
    }

    /// The commands one call sends to an indirect command buffer are its last argument: those it
    /// took, in the order it took them, each a map of its selector to its arguments, its objects
    /// their stream ids, through the stream and back; a command it never took is not among them.
    /// A command sent on its own is a list of one.
    #[test]
    fn an_icb_call_lists_the_commands_it_took() {
        use crate::core::platform::Handle as _;
        use crate::core::platform::icb::IcbRenderCmd as Ir;
        use crate::core::testkit::objects::Model;
        use crate::core::testkit::{FakePlatform, Fk, kind};
        let p = FakePlatform::new();
        let w = p.world();
        let buf: Fk<kind::Buffer> = w.new_obj(Model::None);
        let lost: Fk<kind::Buffer> = w.new_obj(Model::None);
        let ids = |ptr: u64| (ptr == buf.id().0).then_some(9);
        let cmds: Vec<Ir<'_, FakePlatform>> = vec![
            Ir::SetVertexBuffer {
                buffer: Some(&buf),
                offset: 16,
                index: 1,
            },
            Ir::SetFragmentBuffer {
                buffer: Some(&lost),
                offset: 0,
                index: 0,
            },
            Ir::SetBarrier,
            Ir::Reset,
        ];
        let mut active = begin(enter_recording(), || {
            ("indirectRenderCommandAtIndex:", 4, vec![Value::Uint(3)])
        });
        // A send that stops after three commands, as one whose third raised.
        let took = take_commands(&mut active, &ids, cmds.into_iter(), |c| c.take(3).count());
        assert_eq!(took, 3);
        let (calls, _) = written(vec![queued(Some(active))]);
        let cmd =
            |sel: &str, args: Vec<Value>| Value::Map(vec![(sel.to_owned(), Value::List(args))]);
        assert_eq!(
            calls,
            [Call {
                selector: "indirectRenderCommandAtIndex:".to_owned(),
                receiver: 4,
                args: vec![
                    Value::Uint(3),
                    Value::List(vec![
                        cmd(
                            "setVertexBuffer:offset:atIndex:",
                            vec![Value::Object(9), Value::Uint(16), Value::Uint(1)]
                        ),
                        cmd(
                            "setFragmentBuffer:offset:atIndex:",
                            vec![Value::Nil, Value::Uint(0), Value::Uint(0)]
                        ),
                        cmd("setBarrier", vec![]),
                    ]),
                ],
                ..Call::default()
            }]
        );
        let alone: Ir<'_, FakePlatform> = Ir::Reset;
        assert_eq!(
            crate::recorder::rec_val!(IcbCommands(std::slice::from_ref(&alone))),
            Value::List(vec![cmd("reset", vec![])])
        );
    }

    /// The call records of `segment`, read with the names the segments before it defined
    /// (`names`, which grows).
    fn calls_of(segment: &[u8], names: &mut Vec<String>) -> Vec<Call> {
        let read = format::read(segment, names).unwrap();
        assert!(!read.truncated);
        (read.records.into_iter())
            .filter_map(|r| match r {
                format::Record::Call(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    /// A device's identity record is written right after the record that made the device; each
    /// later segment starts with the identity record of each device then live, after its header,
    /// and a released device's is not among them. Read across the segments, the names those
    /// records use are the ones the first defined.
    #[test]
    fn each_segment_starts_with_the_live_devices_identities() {
        let dir =
            std::env::temp_dir().join(format!("rvg-recorder-segments-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let state = State::new(dir.join("streams"));
        let mut segments = Segments::new(dir.join("stream"), 256);
        let made = |id| Call {
            selector: "MTLCreateSystemDefaultDevice".to_owned(),
            result: id,
            ..Call::default()
        };
        let discrete = Identity {
            name: "A discrete GPU".to_owned(),
            unified: false,
        };
        let integrated = Identity {
            name: "An integrated GPU".to_owned(),
            unified: true,
        };
        state.push_made(Item::Plain(made(1)), 1, Some(discrete.clone()));
        state.push_made(Item::Plain(made(2)), 2, Some(integrated.clone()));
        state.push(Item::Release(2));
        // Long enough to fill the first segment.
        let commit = |args| Call {
            selector: "commit".to_owned(),
            receiver: 1,
            args,
            ..Call::default()
        };
        state.push_plain(commit(vec![Value::Bytes(vec![1; 300])]));
        state.drain(true);
        state.write_out(&mut segments);
        state.push_plain(commit(Vec::new()));
        state.drain(true);
        state.write_out(&mut segments);
        let segment = |n: u32| std::fs::read(dir.join(format!("stream.{n:04}.m2vr"))).unwrap();
        let (first, second) = (segment(0), segment(1));
        let _ = std::fs::remove_dir_all(&dir);

        let mut names = Vec::new();
        let release = Call {
            selector: "release".to_owned(),
            receiver: 2,
            ..Call::default()
        };
        let first = calls_of(&first, &mut names);
        assert_eq!(
            first[..5],
            [
                made(1),
                discrete.record(1),
                made(2),
                integrated.record(2),
                release
            ]
        );
        assert_eq!(first.len(), 6, "{first:?}");
        assert_eq!(second.get(..16), Some(&format::HEADER[..]));
        let second = calls_of(&second, &mut names);
        assert_eq!(second, [discrete.record(1), commit(Vec::new())]);
        assert_eq!(
            discrete.record(1).answers,
            [
                ("name".to_owned(), Value::Str("A discrete GPU".to_owned())),
                ("hasUnifiedMemory".to_owned(), Value::Bool(false)),
            ]
        );
        assert_eq!(
            state.stream.lock().calls(),
            8,
            "the identity counts as a call"
        );
    }

    /// The marker that makes a run of a test that records its own process the child, which
    /// records.
    const RECORD_CHILD: &str = "REIMS_VGPU_RECORDER_CHILD";

    /// Runs the test `name` (its path in the crate) in a child process of its own that records
    /// into a scratch directory, since the recorder is the process's, and fails where the child
    /// fails or runs no test. In the child, runs `child` with the directory its streams go to.
    fn in_recording_child(name: &str, child: impl FnOnce(&std::path::Path)) {
        if std::env::var_os(RECORD_CHILD).is_some() {
            let streams = std::env::var_os(ENV_VAR).expect("the child records");
            child(std::path::Path::new(&streams));
            return;
        }
        let leaf = name.rsplit("::").next().unwrap_or(name);
        let dir = std::env::temp_dir().join(format!("rvg-recorder-{leaf}-{}", std::process::id()));
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture", "--test-threads=1"])
            .env(RECORD_CHILD, "1")
            .env(ENV_VAR, dir.join("streams"))
            .output()
            .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(out.status.success(), "{out:?}");
        let ran = String::from_utf8_lossy(&out.stdout).contains("1 passed");
        assert!(ran, "the child ran no test: {out:?}");
    }

    /// The call records and blobs of the first segment this process wrote into `streams`, once
    /// its calls hold what `done` waits for, or after 10 s.
    fn recorded(
        streams: &std::path::Path,
        done: impl Fn(&[Call]) -> bool,
    ) -> (Vec<Call>, Vec<(Hash, Vec<u8>)>) {
        let segment = streams.join(format!("{}.0000.m2vr", std::process::id()));
        let (mut calls, mut blobs) = (Vec::new(), Vec::new());
        for _ in 0..200 {
            let read = std::fs::read(&segment)
                .ok()
                .and_then(|b| format::read(&b, &mut Vec::new()).ok());
            (calls, blobs) = (Vec::new(), Vec::new());
            for r in read.into_iter().flat_map(|r| r.records) {
                match r {
                    format::Record::Call(c) => calls.push(c),
                    format::Record::Blob(hash, data) => blobs.push((hash, data)),
                    format::Record::Name(_) => {}
                }
            }
            if done(&calls) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        (calls, blobs)
    }

    /// The process's platform, recording, and its default device, opened through it as the
    /// device's host opens it.
    fn recording_device() -> (
        crate::platform::VulkanPlatform,
        crate::gpu::device::VulkanDevice,
    ) {
        use crate::core::platform::Platform as _;
        let platform = crate::platform::VulkanPlatform::new();
        assert!(on(), "the child does not record");
        let registry_id = crate::os::default_registry_id()
            .or_else(crate::platform::VulkanPlatform::default_device_registry_id)
            .expect("a Vulkan device");
        let device = platform
            .open_device(registry_id)
            .expect("the default device");
        (platform, device)
    }

    /// A stream starts with the device's making and then its identity: the name the guest's
    /// device info gives (the device's `caps`) and whether its GPU shares the CPU's memory (its
    /// identity's `hasUnifiedMemory`, which the guest is given unchanged), before any call on
    /// the device.
    #[test]
    fn on_vulkan_a_stream_starts_with_the_devices_identity() {
        in_recording_child(
            "recorder::tests::on_vulkan_a_stream_starts_with_the_devices_identity",
            |streams| {
                use crate::core::platform::device::GpuDevice as _;
                let (_platform, device) = recording_device();
                let _queue = device.new_command_queue().expect("a queue");
                let (calls, _) = recorded(streams, |calls| {
                    calls.iter().any(|c| c.selector == "newCommandQueue")
                });
                let made = calls.first().expect("a record");
                assert_eq!(made.selector, "MTLCreateSystemDefaultDevice", "{calls:?}");
                assert_ne!(made.result, 0);
                let name = device.caps().name.clone().expect("a name");
                let unified = device.identity_caps().flags.has_unified_memory;
                let identity = Identity {
                    name,
                    unified: unified.expect("an answer"),
                };
                assert_eq!(
                    calls.get(1),
                    Some(&identity.record(made.result)),
                    "{calls:?}"
                );
                let queue = calls.get(2).expect("the queue's record");
                assert_eq!(
                    (queue.selector.as_str(), queue.receiver),
                    ("newCommandQueue", made.result)
                );
            },
        );
    }

    /// The count `key` of the stats file of this process's stream in `streams`, written once
    /// every queued call is: the writer's own write of it can meet the read, which then reads
    /// again.
    fn stat(streams: &std::path::Path, key: &str) -> u64 {
        let state = state().expect("the child records");
        let path = streams.join(format!("{}.stats.txt", std::process::id()));
        for _ in 0..10 {
            state.drain(true);
            state.write_stats();
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            let value = text.lines().find_map(|line| {
                let (k, n) = line.split_once(' ')?;
                if k != key {
                    return None;
                }
                n.parse().ok()
            });
            if let Some(value) = value {
                return value;
            }
        }
        panic!("no count {key} in {}", path.display());
    }

    /// A texture of the platform's own (the window's cursor glyph) is a `newTextureWithDescriptor:`
    /// on the device, as any other, of a descriptor whose `m2v.textureDescriptor` map is its
    /// `TextureDesc`'s with [`OUTSIDE`] `true` after it, where a guest texture's of the same
    /// value has no such key; the stream names the texture as the receiver of its
    /// `replaceRegion:` and the argument of the `setFragmentTexture:atIndex:` that binds it, and
    /// counts no object it could not name.
    #[test]
    fn on_vulkan_an_own_texture_is_named_where_it_is_used() {
        in_recording_child(
            "recorder::tests::on_vulkan_an_own_texture_is_named_where_it_is_used",
            |streams| {
                use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
                use crate::core::platform::device::GpuDevice as _;
                use crate::core::platform::render::{RenderCmd, RenderEncoder as _};
                use crate::core::platform::resource::Texture as _;
                use crate::core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
                use crate::core::platform::values::TextureDesc;
                use crate::core::platform::{HostRange, Platform as _};
                use crate::core::raise::OnFault;
                const REPLACE: &str =
                    "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:";
                const BIND: &str = "setFragmentTexture:atIndex:";
                const MADE: &str = "newTextureWithDescriptor:";
                const DESCRIPTOR: &str = "m2v.textureDescriptor";
                let (platform, device) = recording_device();
                let (_target, pass) =
                    crate::gpu::encode_render::tests::pass_of(&platform, &device, (4, 4));
                let value = TextureDesc {
                    width: 4,
                    height: 4,
                    ..TextureDesc::default()
                };
                let own = device.new_own_texture(&value).expect("an own texture");
                let pixels = [0x7fu8; 64];
                let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(4, 4, 1));
                (own.replace_region(region, 0, 0, HostRange::of_slice(&pixels), 16, 64)).unwrap();
                let guest_descriptor = platform.texture_descriptor(&value);
                let guest = (device.new_texture(OnFault::Die, &guest_descriptor))
                    .unwrap()
                    .expect("a guest texture");
                let queue = device.new_command_queue().expect("a queue");
                let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                let bind = RenderCmd::SetFragmentTexture {
                    texture: Some(&own),
                    index: 0,
                };
                for cmd in [bind, RenderCmd::EndEncoding] {
                    e.encode(OnFault::Die, cmd).unwrap();
                }
                cb.commit(OnFault::Die).unwrap();
                cb.wait_until_completed(OnFault::Die).unwrap();
                let (calls, _) = recorded(streams, |calls| {
                    calls.iter().any(|c| c.selector == "commit")
                });
                let one = |selector: &str| {
                    let mut of = calls.iter().filter(|c| c.selector == selector);
                    let call = of
                        .next()
                        .unwrap_or_else(|| panic!("no {selector}: {calls:?}"));
                    assert!(of.next().is_none(), "two of {selector}: {calls:?}");
                    call
                };
                // The map of the descriptor the create `made` names.
                let descriptor_of = |made: &Call| {
                    let Some(&Value::Object(id)) = made.args.first() else {
                        panic!("no descriptor named: {made:?}");
                    };
                    let record = (calls.iter())
                        .find(|c| c.selector == DESCRIPTOR && c.result == id)
                        .unwrap_or_else(|| panic!("nothing made descriptor {id}: {calls:?}"));
                    match record.args.first() {
                        Some(Value::Map(pairs)) => pairs.clone(),
                        other => panic!("a descriptor's map: {other:?}"),
                    }
                };
                let replace = one(REPLACE);
                assert_ne!(replace.receiver, 0, "the replace's receiver is unnamed");
                let made = (calls.iter())
                    .find(|c| c.selector == MADE && c.result == replace.receiver)
                    .unwrap_or_else(|| panic!("nothing made the replace's receiver: {calls:?}"));
                let device_id = calls.first().expect("the device's record").result;
                assert_eq!(made.receiver, device_id, "{made:?}");
                let guest_made = (calls.iter())
                    .rfind(|c| c.selector == MADE)
                    .expect("the guest texture's record");
                assert_ne!(guest_made.result, made.result, "{calls:?}");
                let (mut own_map, guest_map) = (descriptor_of(made), descriptor_of(guest_made));
                assert_eq!(
                    own_map.pop(),
                    Some((OUTSIDE.to_owned(), Value::Bool(true))),
                    "{made:?}"
                );
                assert_eq!(own_map, guest_map);
                assert!(guest_map.iter().all(|(key, _)| key != OUTSIDE));
                assert_eq!(
                    one(BIND).args.first(),
                    Some(&Value::Object(replace.receiver)),
                    "{calls:?}"
                );
                assert_eq!(stat(streams, "unrecorded_refs"), 0);
                assert_eq!(stat(streams, "unrecorded_receivers"), 0);
                drop((own, guest));
            },
        );
    }

    /// The descriptor `contents` of a `newBufferWithDescriptor:` record.
    fn contents_of(c: &Call) -> Option<Value> {
        match c.args.first() {
            Some(Value::Map(fields)) => fields
                .iter()
                .find(|(k, _)| k == "contents")
                .map(|(_, v)| v.clone()),
            _ => None,
        }
    }

    /// A compute pipeline's record carries the stage-input descriptor it was made with, through
    /// the stream and back: its map's `stageInput` holds each layout (`bufferIndex`, `stride`,
    /// `stepFunction`, `stepRate`), each attribute (`attributeIndex`, `format`, `offset`,
    /// `bufferIndex`), the `indexType` and the `indexBufferIndex`, each as given, none of them
    /// Metal's default but the attribute's index, which the kernel names.
    #[test]
    fn on_vulkan_a_compute_pipeline_record_carries_its_stage_input() {
        in_recording_child(
            "recorder::tests::on_vulkan_a_compute_pipeline_record_carries_its_stage_input",
            |streams| {
                use crate::core::platform::device::GpuDevice as _;
                use crate::core::platform::resource::Library as _;
                use crate::core::platform::values::{
                    AttributeDesc, BufferLayoutDesc, FullComputePipelineDesc, StageInputDesc,
                };
                use crate::core::raise::OnFault;
                const MADE: &str =
                    "newComputePipelineStateWithDescriptor:options:reflection:error:";
                let (_platform, device) = recording_device();
                let library = (device.new_library(crate::gpu::device::tests::STAGEIN))
                    .unwrap()
                    .unwrap()
                    .unwrap();
                let name = library.function_names().remove(0);
                let function = (library.new_function(OnFault::Return, &name))
                    .unwrap()
                    .unwrap();
                let desc = FullComputePipelineDesc {
                    stage_input: Some(StageInputDesc {
                        layouts: vec![BufferLayoutDesc {
                            buffer_index: 1,
                            stride: 8,
                            step_function: 5,
                            step_rate: 2,
                        }],
                        attributes: vec![AttributeDesc {
                            attribute_index: 0,
                            format: 36,
                            offset: 4,
                            buffer_index: 1,
                        }],
                        index_type: 1,
                        index_buffer_index: 2,
                    }),
                    ..FullComputePipelineDesc::new(&function)
                };
                let pipeline = (device.new_compute_pipeline_state_full(&desc))
                    .unwrap()
                    .expect("a pipeline");
                let (calls, _) =
                    recorded(streams, |calls| calls.iter().any(|c| c.selector == MADE));
                let made = (calls.iter())
                    .find(|c| c.selector == MADE)
                    .unwrap_or_else(|| panic!("no {MADE}: {calls:?}"));
                let Some(Value::Map(fields)) = made.args.first() else {
                    panic!("a descriptor's map: {made:?}");
                };
                let stage_input = (fields.iter())
                    .find(|(k, _)| k == "stageInput")
                    .map(|(_, v)| v.clone());
                let map = |pairs: &[(&str, Value)]| {
                    let mut out = Vec::new();
                    for (key, value) in pairs {
                        out.push(((*key).to_owned(), value.clone()));
                    }
                    Value::Map(out)
                };
                let layout = map(&[
                    ("bufferIndex", Value::Uint(1)),
                    ("stride", Value::Uint(8)),
                    ("stepFunction", Value::Uint(5)),
                    ("stepRate", Value::Uint(2)),
                ]);
                let attribute = map(&[
                    ("attributeIndex", Value::Uint(0)),
                    ("format", Value::Uint(36)),
                    ("offset", Value::Uint(4)),
                    ("bufferIndex", Value::Uint(1)),
                ]);
                let expected = map(&[
                    ("layouts", Value::List(vec![layout])),
                    ("attributes", Value::List(vec![attribute])),
                    ("indexType", Value::Uint(1)),
                    ("indexBufferIndex", Value::Uint(2)),
                ]);
                assert_eq!(stage_input, Some(expected), "{made:?}");
                drop((pipeline, function, library));
            },
        );
    }

    /// A render pipeline's record carries its tessellation state and its call stack depths
    /// through the stream and back, as its map's last three keys, after `privateState`:
    /// `tessellation`, a map of `partitionMode`, `maxFactor`, `factorScaleEnabled`,
    /// `factorFormat`, `controlPointIndexType`, `factorStepFunction` and `outputWinding`, then
    /// `maxVertexCallStackDepth` and `maxFragmentCallStackDepth`, each Metal's raw value as given.
    /// Every value but the factor format, whose one value is Metal's `Half`, is other than
    /// Metal's default. The pipeline is of `render_tess`, a post-tessellation vertex function.
    #[test]
    fn on_vulkan_a_render_pipeline_record_carries_its_tessellation_state() {
        in_recording_child(
            "recorder::tests::on_vulkan_a_render_pipeline_record_carries_its_tessellation_state",
            |streams| {
                use crate::core::platform::device::GpuDevice as _;
                use crate::core::platform::resource::Library as _;
                use crate::core::platform::values::{
                    ColorAttachmentDesc, FullRenderPipelineDesc, TessellationDesc,
                };
                use crate::core::raise::OnFault;
                const MADE: &str = "newRenderPipelineStateWithDescriptor:error:";
                const TESS: &[u8] = include_bytes!("../../../tests/shaders/render_tess.metallib");
                const FRAG: &[u8] = include_bytes!("../../../tests/shaders/frag.metallib");
                let (_platform, device) = recording_device();
                let function = |bytes: &[u8], name: &str| {
                    let library = (device.new_library(bytes)).unwrap().unwrap().unwrap();
                    let function = (library.new_function(OnFault::Return, name))
                        .unwrap()
                        .unwrap();
                    (library, function)
                };
                let (vertex_library, vertex) = function(TESS, "render_tess");
                let (fragment_library, fragment) = function(FRAG, "tri_frag");
                let color = ColorAttachmentDesc {
                    pixel_format: 80,
                    ..ColorAttachmentDesc::default()
                };
                let desc = FullRenderPipelineDesc {
                    colors: vec![(0, color)],
                    tessellation: TessellationDesc {
                        partition_mode: 3,
                        max_factor: 32,
                        factor_scale_enabled: true,
                        factor_format: 0,
                        control_point_index_type: 2,
                        factor_step_function: 1,
                        output_winding: 1,
                    },
                    max_vertex_call_stack_depth: 2,
                    max_fragment_call_stack_depth: 3,
                    ..FullRenderPipelineDesc::new(&vertex, Some(&fragment))
                };
                let pipeline = (device.new_render_pipeline_state_full(&desc))
                    .unwrap()
                    .expect("a pipeline");
                let (calls, _) =
                    recorded(streams, |calls| calls.iter().any(|c| c.selector == MADE));
                let made = (calls.iter())
                    .find(|c| c.selector == MADE)
                    .unwrap_or_else(|| panic!("no {MADE}: {calls:?}"));
                let Some(Value::Map(fields)) = made.args.first() else {
                    panic!("a descriptor's map: {made:?}");
                };
                let keys: Vec<&str> = (fields.iter()).map(|(k, _)| k.as_str()).collect();
                let want = [
                    "label",
                    "vertex",
                    "fragment",
                    "vertexDescriptor",
                    "colors",
                    "depthPixelFormat",
                    "stencilPixelFormat",
                    "rasterSampleCount",
                    "alphaToCoverage",
                    "alphaToOne",
                    "rasterizationEnabled",
                    "inputPrimitiveTopology",
                    "maxVertexAmplificationCount",
                    "supportIndirectCommandBuffers",
                    "vertexLinked",
                    "fragmentLinked",
                    "privateState",
                    "tessellation",
                    "maxVertexCallStackDepth",
                    "maxFragmentCallStackDepth",
                ];
                assert_eq!(keys, want, "{made:?}");
                let pairs = |pairs: &[(&str, Value)]| {
                    let mut out = Vec::new();
                    for (key, value) in pairs {
                        out.push(((*key).to_owned(), value.clone()));
                    }
                    out
                };
                let tessellation = Value::Map(pairs(&[
                    ("partitionMode", Value::Uint(3)),
                    ("maxFactor", Value::Uint(32)),
                    ("factorScaleEnabled", Value::Bool(true)),
                    ("factorFormat", Value::Uint(0)),
                    ("controlPointIndexType", Value::Uint(2)),
                    ("factorStepFunction", Value::Uint(1)),
                    ("outputWinding", Value::Uint(1)),
                ]));
                let expected = pairs(&[
                    ("tessellation", tessellation),
                    ("maxVertexCallStackDepth", Value::Uint(2)),
                    ("maxFragmentCallStackDepth", Value::Uint(3)),
                ]);
                assert_eq!(fields[17..], expected, "{made:?}");
                drop((pipeline, vertex, fragment, vertex_library, fragment_library));
            },
        );
    }

    /// A render pipeline that sets none of them still carries its tessellation state and its
    /// call stack depths in its record, each Metal's default, as its map's last three keys:
    /// `tessellation`, a map of `partitionMode` 0 (`Pow2`), `maxFactor` 16,
    /// `factorScaleEnabled` false, `factorFormat` 0 (`Half`), `controlPointIndexType` 0
    /// (`None`), `factorStepFunction` 0 (`Constant`) and `outputWinding` 0 (`Clockwise`), then
    /// `maxVertexCallStackDepth` and `maxFragmentCallStackDepth`, 1 each. The record is the
    /// descriptor the platform was given, where the runtime is given no tessellation state for
    /// Metal's defaults (`gpu::states`). The pipeline is of `tri_vert`, a vertex function of no
    /// tessellation.
    #[test]
    fn on_vulkan_a_render_pipeline_record_carries_metal_s_default_tessellation_state() {
        in_recording_child(
            "recorder::tests::on_vulkan_a_render_pipeline_record_carries_metal_s_default_tessellation_state",
            |streams| {
                use crate::core::platform::device::GpuDevice as _;
                use crate::core::platform::resource::Library as _;
                use crate::core::platform::values::{ColorAttachmentDesc, FullRenderPipelineDesc};
                use crate::core::raise::OnFault;
                const MADE: &str = "newRenderPipelineStateWithDescriptor:error:";
                const VERT: &[u8] = include_bytes!("../../../tests/shaders/vert.metallib");
                const FRAG: &[u8] = include_bytes!("../../../tests/shaders/frag.metallib");
                let (_platform, device) = recording_device();
                let function = |bytes: &[u8], name: &str| {
                    let library = (device.new_library(bytes)).unwrap().unwrap().unwrap();
                    let function = (library.new_function(OnFault::Return, name))
                        .unwrap()
                        .unwrap();
                    (library, function)
                };
                let (vertex_library, vertex) = function(VERT, "tri_vert");
                let (fragment_library, fragment) = function(FRAG, "tri_frag");
                let color = ColorAttachmentDesc {
                    pixel_format: 80,
                    ..ColorAttachmentDesc::default()
                };
                let desc = FullRenderPipelineDesc {
                    colors: vec![(0, color)],
                    ..FullRenderPipelineDesc::new(&vertex, Some(&fragment))
                };
                let pipeline = (device.new_render_pipeline_state_full(&desc))
                    .unwrap()
                    .expect("a pipeline");
                let (calls, _) =
                    recorded(streams, |calls| calls.iter().any(|c| c.selector == MADE));
                let made = (calls.iter())
                    .find(|c| c.selector == MADE)
                    .unwrap_or_else(|| panic!("no {MADE}: {calls:?}"));
                let Some(Value::Map(fields)) = made.args.first() else {
                    panic!("a descriptor's map: {made:?}");
                };
                let pairs = |pairs: &[(&str, Value)]| {
                    let mut out = Vec::new();
                    for (key, value) in pairs {
                        out.push(((*key).to_owned(), value.clone()));
                    }
                    out
                };
                let tessellation = Value::Map(pairs(&[
                    ("partitionMode", Value::Uint(0)),
                    ("maxFactor", Value::Uint(16)),
                    ("factorScaleEnabled", Value::Bool(false)),
                    ("factorFormat", Value::Uint(0)),
                    ("controlPointIndexType", Value::Uint(0)),
                    ("factorStepFunction", Value::Uint(0)),
                    ("outputWinding", Value::Uint(0)),
                ]));
                let expected = pairs(&[
                    ("tessellation", tessellation),
                    ("maxVertexCallStackDepth", Value::Uint(1)),
                    ("maxFragmentCallStackDepth", Value::Uint(1)),
                ]);
                assert_eq!(fields.len(), 20, "{made:?}");
                assert_eq!(fields[17..], expected, "{made:?}");
                drop((pipeline, vertex, fragment, vertex_library, fragment_library));
            },
        );
    }

    /// On a device that keeps two copies of a `Managed` buffer (the answer forced here), a
    /// no-copy `Managed` buffer's record carries the bytes its create copied into the device's
    /// copy, as a blob through the stream and back: a descriptor's `contents`, those before its
    /// `zeroFrom` (none where that is 0), and the first argument of
    /// `newBufferWithBytesNoCopy:length:options:deallocator:`, all of its bytes. On a device that
    /// keeps one copy, and for a `Shared` buffer, both stay nil.
    #[test]
    fn a_no_copy_managed_buffer_carries_its_create_bytes_where_the_device_keeps_two_copies() {
        use crate::core::platform::HostRange;
        use crate::core::platform::values::{BufferDesc, OptionSet};
        let pages: Vec<u8> = (0..8192u32).map(|i| (i / 7) as u8).collect();
        let (managed, shared) = (0x10, 0);
        let by_desc = |two_copies, options, zero_from| {
            let desc = BufferDesc {
                length: 8192,
                options: vec![OptionSet::ResourceOptions(options)],
                pinned_gpu_address: None,
                contents: Some(HostRange::of_slice(&pages)),
                no_copy: true,
                deallocator: None,
                zero_from,
            };
            let mut active = Some(begin(enter_recording(), || {
                (
                    "newBufferWithDescriptor:",
                    1,
                    vec![crate::recorder::rec_val!(desc)],
                )
            }));
            let bytes = HostRange::of_slice(&pages);
            no_copy_contents_with(&mut active, two_copies, options, bytes, zero_from);
            queued(active)
        };
        let by_bytes = |two_copies, options: u64| {
            let bytes = HostRange::of_slice(&pages);
            let mut active = Some(begin(enter_recording(), || {
                (
                    NO_COPY_BYTES,
                    1,
                    vec![
                        crate::recorder::rec_val!(bytes),
                        crate::recorder::rec_val!(options),
                        Value::Nil,
                    ],
                )
            }));
            no_copy_contents_with(&mut active, two_copies, options, bytes, None);
            queued(active)
        };
        let calls = vec![
            by_desc(true, managed, Some(4096)),
            by_desc(true, managed, None),
            by_desc(true, managed, Some(0)),
            by_desc(false, managed, None),
            by_desc(true, shared, None),
            by_bytes(true, managed),
            by_bytes(false, managed),
            by_bytes(true, shared),
        ];
        let (calls, blobs) = written(calls);
        let blob = |data: &[u8]| Value::Blob(format::sha256(data));
        let descs: Vec<_> = calls[..5].iter().map(contents_of).collect();
        assert_eq!(
            descs,
            [
                Some(blob(&pages[..4096])),
                Some(blob(&pages)),
                Some(Value::Nil),
                Some(Value::Nil),
                Some(Value::Nil),
            ]
        );
        let args =
            |first, options| vec![first, Value::Uint(8192), Value::Uint(options), Value::Nil];
        assert_eq!(calls[5].args, args(blob(&pages), managed));
        assert_eq!(calls[6].args, args(Value::Nil, managed));
        assert_eq!(calls[7].args, args(Value::Nil, shared));
        let hashes: Vec<_> = blobs.iter().map(|(h, _)| *h).collect();
        assert_eq!(
            hashes,
            [format::sha256(&pages[..4096]), format::sha256(&pages)]
        );
    }

    /// Recording a host's own device, a no-copy `Managed` buffer's record carries the bytes the
    /// create copied into the device's own copy where the device keeps two (its
    /// `hasUnifiedMemory` NO): a descriptor's those before its `zeroFrom`, a
    /// `newBufferWithBytesNoCopy:` one's all; where it keeps one, neither does. A no-copy
    /// `Shared` buffer's carries none on either.
    #[test]
    fn on_vulkan_a_no_copy_managed_buffer_carries_what_its_create_copied() {
        in_recording_child(
            "recorder::tests::on_vulkan_a_no_copy_managed_buffer_carries_what_its_create_copied",
            |streams| {
                use crate::core::platform::HostRange;
                use crate::core::platform::device::GpuDevice as _;
                use crate::core::platform::values::{BufferDesc, OptionSet};
                use crate::core::raise::OnFault;
                use crate::os::PAGE;
                let (_platform, device) = recording_device();
                let two_copies = device.identity_caps().flags.has_unified_memory == Some(false);
                let length = 2 * PAGE;
                let n = usize::try_from(length).unwrap();
                // Three runs of pages of the test's own, which outlive the buffers: the process
                // ends with them.
                let pages = crate::os::allocate_aliasable(3 * length).unwrap();
                let run = |i: u64| {
                    pages
                        .as_ptr::<u8>()
                        .wrapping_add(usize::try_from(i * length).unwrap())
                };
                for i in 0..3u64 {
                    // SAFETY: fresh pages of the test's own, which nothing uses yet.
                    let bytes = unsafe { std::slice::from_raw_parts_mut(run(i), n) };
                    for (at, b) in bytes.iter_mut().enumerate() {
                        *b = u8::try_from((at / 64 + usize::try_from(i).unwrap() * 3) % 251)
                            .unwrap();
                    }
                }
                // SAFETY: the pages are the test's own, never freed.
                let range = |i| unsafe { HostRange::new(HostPtr::from_ptr(run(i)), length) };
                // SAFETY: as above, and nothing writes them any more.
                let data = |i| unsafe { std::slice::from_raw_parts(run(i), n) }.to_vec();
                let (managed, shared) = (0x10, 0);
                let by_desc = device
                    .new_buffer(BufferDesc {
                        length,
                        options: vec![OptionSet::ResourceOptions(managed)],
                        pinned_gpu_address: None,
                        contents: Some(range(0)),
                        no_copy: true,
                        deallocator: None,
                        zero_from: Some(PAGE),
                    })
                    .unwrap()
                    .expect("a Managed buffer over the pages");
                let by_bytes =
                    (device.new_buffer_over_host_memory(OnFault::Die, range(1), managed, None))
                        .unwrap()
                        .expect("a Managed buffer over the pages");
                let shared_bytes =
                    (device.new_buffer_over_host_memory(OnFault::Die, range(2), shared, None))
                        .unwrap()
                        .expect("a Shared buffer over the pages");
                let (calls, blobs) = recorded(streams, |calls| {
                    calls.iter().filter(|c| c.selector == NO_COPY_BYTES).count() == 2
                });
                let desc = (calls.iter())
                    .find(|c| c.selector == "newBufferWithDescriptor:")
                    .expect("the descriptor's record");
                let over: Vec<&Call> = calls
                    .iter()
                    .filter(|c| c.selector == NO_COPY_BYTES)
                    .collect();
                let blob = |bytes: &[u8]| Value::Blob(format::sha256(bytes));
                let (first, all) = (data(0), data(1));
                let held = |bytes: &[u8]| {
                    blobs
                        .iter()
                        .any(|(h, d)| *h == format::sha256(bytes) && d == bytes)
                };
                if two_copies {
                    assert_eq!(
                        contents_of(desc),
                        Some(blob(&first[..usize::try_from(PAGE).unwrap()]))
                    );
                    assert_eq!(over[0].args.first(), Some(&blob(&all)));
                    assert!(held(&first[..usize::try_from(PAGE).unwrap()]) && held(&all));
                } else {
                    assert_eq!(contents_of(desc), Some(Value::Nil));
                    assert_eq!(over[0].args.first(), Some(&Value::Nil));
                }
                assert_eq!(
                    over[0].args.get(1..3),
                    Some(&[Value::Uint(length), Value::Uint(managed)][..])
                );
                assert_eq!(over[1].args.first(), Some(&Value::Nil));
                drop((by_desc, by_bytes, shared_bytes));
            },
        );
    }
}
