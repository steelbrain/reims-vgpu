//! The call recorder: with `REIMS_VGPU_RECORD` naming a directory, the process writes there a
//! replay stream of every call core makes of the Vulkan platform, in metal2vulkan's replay format
//! version 2 ([`mod@format`]), so a player can run the device's work on any Vulkan node with no VM.
//! It is a diagnostic the host switches on, like `REIMS_VGPU_TRACE`: with the variable unset, each
//! hooked method costs one relaxed load and a branch, and nothing else changes.
//!
//! - **What is recorded.** Each method of the platform's handles that core calls (the
//!   `impl <Trait><VulkanPlatform>` methods of `gpu`) opens with [`rec_call!`] and ends with
//!   [`rec_done!`]: one call record per call core made, the outermost call alone where the
//!   platform calls itself (a thread's depth, [`enter`]). Getters and queries are not hooked.
//!   The selector is Metal's, as the trait's `sel:` line names it, or an `m2v.` name for what
//!   Metal has no call for; an encoder's command is its enum's selector. The receiver and every
//!   object an argument names are the objects' stream ids, which the recorder gives each object
//!   as a call makes it (`State::register`) and never gives again; an object's last reference going
//!   (`Obj`'s drop) is its `release` record.
//! - **Arguments.** The hottest commands and the pass descriptor's sets are read from their fields
//!   ([`typed`]), integers, booleans and ranges as themselves, and host memory, bytes, strings and
//!   events as such ([`spec`]); the rest is formatted as its `Debug` text, its objects as their
//!   stream ids, which the writer reads into values ([`debug_value`]). Text that names objects by
//!   pointer (`ObjId(..)`) is read at once, while the pointers are the objects'.
//! - **Memory.** Before each commit, the recorder waits for the earlier command buffers that can
//!   run, then writes `m2v.write` records of every 4 KiB of host memory the command buffer can
//!   reach that changed since the stream last held it ([`before_commit`]). A buffer in a heap over
//!   the guest's memory is written as its block's bytes. A copy out of the guest's RAM carries the
//!   bytes it reads as its arguments ([`guest_ram_source`]).
//! - **The calling thread captures; the writer shapes.** Each call's record is queued, owned,
//!   under one lock as the call returns, so the stream's order keeps each thread's, with what
//!   only the call's moment has: its objects' ids, the bytes it consumes and the memory a commit
//!   reaches. A thread of the recorder's own (`reims-vgpu.recorder`) takes the queue every
//!   [`DRAIN`] and, in its order, reads the text, shapes and tracks each call and encodes it. A
//!   commit or a copy out of the guest's RAM first takes what is queued on its own thread, since
//!   its capture needs every earlier call tracked.
//! - **The file.** The writer writes the stream out every [`FLUSH`], into segments
//!   `<pid>.NNNN.m2vr` of at most [`SEGMENT`] bytes, each starting with the header. Libraries go
//!   once each into the `blobs` directory beside the stream's, by the hex of their SHA-256.
//!   `<pid>.stats.txt` says what was written, every few seconds.
//!
//! Recordings hold the guest's data: they stay on the host that made them and are never
//! committed.

pub mod debug_value;
pub mod format;
pub mod typed;

use std::any::Any;
use std::borrow::Cow;
use std::cell::Cell;
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
use crate::platform::Obj;
use format::{Call, Hash, Stream, Value};

/// The environment variable naming the directory streams are written to.
pub const ENV_VAR: &str = "REIMS_VGPU_RECORD";

/// How long a record may wait in the buffer.
const FLUSH: Duration = Duration::from_millis(50);

/// How long a call may wait in the queue before the writer shapes it.
const DRAIN: Duration = Duration::from_millis(1);

/// How often the counts are written.
const STATS: Duration = Duration::from_secs(5);

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
}

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
            (state.stats.read_ns).fetch_add(now_ns() - t0, Ordering::Relaxed);
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
        };
    }
    let t0 = now_ns();
    let receiver = receiver();
    let (args, text) = cmd_call(cmd);
    if let Some(state) = state() {
        (state.stats.read_ns).fetch_add(now_ns() - t0, Ordering::Relaxed);
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
    }
}

/// A command's arguments: read from its fields where it is one [`typed`] reads; else its text,
/// read now where it names objects by pointer, or left for the writer.
fn cmd_call<C: typed::Command + std::fmt::Debug>(cmd: &C) -> (Vec<Value>, Option<String>) {
    let mut cx = typed::Cx::new(&id_at);
    if let Some(args) = cmd.typed(&mut cx) {
        count_unrecorded(cx.unrecorded);
        return (args, None);
    }
    let text = debug_text(cmd);
    if text.contains("ObjId(") {
        return (cmd_args_text(&text), None);
    }
    (Vec::new(), Some(text))
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
            ViaBytes as _, ViaDebug as _, ViaEvent as _, ViaHost as _, ViaNil as _, ViaScalar as _,
            ViaSet as _, ViaStr as _,
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
    /// writer reads ([`Value::Debug`]), but where it names objects by pointer, read now.
    pub fn debug_value_of<T: std::fmt::Debug + ?Sized>(x: &T) -> Value {
        let text = super::debug_text(x);
        if text.contains("ObjId(") {
            return super::parse_text(&text);
        }
        Value::Debug(text)
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

/// An object a call made: its pointer, a weak reference and its type.
pub struct Made {
    ptr: u64,
    weak: Weak<dyn Any + Send + Sync>,
    type_name: &'static str,
}

impl Made {
    fn of<T: Send + Sync + 'static>(o: &Obj<T>) -> Made {
        let weak: Weak<dyn Any + Send + Sync> = Obj::downgrade(o);
        Made {
            ptr: o.id().0,
            weak,
            type_name: std::any::type_name::<T>(),
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

/// The stream id of the object at `ptr`.
fn id_at(ptr: u64) -> Option<u64> {
    state()?.objects.lock().by_ptr.get(&ptr).copied()
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
/// `MTLCreateSystemDefaultDevice` record.
fn receiver_id<T: Send + Sync + 'static>(o: &Obj<T>) -> u64 {
    let Some(state) = state() else { return 0 };
    if let Some(id) = id_at(o.id().0) {
        return id;
    }
    if std::any::type_name::<T>() == std::any::type_name::<crate::gpu::device::DeviceObject>() {
        let id = state.register(Made::of(o));
        state.push_plain(Call {
            selector: "MTLCreateSystemDefaultDevice".to_owned(),
            result: id,
            ..Call::default()
        });
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

/// An object's last reference went, at `ptr`: its `release` record, after every record queued
/// before it. Its pointer is no longer the object's at once, and memory reached through it is
/// no longer read.
pub fn released(ptr: u64) {
    let Some(state) = state() else { return };
    let id = {
        let mut objects = state.objects.lock();
        let Some(id) = objects.by_ptr.remove(&ptr) else {
            return;
        };
        if let Some(info) = objects.info.get_mut(&id) {
            info.dead = true;
        }
        id
    };
    state.baselines.lock().remove(&id);
    state.push(Item::Release(id));
}

/// Records the call `active` opened, with the object it made: queued, for the writer to shape.
pub fn finish(active: Active, made: Option<Made>) {
    let Active {
        token,
        pending,
        text,
    } = active;
    let Some(mut call) = pending else { return };
    let Some(state) = state() else { return };
    let t0 = now_ns();
    if let Some(made) = made {
        call.result = state.register(made);
    }
    state.capture_now(&mut call);
    state.push(Item::Call {
        call,
        text,
        shape: true,
    });
    (state.stats.finish_ns).fetch_add(now_ns() - t0, Ordering::Relaxed);
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
            });
        }
    }
    (state.stats.read_ns).fetch_add(now_ns() - t0, Ordering::Relaxed);
}

/// Before a commit: shapes what is queued, waits for the earlier command buffers that can run,
/// then writes the memory the command buffer reaches that changed.
pub fn before_commit(active: &Option<Active>, cb: &VulkanCommandBuffer) {
    let Some(a) = active else { return };
    if !a.recorded() {
        return;
    }
    let Some(state) = state() else { return };
    let Some(id) = id_at(cb.id().0) else { return };
    state.drain(true);
    let t0 = now_ns();
    let mut written = state.reap();
    state.wait_for_earlier();
    written.extend(state.reap());
    let t1 = now_ns();
    state.refresh(&written);
    state.stats.wait_ns.fetch_add(t1 - t0, Ordering::Relaxed);
    let reach: Vec<u64> = state
        .sync
        .lock()
        .cbs
        .get(&id)
        .map(|c| c.reach.iter().copied().collect())
        .unwrap_or_default();
    let ranges = state.ranges_of(&reach);
    state.capture(&ranges);
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

/// The call `active` opened encoded nothing, so the stream holds no record of it: a copy between
/// a buffer and the guest's RAM that the platform did not make (it answered false, or a refusal).
pub fn unmade(active: &mut Option<Active>) {
    if let Some(a) = active {
        a.pending = None;
    }
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
    let mut written = state.reap();
    state.wait_for_earlier();
    written.extend(state.reap());
    state.refresh(&written);
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

/// Before `didModifyRange:` of a `Managed` buffer: the range's bytes, as a write.
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
    state.capture(&ranges);
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
    /// The objects a descriptor's calls named.
    refs: HashSet<u64>,
    /// Whether its last reference went: its release is queued, and its memory no longer read.
    dead: bool,
}

/// What a calling thread hands the writer, in the stream's order.
enum Item {
    /// A hooked call: its text read (the command's where `text`), shaped where `shape`, tracked
    /// and written.
    Call {
        call: Call,
        text: Option<String>,
        shape: bool,
    },
    /// A record written as it is, counted by its selector where `count`.
    Plain { call: Call, count: bool },
    /// An object's release: its last record, after which the recorder forgets it.
    Release(u64),
}

#[derive(Default)]
struct Objects {
    by_ptr: HashMap<u64, u64>,
    info: HashMap<u64, Info>,
    next: u64,
    /// Live pinned ranges: start, then end and id.
    pins: BTreeMap<u64, (u64, u64)>,
}

/// A command buffer's reach and its event waits and signals.
#[derive(Default)]
struct Cb {
    reach: HashSet<u64>,
    /// What its passes render into and its blits write: bytes only the GPU writes.
    gpu_writes: HashSet<u64>,
    waits: Vec<(u64, u64)>,
    signals: Vec<(u64, u64)>,
}

/// A committed command buffer not yet seen complete.
struct Pending {
    queue: u64,
    gpu_writes: Vec<u64>,
    waits: Vec<(u64, u64)>,
    signals: Vec<(u64, u64)>,
    /// The command buffer, never upgraded by the writer ([`upgrade_cb`]).
    weak: Weak<dyn Any + Send + Sync>,
}

#[derive(Default)]
struct SyncState {
    cbs: HashMap<u64, Cb>,
    pending: Vec<Pending>,
    /// The highest value the CPU, or a completed command buffer, signalled each event to.
    signalled: HashMap<u64, u64>,
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
    /// writes.
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
}

struct State {
    base: PathBuf,
    blobs_dir: PathBuf,
    /// The records not yet shaped, in the stream's order.
    queue: Mutex<Vec<Item>>,
    /// Held while queued records are shaped, so they are shaped in order.
    shaping: Mutex<()>,
    stream: Mutex<Stream>,
    objects: Mutex<Objects>,
    sync: Mutex<SyncState>,
    baselines: Mutex<HashMap<u64, Vec<u64>>>,
    selectors: Mutex<BTreeMap<String, u64>>,
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
            objects: Mutex::new(Objects::default()),
            sync: Mutex::new(SyncState::default()),
            baselines: Mutex::new(HashMap::new()),
            selectors: Mutex::new(BTreeMap::new()),
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
        let mut segment = 0u32;
        let mut file: Option<std::fs::File> = None;
        let mut written = 0u64;
        let flush = u64::try_from(FLUSH.as_nanos()).unwrap_or(u64::MAX);
        let stats = u64::try_from(STATS.as_nanos()).unwrap_or(u64::MAX);
        let mut flushed = now_ns();
        let mut counted = flushed;
        loop {
            std::thread::sleep(DRAIN);
            self.drain(false);
            let now = now_ns();
            if now - flushed < flush {
                continue;
            }
            flushed = now;
            let bytes = self.stream.lock().take();
            let t0 = now_ns();
            if !bytes.is_empty() {
                if file.is_none() || written >= SEGMENT {
                    let path = PathBuf::from(format!("{}.{segment:04}.m2vr", self.base.display()));
                    segment += 1;
                    file = std::fs::File::create(&path)
                        .inspect_err(|e| {
                            log::warn!(target: "reims_vgpu_vulkan::recorder", "{}: {e}", path.display());
                        })
                        .ok();
                    written = 0;
                    if let Some(f) = file.as_mut() {
                        let _ = f.write_all(&format::HEADER);
                        written += format::HEADER.len() as u64;
                    }
                }
                if let Some(f) = file.as_mut() {
                    let _ = f.write_all(&bytes);
                    written += bytes.len() as u64;
                }
            }
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

    /// Queues a record written as it is.
    fn push_plain(&self, call: Call) {
        self.push(Item::Plain { call, count: false });
    }

    /// Shapes and writes every queued record, in order: on the writer's thread, or on a calling
    /// thread (`here`) whose capture needs them tracked.
    fn drain(&self, here: bool) {
        let t0 = now_ns();
        let _shaping = self.shaping.lock();
        let items = std::mem::take(&mut *self.queue.lock());
        let t1 = now_ns();
        let n = items.len() as u64;
        for item in items {
            self.process(item);
        }
        let t2 = now_ns();
        self.stats.process_ns.fetch_add(t2 - t1, Ordering::Relaxed);
        if here {
            self.stats.drain_ns.fetch_add(t2 - t0, Ordering::Relaxed);
            self.stats.drained_here.fetch_add(n, Ordering::Relaxed);
        }
    }

    fn process(&self, item: Item) {
        match item {
            Item::Call {
                mut call,
                text,
                shape,
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
                self.emit(call, false);
            }
            Item::Plain { call, count } => self.emit(call, count),
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
                self.emit(
                    Call {
                        selector: "release".to_owned(),
                        receiver: id,
                        ..Call::default()
                    },
                    false,
                );
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
        let _ = writeln!(out, "live_objects {}", self.objects.lock().info.len());
        for (sel, n) in self.selectors.lock().iter() {
            let _ = writeln!(out, "selector {n} {sel}");
        }
        let path = PathBuf::from(format!("{}.stats.txt", self.base.display()));
        let _ = std::fs::write(path, out);
    }

    fn register(&self, made: Made) -> u64 {
        let (id, stale) = {
            let mut objects = self.objects.lock();
            objects.next += 1;
            let id = objects.next;
            let stale = objects.by_ptr.insert(made.ptr, id);
            if let Some(old) = stale
                && let Some(info) = objects.info.get_mut(&old)
            {
                info.dead = true;
            }
            objects.info.insert(
                id,
                Info {
                    type_name: made.type_name,
                    weak: Some(made.weak),
                    ..Info::default()
                },
            );
            (id, stale)
        };
        if let Some(old) = stale {
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
            "newBufferWithBytesNoCopy:length:options:deallocator:" => {
                // new_buffer_over_host_memory(bytes, options, deallocator)
                let len = match call.args.first() {
                    Some(Value::Host(_, l)) => *l,
                    _ => 0,
                };
                let options = call.args.get(1).cloned().unwrap_or(Value::Nil);
                call.args = vec![Value::Nil, Value::Uint(len), options, Value::Nil];
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
                    let hash = format::sha256(data);
                    let path = self.blobs_dir.join(hex(&hash));
                    if !path.exists() {
                        let tmp =
                            self.blobs_dir
                                .join(format!("{}.{}", hex(&hash), std::process::id()));
                        if std::fs::write(&tmp, data).is_ok() {
                            let _ = std::fs::rename(&tmp, &path);
                        }
                    }
                    call.args = vec![Value::Blob(hash)];
                }
            }
            _ => {}
        }
        for v in &mut call.args {
            read_host(v, &call.selector);
        }
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
        *self
            .selectors
            .lock()
            .entry(call.selector.clone())
            .or_insert(0) += 1;
        let recv_type = self.type_of(call.receiver);
        let cb_type = std::any::type_name::<CommandBufferObject>();
        let mut named = Vec::new();
        for a in &call.args {
            a.each_object(&mut |id| named.push(id));
        }
        // An object a command buffer or its encoder makes is the command buffer's; a command
        // buffer is its queue's.
        if call.result != 0 {
            let mut objects = self.objects.lock();
            if let Some(info) = objects.info.get_mut(&call.result) {
                info.parent = call.receiver;
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
            if recv_type.contains("Descriptor") && !named.is_empty() {
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
            let objects = self.objects.lock();
            for id in &named {
                if let Some(info) = objects.info.get(id) {
                    reach.extend(info.refs.iter().copied());
                    // A pass's descriptor names what the pass renders into.
                    if call.selector.starts_with("renderCommandEncoder") {
                        gpu_writes.extend(info.refs.iter().copied());
                    }
                }
            }
        }
        // A blit's destination: its one object, or each after the first (its source).
        if recv_type.contains("BlitCommandEncoder") {
            gpu_writes.extend(named.iter().skip(usize::from(named.len() > 1)).copied());
        }
        let mut sync = self.sync.lock();
        let entry = sync.cbs.entry(cb).or_default();
        entry.reach.extend(reach);
        entry.gpu_writes.extend(gpu_writes);
        let event = |args: &[Value]| match (args.first(), args.get(1)) {
            (Some(Value::Object(e)), Some(Value::Uint(v))) => Some((*e, *v)),
            _ => None,
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
            "commit" => {
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
                        queue,
                        gpu_writes: cb_info.gpu_writes.into_iter().collect(),
                        waits: cb_info.waits,
                        signals: cb_info.signals,
                        weak,
                    });
                }
            }
            _ => {}
        }
    }

    /// Drops the committed command buffers that completed, their signals folded in; the objects
    /// they wrote as attachments and blit destinations, whose bytes are the GPU's.
    fn reap(&self) -> Vec<u64> {
        let mut held = Vec::new();
        let mut written = Vec::new();
        {
            let mut sync = self.sync.lock();
            // The command buffers looked at are let go once the lock is (`let_go_cb` may record a
            // release).
            let mut done = Vec::new();
            sync.pending.retain(|p| {
                let finished = match upgrade_cb(&p.weak) {
                    None => true,
                    Some(cb) => {
                        let finished = finished(&cb);
                        held.push(cb);
                        finished
                    }
                };
                if finished {
                    done.extend(p.signals.iter().copied());
                    written.extend(p.gpu_writes.iter().copied());
                }
                !finished
            });
            for (e, v) in done {
                let s = sync.signalled.entry(e).or_insert(0);
                *s = (*s).max(v);
            }
        }
        held.into_iter().for_each(let_go_cb);
        written
    }

    /// Takes the bytes the GPU wrote into `ids` as the stream's: the comparison point of the
    /// ranges only the GPU writes moves at their command buffer's completion.
    fn refresh(&self, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        let spans = self.ranges_of(ids);
        self.compare(&spans, false);
    }

    /// Waits, at most [`WAIT_BOUND`], for each committed command buffer that can run: one whose
    /// event waits a CPU signal, a completed command buffer or another that can run satisfies,
    /// and whose queue's earlier one can run.
    fn wait_for_earlier(&self) {
        let runnable: Vec<Weak<dyn Any + Send + Sync>> = {
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
                .map(|(p, _)| p.weak.clone())
                .collect()
        };
        let start = now_ns();
        let bound = u64::try_from(WAIT_BOUND.as_nanos()).unwrap_or(u64::MAX);
        for weak in runnable {
            self.stats.waits.fetch_add(1, Ordering::Relaxed);
            loop {
                let Some(cb) = upgrade_cb(&weak) else { break };
                let done = finished(&cb);
                let_go_cb(cb);
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
                self.buffer_span(id, b, None, &mut spans);
            } else if let Some(t) = any.downcast_ref::<TextureObject>() {
                match &t.under {
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
        self.compare(spans, true);
    }

    /// Compares each 4 KiB of `spans` with the stream's: where `emit`, a changed one is written
    /// as a write; else it is only taken as the stream's.
    fn compare(&self, spans: &[Span], emit: bool) {
        let t0 = now_ns();
        let mut writes: BTreeMap<u64, Vec<(u64, Vec<u8>)>> = BTreeMap::new();
        {
            let mut baselines = self.baselines.lock();
            let mut buf = vec![0u8; CHUNK as usize];
            for s in spans {
                if s.start >= s.end {
                    continue;
                }
                let first = s.start / CHUNK;
                let last = (s.end - 1) / CHUNK;
                let chunks = s.len.div_ceil(CHUNK) as usize;
                let base = baselines.entry(s.target).or_default();
                if base.len() < chunks {
                    base.resize(chunks, 0);
                }
                for c in first..=last {
                    let off = c * CHUNK;
                    let n = CHUNK.min(s.len - off) as usize;
                    let Some(dst) = buf.get_mut(..n) else {
                        continue;
                    };
                    // SAFETY: `base..base + len` is host memory a live object of the stream lies
                    // over (a buffer's contents, a block's range), mapped while the command
                    // buffer that reaches it is being committed; the guest may write it
                    // meanwhile, which a copy of bytes tolerates.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            HostPtr::from_addr(s.base + off as usize).as_ptr::<u8>(),
                            dst.as_mut_ptr(),
                            n,
                        );
                    }
                    self.stats.chunks_hashed.fetch_add(1, Ordering::Relaxed);
                    let h = hash64(dst);
                    let Some(slot) = base.get_mut(c as usize) else {
                        continue;
                    };
                    if *slot == h {
                        continue;
                    }
                    *slot = h;
                    if emit {
                        self.scan_addresses(dst);
                        writes
                            .entry(s.target)
                            .or_default()
                            .push((off, dst.to_vec()));
                    } else {
                        self.stats.chunks_refreshed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        let t1 = now_ns();
        self.stats.hash_ns.fetch_add(t1 - t0, Ordering::Relaxed);
        for (target, chunks) in writes {
            let mut args = Vec::with_capacity(chunks.len() * 2);
            for (off, data) in chunks {
                self.stats.chunks_written.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .write_bytes
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                args.push(Value::Uint(off));
                // A chunk of `BLOB_MIN` bytes or more is written as a blob ([`blob_long`]).
                args.push(Value::Bytes(data));
            }
            self.push(Item::Plain {
                call: Call {
                    selector: "m2v.write".to_owned(),
                    receiver: target,
                    args,
                    ..Call::default()
                },
                count: true,
            });
        }
        (self.stats.write_ns).fetch_add(now_ns() - t1, Ordering::Relaxed);
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

    /// Writes a record into the stream, counted by its selector where `count`.
    fn emit(&self, mut call: Call, count: bool) {
        let mut stream = self.stream.lock();
        // Long inline data goes as a blob.
        for v in &mut call.args {
            blob_long(v, &mut stream);
        }
        let n = stream.call(&call);
        drop(stream);
        self.count(n, if count { &call.selector } else { "" });
    }

    fn count(&self, n: usize, selector: &str) {
        self.stats.records.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes.fetch_add(n as u64, Ordering::Relaxed);
        if !selector.is_empty() {
            *self
                .selectors
                .lock()
                .entry(selector.to_owned())
                .or_insert(0) += 1;
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

/// Whether the command buffer completed or failed.
fn finished(cb: &CommandBufferObject) -> bool {
    let status = cb.raw.status().raw();
    status == CommandBufferStatus::COMPLETED.raw() || status == CommandBufferStatus::ERROR.raw()
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
        assert_eq!(checked, render.len() + compute.len() + blit.len());
        let mut cx = typed::Cx::new(&ids);
        let untyped: R<'_, F> = R::SetLineWidth { line_width: 1.5 };
        assert!(
            untyped.typed(&mut cx).is_none(),
            "an f32 is read from its text"
        );
    }

    /// The pass descriptor's sets read from their fields are the values their `Debug` text,
    /// which names objects by pointer, reads as: objects made and not, nil sets, set values and
    /// non-finite colours
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
        let check = |read: &dyn Fn(&mut typed::Cx<'_>) -> Value, text: String| {
            let mut cx = typed::Cx::new(&ids);
            let v = read(&mut cx);
            let (want, notes) = debug_value::parse(&text, &ids);
            assert!(!notes.unread, "{text}");
            assert_eq!(cx.unrecorded, notes.unrecorded, "{text}");
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
        assert_eq!(
            rec_val!(by_ptr),
            Value::Nil,
            "read at once: no object at 0x1000"
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
        });
        state.push(Item::Plain {
            call: Call {
                selector: "m2v.write".to_owned(),
                receiver: 6,
                args: vec![Value::Uint(4096), Value::Bytes(vec![7; 4096])],
                ..Call::default()
            },
            count: true,
        });
        state.push(Item::Release(6));
        assert_eq!(state.queue.lock().len(), 4);
        state.drain(true);
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
}
