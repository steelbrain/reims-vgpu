//! A GPU that does not finish, for the tests of the crate's bounded waits
//! (`super::wait`): a Vulkan compute shader of one invocation that spins on
//! a flag in host memory until the host sets it, on the GPU of the render
//! node the test's display is on. Queued on a ring ahead of the driver's
//! work, it holds that work back for as long: on the graphics queue, the
//! ring radeonsi's video processing and image copies run on
//! ([`Ring::Graphics`]), or on a compute queue ([`Ring::Compute`]).
//!
//! The shader is SPIR-V, written here word by word ([`shader`]). The flag
//! is host-visible memory the GPU reads and writes uncached
//! (`VK_AMD_device_coherent_memory`'s `DEVICE_UNCACHED_AMD`), which the
//! shader reads with volatile loads, so that it sees the host's write; it
//! writes that it has begun beside the flag, so that a test times its calls
//! only once the ring is held ([`Spin::hold`]). A spin still held after
//! [`HELD_AT_MOST`] is let go by a thread of its own, so that a test whose
//! wait does not end fails rather than hangs.
//!
//! **Where the GPU has no such memory, the spin is timed** ([`Spin::timed`]):
//! the GPU's caches would keep the shader's start from the host until it
//! ended, and the host's flag from the shader, so it holds the ring for
//! [`SPUN_AT_MOST`] by the GPU's clock, whatever the host does, and the host
//! learns that it has begun from a timestamp the queue writes just before
//! it, once the work ahead of it is done, which `vkGetQueryPoolResults`
//! reads. A test lets it go by waiting for it to end. **Measured** on an
//! RX 550 (RADV on GFX8, Mesa 26.2.6, 2026-11-07), which RADV gives no
//! uncached memory: in host-visible memory, the host saw a spin begin 652 ms
//! after its submit, as its 1.8 million reads of the flag ended it (0.36 µs
//! a read); with the reads and writes nontemporal as well, it saw some begin
//! in 0.3 ms and others as late, and no spin saw the host let it go. Timed,
//! a spin on it held the graphics ring and a compute ring for 30,000,000
//! ticks of its clock in 1.1999 s, the device's timestamp period of 40 ns,
//! reading the flag 3.34 million times, and four spins on compute queues
//! ran at once: radeonsi's compute work waited behind them. A GPU whose
//! queue writes no timestamps, or whose host cannot reset a query, has no
//! spin ([`NoSpin::Unheld`]), and the tests of a held GPU say so and end.
//!
//! **It ends on its own**, whatever the host does: after [`SPUN_AT_MOST`]
//! by the GPU's own clock (`OpReadClockKHR` at device scope, at the period
//! the device gives its timestamps), or after [`TURNS_AT_MOST`] reads of
//! the flag ([`TIMED_TURNS_AT_MOST`] for a timed spin), whichever comes
//! first, so that a process that never sets the
//! flag holds the GPU for a second, not for good, and under amdgpu's limit
//! for a job on a ring, past which it resets the ring: 2 s unless
//! `amdgpu.lockup_timeout` says otherwise (kernel 7.2's `modinfo`), and a
//! spin is refused where it says less ([`LIMIT`]). It faults nothing: one
//! invocation reads four bytes of the host's and writes what it counted
//! beside them as it ends ([`Spin::ran`]).
//!
//! **Measured** on the RX 7900 XTX and the Radeon 780M (RADV, Mesa 26.2.4,
//! kernel 7.2.9, 2026-10-26): the clock ran at 100 MHz on both, and a read
//! of the flag took 0.95 µs on the RX 7900 XTX and 0.33 µs on the Radeon
//! 780M (528,000 and 1,500,000 reads in 500 ms). A spin held 3 s had amdgpu
//! reset the RX 7900 XTX's graphics ring at 2 s.
//!
//! A render node another driver than amdgpu runs gets no spin ([`NoSpin::Unheld`]): no limit of
//! amdgpu's bounds it there, and the tests of a held GPU say so and end (on iHD, measured on Iris
//! Xe under i915 2026-11-07, every one of them failed instead, at amdgpu's missing parameter).
//!
//! **It stops at once on a reset.** While a spin lives, a thread reads the
//! kernel's log (`journalctl -k -f`), and at any amdgpu line of a reset, a
//! timeout or a hang it lets every spin go and says so
//! ([`reset_seen`]), which fails the test.

use std::ffi::CStr;
use std::io::{BufRead, BufReader};
use std::os::unix::fs::MetadataExt;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use crate::core::runtime::{Runtime, StdRuntime};
use crate::core::sync::Mutex;
use ash::vk;

/// How long a spin is held at most, unless the test lets it go first.
pub(crate) const HELD_AT_MOST: Duration = Duration::from_secs(1);

/// How long the shader spins at most, by the GPU's clock, whether or not
/// the flag is ever set.
pub(crate) const SPUN_AT_MOST: Duration = Duration::from_millis(1200);

/// How many times the shader reads the flag at most, should the GPU's
/// clock not move: 1.7 s of the RX 7900 XTX's reads, 0.6 s of the Radeon
/// 780M's (module docs).
pub(crate) const TURNS_AT_MOST: u64 = 1_800_000;

/// How many times a timed spin reads the flag at most, should the GPU's
/// clock not move: 1.8 s of the RX 550's reads (module docs), whose
/// 3.34 million in [`SPUN_AT_MOST`] leave it a margin of half as many again.
pub(crate) const TIMED_TURNS_AT_MOST: u64 = 5_000_000;

/// amdgpu's limit for a job on a ring, below which a spin is refused.
const LIMIT: Duration = Duration::from_secs(2);

/// The ring a spin holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ring {
    /// The queue family that draws and computes: AMD's graphics ring.
    Graphics,
    /// A queue family that computes and does not draw: a compute ring.
    Compute,
}

/// The flag's words: the flag the host sets, the word the shader sets as
/// it begins, and what it counted as it ends.
const FLAG: usize = 0;
const BEGUN: usize = 4;
const TURNS: usize = 8;
const SPUN: usize = 16;
const FLAG_BYTES: u64 = 24;

/// The shader: one invocation that sets the word at [`BEGUN`], then reads
/// the flag until it is not 0, or until it has spun `ticks` of the GPU's
/// clock or read it `turns_most` times, then writes the reads it made
/// and the ticks it spun at [`TURNS`] and [`SPUN`]. SPIR-V 1.0, with the
/// GLSL450 memory model and the flag a `BufferBlock`; volatile loads and
/// stores.
fn shader(ticks: u64, turns_most: u64) -> Vec<u32> {
    let mut s = Spirv::default();
    // Ids, in the order they are numbered.
    let [
        main,
        void,
        func,
        uint,
        ulong,
        boolean,
        block,
        p_block,
        p_uint,
        p_ulong,
        flag,
    ] = s.ids();
    let [c0, c1, c2, c3, l0, l1, most_turns, ticks_most, device] = s.ids();
    let [
        entry,
        head,
        body,
        cont,
        done,
        start,
        turns,
        p_flag,
        f,
        next,
        now,
    ] = s.ids();
    let [spun, set, many, long, either, end, p_begun, p_turns, p_spun] = s.ids();
    // Capabilities Shader, Int64 and ShaderClockKHR.
    for capability in [1, 11, 5055] {
        s.op(17, &[capability]);
    }
    s.op_str(10, &[], "SPV_KHR_shader_clock");
    // Logical addressing, the GLSL450 memory model.
    s.op(14, &[0, 1]);
    // A GLCompute entry point named "main", of one invocation.
    s.op_str(15, &[5, main], "main");
    s.op(16, &[main, 17, 1, 1, 1]);
    // The flag's block: a BufferBlock at set 0, binding 0, coherent and
    // volatile, its members at their offsets.
    s.op(71, &[block, 3]);
    for (member, offset) in [(0, FLAG), (1, BEGUN), (2, TURNS), (3, SPUN)] {
        s.op(72, &[block, member, 35, offset as u32]);
    }
    s.op(71, &[flag, 34, 0]);
    s.op(71, &[flag, 33, 0]);
    s.op(71, &[flag, 23]);
    s.op(71, &[flag, 21]);
    // Types.
    s.op(19, &[void]);
    s.op(33, &[func, void]);
    s.op(21, &[uint, 32, 0]);
    s.op(21, &[ulong, 64, 0]);
    s.op(20, &[boolean]);
    s.op(30, &[block, uint, uint, ulong, ulong]);
    // Pointers in the Uniform storage class (2), and the flag's variable.
    s.op(32, &[p_block, 2, block]);
    s.op(32, &[p_uint, 2, uint]);
    s.op(32, &[p_ulong, 2, ulong]);
    s.op(59, &[p_block, flag, 2]);
    // Constants: 64-bit ones take two words, the low first.
    let low = |v: u64| (v & 0xffff_ffff) as u32;
    let high = |v: u64| (v >> 32) as u32;
    for (id, value) in [(c0, 0), (c1, 1), (c2, 2), (c3, 3), (device, 1)] {
        s.op(43, &[uint, id, value]);
    }
    for (id, value) in [
        (l0, 0),
        (l1, 1),
        (most_turns, turns_most),
        (ticks_most, ticks),
    ] {
        s.op(43, &[ulong, id, low(value), high(value)]);
    }
    // main.
    s.op(54, &[void, main, 0, func]);
    s.op(248, &[entry]);
    s.op(65, &[p_uint, p_begun, flag, c1]);
    s.op(62, &[p_begun, c1, 1]);
    s.op(5056, &[ulong, start, device]);
    s.op(249, &[head]);
    // The loop's header: the reads so far.
    s.op(248, &[head]);
    s.op(245, &[ulong, turns, l0, entry, next, cont]);
    s.op(246, &[done, cont, 0]);
    s.op(249, &[body]);
    s.op(248, &[body]);
    s.op(65, &[p_uint, p_flag, flag, c0]);
    s.op(61, &[uint, f, p_flag, 1]);
    s.op(128, &[ulong, next, turns, l1]);
    s.op(5056, &[ulong, now, device]);
    s.op(130, &[ulong, spun, now, start]);
    s.op(171, &[boolean, set, f, c0]);
    s.op(174, &[boolean, many, next, most_turns]);
    s.op(174, &[boolean, long, spun, ticks_most]);
    s.op(166, &[boolean, either, set, many]);
    s.op(166, &[boolean, end, either, long]);
    s.op(250, &[end, done, cont]);
    s.op(248, &[cont]);
    s.op(249, &[head]);
    // After the loop: what it counted.
    s.op(248, &[done]);
    s.op(65, &[p_ulong, p_turns, flag, c2]);
    s.op(62, &[p_turns, next, 1]);
    s.op(65, &[p_ulong, p_spun, flag, c3]);
    s.op(62, &[p_spun, spun, 1]);
    s.op(253, &[]);
    s.op(56, &[]);
    s.words()
}

/// A SPIR-V module as it is written, instruction by instruction.
#[derive(Default)]
struct Spirv {
    bound: u32,
    body: Vec<u32>,
}

impl Spirv {
    /// `N` new ids.
    fn ids<const N: usize>(&mut self) -> [u32; N] {
        std::array::from_fn(|_| {
            self.bound += 1;
            self.bound
        })
    }

    /// The instruction `opcode` with `operands`.
    fn op(&mut self, opcode: u32, operands: &[u32]) {
        let count = u32::try_from(operands.len() + 1).unwrap_or(0);
        self.body.push((count << 16) | opcode);
        self.body.extend_from_slice(operands);
    }

    /// The instruction `opcode` with `operands`, then a literal string.
    fn op_str(&mut self, opcode: u32, operands: &[u32], text: &str) {
        let mut words = operands.to_vec();
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        bytes.resize(bytes.len().next_multiple_of(4), 0);
        let (chunks, _) = bytes.as_chunks::<4>();
        words.extend(chunks.iter().map(|&c| u32::from_le_bytes(c)));
        self.op(opcode, &words);
    }

    /// The module: its header, then its instructions.
    fn words(self) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, self.bound + 1, 0];
        words.extend(self.body);
        words
    }
}

/// A spin on the GPU of one render node.
pub(crate) struct Spin {
    gpu: Arc<Gpu>,
    /// When the shader was last queued, which its hold's [`HELD_AT_MOST`] counts from.
    queued: Mutex<Option<crate::core::runtime::MonoTime>>,
}

/// The device, the shader's pipeline and the flag, freed once the shader
/// has ended and nothing holds them.
struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    /// The flag's 24 bytes, mapped.
    mapped: *mut u8,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    commands: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    /// A timed spin's query, the timestamp written just before it, which
    /// says that it has begun; `None` where the flag says so.
    query: Option<vk::QueryPool>,
    /// Whether the shader was submitted and its fence not yet seen.
    submitted: AtomicBool,
}

// SAFETY: the handles are Vulkan's, used from one thread at a time
// (`Spin`'s calls, then its drop); the mapped flag is reached through
// atomics alone.
unsafe impl Send for Gpu {}
// SAFETY: as above.
unsafe impl Sync for Gpu {}

impl Gpu {
    fn word(&self, at: usize) -> &AtomicU32 {
        // SAFETY: four aligned bytes of the mapped 24, live while `self` is.
        unsafe { &*self.mapped.add(at).cast::<AtomicU32>() }
    }

    fn long(&self, at: usize) -> u64 {
        // SAFETY: eight bytes at an 8-aligned offset of the mapped 24.
        unsafe { &*self.mapped.add(at).cast::<AtomicU64>() }.load(Ordering::SeqCst)
    }

    /// Waits for the shader to end, `most` at most: whether it has.
    fn ended(&self, most: Duration) -> bool {
        if !self.submitted.load(Ordering::SeqCst) {
            return true;
        }
        let nanos = u64::try_from(most.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: the device's own fence.
        let done = unsafe { self.device.wait_for_fences(&[self.fence], true, nanos) }.is_ok();
        if done {
            self.submitted.store(false, Ordering::SeqCst);
        }
        done
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.word(FLAG).store(1, Ordering::SeqCst);
        // The shader ends by its bounds; past them the device is lost, and
        // its objects are destroyed all the same.
        let _ = self.ended(SPUN_AT_MOST + Duration::from_secs(2));
        // SAFETY: the device's objects, each destroyed once, with no work
        // using them.
        unsafe {
            self.device.destroy_fence(self.fence, None);
            if let Some(query) = self.query {
                self.device.destroy_query_pool(query, None);
            }
            self.device.destroy_command_pool(self.commands, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_shader_module(self.module, None);
            self.device.destroy_buffer(self.buffer, None);
            self.device.unmap_memory(self.memory);
            self.device.free_memory(self.memory, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        LIVE.fetch_sub(1, Ordering::SeqCst);
        stop_watching();
    }
}

/// Spins that live, process-wide: the kernel's log is read while any does.
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// Set once the kernel's log has said amdgpu reset, timed out or hung.
static RESET: AtomicBool = AtomicBool::new(false);

/// The log reader while a spin lives.
static WATCH: Mutex<Option<Child>> = Mutex::new(None);

/// Whether the kernel's log said amdgpu reset, timed out or hung while a
/// spin lived.
pub(crate) fn reset_seen() -> bool {
    RESET.load(Ordering::SeqCst)
}

/// Whether `line` of the kernel's log says amdgpu reset a GPU, or a job
/// of it timed out or hung.
fn says_reset(line: &str) -> bool {
    let line = line.to_ascii_lowercase();
    line.contains("amdgpu")
        && ["reset", "timeout", "timed out", "hang", "hung"]
            .iter()
            .any(|word| line.contains(word))
}

/// Reads the kernel's log from now on, while a spin lives, unless it is
/// read already: at a reset's line, every spin is let go.
fn watch() {
    let mut watch = WATCH.lock();
    if watch.is_some() {
        return;
    }
    let child = Command::new("journalctl")
        .args(["-k", "-f", "-n", "0", "-o", "cat"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            eprintln!("spin: the kernel's log is not read: {error}");
            return;
        }
    };
    if let Some(out) = child.stdout.take() {
        let read = StdRuntime::thread("test-spin-kernel-log", move || {
            for line in BufReader::new(out).lines() {
                let Ok(line) = line else { return };
                if says_reset(&line) {
                    eprintln!("spin: the kernel's log says: {line}");
                    RESET.store(true, Ordering::SeqCst);
                }
            }
        });
        if read.is_err() {
            eprintln!("spin: no thread reads the kernel's log");
        }
    }
    *watch = Some(child);
}

/// Stops reading the kernel's log once no spin lives.
fn stop_watching() {
    if LIVE.load(Ordering::SeqCst) != 0 {
        return;
    }
    if let Some(mut child) = WATCH.lock().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Whether `amdgpu.lockup_timeout`, `value`, leaves every ring [`LIMIT`]
/// at least: empty or 0 is the default, 2 s; a value under 0 is no limit;
/// one value is every ring's, four are the graphics, compute, SDMA and
/// video rings'.
fn limit_allows(value: &str) -> bool {
    value.split(',').all(|ms| match ms.trim().parse::<i64>() {
        Ok(ms) => ms <= 0 || u128::try_from(ms).is_ok_and(|ms| ms >= LIMIT.as_millis()),
        Err(_) => ms.trim().is_empty(),
    })
}

/// The device number of the render node at `path`.
fn render_node(path: &str) -> Option<(i64, i64)> {
    let rdev = std::fs::metadata(path).ok()?.rdev();
    // glibc's gnu_dev_major and gnu_dev_minor.
    let major = ((rdev >> 8) & 0xfff) | ((rdev >> 32) & !0xfff);
    let minor = (rdev & 0xff) | ((rdev >> 12) & !0xff);
    Some((i64::try_from(major).ok()?, i64::try_from(minor).ok()?))
}

fn failed(what: &str) -> impl FnOnce(vk::Result) -> String + '_ {
    move |result| format!("{what}: {result}")
}

/// Why no spin is made.
#[derive(Debug)]
pub(crate) enum NoSpin {
    /// The GPU has no memory that its shaders and the host both see while a
    /// shader runs, so no spin holds it, as its own answer says (module
    /// docs), and a test of a held GPU does not apply to it.
    Unheld(String),
    /// Anything else, which fails the test.
    Failed(String),
}

impl From<String> for NoSpin {
    fn from(why: String) -> Self {
        Self::Failed(why)
    }
}

/// Why no spin holds a GPU without uncached memory whose start the host
/// cannot see either ([`NoSpin::Unheld`]); what it lacks follows.
const UNHELD: &str = "no host-visible memory the GPU reads and writes uncached \
    (VK_AMD_device_coherent_memory's DEVICE_UNCACHED_AMD), which its caches would keep a spin's \
    start and the host's flag out of, and no timestamp by which a timed spin's start is seen";

impl Spin {
    /// The shader, ready on `ring` of the GPU of the render node at `node`.
    pub(crate) fn new(node: &str, ring: Ring) -> Result<Self, NoSpin> {
        let (major, minor) = render_node(node).ok_or_else(|| format!("no render node {node}"))?;
        // The spin is bounded by amdgpu's limit for a job on a ring (module docs): a GPU another
        // driver runs has no such limit, so no spin is made for it.
        let driver = std::fs::read_link(format!("/sys/dev/char/{major}:{minor}/device/driver"))
            .ok()
            .and_then(|path| Some(path.file_name()?.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "no driver the kernel names".to_owned());
        if driver != "amdgpu" {
            return Err(NoSpin::Unheld(format!(
                "{node} is {driver}'s, and a spin is held under amdgpu's lockup_timeout alone"
            )));
        }
        let limit = std::fs::read_to_string("/sys/module/amdgpu/parameters/lockup_timeout")
            .map_err(|e| format!("no amdgpu.lockup_timeout: {e}"))?;
        if !limit_allows(limit.trim()) {
            return Err(NoSpin::Failed(format!(
                "amdgpu.lockup_timeout {limit:?} is under {LIMIT:?}"
            )));
        }
        // SAFETY: loads the system's Vulkan loader.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("no Vulkan: {e}"))?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: a valid create info, for the call.
        let instance =
            unsafe { entry.create_instance(&info, None) }.map_err(failed("vkCreateInstance"))?;
        // On an error, what was made is left to the process's end: the
        // test fails.
        let gpu = Self::on(entry, instance, (major, minor), ring)?;
        LIVE.fetch_add(1, Ordering::SeqCst);
        watch();
        Ok(Spin {
            gpu: Arc::new(gpu),
            queued: Mutex::new(None),
        })
    }

    /// The device and the shader on `instance`'s device of the render node
    /// `node`.
    fn on(
        entry: ash::Entry,
        instance: ash::Instance,
        node: (i64, i64),
        ring: Ring,
    ) -> Result<Gpu, NoSpin> {
        // SAFETY: the instance's own calls, with valid out-pointers.
        let physicals = unsafe { instance.enumerate_physical_devices() }
            .map_err(failed("vkEnumeratePhysicalDevices"))?;
        let has = |physical, name: &CStr| {
            // SAFETY: a physical device of the instance.
            unsafe { instance.enumerate_device_extension_properties(physical) }
                .unwrap_or_default()
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let physical = physicals
            .into_iter()
            .find(|&physical| {
                if !has(physical, ash::ext::physical_device_drm::NAME) {
                    return false;
                }
                let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
                let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
                // SAFETY: a physical device of the instance, and a chain
                // of valid structures.
                unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
                drm.has_render != 0 && (drm.render_major, drm.render_minor) == node
            })
            .ok_or_else(|| format!("no Vulkan device of render node {node:?}"))?;
        if !has(physical, ash::khr::shader_clock::NAME) {
            return Err(NoSpin::Failed("no VK_KHR_shader_clock".into()));
        }
        // SAFETY: a physical device of the instance.
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let period = f64::from(properties.limits.timestamp_period);
        let ticks = (SPUN_AT_MOST.as_secs_f64() * 1e9 / period.max(1.0)) as u64;
        // SAFETY: as above.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = families
            .iter()
            .position(|f| {
                let flags = f.queue_flags;
                match ring {
                    Ring::Graphics => flags.contains(vk::QueueFlags::GRAPHICS),
                    Ring::Compute => {
                        flags.contains(vk::QueueFlags::COMPUTE)
                            && !flags.contains(vk::QueueFlags::GRAPHICS)
                    }
                }
            })
            .and_then(|f| u32::try_from(f).ok())
            .ok_or_else(|| format!("no {ring:?} queue family"))?;
        let coherent = has(physical, ash::amd::device_coherent_memory::NAME);
        let mut reset = vk::PhysicalDeviceHostQueryResetFeatures::default();
        let mut supported = vk::PhysicalDeviceFeatures2::default().push_next(&mut reset);
        // SAFETY: a physical device of the instance, and a chain of valid
        // structures.
        unsafe { instance.get_physical_device_features2(physical, &mut supported) };
        let timestamps = families[family as usize].timestamp_valid_bits > 0;
        let timeable = timestamps && reset.host_query_reset != 0;
        let mut extensions = vec![ash::khr::shader_clock::NAME.as_ptr()];
        if coherent {
            extensions.push(ash::amd::device_coherent_memory::NAME.as_ptr());
        }
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut clock =
            vk::PhysicalDeviceShaderClockFeaturesKHR::default().shader_device_clock(true);
        let mut uncached =
            vk::PhysicalDeviceCoherentMemoryFeaturesAMD::default().device_coherent_memory(true);
        let mut host_reset =
            vk::PhysicalDeviceHostQueryResetFeatures::default().host_query_reset(timeable);
        let features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .enabled_features(&features)
            .push_next(&mut clock)
            .push_next(&mut host_reset);
        if coherent {
            info = info.push_next(&mut uncached);
        }
        // SAFETY: a valid create info, for the call.
        let device = unsafe { instance.create_device(physical, &info, None) }
            .map_err(failed("vkCreateDevice"))?;
        // SAFETY: the family's one queue, made with the device.
        let queue = unsafe { device.get_device_queue(family, 0) };
        // The flag's buffer, in host-visible coherent memory, uncached on
        // the device where it can be.
        let buffer_info = vk::BufferCreateInfo::default()
            .size(FLAG_BYTES)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        // SAFETY: valid create infos, for each call; each object is
        // destroyed with the device, in `Gpu`'s drop, or below on an error.
        unsafe {
            let buffer = device
                .create_buffer(&buffer_info, None)
                .map_err(failed("vkCreateBuffer"))?;
            let needs = device.get_buffer_memory_requirements(buffer);
            let types = instance.get_physical_device_memory_properties(physical);
            let host =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let device_uncached = vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD
                | vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD;
            let find = |wanted: vk::MemoryPropertyFlags| {
                (0..types.memory_type_count).find(|&i| {
                    needs.memory_type_bits & (1 << i) != 0
                        && types.memory_types[i as usize]
                            .property_flags
                            .contains(wanted)
                })
            };
            // Uncached, the flag holds the spin; else, the spin is timed.
            let (memory_type, timed) = match (find(host | device_uncached), find(host)) {
                (Some(uncached), _) if coherent => (uncached, false),
                (_, Some(cached)) if timeable => (cached, true),
                _ => {
                    device.destroy_buffer(buffer, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                    let lacks = if timestamps {
                        "the host resets no query (hostQueryReset)"
                    } else {
                        "the queue family writes no timestamps"
                    };
                    return Err(NoSpin::Unheld(format!("{UNHELD}: {lacks}")));
                }
            };
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(needs.size)
                        .memory_type_index(memory_type),
                    None,
                )
                .map_err(failed("vkAllocateMemory"))?;
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(failed("vkBindBufferMemory"))?;
            let mapped = device
                .map_memory(memory, 0, FLAG_BYTES, vk::MemoryMapFlags::empty())
                .map_err(failed("vkMapMemory"))?
                .cast::<u8>();
            std::ptr::write_bytes(mapped, 0, FLAG_BYTES as usize);
            (*mapped.cast::<AtomicU32>()).store(1, Ordering::SeqCst);
            let turns_most = if timed {
                TIMED_TURNS_AT_MOST
            } else {
                TURNS_AT_MOST
            };
            let code = shader(ticks, turns_most);
            let module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .map_err(failed("vkCreateShaderModule"))?;
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(failed("vkCreateDescriptorSetLayout"))?;
            let set_layouts = [set_layout];
            let layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
                .map_err(failed("vkCreatePipelineLayout"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, e)| format!("vkCreateComputePipelines: {e}"))?[0];
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)];
            let pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(failed("vkCreateDescriptorPool"))?;
            let set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&set_layouts),
                )
                .map_err(failed("vkAllocateDescriptorSets"))?[0];
            let buffers = [vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(0)
                .range(FLAG_BYTES)];
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffers)],
                &[],
            );
            let commands = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                    None,
                )
                .map_err(failed("vkCreateCommandPool"))?;
            let command = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(commands)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(failed("vkAllocateCommandBuffers"))?[0];
            let query = if timed {
                let pool = device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(1),
                        None,
                    )
                    .map_err(failed("vkCreateQueryPool"))?;
                device.reset_query_pool(pool, 0, 1);
                Some(pool)
            } else {
                None
            };
            device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .map_err(failed("vkBeginCommandBuffer"))?;
            if let Some(pool) = query {
                // Written once the work ahead of the spin is done, just
                // before the spin.
                device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    0,
                );
            }
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(command, 1, 1, 1);
            device
                .end_command_buffer(command)
                .map_err(failed("vkEndCommandBuffer"))?;
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(failed("vkCreateFence"))?;
            Ok(Gpu {
                _entry: entry,
                instance,
                device,
                queue,
                memory,
                buffer,
                mapped,
                module,
                set_layout,
                layout,
                pipeline,
                pool,
                commands,
                command,
                fence,
                query,
                submitted: AtomicBool::new(false),
            })
        }
    }

    /// Queues the shader on its ring, held until [`Self::release`] or
    /// [`HELD_AT_MOST`] from now, and waits for it to begin on the GPU, a
    /// second at most: `Err` where it does not.
    pub(crate) fn hold(&self) -> Result<(), String> {
        self.queue()?;
        let start = StdRuntime.now();
        while !self.begun() {
            if StdRuntime.elapsed_since(start) > Duration::from_secs(1) {
                self.release();
                return Err("the spin did not begin in a second".into());
            }
            std::thread::sleep(Duration::from_micros(100));
        }
        Ok(())
    }

    /// Whether the shader last queued has begun on the GPU.
    pub(crate) fn begun(&self) -> bool {
        let gpu = &self.gpu;
        let Some(query) = gpu.query else {
            return gpu.word(BEGUN).load(Ordering::SeqCst) != 0;
        };
        // The timestamp, and whether it is written.
        let mut written = [[0u64; 2]];
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY;
        // SAFETY: the device's query, and room for its result and its
        // availability.
        let read = unsafe {
            gpu.device
                .get_query_pool_results(query, 0, &mut written, flags)
        };
        read.is_ok() || written[0][1] != 0
    }

    /// Whether the spin is timed: it holds its ring for [`SPUN_AT_MOST`]
    /// from its start, and [`Self::release`] does not end it (module docs).
    pub(crate) fn timed(&self) -> bool {
        self.gpu.query.is_some()
    }

    /// Queues the shader as [`Self::hold`] does, without waiting for it to
    /// begin: behind other work on its ring, it begins once that is done.
    pub(crate) fn queue(&self) -> Result<(), String> {
        if reset_seen() {
            return Err("the kernel's log said amdgpu reset".into());
        }
        let gpu = &self.gpu;
        if !gpu.ended(SPUN_AT_MOST) {
            return Err("the last spin has not ended".into());
        }
        gpu.word(FLAG).store(0, Ordering::SeqCst);
        gpu.word(BEGUN).store(0, Ordering::SeqCst);
        if let Some(query) = gpu.query {
            // SAFETY: the device's query, which no work in flight uses: the
            // last spin has ended.
            unsafe { gpu.device.reset_query_pool(query, 0, 1) };
        }
        let commands = [gpu.command];
        let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
        // SAFETY: the device's fence, signalled and waited for or never
        // submitted; its recorded command buffer, which is not in flight.
        unsafe {
            gpu.device
                .reset_fences(&[gpu.fence])
                .map_err(failed("vkResetFences"))?;
            gpu.device
                .queue_submit(gpu.queue, &submits, gpu.fence)
                .map_err(failed("vkQueueSubmit"))?;
        }
        gpu.submitted.store(true, Ordering::SeqCst);
        *self.queued.lock() = Some(StdRuntime.now());
        let held = Arc::clone(gpu);
        StdRuntime::thread("test-spin-release", move || {
            let start = StdRuntime.now();
            while StdRuntime.elapsed_since(start) < HELD_AT_MOST && !reset_seen() {
                if held.word(FLAG).load(Ordering::SeqCst) != 0 {
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            held.word(FLAG).store(1, Ordering::SeqCst);
        })
        .map_err(|e| format!("no thread lets the spin go: {e}"))?;
        Ok(())
    }

    /// How much of its hold the shader last queued has left at least: [`HELD_AT_MOST`] from its
    /// queueing, past which a thread of its own lets it go (a timed one holds to [`SPUN_AT_MOST`]
    /// from its start, which is later); none where none was queued. A host under load can spend
    /// it before a test's timed call is made, which then measures nothing.
    pub(crate) fn left(&self) -> Duration {
        (*self.queued.lock()).map_or(Duration::ZERO, |at| {
            HELD_AT_MOST.saturating_sub(StdRuntime.elapsed_since(at))
        })
    }

    /// Lets the shader end; a timed one ends by its own bound.
    pub(crate) fn release(&self) {
        self.gpu.word(FLAG).store(1, Ordering::SeqCst);
    }

    /// Waits for the shader to end, `most` at most: whether it has.
    pub(crate) fn ended(&self, most: Duration) -> bool {
        self.gpu.ended(most)
    }

    /// What the shader last held counted as it ended: how many times it
    /// read the flag, and how many ticks of the GPU's clock it spun.
    pub(crate) fn ran(&self) -> (u64, u64) {
        (self.gpu.long(TURNS), self.gpu.long(SPUN))
    }
}

impl Drop for Spin {
    /// Lets the shader end: a test that fails holds the GPU no longer.
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_ends_by_its_bounds() {
        let words = shader(800_000_000, TURNS_AT_MOST);
        assert_eq!(words[0], 0x0723_0203);
        // The two bounds, as 64-bit constants.
        let has = |v: u64| {
            words
                .windows(2)
                .any(|w| w[0] == (v & 0xffff_ffff) as u32 && w[1] == (v >> 32) as u32)
        };
        assert!(has(TURNS_AT_MOST));
        assert!(has(800_000_000));
        // Each bound ends the shader under amdgpu's limit: the turns at the
        // RX 7900 XTX's 0.95 µs a read (module docs).
        assert!(HELD_AT_MOST < SPUN_AT_MOST && SPUN_AT_MOST < LIMIT);
        assert!(Duration::from_nanos(TURNS_AT_MOST * 950) < LIMIT);
        // A timed spin's turns at the RX 550's 0.36 µs a read.
        assert!(Duration::from_nanos(TIMED_TURNS_AT_MOST * 360) < LIMIT);
        assert!(Duration::from_nanos(TIMED_TURNS_AT_MOST * 360) > SPUN_AT_MOST);
    }

    #[test]
    fn a_limit_under_two_seconds_refuses_the_spin() {
        for allowed in [
            "",
            "0",
            "2000",
            "10000",
            "-1",
            "10000,60000,10000,10000",
            "0,0,0,0",
        ] {
            assert!(limit_allows(allowed), "{allowed:?}");
        }
        for refused in ["1000", "10000,1000,10000,10000", "500,0", "x"] {
            assert!(!limit_allows(refused), "{refused:?}");
        }
    }

    #[test]
    fn a_reset_line_is_seen() {
        assert!(says_reset(
            "amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout, signaled seq=1"
        ));
        assert!(says_reset("amdgpu 0000:03:00.0: amdgpu: GPU reset begin!"));
        assert!(!says_reset(
            "amdgpu 0000:03:00.0: amdgpu: SMU is resumed successfully!"
        ));
        assert!(!says_reset("[UFW BLOCK] IN=enp5s0 OUT="));
    }

    /// Dumps the shader for spirv-val: `RVG_SPIN_SPIRV=PATH`. Without it,
    /// as in the VA-API host's run of the ignored tests, it writes nothing.
    #[test]
    #[ignore = "writes the shader where RVG_SPIN_SPIRV names, for spirv-val"]
    fn the_shader_is_written_out() {
        let Some(path) = std::env::var_os("RVG_SPIN_SPIRV") else {
            return;
        };
        let bytes: Vec<u8> = shader(800_000_000, TURNS_AT_MOST)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        std::fs::write(path, bytes).expect("written");
    }

    /// The spin holds its ring, its clock moving at the device's
    /// timestamp period, and ends when let go, or, timed, by its bound.
    #[test]
    #[ignore = "needs an AMD GPU: run on the VA-API host"]
    fn a_spin_holds_until_let_go() {
        let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
        for ring in [Ring::Graphics, Ring::Compute] {
            let spin = match Spin::new(&node, ring) {
                Err(NoSpin::Unheld(why)) => {
                    eprintln!("no spin holds {node}, as its answer says: {why}");
                    return;
                }
                made => made.expect("a spin"),
            };
            // A host under load can sleep through the hold's whole bound, which then says nothing
            // of the spin: such an attempt is made again, once the spin it held has ended.
            let mut attempts = 0;
            let start = loop {
                attempts += 1;
                spin.hold().expect("held");
                let start = StdRuntime.now();
                std::thread::sleep(Duration::from_millis(300));
                let left = spin.left();
                if left > Duration::from_millis(100) {
                    break start;
                }
                eprintln!("{ring:?}: {left:?} of the hold left after the sleep, so again");
                assert!(
                    attempts < 3,
                    "{ring:?}: no attempt of 3 slept within the hold"
                );
                spin.release();
                assert!(spin.ended(Duration::from_secs(2)), "{ring:?}: not ended");
            };
            assert!(!spin.ended(Duration::ZERO), "{ring:?}: ended unasked");
            spin.release();
            assert!(spin.ended(Duration::from_secs(2)), "{ring:?}: not ended");
            let wall = StdRuntime.elapsed_since(start);
            let (turns, ticks) = spin.ran();
            let timed = spin.timed();
            eprintln!("{ring:?}: {turns} reads, {ticks} ticks in {wall:?}, timed {timed}");
            if timed {
                // Held to its bound, by the clock rather than the reads.
                assert!(wall > SPUN_AT_MOST / 2, "{ring:?}: ended in {wall:?}");
                assert!(turns < TIMED_TURNS_AT_MOST, "{ring:?}: ended by its reads");
            }
            assert!(!reset_seen());
        }
    }
}
