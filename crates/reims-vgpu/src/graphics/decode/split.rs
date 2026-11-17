//! A command buffer committed in parts as it is walked, so the GPU starts on its first encoders
//! while the host walks the rest (docs/graphics-kext/design.md, "A command buffer in parts").
//!
//! Where the platform starts a command buffer's GPU work only at its commit (the Vulkan
//! platform's runtime records it and submits it then), a long buffer's walk and its GPU work run
//! in series: WindowServer's composite of a Liquid Glass desktop took a walk of 2.6 to 3.5 ms and
//! then 1.6 to 3.3 ms of GPU work a frame on an RTX 5080 Laptop GPU (measured),
//! against a refresh of 8.33 ms. So at an encoder's end, once the part walked so far took
//! [`SPLIT_AFTER`], the walk hands the platform that part (`CommandBuffer::split`), which commits
//! it on the buffer's queue and answers a command buffer of the queue that goes on from it, and
//! the walk goes on in that one. The platform keeps a buffer whole where a part would answer
//! otherwise than the whole (its docs list where).
//!
//! **What it gains** (measured on the M3 Max under macOS 27.0.1, the Vulkan platform on
//! MoltenVK, release, by `composite_cost`: a buffer of 60 render passes of 3 blended quads each,
//! frames one at a time, 200 a run, two runs). From the take of a frame's message to its
//! `CB_COMPLETE`, the median fell from 9.5-9.7 ms whole to 6.0 ms in parts over a 3840 x 2400
//! target, every 4th pass reading it; from 8.5-8.8 to 4.7 ms over 1920 x 1200; and from 10.5-10.6
//! to 7.0 ms over 1920 x 1200 with every pass clearing its target and reading it first. More
//! than the walk itself (1.3-1.5 ms) came off: the runtime's own work at a commit, its recording
//! and submission of the buffer's batches, runs beside the walk too. The walk took 1.5% more
//! instructions. Parts walked [`SPLIT_AFTER`] at least were the fastest of 100, 250 and 500 µs in
//! each shape and run: 100 µs cost the walk 5% more instructions and gained nothing, 500 µs left
//! 0.2-0.4 ms on the table.
//!
//! A buffer of compute encoders alone is cut too, its unit of command predication going on in the
//! next part (`gpu/command.rs` of the Vulkan platform, "A command buffer in parts"). Measured the
//! same way by `dispatch_cost`'s `fluid` shape, a buffer shaped like Geekbench 7's Fluid
//! Simulation's, of 2,600 encoders of one dispatch each, 6 a run, five runs: from the first
//! message's take to its completion, the median fell from 87.0 ms whole to 34.8 ms in parts, and
//! the walk's CPU time a dispatch from 14.7 to 8.8 µs (164.2k to 110.6k instructions): the runtime
//! measured (metal2vulkan c796a88d6a) does more work for an encoder the more encoders its command
//! buffer holds before it, and a part holds fewer.
//!
//! **How many parts.** A part is cut each [`SPLIT_AFTER`] while nothing the buffer's queue
//! committed, its earlier parts and buffers, is incomplete (`QueueCounts::out`, counted from each
//! commit to its completion's handler), up to its queue's room for command buffers
//! (`TAKE_IN_FLIGHT_MAX`, [`super::cmdbuf`]). Behind such work a part is cut once its estimated
//! GPU time reaches [`PART_GPU`] ([`Estimate::cuts`]): its walk times the rate of GPU time to
//! walk time that the buffer's parts seen complete measured, else its queue's last walk in parts
//! ([`QueueRate`]); with neither, each [`SPLIT_AFTER`], until a part is seen complete; and never
//! where its parts had no GPU times. Its walk is cut at [`PART_WALK_MOST`] whatever the estimate:
//! a compute encoder's GPU time follows its grid and not its walk, so the parts before measure
//! the next one poorly. On the UHD Graphics 630, consecutive parts of an ML burst's buffer took
//! 0.4 to 102 times their walk, and the rate its first parts measured let a later one run for
//! 16.6 ms of walk, which took 642 ms of GPU time (lane splitcut, m3-max-38430). Such a part is committed unmarked ("Its parts'
//! completions", below), so its completion is seen at its own end. A buffer dealt to two threads
//! behind such work is dealt in two chunks, one a thread ("Two threads at once", below).
//! Each part is a submission to the driver and a job of its scheduler, and a part committed
//! behind work only adds one where the GPU keeps up with the walk: the GPU starts it no earlier
//! than the work before it ends. On an
//! RX 7900 XTX, where a job waits 0.12 to 0.19 ms from its submission to its start on the GPU
//! (amdgpu's scheduler, measured by metal2vulkan's jobstart lane), Geekbench 7's Metal pass took 1,437 parts
//! for Particle Physics' 111 command buffers of its own and 2,631 for Fluid Simulation's 882,
//! 1,256 and 1,703 of their cuts made while an earlier part was not yet seen complete, where
//! Geekbench's own Vulkan pass on the host ran 111 and 882 jobs (perfscout9's gb-xtx-2 and its
//! native run, measured). On an Iris Xe,
//! WindowServer's frames of an animating page took 8.96 parts a command buffer, 22,155 of
//! 25,856 cuts behind incomplete work, 847 of its 2,394 dealt walks begun behind it (lane joins'
//! ja-dev, measured). So a part behind work is cut only where its GPU time is long: where the GPU
//! is far slower than the walk, a walk that cut nothing behind its first part went on into parts
//! of several hundred milliseconds of GPU time. On the UHD Graphics 630 of a Mac mini 2018 host
//! (Windows, macOS 26.7.2 guest), an ML burst's command buffer of 794 dispatches in 582 compute
//! encoders went to the GPU in three parts of 22.3, 855.1 and 923.8 ms, one of 541 dispatches in
//! parts of 0.2, 66.2 and 507.3 ms, and the compositor's buffers committed behind them waited 442
//! to 523 ms to start, the desktop's frames 550 ms apart (m3-max-38035, read by lane splitcut).
//!
//! **How far ahead of the GPU.** Every command queue of the guest's may share one `VkQueue`
//! (metal2vulkan's `queue_share.rs`: RADV's single graphics queue, any device with fewer queues
//! than the guest's), whose work runs in its submission order, and the runtime submits each
//! command buffer's batches as they are ready, at its commit. So a buffer of another queue,
//! WindowServer's among them, committed after a walk's parts runs after all of them: in the burst
//! above, the walk committed its parts within 39 ms, and the probe's buffers committed 1.3 s
//! later still waited 460 ms for the last. Cutting smaller parts alone would not have helped
//! them. A walk on its queue's lane therefore waits, after a cut, while two or more of its parts
//! are out and their estimated GPU time is over [`PACE_AHEAD`] ([`Estimate::waits`]; with
//! nothing to estimate it from yet, while their walk is over [`PACE_UNKNOWN_WALK`]), or the
//! oldest of them was committed [`PACE_AHEAD`] ago or more, and goes on as its parts are seen
//! complete. The oldest's age is what the GPU itself shows: an estimate from the parts seen
//! complete is blind while none completes, as the GPU runs a part far longer than it estimated.
//! In the burst above, a buffer of 541 dispatches walked parts that its first four, of 0.0 to 0.6
//! times their walk, estimated at 0.6 ms each, and that took 164, 23, 55, 38 and up to 120 ms,
//! none seen complete as its 22 parts were committed within 44 ms (lane splitcut, m3-max-38627).
//! One part out never holds the walk, so the GPU runs a part while the walk walks the next, and
//! a buffer of another queue committed meanwhile waits for about two of the walk's parts at most
//! ([`PART_GPU`]), or for an encoder longer than that, which no cut divides. Where the GPU keeps
//! up with the walk, its parts complete well within [`PACE_AHEAD`] and nothing waits. The wait is
//! the lane's job's (`Lane::waiting`), so the lane's next buffer walks meanwhile, and only a
//! lane's thread waits: never a queue's service thread or a doorbell's, which walk only short
//! buffers.
//!
//! A walk is never paced once it waited for or signalled an event by an op: a part's completion
//! may then wait for a buffer committed only after the walk's last part, as one whose process
//! signals the event once it sees the walk's buffer scheduled; and a buffer that waits for an
//! event by an op is not cut at all ("Which buffers"). Nor is it paced once a buffer of its
//! context waits for its last commit: a commit at the gate below, a buffer dealt to two threads
//! that is to take the gate, a commit of another of the process's queues that names the walk's
//! queue in the commit order ([`super::commit_order`]), or one that waits for the context's
//! tickets or range messages ([`super::sequencer`]). Such a buffer would wait for a paced walk's
//! end, and a part may wait for it on the GPU: a part's waits from its begin, for a cache's fill
//! (which waits for the buffers begun that use the cache) or for another queue's buffer in the
//! surface order, are for buffers committed or still to be committed, and the walk goes on
//! unpaced once one of those waits for it. A wait ends at once where a reset of the device
//! begins, the buffer's queue ends or a part fails, within a [`WAIT_SLICE`] where such a buffer
//! waits, and once [`PACE_WAIT_MOST`] passes with none of its parts seen complete; the walk then
//! goes on unpaced to its end, cutting and committing its parts as before. A lost device fails
//! its parts, whose completions end the wait. What the guest sees changes only in time: a paced
//! buffer's last part, and so its `CB_SCHEDULED` where it waits for that, comes once the GPU has
//! nearly finished its earlier parts rather than at the walk's speed.
//!
//! Before that rule, a limit on the parts out, under which the walk went on into a longer part
//! while the GPU was behind, gained nothing measured by the decoder alone: `dispatch_cost`
//! (release, the median of 5 runs of 6 repetitions) measured its fluid shape and its heavy one,
//! whose GPU work outlasts its walk as Fluid Simulation's does on a Radeon 780M, with at most 2
//! or 16 parts out against no limit, on metal2vulkan afba68e6e4:
//! - Radeon 780M (RADV): the heavy shape's latency 166.8 and 167.0 ms against 166.2 ms, its walk
//!   6.6 and 7.0 µs a dispatch against 7.3 µs; the fluid shape's GPU keeps up, so neither limit
//!   was reached;
//! - Iris Xe (ANV): heavy 252.4 and 252.2 ms against 253.0 ms; fluid as on the 780M;
//! - M3 Max (MoltenVK): the latency rose from 33.8 to 44.3 and 35.7 ms (fluid) and from 121.9 to
//!   132.2 and 127.2 ms (heavy), the cuts waiting for MoltenVK's late completions. Parts that
//!   double in length (250, 500, 1,000 µs and on) lost there too: 45.6 and 139.2 ms against 34.4
//!   and 122.3 ms.
//!
//! A walk in parts takes no more instructions than a whole one, but it stalls longer. Each
//! part's commit, and its GPU work beside the walk, cost the walk 110 to 210 thousand cycles on
//! the Iris Xe and the Radeon 780M (the fluid shape in 88 and 72 parts, against whole).
//! At metal2vulkan bc12f6d95a a busy GPU stalled the walk far more: on the Radeon 780M the heavy
//! shape's walk took 55.8 thousand cycles a dispatch against fluid's 34.5 thousand, for 55.7 and
//! 59.0 thousand instructions. Fluid Simulation's walk there took 33.6 µs a dispatch in parts
//! against 17.2 µs whole (perfscout5's dp-gbr-780m against perfscout4's gbr-780m). At
//! afba68e6e4 the heavy shape's walk takes 30.2 thousand cycles against fluid's 33.3 thousand.
//!
//! **What stays as it was.** The parts run in their order on the queue, as the encoders of one
//! buffer do, and each part is committed only where the whole buffer could be committed then,
//! waiting for nothing (`GraphicsDecoder::may_commit_now`): its turn on its queue is next, every
//! ticket and range message before its begin is done, and the buffers its `COMMIT_ORDER` names
//! are committed. Every wait of the last commit was met at the first part, so the walk waits
//! between parts for nothing but its own parts' completions where it is paced ("How far ahead of
//! the GPU"). Its handlers, records and tickets are the last part's, and come as they
//! came: its `CB_SUBMITTED`, `CB_SCHEDULED` and `CB_COMPLETE` once each, its ticket given back
//! at the last commit, its commit marks raised there. Its outcome is all the parts' ([`Parts`]):
//! the first failed part's error, else the last part's; and GPU times that start at the first
//! part's start and span the parts' GPU time summed, never ending after the last part's end
//! (below, "Its GPU times"). A signal of a shared event stays in the part it was encoded in, before the
//! cut, and its value goes to the guest as the event reaches it, as a whole buffer's does; one
//! of a shared event whose create the host refused goes with the buffer's completion, after its
//! last part. A buffer whose walk faults after a part has its failure at its stand-in's
//! completion, as any buffer whose walk faults; the parts committed before the fault ran, as the
//! encoders before a fault of Metal's GPU run.
//!
//! **Its GPU times.** Each part runs once the walk has handed it over, so the GPU waits between
//! parts for the walk of the next: from the first part's start to the last part's end the buffer
//! spans the host's walk of it, which a Mac, whose GPU is handed the whole buffer at its commit,
//! never waits for. The guest's compositor reads these times: macOS's SkyLight and QuartzCore,
//! which composite WindowServer's frames, reference `GPUStartTime` and `GPUEndTime` (read from
//! their selector references on macOS 27), and the bundle answers them with the record's span
//! (`rgfxNoteGPUTimes:`). On a Mac one buffer spans its encoders' GPU time summed: on both x86
//! Macs (macOS 26.7.1, an AMD Radeon Pro 5300M and an Intel UHD Graphics 630), a buffer of 8 or
//! 32 compute encoders or render passes spanned what the same encoders committed one buffer each
//! summed (AMD: 32 compute encoders 6.31 ms whole, 7.37 summed; 32 passes 11.04 and 10.89; Intel:
//! 13.05 and 12.41; 59.56 and 59.99), the CPU's time encoding it never counted, and its
//! `GPUStartTime` came 0.08 to 0.40 ms after its commit (measured, `oracle_cmdbuf DEVICE
//! gputimes`; docs/graphics-kext/dgpu-oracle.md, Q7). So the record starts at the first part's
//! start, which the GPU reached soonest after the commit, and spans the parts' GPU time summed,
//! each part's end less its start: the time the buffer's work kept the GPU, as a buffer committed
//! whole answers it. It never ends after the last part's end, where parts that overlap on the GPU
//! sum to more.
//!
//! So the record's end (`gpu_end_ns`) falls before the last part's own end by the GPU's waits
//! between parts, and no guest code reads it as a time: the bundle answers `GPUEndTime` with the
//! guest's own clock as it takes the record, and `GPUStartTime` with the record's end less its
//! start before that, no earlier than the buffer's post (`rgfxNoteGPUTimes:`, held by the bundle's
//! `test_reply_reader_completion`). The guest so sees the parts' GPU time summed as the buffer's
//! run, ending after its last part's end by the record's delivery, as a buffer committed whole
//! does. In steady Launchpad on a Radeon 780M (macOS 26.7.2, metal2vulkan 574223a971, 11-16), 63%
//! of the guest's command buffers, all WindowServer's, were committed in parts (16 at the median,
//! 28 at most). Their gaps came to 2.3 to 2.6 ms at the median, and their records were posted 79
//! to 80 us after the last part's own end at the median and 0.44 to 0.95 ms at the 99th
//! percentile, against 66 to 69 and 115 us for a buffer committed whole (measured, lane splitend's
//! walk sb-base1, m3-max-35830). Of the readers of these times in macOS 27's frameworks
//! (QuartzCore's `CA::OGL::MetalContext` completed handlers, SkyLight's hitch telemetry and debug
//! HUD, RenderBox's frame statistics), none paces frames from them (read from their code).
//!
//! **Its parts' completions.** Each part before the last is marked to join the next one's
//! completion as it is committed (`CommandBuffer::set_completion_joins_next`), so the platform may
//! send several of a queue's parts found ready together, and the part behind them, to its driver
//! in one submission, rather than each in a submission of its own: each costs the driver's
//! submission, which grows with the memory the part's work lists. A marked part's work, its order
//! on the queue and its GPU times are unchanged, but it is seen complete, and its handlers run,
//! only once the last of its submission completed, late by the GPU time of at most three
//! batches of work after it (the runtime's bound of a submission). Nothing
//! observes that but its own handler ([`Parts::part_handler`]), which notes its outcome for the
//! last part's and lets its place among its queue's command buffers go, a hold the guest never
//! sees. So a part is marked only where its completion carries nothing else: never the last part,
//! which carries the buffer's handlers, records, tickets and its queue's surface signal; never a
//! part that waits for or signals an event, at its begin (a device cache's fill, a buffer of
//! another queue in the surface order) or by an op, whose value another queue's GPU, the guest
//! or the device's listener waits for; never a part cut behind incomplete work its queue committed
//! ("How many parts"), whose completion a paced walk waits for ("How far ahead of the GPU"); and
//! a part the platform keeps whole is unmarked again, as its commit is then the whole's. A fence a marked part updates is reached as late as its
//! completion: the GPU's order of its work is unchanged, and a wait of another queue's for it is
//! met at the end of the part's submission, never early. A fault in a marked part fails the
//! buffer as before: its error reaches the last part's record, whose handler runs after every
//! earlier part's in the queue's commit order. The guest sees its command buffer complete at its
//! last part's own end, as before: the platform sends the first unmarked command buffer behind
//! marked ones last in their submission, seen complete at its own end, and never holds it for
//! later work.
//!
//! **What another message sees** ([`SplitGate`]). While a buffer of a context has parts out, from
//! its first part's commit to its last, no other command buffer of the context is committed, and
//! the context's range messages that come are carried out after its last commit: each other
//! message comes wholly before the buffer's parts or wholly after them, as it would before a
//! whole buffer's commit or after it ("What a walk sees"). A buffer splits only while no other
//! commit or range message of its context is under way; the others' commits wait at their
//! commit, never the walk, and ring 0 never waits.
//!
//! **Which buffers.** One of a context, whose walk is not ordered on its queue (a video buffer's
//! is), that waits for no event and is behind no buffer of its queue that does ([`super::cmdbuf`]'s
//! `Held`), and whose queue holds room for another command buffer, so the platform's make of the
//! next part waits for none.
//!
//! **Two threads at once.** A walk in parts still runs on one thread, and the runtime's lowering
//! is most of it: WindowServer's frame in Launchpad's open view, one command buffer of about 107
//! render passes, took a walk of 14.9 ms at the median, 91% of it in the runtime's calls, so its
//! present was answered two refreshes on (an RTX 3090 host, every driver listed, metal2vulkan
//! pin 92; m3-max-30753, measured). So a buffer that may be cut, of
//! [`DEAL_MIN_BYTES`](super::cmdbuf::DEAL_MIN_BYTES) of ops and
//! [`DEAL_MIN_ENCODERS`](super::cmdbuf::DEAL_MIN_ENCODERS) at least, all of them render and
//! blit encoders that hold no event, rate map, indirect command buffer or residency of
//! resources (`cmdbuf.rs`'s `deal`), is cut at encoder ends into chunks of
//! [`DEAL_CHUNK_BYTES`](super::cmdbuf::DEAL_CHUNK_BYTES) of ops or more (into two chunks, at the
//! first encoder's end past half its ops, where its walk starts behind work its queue committed
//! and not yet complete: "How many parts"), dealt to the lane's
//! thread and a helper thread ([`super::helpers`]), handed over at once as the walk starts: the
//! first chunk to the lane's thread, then the longest first, each to the thread dealt fewer bytes
//! so far (`cmdbuf.rs`'s `helper_chunks`), so the longest starts at once on the thread that is
//! free (chunks of one length go in turn, the odd ones to the helper), and each thread walks its
//! chunks in their order. Each chunk goes into a command buffer of its own on the
//! buffer's queue, which the platform makes before any chunk is walked (`CommandBuffer::part`),
//! each while the queue holds room, as a cut's make does, so no make waits on the queue's room
//! while parts of the buffer wait uncommitted (where the room runs out first, the buffer is
//! walked as a whole is); the first chunk goes into the buffer's own;
//! the helper walks its chunks on a handle whose view of the context's table is a twin of the
//! buffer's (`View::twin`), so it names what the buffer names. After each of its own chunks the
//! lane's thread commits every chunk walked, in their order, each but the last as a part
//! (`CommandBuffer::commit_part`), waiting for the helper only for the chunk it is to commit
//! next, so the GPU runs each part while the rest is walked: a split into two ranges walked at
//! once, the later range's work reaching the GPU only at their join, took the walk's end from
//! 14.8 to 8.2 ms but left the later range's GPU work, 3.7 ms, after it (m3-max-31216 against
//! m3-max-31226). A part goes where the whole buffer could be committed then, as a cut does; at
//! the walk's end, once every wait of the buffer's commit is met. The last chunk's command buffer
//! is the buffer's last part. The runtime takes command buffers of one queue encoded on several
//! threads at once and committed in order from one (metal2vulkan's answer, 11-14).
//!
//! What stays as it was is what a walk in parts keeps (above): the parts run in their order on
//! the queue, their outcome is all of theirs, the last part carries the buffer's handlers,
//! records and tickets, and the context's gate is held from its first part's commit to its last.
//! No chunk signals or waits for an event, so no value goes to the guest from a helper. A walk
//! that faults or stops in a chunk answers as the first such chunk in their order does, as a
//! whole walk that faults there after its parts: the parts committed before it ran, and nothing
//! at or after it is committed. Where a part fails from one of its encoders on (a draw of
//! primitive type 9, a store into read-only memory), the platform keeps it whole and every later
//! chunk's command buffer is folded into it in their order (`CommandBuffer::fold`): every
//! encoder after the failing one runs nothing, in it as in a whole buffer, and its commit answers
//! as one buffer's. Chunks no helper has started by the end of the lane's first are taken back
//! and walked by the lane's thread into the buffer's current part, as a walk goes on, so a walk
//! never waits for a helper to come free. Each op is walked once, on one thread or the other:
//! what a walk does once a buffer (its names found, its uses of laid-out device caches told at
//! the commit, in the chunks' order) is done once. A `cbpar` trace event gives each buffer's
//! chunks and their walks, and a `cbdeal` event each chunk's.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reims_vgpu_vulkan::core::platform::command::{CompletedHandler, CompletedInfo};
use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, NsErrorText};
use reims_vgpu_vulkan::core::sync::{Condvar, Mutex};

use super::commit_order::WAIT_SLICE;
use super::sequencer::RangeOp;

/// How long a part of a command buffer is walked, at least, before the walk commits it at an
/// encoder's end: each part costs its commit and the next part's make on the walk's thread, and
/// the GPU's wait for the part before it at their boundary. The fastest of those measured (the
/// module docs, "What it gains").
pub(crate) const SPLIT_AFTER: Duration = Duration::from_micros(250);

/// The GPU time, as estimated ([`Estimate`]), a part cut behind its queue's incomplete work is
/// walked to ("How many parts"): a refresh at 60 Hz, as [`PACE_AHEAD`], so a buffer of another
/// queue committed behind a paced walk's parts waits for about two of them at most. Each part
/// costs its boundary: on a Radeon 780M (RADV), `dispatch_cost`'s heavy shape cut behind work at
/// 4 ms took 51.0 parts a command buffer against 3.0, and 186.4 ms against 181.8 ms, and at
/// 16 ms 14.3 parts and 183.5 ms against 181.5 ms (lane splitcut, m3-max-38121 and
/// m3-max-38356); Geekbench 7's Fluid Simulation took 2% more GPU time at 4 ms (m3-max-38192
/// against m3-max-38191); an Iris Xe and the M3 Max lost nothing. A boundary's own GPU cost there
/// is smaller, a full barrier at the part's end and two timestamps, about 34 µs of span at the
/// median (metal2vulkan's fills of 64 MiB in 3 or 51 parts, m3-max-38426); the rest is not
/// measured, the overlap of independent dispatches lost across a cut its expected cause. Chosen
/// from those.
pub(crate) const PART_GPU: Duration = Duration::from_millis(16);

/// The longest a part cut behind its queue's incomplete work is walked, whatever its estimated
/// GPU time ("How many parts"): on the UHD Graphics 630, a rate of GPU time to walk time of 100
/// makes it 200 ms of GPU time, as long as one of its longest compute encoders; on a Radeon 780M,
/// `dispatch_cost`'s heavy shape reaches [`PART_GPU`] in less. Chosen, not measured.
pub(crate) const PART_WALK_MOST: Duration = Duration::from_millis(2);

/// The GPU time, as estimated, of a walk's parts committed and not seen complete above which the
/// walk waits before it goes on, and the age of the oldest of them at which it waits whatever
/// the estimate ("How far ahead of the GPU"): a refresh at 60 Hz, which a compositor's buffer
/// committed behind them waits at most besides the part the GPU is running. Chosen, not
/// measured.
pub(crate) const PACE_AHEAD: Duration = Duration::from_millis(16);

/// The walk of a buffer's parts committed and not seen complete above which the walk waits,
/// where nothing estimates their GPU time yet (no part of the buffer is seen complete, and no
/// buffer of its queue was measured): on the UHD Graphics 630 of a Mac mini 2018 host, a part of
/// an ML burst's compute encoders took 37 to 69 times its walk in GPU time (lane splitcut's
/// reading of m3-max-38035), so 1 ms of walk is about 55 ms of its GPU's. Chosen, not measured.
pub(crate) const PACE_UNKNOWN_WALK: Duration = Duration::from_millis(1);

/// The longest a walk waits at one cut with none of its parts seen complete ("How far ahead of
/// the GPU"), after which it goes on unpaced to its end, as a walk did before it was paced: a
/// bound on a wait whose completion never comes. A part's own GPU time is what a wait for it
/// needs, and on the UHD Graphics 630 of a Mac mini 2018 host one compute encoder of an ML burst,
/// which no cut divides, took 267 ms (lane splitcut, m3-max-38307); twice that. Chosen, not
/// measured.
pub(crate) const PACE_WAIT_MOST: Duration = Duration::from_millis(500);

/// What a walk in parts estimates a part's GPU time from ("How far ahead of the GPU").
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Estimate {
    /// GPU time a walk time: the buffer's parts' seen complete with GPU times, summed, against
    /// their walks summed; else the last such rate a walk of its queue measured.
    Rate(f64),
    /// None yet: no part of the buffer is seen complete, and no walk of its queue measured a rate.
    Unknown,
    /// None: the buffer's parts seen complete had no GPU times.
    Untimed,
}

impl Estimate {
    /// Whether a walk cuts its part at an encoder's end, the part walked `walked`, `behind`
    /// incomplete work its queue committed ("How many parts").
    pub(crate) fn cuts(self, walked: Duration, behind: bool) -> bool {
        if !behind {
            return walked >= SPLIT_AFTER;
        }
        match self {
            Estimate::Rate(rate) => {
                walked.as_secs_f64() * rate >= PART_GPU.as_secs_f64() || walked >= PART_WALK_MOST
            }
            Estimate::Unknown => walked >= SPLIT_AFTER,
            Estimate::Untimed => false,
        }
    }

    /// Whether a walk waits before it goes on, `out` of its parts committed and not seen complete,
    /// walked `walked` summed, the oldest of them committed `oldest` ago ("How far ahead of the
    /// GPU"): their estimated GPU time is over [`PACE_AHEAD`], or the oldest is that old. One
    /// part out never holds a walk: the GPU runs it while the walk walks the next.
    pub(crate) fn waits(self, out: usize, walked: Duration, oldest: Duration) -> bool {
        if out < 2 {
            return false;
        }
        match self {
            Estimate::Rate(rate) => {
                walked.as_secs_f64() * rate > PACE_AHEAD.as_secs_f64() || oldest >= PACE_AHEAD
            }
            Estimate::Unknown => walked > PACE_UNKNOWN_WALK || oldest >= PACE_AHEAD,
            Estimate::Untimed => false,
        }
    }
}

/// How long a walk's wait has gone with none of its parts seen complete ([`PACE_WAIT_MOST`]):
/// how many were seen complete when the count last rose, and when that was.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Progress {
    completed: Option<u64>,
    at: MonoTime,
}

impl From<MonoTime> for Progress {
    /// A wait that began at `began`.
    fn from(began: MonoTime) -> Progress {
        Progress {
            completed: None,
            at: began,
        }
    }
}

impl Progress {
    /// The time since the wait's begin or the last part seen complete, at `now`, with
    /// `completed` of its parts seen complete so far.
    pub(crate) fn since(&mut self, completed: u64, now: MonoTime) -> Duration {
        match self.completed {
            Some(seen) if seen == completed => {}
            None => self.completed = Some(completed),
            Some(_) => {
                self.completed = Some(completed);
                self.at = now;
            }
        }
        now.saturating_duration_since(self.at)
    }
}

/// A queue's last rate of GPU time to walk time, as a walk in parts of one of its buffers
/// measured it ([`Estimate::Rate`]), its `f64`'s bits; 0 before the first.
#[derive(Default)]
pub(crate) struct QueueRate(AtomicU64);

impl QueueRate {
    /// The last rate measured, where one was.
    pub(crate) fn get(&self) -> Option<f64> {
        let bits = self.0.load(Ordering::Relaxed);
        (bits != 0).then(|| f64::from_bits(bits))
    }

    /// Notes `rate`, a walk's measure.
    pub(crate) fn set(&self, rate: f64) {
        if rate.is_finite() && rate > 0.0 {
            self.0.store(rate.to_bits(), Ordering::Relaxed);
        }
    }
}

/// The contexts with a buffer whose parts are out, and the commits and range messages of each
/// context under way (the module docs).
#[derive(Default)]
pub(crate) struct SplitGate {
    contexts: Mutex<HashMap<u32, Gate>>,
    changed: Condvar,
}

/// One context's gate.
#[derive(Default)]
struct Gate {
    /// A buffer of the context has parts out.
    split: bool,
    /// The context's commits and range messages under way.
    under_way: usize,
    /// The range messages that came while `split`, carried out at its end in the order they
    /// came.
    deferred: Vec<RangeOp>,
    /// The commits waiting for `split`'s end.
    waiting: usize,
    /// The buffers waiting to take the gate for their parts ([`SplitGate::split`]).
    splitting: usize,
}

impl SplitGate {
    /// A command buffer of `context` is to be committed: waits, with no lock of the caller's
    /// held, while a buffer of the context has parts out, and holds a split of the context off
    /// until the answer drops, after the commit.
    pub(crate) fn commit(self: &Arc<Self>, context: u32) -> UnderWay {
        let mut contexts = self.contexts.lock();
        if let Some(g) = contexts.get_mut(&context).filter(|g| g.split) {
            g.waiting += 1;
            while contexts.get(&context).is_some_and(|g| g.split) {
                self.changed.wait(&mut contexts);
            }
            contexts.entry(context).or_default().waiting -= 1;
        }
        contexts.entry(context).or_default().under_way += 1;
        UnderWay {
            gate: Arc::clone(self),
            context,
        }
    }

    /// Carries out `op`, a range message of `context`'s: now, holding a split of the context off
    /// meanwhile, unless a buffer of the context has parts out, after whose last commit it is
    /// carried out then, on that thread. It waits for nothing.
    pub(crate) fn range(self: &Arc<Self>, context: u32, op: RangeOp) {
        {
            let mut contexts = self.contexts.lock();
            let gate = contexts.entry(context).or_default();
            if gate.split {
                gate.deferred.push(op);
                return;
            }
            gate.under_way += 1;
        }
        let under_way = UnderWay {
            gate: Arc::clone(self),
            context,
        };
        op();
        drop(under_way);
    }

    /// The gate of `context` for a buffer's parts, where no buffer of the context has parts out
    /// and no commit or range message of it is under way; `None` otherwise.
    pub(crate) fn try_split(self: &Arc<Self>, context: u32) -> Option<SplitHold> {
        let mut contexts = self.contexts.lock();
        let gate = contexts.entry(context).or_default();
        if gate.split || gate.under_way > 0 {
            return None;
        }
        gate.split = true;
        Some(SplitHold {
            gate: Arc::clone(self),
            context,
        })
    }

    /// The gate of `context` for a buffer's parts, once no buffer of the context has parts out
    /// and no commit or range message of it is under way: waits, with no lock of the caller's
    /// held. Taken by a buffer dealt to two threads at once ("Two threads at once") whose parts
    /// could not take it as they were walked ([`Self::try_split`]), at the walk's end, once every
    /// wait of its commit is met: a buffer whose parts are out commits its last part, and a
    /// commit or range message under way ends, waiting for nothing of it.
    pub(crate) fn split(self: &Arc<Self>, context: u32) -> SplitHold {
        let mut contexts = self.contexts.lock();
        loop {
            let gate = contexts.entry(context).or_default();
            if !gate.split && gate.under_way == 0 {
                gate.split = true;
                return SplitHold {
                    gate: Arc::clone(self),
                    context,
                };
            }
            gate.splitting += 1;
            self.changed.wait(&mut contexts);
            contexts.entry(context).or_default().splitting -= 1;
        }
    }

    /// The gate of `context` is free and holds nothing, so its entry goes.
    fn prune(contexts: &mut HashMap<u32, Gate>, context: u32) {
        if contexts.get(&context).is_some_and(|g| {
            !g.split
                && g.under_way == 0
                && g.deferred.is_empty()
                && g.waiting == 0
                && g.splitting == 0
        }) {
            contexts.remove(&context);
        }
    }

    /// Whether a buffer of `context` waits for the last commit of the context's buffer whose parts
    /// are out: a commit, or a buffer dealt to two threads that is to take the gate for its parts
    /// ([`Self::split`]).
    pub(crate) fn holds_back(&self, context: u32) -> bool {
        (self.contexts.lock().get(&context)).is_some_and(|g| g.waiting > 0 || g.splitting > 0)
    }

    /// How many commits of `context` wait for a buffer's parts (for the tests).
    #[cfg(test)]
    pub(crate) fn waiting(&self, context: u32) -> usize {
        self.contexts.lock().get(&context).map_or(0, |g| g.waiting)
    }
}

/// A commit or range message of a context under way ([`SplitGate::commit`],
/// [`SplitGate::range`]), which holds a split of the context off until it drops.
pub(crate) struct UnderWay {
    gate: Arc<SplitGate>,
    context: u32,
}

impl Drop for UnderWay {
    /// The last commit or range message under way lets a buffer waiting to take the gate go
    /// ([`SplitGate::split`]).
    fn drop(&mut self) {
        let mut contexts = self.gate.contexts.lock();
        if let Some(g) = contexts.get_mut(&self.context) {
            g.under_way -= 1;
            if g.under_way == 0 && g.splitting > 0 {
                self.gate.changed.notify_all();
            }
        }
        SplitGate::prune(&mut contexts, self.context);
    }
}

/// A buffer's hold of its context's gate while its parts are out ([`SplitGate::try_split`]):
/// dropped after its last commit, or where its walk stops, it carries out the range messages
/// that came meanwhile, in the order they came, and then lets the context's commits go.
pub(crate) struct SplitHold {
    gate: Arc<SplitGate>,
    context: u32,
}

impl Drop for SplitHold {
    fn drop(&mut self) {
        loop {
            let deferred = {
                let mut contexts = self.gate.contexts.lock();
                let Some(g) = contexts.get_mut(&self.context) else {
                    return;
                };
                if g.deferred.is_empty() {
                    g.split = false;
                    SplitGate::prune(&mut contexts, self.context);
                    self.gate.changed.notify_all();
                    return;
                }
                std::mem::take(&mut g.deferred)
            };
            for op in deferred {
                op();
            }
        }
    }
}

/// What the completions of a command buffer's parts before its last found, which the last
/// part's completion makes the buffer's ([`Parts::merged`]), and how far its parts run ahead of
/// the GPU, which its walk waits on ([`Parts::wait_ahead`]).
#[derive(Default)]
pub(crate) struct Parts {
    seen: Mutex<Seen>,
    ahead: Mutex<Ahead>,
    /// Woken as a part the walk counts is seen complete.
    moved: Condvar,
    /// Its queue's rate of GPU time to walk time, which its parts' completions update.
    rate: Arc<QueueRate>,
}

#[derive(Default)]
struct Seen {
    /// The GPU start of the first part that has GPU times.
    gpu_start: Option<u64>,
    /// The GPU time of the parts seen that have GPU times, summed: each one's end less its start.
    gpu_busy: u64,
    /// The error of the first part that failed.
    error: Option<NsErrorText>,
}

/// How far a walk's parts run ahead of the GPU ("How far ahead of the GPU"): of the parts it cut
/// with their walk's time ([`Parts::part_handler`]).
#[derive(Default)]
struct Ahead {
    /// Those committed and not seen complete.
    out: usize,
    /// Their walks, summed.
    out_walk: Duration,
    /// When each was committed, the oldest first: a queue's parts are seen complete in their
    /// commit order.
    out_at: VecDeque<MonoTime>,
    /// One was seen complete.
    seen: bool,
    /// The walks of those seen complete with GPU times, summed.
    timed_walk: Duration,
    /// Their GPU times, summed.
    timed_gpu: Duration,
    /// How many were seen complete.
    completed: u64,
    /// One failed.
    failed: bool,
}

impl Ahead {
    /// The walk's estimate: its parts' own rate, else, where none was seen complete, its queue's.
    fn estimate(&self, queue: &QueueRate) -> Estimate {
        if !self.timed_walk.is_zero() {
            return Estimate::Rate(self.timed_gpu.as_secs_f64() / self.timed_walk.as_secs_f64());
        }
        if self.seen {
            return Estimate::Untimed;
        }
        queue.get().map_or(Estimate::Unknown, Estimate::Rate)
    }
}

impl Parts {
    /// A buffer's parts, on a queue whose rate of GPU time to walk time is `rate`.
    pub(crate) fn on(rate: Arc<QueueRate>) -> Parts {
        Parts {
            rate,
            ..Parts::default()
        }
    }

    /// The completion handler of a part before the last: it notes what the part's completion
    /// found, then lets `keep` go (the part's place among its queue's command buffers). A part cut
    /// with its walk's time and its commit's, `cut`, counts as out from then to its completion,
    /// which notes its GPU time against that walk ("How far ahead of the GPU"); one cut without
    /// (a chunk walked at once with others) counts for none of it.
    pub(crate) fn part_handler<K: Send + 'static>(
        self: &Arc<Self>,
        keep: K,
        cut: Option<(Duration, MonoTime)>,
    ) -> CompletedHandler {
        let walk = cut.map(|(walk, _)| walk);
        if let Some((walk, at)) = cut {
            let mut ahead = self.ahead.lock();
            ahead.out += 1;
            ahead.out_walk = ahead.out_walk.saturating_add(walk);
            ahead.out_at.push_back(at);
        }
        let parts = Arc::clone(self);
        Box::new(move |info: CompletedInfo| {
            let mut seen = parts.seen.lock();
            if let Some((start, end)) = info.gpu_times_ns {
                seen.gpu_start.get_or_insert(start);
                seen.gpu_busy = seen.gpu_busy.saturating_add(end.saturating_sub(start));
            }
            if let Some(walk) = walk {
                let mut ahead = parts.ahead.lock();
                ahead.out = ahead.out.saturating_sub(1);
                ahead.out_walk = ahead.out_walk.saturating_sub(walk);
                ahead.out_at.pop_front();
                ahead.seen = true;
                ahead.completed += 1;
                ahead.failed |= info.error.is_some();
                if let Some((start, end)) = info.gpu_times_ns
                    && !walk.is_zero()
                {
                    ahead.timed_walk = ahead.timed_walk.saturating_add(walk);
                    let gpu = Duration::from_nanos(end.saturating_sub(start));
                    ahead.timed_gpu = ahead.timed_gpu.saturating_add(gpu);
                    let rate = ahead.timed_gpu.as_secs_f64() / ahead.timed_walk.as_secs_f64();
                    parts.rate.set(rate);
                }
                drop(ahead);
                parts.moved.notify_all();
            }
            if seen.error.is_none() {
                seen.error = info.error;
            }
            drop(seen);
            drop(keep);
        })
    }

    /// The walk's estimate of its parts' GPU time ([`Estimate`]).
    pub(crate) fn estimate(&self) -> Estimate {
        self.ahead.lock().estimate(&self.rate)
    }

    /// Waits, with no lock of the caller's held, while the walk's parts run so far ahead of the
    /// GPU that it is to wait ([`Estimate::waits`]), its estimate taken again as each part is seen
    /// complete and the oldest out's age at `now`, in slices of [`WAIT_SLICE`] at most, until
    /// `over`, handed how many of the parts were seen complete so far, says the wait ends (a
    /// reset, its queue's end, [`PACE_WAIT_MOST`] passed with none seen complete). Whether the
    /// walk may go on paced: `false` where `over` ended the wait, or a part failed.
    pub(crate) fn wait_ahead(
        &self,
        now: impl Fn() -> MonoTime,
        over: impl Fn(u64) -> bool,
    ) -> bool {
        let mut ahead = self.ahead.lock();
        loop {
            if ahead.failed {
                return false;
            }
            let oldest = (ahead.out_at.front())
                .map_or(Duration::ZERO, |&at| now().saturating_duration_since(at));
            if !(ahead.estimate(&self.rate)).waits(ahead.out, ahead.out_walk, oldest) {
                return true;
            }
            if over(ahead.completed) {
                return false;
            }
            let _ = self.moved.wait_for(&mut ahead, WAIT_SLICE);
        }
    }

    /// `handler`, the last part's, handed the whole buffer's outcome: the first failed part's
    /// error, and status, where a part before the last failed, else the last part's; and GPU
    /// times that start at the first part's start and span every part's GPU time summed, ending no
    /// later than the last part's end (the module docs, "Its GPU times"). The platform runs
    /// a queue's completion handlers in its commit order, so every part before the last has been
    /// seen.
    pub(crate) fn merged(self: Arc<Self>, handler: CompletedHandler) -> CompletedHandler {
        Box::new(move |mut info: CompletedInfo| {
            let seen = std::mem::take(&mut *self.seen.lock());
            if let Some(error) = seen.error {
                info.error = Some(error);
                info.status = CommandBufferStatus::ERROR;
            }
            if let (Some(first), Some((start, end))) = (seen.gpu_start, info.gpu_times_ns) {
                let busy = seen.gpu_busy.saturating_add(end.saturating_sub(start));
                let first = first.min(end);
                info.gpu_times_ns = Some((first, first.saturating_add(busy).min(end)));
            }
            handler(info);
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    /// When the tests' parts are committed, and when they wait, unless they say otherwise.
    const AT: MonoTime = MonoTime::from_nanos(1_000_000_000);

    /// While a buffer of a context has parts out, a commit of the context waiting at the gate, or
    /// a buffer dealt to two threads waiting to take it, holds the context back
    /// ([`SplitGate::holds_back`]), and nothing else does: a walk in parts then goes on unpaced.
    #[test]
    fn a_commit_or_split_waiting_at_the_gate_holds_its_context_back() {
        let gate = Arc::new(SplitGate::default());
        assert!(!gate.holds_back(1));
        let hold = gate.try_split(1).expect("the gate is free");
        gate.range(1, Box::new(|| {}));
        assert!(!gate.holds_back(1), "a range message waits for no one");
        std::thread::scope(|scope| {
            let commit = scope.spawn(|| drop(gate.commit(1)));
            assert!(crate::graphics::tests::eventually(|| gate.holds_back(1)));
            assert!(!gate.holds_back(2), "another context");
            drop(hold);
            commit.join().unwrap();
        });
        assert!(!gate.holds_back(1));
        let hold = gate.try_split(1).expect("the gate is free");
        std::thread::scope(|scope| {
            let split = scope.spawn(|| drop(gate.split(1)));
            assert!(crate::graphics::tests::eventually(|| gate.holds_back(1)));
            drop(hold);
            split.join().unwrap();
        });
        assert!(!gate.holds_back(1));
    }

    /// A buffer's take of its context's gate for its parts waits while a commit of the context is
    /// under way, and while another buffer of it has parts out, and takes the gate once that
    /// ended; the gate's entry goes once it is let go.
    #[test]
    fn a_take_of_the_gate_waits_for_commits_under_way_and_parts_out() {
        let gate = Arc::new(SplitGate::default());
        for parts_out in [false, true] {
            let commit = (!parts_out).then(|| gate.commit(1));
            let hold = parts_out.then(|| gate.try_split(1).expect("the gate is free"));
            let took = AtomicBool::new(false);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    let _hold = gate.split(1);
                    took.store(true, Ordering::Release);
                });
                std::thread::sleep(Duration::from_millis(50));
                assert!(
                    !took.load(Ordering::Acquire),
                    "taken early (parts out: {parts_out})"
                );
                drop((commit, hold));
            });
            assert!(took.load(Ordering::Acquire));
            assert!(gate.contexts.lock().is_empty(), "its entry gone");
        }
    }

    /// A commit of a context waits while a buffer of it has parts out, and a commit of another
    /// context does not.
    #[test]
    fn a_commit_waits_for_its_context_s_parts_alone() {
        let gate = Arc::new(SplitGate::default());
        let hold = gate.try_split(1).expect("the gate is free");
        drop(gate.commit(2));
        let went = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let _commit = gate.commit(1);
                went.store(true, Ordering::Release);
            });
            std::thread::sleep(Duration::from_millis(50));
            assert!(!went.load(Ordering::Acquire), "committed between the parts");
            drop(hold);
        });
        assert!(went.load(Ordering::Acquire));
    }

    /// No buffer splits while another commit or range message of its context is under way, or
    /// another buffer of it has parts out.
    #[test]
    fn a_split_waits_for_nothing_and_takes_a_free_gate_alone() {
        let gate = Arc::new(SplitGate::default());
        let commit = gate.commit(1);
        assert!(gate.try_split(1).is_none(), "split beside a commit");
        drop(commit);
        let hold = gate.try_split(1).expect("the gate is free");
        assert!(
            gate.try_split(1).is_none(),
            "two buffers' parts out at once"
        );
        assert!(gate.try_split(2).is_some(), "another context's gate");
        drop(hold);
        assert!(gate.try_split(1).is_some());
        assert!(gate.contexts.lock().is_empty(), "a free gate's entry stays");
    }

    /// A range message that comes while a buffer's parts are out is carried out at the last
    /// part's commit, in the order they came, before the context's other commits go; one that
    /// comes with none out is carried out at once.
    #[test]
    fn a_range_message_comes_wholly_before_or_after_a_buffer_s_parts() {
        let gate = Arc::new(SplitGate::default());
        let done = Arc::new(Mutex::new(Vec::new()));
        let range = |n: u32| {
            let done = Arc::clone(&done);
            Box::new(move || done.lock().push(n)) as RangeOp
        };
        gate.range(1, range(1));
        assert_eq!(*done.lock(), [1]);
        let hold = gate.try_split(1).unwrap();
        gate.range(1, range(2));
        gate.range(1, range(3));
        gate.range(2, range(4));
        assert_eq!(*done.lock(), [1, 4], "carried out while the parts are out");
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let _commit = gate.commit(1);
                done.lock().push(5);
            });
            std::thread::sleep(Duration::from_millis(50));
            drop(hold);
        });
        assert_eq!(*done.lock(), [1, 4, 2, 3, 5]);
    }

    fn error(code: i64) -> NsErrorText {
        NsErrorText {
            domain: "MTLCommandBufferErrorDomain".to_owned(),
            code,
            description: String::new(),
        }
    }

    /// The whole buffer's outcome: the first failed part's error, and GPU times that start at the
    /// first part's start and span the parts' GPU time summed, no later than the last part's end:
    /// the gaps between parts, where the GPU waited for the walk, are not the buffer's.
    #[test]
    fn a_buffer_s_outcome_is_all_its_parts() {
        let outcome = |parts: [CompletedInfo; 3]| {
            let p = Arc::new(Parts::default());
            let [first, second, last] = parts;
            p.part_handler((), None)(first);
            p.part_handler((), None)(second);
            let got = Arc::new(Mutex::new(None));
            let into = Arc::clone(&got);
            Arc::clone(&p).merged(Box::new(move |info| *into.lock() = Some(info)))(last);
            got.lock().take().unwrap()
        };
        let ok = |start, end| CompletedInfo {
            status: CommandBufferStatus::COMPLETED,
            error: None,
            gpu_times_ns: Some((start, end)),
        };
        let failed = |start, end, code| CompletedInfo {
            status: CommandBufferStatus::ERROR,
            error: Some(error(code)),
            gpu_times_ns: Some((start, end)),
        };
        assert_eq!(
            outcome([ok(10, 20), ok(25, 40), ok(45, 60)]),
            ok(10, 50),
            "10 + 15 + 15 of GPU time, from the first part's start"
        );
        assert_eq!(
            outcome([ok(10, 20), failed(25, 40, 8), failed(45, 60, 3)]),
            failed(10, 50, 8),
            "the first failed part's error"
        );
        assert_eq!(
            outcome([ok(10, 20), ok(25, 40), failed(45, 60, 3)]),
            failed(10, 50, 3)
        );
        assert_eq!(
            outcome([ok(10, 30), ok(20, 45), ok(40, 60)]),
            ok(10, 60),
            "parts that overlap end no later than the last part"
        );
        let untimed = CompletedInfo {
            status: CommandBufferStatus::COMPLETED,
            error: None,
            gpu_times_ns: None,
        };
        assert_eq!(
            outcome([untimed.clone(), ok(30, 40), ok(50, 60)]),
            ok(30, 50),
            "a part with no GPU times adds none"
        );
    }

    /// Where a walk cuts its part ("How many parts"): with nothing out, at [`SPLIT_AFTER`];
    /// behind incomplete work, once the part's estimated GPU time reaches [`PART_GPU`], or with
    /// nothing to estimate it from yet at [`SPLIT_AFTER`], and never where its parts had no GPU
    /// times.
    #[test]
    fn a_part_behind_incomplete_work_is_cut_at_its_estimated_gpu_time() {
        let ms = Duration::from_millis;
        let us = Duration::from_micros;
        for estimate in [Estimate::Rate(0.1), Estimate::Unknown, Estimate::Untimed] {
            assert!(!estimate.cuts(us(249), false), "{estimate:?}");
            assert!(estimate.cuts(SPLIT_AFTER, false), "{estimate:?}");
        }
        assert!(!Estimate::Rate(0.1).cuts(us(1_999), true));
        assert!(
            Estimate::Rate(0.1).cuts(PART_WALK_MOST, true),
            "its walk's bound"
        );
        assert!(!Estimate::Rate(10.0).cuts(us(1_599), true));
        assert!(
            Estimate::Rate(10.0).cuts(us(1_600), true),
            "16 ms at 10 times its walk"
        );
        assert!(
            !Estimate::Rate(55.0).cuts(us(290), true),
            "16 ms at 55 times its walk"
        );
        assert!(Estimate::Rate(55.0).cuts(us(291), true));
        assert!(!Estimate::Unknown.cuts(us(249), true));
        assert!(Estimate::Unknown.cuts(SPLIT_AFTER, true));
        assert!(!Estimate::Untimed.cuts(ms(1000), true));
    }

    /// Where a walk waits after a cut ("How far ahead of the GPU"): never with one part out;
    /// with two or more, while their estimated GPU time is over [`PACE_AHEAD`], or, with nothing
    /// to estimate it from yet, while their walk is over [`PACE_UNKNOWN_WALK`], or while the
    /// oldest of them was committed [`PACE_AHEAD`] ago or more; never where its parts had no GPU
    /// times.
    #[test]
    fn a_walk_waits_while_its_parts_out_run_far_ahead_of_the_gpu() {
        let ms = Duration::from_millis;
        let young = ms(15);
        assert!(
            !Estimate::Rate(100.0).waits(1, ms(100), ms(100)),
            "one part out"
        );
        assert!(!Estimate::Rate(1.0).waits(2, ms(16), young));
        assert!(Estimate::Rate(1.0).waits(2, ms(17), young));
        assert!(Estimate::Rate(55.0).waits(4, ms(1), young), "55 ms ahead");
        assert!(!Estimate::Rate(0.5).waits(48, ms(30), young), "15 ms ahead");
        assert!(
            Estimate::Rate(0.5).waits(2, ms(1), PACE_AHEAD),
            "the oldest out that old, whatever the estimate"
        );
        assert!(!Estimate::Unknown.waits(8, PACE_UNKNOWN_WALK, young));
        assert!(Estimate::Unknown.waits(2, PACE_UNKNOWN_WALK + Duration::from_nanos(1), young));
        assert!(Estimate::Unknown.waits(2, ms(0), PACE_AHEAD));
        assert!(!Estimate::Untimed.waits(48, ms(1000), ms(1000)));
    }

    /// A walk's estimate is its queue's last rate until a part of its own is seen complete with
    /// GPU times, then its own parts' GPU time against their walk, which its queue keeps; parts
    /// seen complete with no GPU times leave it none.
    #[test]
    fn a_walk_estimates_from_its_own_parts_then_its_queue_s() {
        let ms = Duration::from_millis;
        let rate = Arc::new(QueueRate::default());
        let done = |gpu_times_ns| CompletedInfo {
            status: CommandBufferStatus::COMPLETED,
            error: None,
            gpu_times_ns,
        };
        let parts = Arc::new(Parts::on(Arc::clone(&rate)));
        assert_eq!(parts.estimate(), Estimate::Unknown);
        rate.set(2.0);
        assert_eq!(parts.estimate(), Estimate::Rate(2.0), "its queue's");
        let first = parts.part_handler((), Some((ms(1), AT)));
        let second = parts.part_handler((), Some((ms(3), AT)));
        let near = |estimate: Estimate, want: f64| matches!(estimate, Estimate::Rate(r) if (r - want).abs() < 1e-9);
        first(done(Some((0, 10_000_000))));
        assert!(
            near(parts.estimate(), 10.0),
            "10 ms of GPU for 1 ms of walk"
        );
        second(done(Some((10_000_000, 16_000_000))));
        assert!(near(parts.estimate(), 4.0), "16 ms for 4 ms");
        assert!(
            near(Estimate::Rate(rate.get().unwrap()), 4.0),
            "kept by its queue"
        );
        let untimed = Arc::new(Parts::on(Arc::new(QueueRate::default())));
        untimed.part_handler((), Some((ms(1), AT)))(done(None));
        assert_eq!(untimed.estimate(), Estimate::Untimed);
        let dealt = Arc::new(Parts::on(Arc::new(QueueRate::default())));
        dealt.part_handler((), None)(done(Some((0, 10_000_000))));
        assert_eq!(
            dealt.estimate(),
            Estimate::Unknown,
            "a chunk dealt counts for none"
        );
    }

    /// A wait's bound runs from its begin, then from the last of its parts seen complete
    /// ([`PACE_WAIT_MOST`]).
    #[test]
    fn a_wait_s_bound_runs_from_its_last_part_seen_complete() {
        let at = |ms: u64| MonoTime::from_nanos(ms * 1_000_000);
        let ms = Duration::from_millis;
        let mut progress = Progress::from(at(1_000));
        assert_eq!(
            progress.since(3, at(1_000)),
            ms(0),
            "the count it began with"
        );
        assert_eq!(progress.since(3, at(1_400)), ms(400));
        assert_eq!(progress.since(4, at(1_450)), ms(0), "a part seen complete");
        assert_eq!(progress.since(4, at(1_900)), ms(450));
        assert_eq!(progress.since(4, at(1_950)), PACE_WAIT_MOST);
    }

    /// A walk whose parts' estimate says they are close to the GPU waits while the oldest of them
    /// out was committed [`PACE_AHEAD`] ago or more, and goes on as it is seen complete.
    #[test]
    fn a_walk_waits_while_its_oldest_part_out_is_a_refresh_old() {
        let ms = Duration::from_millis;
        let rate = Arc::new(QueueRate::default());
        rate.set(0.1);
        let parts = Arc::new(Parts::on(rate));
        let first = parts.part_handler((), Some((ms(1), AT)));
        let _second = parts.part_handler((), Some((ms(1), AT)));
        let young = || AT.saturating_add(ms(15));
        assert!(
            parts.wait_ahead(young, |_| false),
            "0.2 ms ahead, 15 ms old"
        );
        let went = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                assert!(parts.wait_ahead(|| AT.saturating_add(PACE_AHEAD), |_| false));
                went.store(true, Ordering::Release);
            });
            std::thread::sleep(Duration::from_millis(50));
            assert!(
                !went.load(Ordering::Acquire),
                "went on with its oldest 16 ms old"
            );
            first(CompletedInfo {
                status: CommandBufferStatus::COMPLETED,
                error: None,
                gpu_times_ns: Some((0, 100_000)),
            });
        });
        assert!(went.load(Ordering::Acquire));
    }

    /// A walk's wait for its parts ends as a part's completion leaves them close enough to the
    /// GPU, and at once where `over`, handed how many were seen complete, says so or a part
    /// failed.
    #[test]
    fn a_walk_s_wait_for_its_parts_ends_at_a_completion_its_end_or_a_failure() {
        let ms = Duration::from_millis;
        let rate = Arc::new(QueueRate::default());
        rate.set(10.0);
        let parts = Arc::new(Parts::on(rate));
        let first = parts.part_handler((), Some((ms(1), AT)));
        let _second = parts.part_handler((), Some((ms(1), AT)));
        // 20 ms ahead: waits until the first part is seen complete.
        let went = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                assert!(parts.wait_ahead(|| AT, |_| false));
                went.store(true, Ordering::Release);
            });
            std::thread::sleep(Duration::from_millis(50));
            assert!(!went.load(Ordering::Acquire), "went on 20 ms ahead");
            first(CompletedInfo {
                status: CommandBufferStatus::COMPLETED,
                error: None,
                gpu_times_ns: Some((0, 10_000_000)),
            });
        });
        assert!(went.load(Ordering::Acquire));
        let _third = parts.part_handler((), Some((ms(2), AT)));
        assert!(
            !parts.wait_ahead(|| AT, |completed| completed == 1),
            "ended by `over`, handed the one part seen complete"
        );
        let failed = parts.part_handler((), Some((ms(1), AT)));
        failed(CompletedInfo {
            status: CommandBufferStatus::ERROR,
            error: Some(error(3)),
            gpu_times_ns: None,
        });
        assert!(!parts.wait_ahead(|| AT, |_| false), "a part failed");
    }
}
