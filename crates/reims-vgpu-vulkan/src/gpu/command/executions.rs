//! The commands an execution of an indirect command buffer runs, checked at its command buffer's
//! commit where a direct command of their kind dies at its call.
//!
//! - **What is checked.** Two deaths of a direct command depend, for an execution, on the
//!   records of the commands it runs: their draws, and the pipeline states they name. The
//!   runtime runs the records as they stand at the commit (`crate::gpu::icb`), so the commit
//!   reads them (`IndirectCommandBuffer::commands`, of the execution's range, or of every
//!   command where a buffer holds the range) and dies, at `commit`, naming the execution and the
//!   command, where a direct command of its kind dies:
//!   - a draw of points, of some vertices and instances, under a state whose points Metal sizes
//!     none of ([`Points`], `crate::gpu::encode_render`'s `sized`).
//!
//!   Commands that inherit the encoder's state are decided at the execution's call where the
//!   state alone decides (`crate::gpu::encode_compute`, `crate::gpu::encode_render`). A command
//!   of no draw or dispatch, or of a record that names no state, runs nothing, and is not asked;
//!   nor is an execution in an encoder its command buffer's failure runs nothing of
//!   ([`rt::FailedEncoders`]). The state a command runs is the live state of the ID its record
//!   names, where that state's identifier is the one the record answers; else the runtime runs
//!   the state the program set on the command and released since, of which nothing here holds
//!   what decides, and a command a death would ask of dies.
//!
//!   The commit reads the records only where what every route wrote into them so far
//!   (`IndirectCommandBuffer::command_writes`) leaves it something to find ([`may_find`]), as
//!   the runtime's host step reads them for its check only so (`IndirectCommandCheckNeeds`). A
//!   buffer into whose records no route wrote a draw or a dispatch runs nothing: a buffer of no
//!   command types among them, whose execution in a pass Metal runs nothing of, and whose
//!   records the runtime refuses to read. Where the execution asks of no ID of the guest's and
//!   its encoder made no buffer resident for writing, a draw of points is the only death to
//!   find, so a buffer whose records hold none, as one of draws of triangles, is read at the
//!   commit by the runtime alone.
//! - **Records the GPU may write** ([`gpu_writes`]): a blit's copy into a buffer or reset of
//!   it, and a command, direct or executed, whose state can encode indirect commands in an
//!   encoder that made the buffer resident for writing, may write its records as the GPU runs,
//!   and what it wrote may stay. Once the runtime's buffer says so (`gpu_written`, which each
//!   of those sets at its call, as does any encoder's making the buffer resident for writing),
//!   it records an execution committed since once the work before it completed (its host step),
//!   from the records as the GPU and the CPU left them, and so it does an execution whose range
//!   a buffer holds, and a compute encoder's execution under predication. The commit then reads
//!   nothing: an execution that a death could ask of hands the runtime a check at its call
//!   ([`Execution::host_step_check`]), which the host step runs on each command it will record,
//!   before it records any, and which dies there where the commit would have
//!   ([`ask`]). Where the commit reads the records, the runtime records them as the commit read
//!   them and never runs the check. An execution whose own commands may write its records dies
//!   at the commit. A write encoded while a committed command buffer that has not ended executes
//!   the buffer as its commit read it dies at its own call: it may land before those commands
//!   run.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::platform::icb::IndirectCommandBuffer as _;
use metal2vulkan::metal::PrimitiveType;
use metal2vulkan::pipeline::CommandWrites;
use metal2vulkan_runtime as rt;

use crate::answer::{get, plain, unrepresentable};
use crate::gpu::command::Progress;
use crate::gpu::icb::{IcbObject, VulkanIcb};

/// The operations of a record that runs something: a draw of any kind or a dispatch
/// (`rt::IndirectCommandBuffer::command_writes`). A record of none runs nothing, and [`ask`] asks
/// nothing of it.
const RUNS: u32 = CommandWrites::DRAW
    | CommandWrites::DRAW_INDEXED
    | CommandWrites::DRAW_PATCHES
    | CommandWrites::DRAW_MESH
    | CommandWrites::DISPATCH;

/// Which states size the points an execution's commands draw ([`Execution::check`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Points {
    /// Not asked: a compute encoder's execution, or commands that inherit a state that sizes
    /// them, or no state.
    Unasked,
    /// The commands inherit the encoder's state, whose points Metal sizes none of: a draw of
    /// points dies.
    Unsized,
    /// Each command's own state decides ([`crate::gpu::states::points_sized`]).
    Own,
}

/// An execution of an indirect command buffer whose commands its commit reads (the module docs).
#[derive(Debug)]
pub(crate) struct Execution {
    /// Its selector, which a death names.
    pub(crate) sel: &'static str,
    /// The buffer it executes.
    pub(crate) icb: VulkanIcb,
    /// Its range; `None` for one a buffer holds, which the GPU reads as it runs, so every command
    /// of the buffer is asked.
    pub(crate) range: Option<Range<u64>>,
    /// Which states size its draws of points.
    pub(crate) points: Points,
    /// The indirect command buffers its encoder made resident for writing, where its commands
    /// name their own states: a command whose state can encode indirect commands may write their
    /// records ([`gpu_writes`]).
    pub(crate) writable: Vec<VulkanIcb>,
}

impl Execution {
    /// Whether its commit reads its commands.
    pub(crate) fn asks(&self) -> bool {
        self.decides() || !self.writable.is_empty()
    }

    /// Whether its commands decide a death.
    fn decides(&self) -> bool {
        self.points != Points::Unasked
    }

    /// Checks its commands at the commit (the module docs): dies where one would die as a direct
    /// command, marks the buffers its commands may write, and answers the buffer whose records
    /// the commit read for a death, which [`reads`] then holds.
    pub(super) fn check(self) -> Option<VulkanIcb> {
        let decides = self.decides();
        let Execution {
            sel: execution,
            icb,
            range,
            points,
            writable,
        } = self;
        if plain(|| icb.raw.gpu_written()) {
            // The host step reads its commands, and asks them where a death could
            // ([`Execution::host_step_check`]); here they are not known: any of them may write.
            writable.iter().for_each(|w| gpu_writes("commit", w));
            return None;
        }
        if !may_find(&icb, points, !writable.is_empty()) {
            return decides.then_some(icb);
        }
        let range = range.unwrap_or(0..icb.size());
        let commands = get("commit", || icb.raw.commands(range.clone()));
        let mut writes = false;
        for (index, command) in (range.start..).zip(&commands) {
            writes |= ask(execution, index, command, points, true) == Some(true);
        }
        if writes && !writable.is_empty() {
            if decides && writable.iter().any(|w| same(w, &icb)) {
                unrepresentable(
                    "commit",
                    &format!(
                        "{execution} of an indirect command buffer its encoder made resident \
                         for writing, a command of which may write its records as the GPU runs \
                         them: what its later commands run is not what the commit read"
                    ),
                );
            }
            writable.iter().for_each(|w| gpu_writes("commit", w));
        }
        decides.then_some(icb)
    }

    /// The check the runtime's host step runs on each command it records of this execution,
    /// where its commit reads nothing (the module docs); `None` where no death could ask of its
    /// commands, which leaves the runtime its other paths. It dies where [`Execution::check`]
    /// would, at `commit`, on the scheduler's thread: the death ends the process there, as `fatal` does outside the tests' in-process capture, and unwinds
    /// nothing into the runtime. A death test's child runs without that capture, as production.
    /// With it, what the check looks for: draws of points where a state sizes them. An execution
    /// of a range
    /// whose buffer's records rule out every kind it names may run on the GPU alone, unchecked;
    /// one of a range the GPU reads looks for every kind.
    pub(crate) fn host_step_check(
        &self,
    ) -> Option<(rt::IndirectCommandCheck, rt::IndirectCommandCheckNeeds)> {
        if !self.decides() {
            return None;
        }
        let (execution, points) = (self.sel, self.points);
        let needs = rt::IndirectCommandCheckNeeds::new(points != Points::Unasked, false);
        let check: rt::IndirectCommandCheck = Box::new(move |index, command| {
            ask(execution, index, command, points, false);
        });
        Some((check, needs))
    }
}

/// Whether the commit's reading of `icb`'s records may find a death or a write ([`ask`]): whether
/// what every route wrote into them so far (`command_writes`), the CPU's setters alone where the
/// GPU may have written none, holds a draw or a dispatch, and, where the execution asks of
/// no buffer made resident for writing (`by_state`), a draw of points
/// where `points` asks of them. Records no route wrote a draw or a dispatch into run nothing and
/// ask nothing: a buffer of no command types among them, whose execution in a pass Metal runs
/// nothing of, and whose records the runtime refuses to read.
fn may_find(icb: &VulkanIcb, points: Points, by_state: bool) -> bool {
    let writes = plain(|| icb.raw.command_writes());
    let point_draws =
        points != Points::Unasked && writes.may_draw_primitive_type(PrimitiveType::Point.as_raw());
    writes.may(RUNS) && (by_state || point_draws)
}

/// Asks command `index` of `execution`'s indirect command buffer, `command`, of the death a direct
/// command of its kind dies of (the module docs), `points` saying which states size its draws
/// of points: dies where one would, at `commit`. Where `encodes` is asked (the commit's), answers
/// whether its state may encode indirect commands (one released since may); `None` for a
/// command of no draw or dispatch, or of no state, which runs nothing, and wherever `encodes` is
/// not asked and the answer decides no death. A command's state is looked at only where a death
/// or the answer needs it: a host step asks each of tens of thousands of commands a frame.
fn ask(
    execution: &str,
    index: u64,
    command: &rt::IndirectCommand,
    points: Points,
    encodes: bool,
) -> Option<bool> {
    let at = || format!("command {index} of {execution}'s indirect command buffer");
    let named =
        match command {
            rt::IndirectCommand::Compute(c) if c.dispatch.is_some() => {
                if c.pipeline_id.is_none() || !encodes {
                    return None;
                }
                let state = c.pipeline.as_ref().filter(|s| {
                    Some(plain(|| s.unique_identifier())) == c.pipeline_unique_identifier
                });
                state.map(|s| plain(|| s.can_encode_indirect_commands()))
            }
            rt::IndirectCommand::Render(r) if r.draw.is_some() => {
                let of_points = points != Points::Unasked && points_drawn(r.draw);
                if points == Points::Unsized && of_points {
                    dies_unsized(&at());
                }
                let sized_asked = points == Points::Own && of_points;
                if r.pipeline_id.is_none() || !(encodes || sized_asked) {
                    return None;
                }
                let state = r.pipeline.as_ref().filter(|s| {
                    Some(plain(|| s.unique_identifier())) == r.pipeline_unique_identifier
                });
                if sized_asked {
                    match state.map(crate::gpu::states::points_sized) {
                        Some(Some(true)) => {}
                        Some(Some(false)) => dies_unsized(&at()),
                        Some(None) | None => unrepresentable(
                            "commit",
                            &format!(
                                "{}, a draw of points under a pipeline state the program released, \
                             which the runtime runs: its input primitive topology class, which \
                             decides whether Metal sizes its points, went with it",
                                at()
                            ),
                        ),
                    }
                }
                if !encodes {
                    return None;
                }
                state.map(|s| plain(|| s.can_encode_indirect_commands()))
            }
            _ => return None,
        };
    // `None`: a state released since, whose functions nothing here holds.
    Some(named != Some(false))
}

/// Whether `draw` is of points, of some vertices or indices and instances: the draw as the record
/// runs it ([`rt::EncodedRenderCommand::draw`], of the type set), not as the arguments getters
/// answer it, which read quads and a quad strip as points ([`rt::IndirectRenderCommand::draw`]).
fn points_drawn(draw: Option<rt::IndirectDraw>) -> bool {
    match draw {
        Some(rt::IndirectDraw::Primitives {
            primitive_type: rt::DrawPrimitiveType::Public(PrimitiveType::Point),
            vertex_count,
            instance_count,
            ..
        }) => vertex_count != 0 && instance_count != 0,
        Some(rt::IndirectDraw::IndexedPrimitives {
            primitive_type: rt::DrawPrimitiveType::Public(PrimitiveType::Point),
            index_count,
            instance_count,
            ..
        }) => index_count != 0 && instance_count != 0,
        _ => false,
    }
}

/// Dies at `commit` at `at`, a draw of points under a state whose points Metal sizes none of.
fn dies_unsized(at: &str) -> ! {
    unrepresentable(
        "commit",
        &format!(
            "{at}, a draw of {}",
            crate::gpu::encode_render::UNSIZED_POINTS
        ),
    )
}

/// Whether `a` and `b` are one buffer.
fn same(a: &VulkanIcb, b: &VulkanIcb) -> bool {
    std::ptr::eq::<IcbObject>(&**a, &**b)
}

/// A command encoded at `sel` may write `icb`'s records on the GPU (the module docs). It dies
/// where a committed command buffer that has not ended executes them as its commit read them
/// ([`reads`]): the write may land before those run.
pub(crate) fn gpu_writes(sel: &'static str, icb: &VulkanIcb) {
    if icb.readers.load(Ordering::SeqCst) != 0 {
        unrepresentable(
            sel,
            "a write on the GPU of the records of an indirect command buffer that a committed \
             command buffer that has not ended executes, whose commands its commit read and \
             checked: the write may land before they run, behind an event or on another queue, \
             and what they run then is not checked",
        );
    }
}

/// A committed command buffer of `progress` executes `icb` as its commit read its records,
/// until it ends ([`gpu_writes`]).
pub(super) fn reads(icb: &VulkanIcb, progress: &Progress) {
    icb.readers.fetch_add(1, Ordering::SeqCst);
    progress.hold(Read(Arc::clone(&icb.readers)));
}

/// A reader of an indirect command buffer's records ([`IcbObject::readers`]), which ends as its
/// command buffer's progress lets go of it.
struct Read(Arc<AtomicUsize>);

impl Drop for Read {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use crate::core::platform::Handle as _;
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::icb::{IcbRenderCmd, IndirectRenderCommand as _};
    use crate::core::platform::render::{RenderCmd, RenderEncoder as _};
    use crate::core::platform::types::NsRange;
    use crate::core::platform::values::IcbDesc;
    use crate::core::raise::OnFault;
    use crate::core::testkit::capture::catch_death;
    use metal2vulkan::metal::PixelFormat;
    use metal2vulkan::pipeline::{
        ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
    };

    use super::*;
    use crate::gpu::device::VulkanDevice;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::VulkanRenderPipeline;

    const VERT: &[u8] = include_bytes!("../../../../../tests/shaders/vert.metallib");
    const FRAG: &[u8] = include_bytes!("../../../../../tests/shaders/frag.metallib");

    /// `tri_vert` and `tri_frag` into `BGRA8Unorm`, with indirect command buffer support: its
    /// vertex function writes no `[[point_size]]`, of no input primitive topology class.
    fn state(d: &VulkanDevice) -> VulkanRenderPipeline {
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        let mut desc =
            RenderPipelineDescriptor::new(StageSource::Function(function(VERT, "tri_vert")));
        desc.fragment = Some(StageSource::Function(function(FRAG, "tri_frag")));
        desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        desc.support_indirect_command_buffers = true;
        crate::gpu::resource::render_pipeline(
            d.raw
                .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                .unwrap(),
        )
    }

    /// A buffer of `n` draws of `primitive_type` (three vertices, one instance) under `state`.
    fn draws(
        d: &VulkanDevice,
        state: &VulkanRenderPipeline,
        n: u64,
        primitive_type: u64,
    ) -> VulkanIcb {
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    max_vertex_buffer_bind_count: 2,
                    max_fragment_buffer_bind_count: 2,
                    ..IcbDesc::default()
                },
                n,
                0,
            )
            .unwrap()
            .unwrap();
        for i in 0..n {
            let c = icb.render_command(i).unwrap().unwrap();
            for cmd in [
                IcbRenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(state),
                },
                IcbRenderCmd::DrawPrimitives {
                    primitive_type,
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 1,
                    base_instance: 0,
                },
            ] {
                c.encode(cmd).unwrap();
            }
        }
        icb
    }

    /// A render encoder's execution of `icb`'s commands `range`, whose commands name their own
    /// states, as `crate::gpu::encode_render` makes it.
    fn execution(icb: &VulkanIcb, range: Range<u64>) -> Execution {
        Execution {
            sel: "executeCommandsInBuffer:withRange:",
            icb: icb.clone_ref(),
            range: Some(range),
            points: Points::Own,
            writable: Vec::new(),
        }
    }

    /// A render pass's execution of a buffer of no command types runs nothing, as Metal's does,
    /// whose records the runtime refuses to read: its commit reads none and completes
    #[test]
    fn an_execution_of_a_buffer_of_no_command_types_runs_nothing() {
        let (p, d) = (crate::platform::tests::platform(), dev());
        let icb = d
            .new_indirect_command_buffer(&IcbDesc::default(), 2, 0)
            .unwrap()
            .unwrap();
        let (_dst, pass) = crate::gpu::encode_render::tests::pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &icb,
                execution_range: NsRange::new(0, 2),
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
    }

    /// The commit reads an execution's records only where what they may hold leaves a death to
    /// find: of a buffer of draws of triangles under a state that sizes no points, none, its
    /// buffer still held as read; once one command draws points there, every command of the
    /// range, dying at that one
    #[test]
    fn the_commit_reads_records_only_where_a_death_may_lie() {
        let d = dev();
        let s = state(&d);
        let icb = draws(&d, &s, 4, 3);
        let reads = || d.raw.work_counts().indirect_records_read;
        let before = reads();
        assert!(execution(&icb, 0..4).check().is_some());
        assert_eq!(reads() - before, 0);
        let c = icb.render_command(2).unwrap().unwrap();
        c.encode(IcbRenderCmd::DrawPrimitives {
            primitive_type: 0,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        })
        .unwrap();
        *crate::answer::LAST_LINE.lock() = None;
        let before = reads();
        let site = catch_death(|| execution(&icb, 0..4).check()).unwrap_err();
        assert_eq!(site.file, "commit");
        assert_eq!(reads() - before, 4);
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert!(
            line.contains("command 2 of executeCommandsInBuffer"),
            "{line}"
        );
    }
}
