//! Single ordered owner of the engine's Vulkan graphics queue.
//!
//! Vulkan requires host access to one queue to be externally synchronized.
//! More importantly for this engine, submission order is part of the guest
//! contract: a readback, present, or compute dispatch must remain behind every
//! draw batch handed off before it.  Sending all queue operations through this
//! FIFO preserves both properties while allowing an ended draw batch to leave
//! the drain worker before the host driver returns from `vkQueueSubmit`.

use ash::vk;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use super::stamp_completion::SubmissionNote;

/// One GPU-side wait on an already accepted timeline point.
///
/// The semaphore, value and destination stage travel together because they
/// occupy the same index in three Vulkan submission arrays. Keeping them in
/// separate call arguments would permit a wait value to be paired with a stage
/// belonging to another semaphore.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TimelineWait {
    semaphore: vk::Semaphore,
    value: u64,
    stage: vk::PipelineStageFlags,
}

impl TimelineWait {
    pub(crate) const fn new(
        semaphore: vk::Semaphore,
        value: u64,
        stage: vk::PipelineStageFlags,
    ) -> Self {
        Self {
            semaphore,
            value,
            stage,
        }
    }
}

/// Owned arrays for one `VkSubmitInfo` and its optional timeline extension.
///
/// Vulkan indexes timeline values by the corresponding semaphore array. The
/// zero entries for binary semaphores are ignored by Vulkan, but are required
/// to keep both value counts equal to their semaphore counts whenever a
/// timeline semaphore is present.
pub(crate) struct SemaphoreSubmitOperands {
    pub(crate) wait_semaphores: Vec<vk::Semaphore>,
    pub(crate) wait_stages: Vec<vk::PipelineStageFlags>,
    pub(crate) wait_values: Vec<u64>,
    pub(crate) signal_semaphores: Vec<vk::Semaphore>,
    pub(crate) signal_values: Vec<u64>,
    pub(crate) has_timeline: bool,
}

pub(crate) fn semaphore_submit_operands(
    binary_waits: &[vk::Semaphore],
    binary_wait_stages: &[vk::PipelineStageFlags],
    binary_signals: &[vk::Semaphore],
    timeline_wait: Option<TimelineWait>,
    timeline_signal: Option<(vk::Semaphore, u64)>,
) -> SemaphoreSubmitOperands {
    assert_eq!(
        binary_waits.len(),
        binary_wait_stages.len(),
        "every wait semaphore requires its own destination stage"
    );
    if let (Some(wait), Some((signal_semaphore, signal_value))) = (timeline_wait, timeline_signal) {
        assert!(
            wait.semaphore != signal_semaphore || wait.value < signal_value,
            "a submission cannot wait for its own or a later timeline signal"
        );
    }

    let mut wait_semaphores = binary_waits.to_vec();
    let mut wait_stages = binary_wait_stages.to_vec();
    let mut wait_values = vec![0; binary_waits.len()];
    if let Some(wait) = timeline_wait {
        wait_semaphores.push(wait.semaphore);
        wait_stages.push(wait.stage);
        wait_values.push(wait.value);
    }

    let mut signal_semaphores = binary_signals.to_vec();
    let mut signal_values = vec![0; binary_signals.len()];
    if let Some((semaphore, value)) = timeline_signal {
        signal_semaphores.push(semaphore);
        signal_values.push(value);
    }

    SemaphoreSubmitOperands {
        wait_semaphores,
        wait_stages,
        wait_values,
        signal_semaphores,
        signal_values,
        has_timeline: timeline_wait.is_some() || timeline_signal.is_some(),
    }
}

type Reply = mpsc::SyncSender<Result<QueueOutcome, vk::Result>>;
type AsyncSubmitReply = mpsc::SyncSender<Result<(), vk::Result>>;
#[cfg(feature = "host-window")]
type PresentReply = mpsc::SyncSender<Result<bool, vk::Result>>;

#[derive(Clone, Copy, Debug)]
enum QueueOutcome {
    Unit,
}

struct OwnedSubmit {
    command_buffers: Vec<vk::CommandBuffer>,
    wait_semaphores: Vec<vk::Semaphore>,
    wait_stages: Vec<vk::PipelineStageFlags>,
    signal_semaphores: Vec<vk::Semaphore>,
    fence: vk::Fence,
    timeline_wait: Option<TimelineWait>,
    timeline: Option<(vk::Semaphore, u64, SubmissionNote)>,
    async_queued_at: Option<std::time::Instant>,
}

/// Completion of one ordered submit-plus-present transaction.
///
/// The transaction is enqueued while the engine still owns the resource-state
/// ordering point, then waited after that lock is released.  Keeping the
/// receiver as a value makes the split explicit: accepting the transaction and
/// completing the host driver calls are two different events.
#[cfg(feature = "host-window")]
pub(crate) struct PendingPresent {
    receiver: mpsc::Receiver<Result<bool, vk::Result>>,
}

/// Receipt for the end of the host driver's `vkQueueSubmit` call.
///
/// Enqueueing transfers host ownership of the fence to the queue thread.  The
/// fence may be polled or waited only after this receipt completes; Vulkan's
/// external-synchronization rule covers the submit call itself, not merely the
/// GPU work it starts.
pub(crate) struct PendingQueueSubmit {
    receiver: mpsc::Receiver<Result<(), vk::Result>>,
}

impl PendingQueueSubmit {
    pub(crate) fn try_complete(&self) -> Option<Result<(), vk::Result>> {
        match self.receiver.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(vk::Result::ERROR_DEVICE_LOST)),
        }
    }

    pub(crate) fn wait(self) -> Result<(), vk::Result> {
        self.receiver
            .recv()
            .unwrap_or(Err(vk::Result::ERROR_DEVICE_LOST))
    }

    #[cfg(test)]
    pub(crate) fn test_pair() -> (Self, mpsc::SyncSender<Result<(), vk::Result>>) {
        let (sender, receiver) = mpsc::sync_channel(1);
        (Self { receiver }, sender)
    }
}

#[cfg(feature = "host-window")]
impl PendingPresent {
    pub(crate) fn wait(self) -> Result<bool, vk::Result> {
        self.receiver
            .recv()
            .unwrap_or(Err(vk::Result::ERROR_DEVICE_LOST))
    }
}

enum Request {
    Submit {
        submit: OwnedSubmit,
        reply: Option<Reply>,
        async_reply: Option<AsyncSubmitReply>,
    },
    #[cfg(feature = "host-window")]
    PresentTransaction {
        submit: OwnedSubmit,
        loader: ash::khr::swapchain::Device,
        wait: vk::Semaphore,
        swapchain: vk::SwapchainKHR,
        image_index: u32,
        queued_at: std::time::Instant,
        reply: PresentReply,
    },
    WaitIdle {
        reply: Reply,
    },
    Barrier {
        reply: Reply,
    },
    Stop,
}

/// The first asynchronous submission failure.  Once one queue operation was
/// lost, later work must surface the same failure instead of running past the
/// missing point and making a corrupted frame look successful.
#[derive(Default)]
struct FailureLatch(Mutex<Option<vk::Result>>);

#[derive(Default)]
struct QueueStats {
    async_submits: AtomicU64,
    async_queue_us: AtomicU64,
    async_driver_us: AtomicU64,
    present_transactions: AtomicU64,
    present_queue_us: AtomicU64,
    present_driver_us: AtomicU64,
}

impl FailureLatch {
    fn get(&self) -> Option<vk::Result> {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set(&self, result: vk::Result) {
        let mut failure = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if failure.is_none() {
            *failure = Some(result);
            reims_vgpu_observe::fail(format!(
                "vk_queue_async_submit_failed reason=vk_queue_async_submit_failed result={result:?}"
            ));
            // Unlike a synchronous submit, this failure arrives after the
            // caller committed cache and ring state to the submission.  No
            // local rollback can reconstruct that point, so every asynchronous
            // failure poisons the context and takes the established recreate
            // path.  The line above retains the driver's exact result.
            super::device_lost::note_device_lost_seen();
        }
    }
}

/// Running queue thread.  `stop` is explicit because its cloned device handle
/// and every queued operation must be gone before `vkDestroyDevice`.
pub(crate) struct QueueOwner {
    sender: mpsc::Sender<Request>,
    failure: Arc<FailureLatch>,
    stats: Arc<QueueStats>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl QueueOwner {
    pub(crate) fn start(device: &ash::Device, queue: vk::Queue) -> std::io::Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let failure = Arc::new(FailureLatch::default());
        let stats = Arc::new(QueueStats::default());
        let thread_failure = Arc::clone(&failure);
        let thread_stats = Arc::clone(&stats);
        let thread_device = device.clone();
        let join = std::thread::Builder::new()
            .name("reims-vgpu-submit".into())
            .spawn(move || {
                run(
                    &thread_device,
                    queue,
                    receiver,
                    &thread_failure,
                    &thread_stats,
                )
            })?;
        Ok(Self {
            sender,
            failure,
            stats,
            join: Some(join),
        })
    }

    fn reply_channel() -> (Reply, mpsc::Receiver<Result<QueueOutcome, vk::Result>>) {
        mpsc::sync_channel(1)
    }

    fn send_sync(
        &self,
        request: impl FnOnce(Reply) -> Request,
    ) -> Result<QueueOutcome, vk::Result> {
        if let Some(result) = self.failure.get() {
            return Err(result);
        }
        let (reply, receiver) = Self::reply_channel();
        self.sender
            .send(request(reply))
            .map_err(|_| vk::Result::ERROR_DEVICE_LOST)?;
        receiver
            .recv()
            .unwrap_or(Err(vk::Result::ERROR_DEVICE_LOST))
    }

    pub(crate) fn submit_sync(
        &self,
        command_buffers: &[vk::CommandBuffer],
        fence: vk::Fence,
        timeline_wait: Option<TimelineWait>,
        timeline: Option<(vk::Semaphore, u64, SubmissionNote)>,
    ) -> Result<(), vk::Result> {
        let submit = OwnedSubmit {
            command_buffers: command_buffers.to_vec(),
            wait_semaphores: Vec::new(),
            wait_stages: Vec::new(),
            signal_semaphores: Vec::new(),
            fence,
            timeline_wait,
            timeline,
            async_queued_at: None,
        };
        self.send_sync(|reply| Request::Submit {
            submit,
            reply: Some(reply),
            async_reply: None,
        })
        .map(|_| ())
    }

    pub(crate) fn submit_async(
        &self,
        command_buffers: &[vk::CommandBuffer],
        fence: vk::Fence,
        timeline_wait: Option<TimelineWait>,
        timeline: Option<(vk::Semaphore, u64, SubmissionNote)>,
    ) -> Result<PendingQueueSubmit, vk::Result> {
        if let Some(result) = self.failure.get() {
            return Err(result);
        }
        let queued_point = timeline
            .as_ref()
            .map(|(_, value, note)| (*value, note.clone()));
        let (async_reply, receiver) = mpsc::sync_channel(1);
        self.sender
            .send(Request::Submit {
                submit: OwnedSubmit {
                    command_buffers: command_buffers.to_vec(),
                    wait_semaphores: Vec::new(),
                    wait_stages: Vec::new(),
                    signal_semaphores: Vec::new(),
                    fence,
                    timeline_wait,
                    timeline,
                    async_queued_at: Some(std::time::Instant::now()),
                },
                reply: None,
                async_reply: Some(async_reply),
            })
            .map_err(|_| vk::Result::ERROR_DEVICE_LOST)?;
        if let Some((value, note)) = queued_point {
            note.queued(value);
        }
        Ok(PendingQueueSubmit { receiver })
    }

    /// Enqueue the blit submission and its presentation as one ordered display
    /// transaction, returning before either host call runs.
    ///
    /// A separate submit followed later by a separate present has an observable
    /// gap in the queue-owner FIFO.  Packaging the pair is what lets the caller
    /// release the engine lock after enqueue without allowing guest work to
    /// appear between the semaphore signal and its consumer.
    #[cfg(feature = "host-window")]
    pub(crate) fn enqueue_present(
        &self,
        transaction: super::context::PresentTransaction<'_>,
    ) -> Result<PendingPresent, vk::Result> {
        if let Some(result) = self.failure.get() {
            return Err(result);
        }
        let submit = OwnedSubmit {
            command_buffers: transaction.command_buffers.to_vec(),
            wait_semaphores: transaction.wait_semaphores.to_vec(),
            wait_stages: transaction.wait_stages.to_vec(),
            signal_semaphores: transaction.signal_semaphores.to_vec(),
            fence: transaction.fence,
            timeline_wait: None,
            timeline: None,
            async_queued_at: None,
        };
        let (reply, receiver) = mpsc::sync_channel(1);
        self.sender
            .send(Request::PresentTransaction {
                submit,
                loader: transaction.loader,
                wait: transaction.present_wait,
                swapchain: transaction.swapchain,
                image_index: transaction.image_index,
                queued_at: std::time::Instant::now(),
                reply,
            })
            .map_err(|_| vk::Result::ERROR_DEVICE_LOST)?;
        Ok(PendingPresent { receiver })
    }

    pub(crate) fn wait_idle(&self) -> Result<(), vk::Result> {
        // Unlike ordinary work this must reach the owner even after its failure
        // latch fired: callers destroy resources after it returns.
        let (reply, receiver) = Self::reply_channel();
        self.sender
            .send(Request::WaitIdle { reply })
            .map_err(|_| vk::Result::ERROR_DEVICE_LOST)?;
        receiver
            .recv()
            .unwrap_or(Err(vk::Result::ERROR_DEVICE_LOST))
            .map(|_| ())
    }

    /// Wait until every request sent before this call has left the queue
    /// thread. Unlike `wait_idle`, this makes no driver call and therefore
    /// remains usable while dismantling an already-lost device.
    pub(crate) fn barrier(&self) {
        let (reply, receiver) = Self::reply_channel();
        if self.sender.send(Request::Barrier { reply }).is_ok() {
            let _ = receiver.recv();
        }
    }

    pub(crate) fn failure(&self) -> Option<vk::Result> {
        self.failure.get()
    }

    pub(crate) fn stats(&self) -> (u64, u64, u64, u64, u64, u64) {
        (
            self.stats.async_submits.load(Ordering::Relaxed),
            self.stats.async_queue_us.load(Ordering::Relaxed),
            self.stats.async_driver_us.load(Ordering::Relaxed),
            self.stats.present_transactions.load(Ordering::Relaxed),
            self.stats.present_queue_us.load(Ordering::Relaxed),
            self.stats.present_driver_us.load(Ordering::Relaxed),
        )
    }

    pub(crate) fn stop(&mut self) {
        let _ = self.sender.send(Request::Stop);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn run(
    device: &ash::Device,
    queue: vk::Queue,
    receiver: mpsc::Receiver<Request>,
    failure: &FailureLatch,
    stats: &QueueStats,
) {
    while let Ok(request) = receiver.recv() {
        match request {
            Request::Submit {
                submit,
                reply,
                async_reply,
            } => {
                let async_queued_at = submit.async_queued_at;
                let driver_started = std::time::Instant::now();
                let result = failure
                    .get()
                    .map_or_else(|| unsafe { execute_submit(device, queue, submit) }, Err);
                if let Some(queued_at) = async_queued_at {
                    stats.async_submits.fetch_add(1, Ordering::Relaxed);
                    stats.async_queue_us.fetch_add(
                        driver_started.duration_since(queued_at).as_micros() as u64,
                        Ordering::Relaxed,
                    );
                    stats.async_driver_us.fetch_add(
                        driver_started.elapsed().as_micros() as u64,
                        Ordering::Relaxed,
                    );
                }
                if reply.is_none() {
                    if let Err(result) = result {
                        failure.set(result);
                    }
                }
                if let Some(async_reply) = async_reply {
                    let _ = async_reply.send(result);
                }
                if let Some(reply) = reply {
                    let _ = reply.send(result.map(|_| QueueOutcome::Unit));
                }
            }
            #[cfg(feature = "host-window")]
            Request::PresentTransaction {
                submit,
                loader,
                wait,
                swapchain,
                image_index,
                queued_at,
                reply,
            } => {
                let driver_started = std::time::Instant::now();
                let result = complete_present_transaction(
                    failure,
                    || unsafe { execute_submit(device, queue, submit) },
                    || {
                        let waits = [wait];
                        let swapchains = [swapchain];
                        let indices = [image_index];
                        unsafe {
                            loader.queue_present(
                                queue,
                                &vk::PresentInfoKHR::default()
                                    .wait_semaphores(&waits)
                                    .swapchains(&swapchains)
                                    .image_indices(&indices),
                            )
                        }
                    },
                );
                stats.present_transactions.fetch_add(1, Ordering::Relaxed);
                stats.present_queue_us.fetch_add(
                    driver_started.duration_since(queued_at).as_micros() as u64,
                    Ordering::Relaxed,
                );
                stats.present_driver_us.fetch_add(
                    driver_started.elapsed().as_micros() as u64,
                    Ordering::Relaxed,
                );
                let _ = reply.send(result);
            }
            Request::WaitIdle { reply } => {
                // Still quiesce already-submitted work after an asynchronous
                // failure.  Callers use this operation before destroying GPU
                // resources; the latch changes the reported result, not that
                // lifetime obligation.
                let waited = unsafe { device.queue_wait_idle(queue) };
                let result = waited.and_then(|_| match failure.get() {
                    Some(result) => Err(result),
                    None => Ok(()),
                });
                let _ = reply.send(result.map(|_| QueueOutcome::Unit));
            }
            Request::Barrier { reply } => {
                let _ = reply.send(Ok(QueueOutcome::Unit));
            }
            Request::Stop => return,
        }
    }
}

#[cfg(any(test, feature = "host-window"))]
fn complete_present_transaction(
    failure: &FailureLatch,
    submit: impl FnOnce() -> Result<(), vk::Result>,
    present: impl FnOnce() -> Result<bool, vk::Result>,
) -> Result<bool, vk::Result> {
    if let Some(result) = failure.get() {
        return Err(result);
    }
    if let Err(result) = submit() {
        // The transaction has no submission point. Later queue work may not
        // run past that missing point.
        failure.set(result);
        return Err(result);
    }
    let result = present();
    // An out-of-date surface invalidates this display transaction, not the
    // ordered graphics queue. A lost device does invalidate later queue work.
    if result == Err(vk::Result::ERROR_DEVICE_LOST) {
        failure.set(vk::Result::ERROR_DEVICE_LOST);
    }
    result
}

unsafe fn execute_submit(
    device: &ash::Device,
    queue: vk::Queue,
    submit: OwnedSubmit,
) -> Result<(), vk::Result> {
    let signal_point = submit
        .timeline
        .as_ref()
        .map(|(semaphore, value, _)| (*semaphore, *value));
    let operands = semaphore_submit_operands(
        &submit.wait_semaphores,
        &submit.wait_stages,
        &submit.signal_semaphores,
        submit.timeline_wait,
        signal_point,
    );
    let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&operands.wait_values)
        .signal_semaphore_values(&operands.signal_values);
    let mut info = vk::SubmitInfo::default()
        .wait_semaphores(&operands.wait_semaphores)
        .wait_dst_stage_mask(&operands.wait_stages)
        .command_buffers(&submit.command_buffers)
        .signal_semaphores(&operands.signal_semaphores);
    if operands.has_timeline {
        info = info.push_next(&mut timeline_info);
    }
    unsafe { device.queue_submit(queue, &[info], submit.fence) }?;
    if let Some((_, value, note)) = submit.timeline {
        note.submitted(value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn owner_with_sender(sender: mpsc::Sender<Request>) -> QueueOwner {
        QueueOwner {
            sender,
            failure: Arc::new(FailureLatch::default()),
            stats: Arc::new(QueueStats::default()),
            join: None,
        }
    }

    #[test]
    fn mixed_binary_and_timeline_operands_keep_every_index_aligned() {
        let binary_waits = [vk::Semaphore::from_raw(1), vk::Semaphore::from_raw(2)];
        let binary_stages = [
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ];
        let binary_signals = [vk::Semaphore::from_raw(3)];
        let timeline = vk::Semaphore::from_raw(4);
        let operands = semaphore_submit_operands(
            &binary_waits,
            &binary_stages,
            &binary_signals,
            Some(TimelineWait::new(
                timeline,
                7,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            )),
            Some((timeline, 8)),
        );

        assert_eq!(
            operands.wait_semaphores,
            vec![binary_waits[0], binary_waits[1], timeline]
        );
        assert_eq!(
            operands.wait_stages,
            vec![
                binary_stages[0],
                binary_stages[1],
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ]
        );
        assert_eq!(operands.wait_values, vec![0, 0, 7]);
        assert_eq!(
            operands.signal_semaphores,
            vec![binary_signals[0], timeline]
        );
        assert_eq!(operands.signal_values, vec![0, 8]);
        assert!(operands.has_timeline);
    }

    #[test]
    fn signal_only_and_plain_submissions_preserve_the_existing_shapes() {
        let timeline = vk::Semaphore::from_raw(9);
        let signal = semaphore_submit_operands(&[], &[], &[], None, Some((timeline, 11)));
        assert!(signal.wait_semaphores.is_empty());
        assert!(signal.wait_stages.is_empty());
        assert!(signal.wait_values.is_empty());
        assert_eq!(signal.signal_semaphores, vec![timeline]);
        assert_eq!(signal.signal_values, vec![11]);
        assert!(signal.has_timeline);

        let plain = semaphore_submit_operands(&[], &[], &[], None, None);
        assert!(plain.wait_semaphores.is_empty());
        assert!(plain.wait_values.is_empty());
        assert!(plain.signal_semaphores.is_empty());
        assert!(plain.signal_values.is_empty());
        assert!(!plain.has_timeline);
    }

    #[test]
    #[should_panic(expected = "cannot wait for its own or a later timeline signal")]
    fn one_submit_cannot_wait_for_the_timeline_value_it_signals() {
        let timeline = vk::Semaphore::from_raw(12);
        let _ = semaphore_submit_operands(
            &[],
            &[],
            &[],
            Some(TimelineWait::new(
                timeline,
                13,
                vk::PipelineStageFlags::ALL_COMMANDS,
            )),
            Some((timeline, 13)),
        );
    }

    #[test]
    fn first_async_failure_is_sticky() {
        let _ = super::super::device_lost::take_device_lost_seen();
        let latch = FailureLatch::default();
        latch.set(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        latch.set(vk::Result::ERROR_DEVICE_LOST);
        assert_eq!(latch.get(), Some(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        assert!(super::super::device_lost::take_device_lost_seen());
    }

    #[test]
    fn async_handoff_separates_queue_acceptance_from_driver_return() {
        let (sender, receiver) = mpsc::channel();
        let owner = owner_with_sender(sender);
        let probe = super::super::stamp_completion::SubmissionProbe::new();

        let pending = owner
            .submit_async(
                &[vk::CommandBuffer::from_raw(1)],
                vk::Fence::from_raw(2),
                None,
                Some((vk::Semaphore::from_raw(3), 7, probe.note())),
            )
            .expect("owner accepted submission");

        assert_eq!(probe.latest_queued(), Some(7));
        let request = receiver
            .try_recv()
            .expect("submission remains in host FIFO");
        let Request::Submit {
            submit,
            reply,
            async_reply,
        } = request
        else {
            panic!("async handoff queued a non-submit request");
        };
        assert!(reply.is_none());
        let async_reply = async_reply.expect("async submit carries its return receipt");
        assert_eq!(
            submit.timeline.as_ref().map(|(_, value, _)| *value),
            Some(7)
        );
        assert!(pending.try_complete().is_none());
        async_reply.send(Ok(())).unwrap();
        assert_eq!(pending.try_complete(), Some(Ok(())));
    }

    #[test]
    fn failed_async_handoff_does_not_publish_its_point() {
        let (sender, receiver) = mpsc::channel();
        drop(receiver);
        let owner = owner_with_sender(sender);
        let probe = super::super::stamp_completion::SubmissionProbe::new();

        assert!(matches!(
            owner.submit_async(
                &[vk::CommandBuffer::from_raw(1)],
                vk::Fence::from_raw(2),
                None,
                Some((vk::Semaphore::from_raw(3), 7, probe.note())),
            ),
            Err(vk::Result::ERROR_DEVICE_LOST)
        ));
        assert_eq!(probe.latest_queued(), None);
    }

    #[cfg(feature = "host-window")]
    #[test]
    fn a_pending_display_transaction_returns_its_exact_completion() {
        let (reply, receiver) = mpsc::sync_channel(1);
        let pending = PendingPresent { receiver };
        let sender = std::thread::spawn(move || reply.send(Ok(true)).unwrap());

        assert_eq!(pending.wait(), Ok(true));
        sender.join().unwrap();
    }

    #[cfg(feature = "host-window")]
    #[test]
    fn a_lost_display_transaction_owner_cannot_look_successful() {
        let (reply, receiver) = mpsc::sync_channel(1);
        drop(reply);
        let pending = PendingPresent { receiver };

        assert_eq!(pending.wait(), Err(vk::Result::ERROR_DEVICE_LOST));
    }

    #[test]
    fn a_display_submission_failure_skips_present_and_stops_later_queue_work() {
        let _ = super::super::device_lost::take_device_lost_seen();
        let latch = FailureLatch::default();
        let presented = std::cell::Cell::new(false);

        let result = complete_present_transaction(
            &latch,
            || Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
            || {
                presented.set(true);
                Ok(false)
            },
        );

        assert_eq!(result, Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        assert!(!presented.get());
        assert_eq!(latch.get(), Some(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        assert!(super::super::device_lost::take_device_lost_seen());
    }

    #[test]
    fn an_out_of_date_display_transaction_does_not_poison_the_queue() {
        let latch = FailureLatch::default();
        let result = complete_present_transaction(
            &latch,
            || Ok(()),
            || Err(vk::Result::ERROR_OUT_OF_DATE_KHR),
        );

        assert_eq!(result, Err(vk::Result::ERROR_OUT_OF_DATE_KHR));
        assert_eq!(latch.get(), None);
    }

    #[test]
    fn device_loss_during_present_stops_later_queue_work() {
        let _ = super::super::device_lost::take_device_lost_seen();
        let latch = FailureLatch::default();
        let result =
            complete_present_transaction(&latch, || Ok(()), || Err(vk::Result::ERROR_DEVICE_LOST));

        assert_eq!(result, Err(vk::Result::ERROR_DEVICE_LOST));
        assert_eq!(latch.get(), Some(vk::Result::ERROR_DEVICE_LOST));
        assert!(super::super::device_lost::take_device_lost_seen());
    }
}
