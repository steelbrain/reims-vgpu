//! Device-local scheduling state for FIFO work held behind shader translation.

/// Decoded page-list state for one child FIFO channel.
#[derive(Clone, Debug, Default)]
pub struct ChannelRing {
    pub valid: bool,
    pub base_pfn: u32,
    pub length: u32,
    pub page_gpas: Vec<u64>,
}

/// Work published by transport ingress and consumed by the ordered worker.
#[derive(Clone, Debug, Default)]
pub struct PendingWork {
    main_drain: bool,
    active_child_mask: u32,
    child_mask: u32,
    iosfc: bool,
    host_action_yield: bool,
}

impl PendingWork {
    pub fn request_main(&mut self) {
        self.main_drain = true;
    }

    pub fn set_main_requested(&mut self, requested: bool) {
        self.main_drain = requested;
    }

    pub fn main_requested(&self) -> bool {
        self.main_drain
    }

    pub fn request_children(&mut self, mask: u32) {
        self.child_mask |= mask;
    }

    pub fn activate_children(&mut self, mask: u32) {
        self.active_child_mask |= mask;
    }

    pub fn activate_and_request_children(&mut self, mask: u32) {
        self.activate_children(mask);
        self.request_children(mask);
    }

    pub fn retire_children(&mut self, mask: u32) {
        self.active_child_mask &= !mask;
        self.child_mask &= !mask;
    }

    pub fn active_child_mask(&self) -> u32 {
        self.active_child_mask
    }

    pub fn active_or_pending_children(&self) -> u32 {
        self.active_child_mask | self.child_mask
    }

    pub fn clear_children(&mut self, mask: u32) {
        self.child_mask &= !mask;
    }

    #[cfg(feature = "test-fixtures")]
    pub fn replace_children(&mut self, mask: u32) {
        self.child_mask = mask;
    }

    #[cfg(feature = "test-fixtures")]
    pub fn replace_active_children(&mut self, mask: u32) {
        self.active_child_mask = mask;
    }

    pub fn take_children(&mut self) -> u32 {
        std::mem::take(&mut self.child_mask)
    }

    pub fn child_mask(&self) -> u32 {
        self.child_mask
    }

    pub fn request_iosfc(&mut self) {
        self.iosfc = true;
    }

    pub fn clear_iosfc(&mut self) {
        self.iosfc = false;
    }

    pub fn iosfc_requested(&self) -> bool {
        self.iosfc
    }

    pub fn yield_for_host_action(&mut self) {
        self.host_action_yield = true;
    }

    pub fn resume_after_host_action(&mut self) {
        self.host_action_yield = false;
    }

    pub fn host_action_yielded(&self) -> bool {
        self.host_action_yield
    }
}

/// Complete channel-ingress and ordering state for one device.
///
/// Defining or freeing a child channel updates its admission bit, translation
/// holds, and decoded ring cache in one transition. Pending requests, nested
/// drain ownership, and per-channel ring contents remain separately typed
/// subobjects within this scheduling boundary.
#[derive(Clone, Debug)]
pub struct WorkSchedulingState {
    pub pending: PendingWork,
    pub child_rings: [ChannelRing; crate::MAX_CHANNELS],
    pub translation: TranslationScheduling,
    pub drains: ChildDrainStack,
}

impl Default for WorkSchedulingState {
    fn default() -> Self {
        Self {
            pending: PendingWork::default(),
            child_rings: std::array::from_fn(|_| ChannelRing::default()),
            translation: TranslationScheduling::default(),
            drains: ChildDrainStack::default(),
        }
    }
}

impl WorkSchedulingState {
    pub fn define_child(&mut self, channel: u32) -> bool {
        let Some(bit) = 1u32.checked_shl(channel) else {
            return false;
        };
        if channel == 0 || channel as usize >= crate::MAX_CHANNELS {
            return false;
        }
        self.pending.activate_children(bit);
        self.translation.retire_channel(bit);
        self.child_rings[channel as usize] = ChannelRing::default();
        true
    }

    pub fn free_child(&mut self, channel: u32) -> bool {
        let Some(bit) = 1u32.checked_shl(channel) else {
            return false;
        };
        if channel == 0 || channel as usize >= crate::MAX_CHANNELS {
            return false;
        }
        self.pending.retire_children(bit);
        self.translation.retire_channel(bit);
        self.child_rings[channel as usize] = ChannelRing::default();
        true
    }
}

/// An invalid transition in the nested child-FIFO drain stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildDrainNestingError {
    InvalidChannel { channel: u32 },
    ReenteredChannel { channel: u32 },
    ExitOutOfOrder { expected: Option<u32>, actual: u32 },
}

impl reims_vgpu_observe::Decline for ChildDrainNestingError {
    fn slug(&self) -> &'static str {
        match self {
            Self::InvalidChannel { .. } => "child_drain_channel_invalid",
            Self::ReenteredChannel { .. } => "child_drain_channel_reentered",
            Self::ExitOutOfOrder { .. } => "child_drain_exit_out_of_order",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::InvalidChannel { channel } | Self::ReenteredChannel { channel } => {
                vec![("channel", channel.to_string())]
            }
            Self::ExitOutOfOrder { expected, actual } => vec![
                (
                    "expected",
                    expected
                        .map(|channel| channel.to_string())
                        .unwrap_or_else(|| "none".to_string()),
                ),
                ("actual", actual.to_string()),
            ],
        }
    }
}

reims_vgpu_observe::decline_display!(ChildDrainNestingError);

/// Owns child-FIFO drain nesting as one ordered state.
///
/// The current channel and active-channel mask are projections of the stack,
/// so an inner drain cannot restore one and forget to restore the other.
#[derive(Clone, Default, Debug)]
pub struct ChildDrainStack {
    channels: Vec<u32>,
}

impl ChildDrainStack {
    pub fn current(&self) -> Option<u32> {
        self.channels.last().copied()
    }

    pub fn active_mask(&self) -> u32 {
        self.channels
            .iter()
            .fold(0, |mask, channel| mask | (1u32 << channel))
    }

    pub fn enter(&mut self, channel: u32) -> Result<(), ChildDrainNestingError> {
        let Some(bit) = 1u32.checked_shl(channel) else {
            return Err(ChildDrainNestingError::InvalidChannel { channel });
        };
        if channel == 0 {
            return Err(ChildDrainNestingError::InvalidChannel { channel });
        }
        if self.active_mask() & bit != 0 {
            return Err(ChildDrainNestingError::ReenteredChannel { channel });
        }
        self.channels.push(channel);
        Ok(())
    }

    pub fn exit(&mut self, channel: u32) -> Result<(), ChildDrainNestingError> {
        if self.current() != Some(channel) {
            return Err(ChildDrainNestingError::ExitOutOfOrder {
                expected: self.current(),
                actual: channel,
            });
        }
        self.channels.pop();
        Ok(())
    }
}

/// One newly established sibling-order hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranslationOrderHold {
    pub held_mask: u32,
    pub new_mask: u32,
    pub producer_mask: u32,
    pub episodes: u64,
}

/// Result of checking whether a present may pass translation-held work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentTranslationBarrier {
    Ready {
        released: bool,
        producer_mask: u32,
    },
    Held {
        pending_mask: u32,
        new_episode: Option<u64>,
    },
}

/// Translation work still owned when the device lifetime ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnreleasedTranslationHold {
    pub held_mask: u32,
    pub producer_mask: u32,
    pub episodes: u64,
}

/// Owns the complete defer/hold/release lifecycle for translated FIFO work.
#[derive(Clone, Default, Debug)]
pub struct TranslationScheduling {
    deferred_mask: u32,
    order_hold_mask: u32,
    order_hold_episodes: u64,
    present_hold_mask: u32,
    present_hold_episodes: u64,
}

impl TranslationScheduling {
    pub fn deferred_mask(&self) -> u32 {
        self.deferred_mask
    }

    pub fn order_hold_mask(&self) -> u32 {
        self.order_hold_mask
    }

    pub fn order_hold_episodes(&self) -> u64 {
        self.order_hold_episodes
    }

    pub fn present_hold_mask(&self) -> u32 {
        self.present_hold_mask
    }

    pub fn present_hold_episodes(&self) -> u64 {
        self.present_hold_episodes
    }

    /// Mark one FIFO timeline as waiting for immutable translation.
    /// Returns the complete producer mask only on the first transition.
    pub fn defer(&mut self, bit: u32) -> Option<u32> {
        if bit == 0 || self.deferred_mask & bit != 0 {
            return None;
        }
        self.deferred_mask |= bit;
        Some(self.deferred_mask)
    }

    /// Mark one FIFO timeline's translation ready.
    /// Returns the complete producer mask only when the bit was owned.
    pub fn ready(&mut self, bit: u32) -> Option<u32> {
        if bit == 0 || self.deferred_mask & bit == 0 {
            return None;
        }
        self.deferred_mask &= !bit;
        Some(self.deferred_mask)
    }

    /// Hold sibling timelines behind the currently deferred producer set.
    pub fn hold_order(&mut self, held_mask: u32) -> Option<TranslationOrderHold> {
        let new_mask = held_mask & !self.order_hold_mask;
        if new_mask == 0 {
            return None;
        }
        let starts_episode = self.order_hold_mask == 0;
        self.order_hold_mask |= new_mask;
        if starts_episode {
            self.order_hold_episodes = self.order_hold_episodes.saturating_add(1);
        }
        Some(TranslationOrderHold {
            held_mask: self.order_hold_mask,
            new_mask,
            producer_mask: self.deferred_mask,
            episodes: self.order_hold_episodes,
        })
    }

    /// Release every sibling hold once no translation producer remains.
    pub fn release_order_if_ready(&mut self) -> Option<u32> {
        if self.deferred_mask != 0 || self.order_hold_mask == 0 {
            return None;
        }
        Some(std::mem::take(&mut self.order_hold_mask))
    }

    /// Decide whether a display FIFO may overtake any other deferred timeline.
    pub fn present_barrier(&mut self, current_bit: u32) -> PresentTranslationBarrier {
        let pending_mask = self.deferred_mask & !current_bit;
        if pending_mask != 0 {
            let new_episode = if self.present_hold_mask & current_bit == 0 {
                self.present_hold_mask |= current_bit;
                self.present_hold_episodes = self.present_hold_episodes.saturating_add(1);
                Some(self.present_hold_episodes)
            } else {
                None
            };
            return PresentTranslationBarrier::Held {
                pending_mask,
                new_episode,
            };
        }

        let released = self.present_hold_mask & current_bit != 0;
        self.present_hold_mask &= !current_bit;
        PresentTranslationBarrier::Ready {
            released,
            producer_mask: self.deferred_mask,
        }
    }

    /// Retire every scheduling edge owned by one FIFO channel.
    pub fn retire_channel(&mut self, bit: u32) {
        self.deferred_mask &= !bit;
        self.order_hold_mask &= !bit;
        self.present_hold_mask &= !bit;
    }

    pub fn unreleased(&self) -> Option<UnreleasedTranslationHold> {
        (self.order_hold_mask != 0 || self.deferred_mask != 0).then_some(
            UnreleasedTranslationHold {
                held_mask: self.order_hold_mask,
                producer_mask: self.deferred_mask,
                episodes: self.order_hold_episodes,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_requests_merge_until_the_worker_takes_the_whole_set() {
        let mut pending = PendingWork::default();
        pending.request_children(1 << 2);
        pending.request_children(1 << 5);
        pending.clear_children(1 << 2);
        assert_eq!(pending.child_mask(), 1 << 5);
        assert_eq!(pending.take_children(), 1 << 5);
        assert_eq!(pending.child_mask(), 0);
    }

    #[test]
    fn child_activation_request_and_retirement_cannot_drift() {
        let mut pending = PendingWork::default();
        pending.activate_and_request_children(1 << 4);
        assert_eq!(pending.active_child_mask(), 1 << 4);
        assert_eq!(pending.child_mask(), 1 << 4);
        pending.retire_children(1 << 4);
        assert_eq!(pending.active_or_pending_children(), 0);
    }

    #[test]
    fn host_action_yield_is_a_named_worker_boundary() {
        let mut pending = PendingWork::default();
        pending.yield_for_host_action();
        assert!(pending.host_action_yielded());
        pending.resume_after_host_action();
        assert!(!pending.host_action_yielded());
    }

    #[test]
    fn channel_definition_and_free_retire_every_prior_scheduling_edge() {
        let mut scheduling = WorkSchedulingState::default();
        let bit = 1 << 4;
        scheduling.pending.activate_and_request_children(bit);
        scheduling.translation.defer(bit);
        scheduling.child_rings[4] = ChannelRing {
            valid: true,
            base_pfn: 7,
            length: 8,
            page_gpas: vec![0x1000],
        };

        assert!(scheduling.define_child(4));
        assert_eq!(scheduling.pending.active_child_mask(), bit);
        assert_eq!(scheduling.pending.child_mask(), bit);
        assert_eq!(scheduling.translation.deferred_mask(), 0);
        assert!(!scheduling.child_rings[4].valid);

        assert!(scheduling.free_child(4));
        assert_eq!(scheduling.pending.active_or_pending_children(), 0);
        assert!(!scheduling.child_rings[4].valid);
    }

    #[test]
    fn child_drain_stack_derives_current_and_mask_from_one_ordered_owner() {
        let mut drains = ChildDrainStack::default();
        drains.enter(2).unwrap();
        drains.enter(5).unwrap();
        assert_eq!(drains.current(), Some(5));
        assert_eq!(drains.active_mask(), (1 << 2) | (1 << 5));

        assert_eq!(
            drains.enter(2),
            Err(ChildDrainNestingError::ReenteredChannel { channel: 2 })
        );
        assert_eq!(
            drains.exit(2),
            Err(ChildDrainNestingError::ExitOutOfOrder {
                expected: Some(5),
                actual: 2,
            })
        );
        assert_eq!(drains.current(), Some(5));
        assert_eq!(drains.active_mask(), (1 << 2) | (1 << 5));

        drains.exit(5).unwrap();
        drains.exit(2).unwrap();
        assert_eq!(drains.current(), None);
        assert_eq!(drains.active_mask(), 0);
    }

    #[test]
    fn one_translation_ownership_interval_is_one_episode() {
        let mut scheduling = TranslationScheduling::default();
        scheduling.defer(0b10);
        assert_eq!(scheduling.hold_order(0b100).unwrap().episodes, 1);
        assert_eq!(scheduling.hold_order(0b001).unwrap().episodes, 1);
        assert!(scheduling.hold_order(0b101).is_none());

        scheduling.ready(0b10);
        assert_eq!(scheduling.release_order_if_ready(), Some(0b101));
        assert_eq!(scheduling.hold_order(0b1000).unwrap().episodes, 2);
    }

    #[test]
    fn channel_retirement_clears_every_hold_kind() {
        let mut scheduling = TranslationScheduling::default();
        scheduling.defer(0b10);
        scheduling.hold_order(0b10);
        assert!(matches!(
            scheduling.present_barrier(0b10),
            PresentTranslationBarrier::Ready { .. }
        ));
        assert!(matches!(
            scheduling.present_barrier(0b100),
            PresentTranslationBarrier::Held { .. }
        ));

        scheduling.retire_channel(0b10);
        scheduling.retire_channel(0b100);
        assert_eq!(scheduling.deferred_mask(), 0);
        assert_eq!(scheduling.order_hold_mask(), 0);
        assert_eq!(scheduling.present_hold_mask(), 0);
    }
}
