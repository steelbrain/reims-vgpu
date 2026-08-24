//! Reusable behavioral fixtures for semantic-core and executor conformance.
//!
//! This crate deliberately knows neither Vulkan nor QEMU. Tests supply the
//! concrete topology type and executor submission type, which keeps the same
//! harness usable at both sides of the composition boundary.

use reims_vgpu_core::ExecutionPort;
use std::collections::VecDeque;
use std::sync::Mutex;

/// An executor which consumes production submission values and returns a
/// prearranged sequence of completions or refusals.
pub struct ScriptedExecutor<Submission, Completion, Error> {
    submissions: Mutex<Vec<Submission>>,
    outcomes: Mutex<VecDeque<Result<Completion, Error>>>,
}

impl<Submission, Completion, Error> std::fmt::Debug
    for ScriptedExecutor<Submission, Completion, Error>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedExecutor").finish_non_exhaustive()
    }
}

impl<Submission, Completion, Error> ScriptedExecutor<Submission, Completion, Error> {
    pub fn new(outcomes: impl IntoIterator<Item = Result<Completion, Error>>) -> Self {
        Self {
            submissions: Mutex::new(Vec::new()),
            outcomes: Mutex::new(outcomes.into_iter().collect()),
        }
    }

    pub fn take_submissions(&self) -> Vec<Submission> {
        std::mem::take(
            &mut *self
                .submissions
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    pub fn remaining_outcomes(&self) -> usize {
        self.outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

impl<Submission, Completion, Error> ExecutionPort
    for ScriptedExecutor<Submission, Completion, Error>
where
    Submission: Send,
    Completion: Send,
    Error: Send,
{
    type Submission = Submission;
    type Completion = Completion;
    type Error = Error;

    fn execute(&self, submission: Submission) -> Result<Completion, Error> {
        self.submissions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(submission);
        self.outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop_front()
            .expect("scripted executor received more submissions than outcomes")
    }
}

/// One cell of the topology × host-pointer-import matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryCell<Topology> {
    pub topology: Topology,
    pub host_pointer_import: bool,
}

pub const fn four_memory_cells<Topology: Copy>(
    unified: Topology,
    discrete: Topology,
) -> [MemoryCell<Topology>; 4] {
    [
        MemoryCell {
            topology: unified,
            host_pointer_import: true,
        },
        MemoryCell {
            topology: unified,
            host_pointer_import: false,
        },
        MemoryCell {
            topology: discrete,
            host_pointer_import: true,
        },
        MemoryCell {
            topology: discrete,
            host_pointer_import: false,
        },
    ]
}

/// Guest-observable result of one semantic trace.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GuestEffects {
    pub memory: Vec<u8>,
    pub stamps: Vec<(u32, u32)>,
    pub interrupts: Vec<u32>,
    pub refusals: Vec<String>,
    pub presented: Vec<u8>,
}

/// Run one trace in all four memory cells and require exact guest equivalence.
///
/// The runner returns its internal metrics separately. They are intentionally
/// not compared: allocation and transfer plans are precisely what the policy
/// is allowed to change.
pub fn assert_four_cell_guest_equivalence<Topology, Metrics>(
    unified: Topology,
    discrete: Topology,
    mut run: impl FnMut(MemoryCell<Topology>) -> (GuestEffects, Metrics),
) -> [Metrics; 4]
where
    Topology: Copy + std::fmt::Debug,
{
    let cells = four_memory_cells(unified, discrete);
    let [(baseline, first), (second_effects, second), (third_effects, third), (fourth_effects, fourth)] =
        cells.map(&mut run);
    assert_eq!(
        second_effects, baseline,
        "guest effects changed in {:#?}",
        cells[1]
    );
    assert_eq!(
        third_effects, baseline,
        "guest effects changed in {:#?}",
        cells[2]
    );
    assert_eq!(
        fourth_effects, baseline,
        "guest effects changed in {:#?}",
        cells[3]
    );
    [first, second, third, fourth]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_executor_records_owned_production_values() {
        let executor = ScriptedExecutor::new([Ok::<_, ()>(7)]);
        assert_eq!(executor.execute(String::from("submission")), Ok(7));
        assert_eq!(executor.take_submissions(), ["submission"]);
        assert_eq!(executor.remaining_outcomes(), 0);
    }

    #[test]
    fn topology_metrics_may_differ_while_guest_effects_do_not() {
        let metrics = assert_four_cell_guest_equivalence("unified", "discrete", |cell| {
            (
                GuestEffects {
                    memory: vec![1, 2, 3],
                    stamps: vec![(0, 1)],
                    interrupts: vec![4],
                    refusals: Vec::new(),
                    presented: vec![9],
                },
                (cell.topology, cell.host_pointer_import),
            )
        });
        assert_eq!(metrics[0], ("unified", true));
        assert_eq!(metrics[3], ("discrete", false));
    }
}
