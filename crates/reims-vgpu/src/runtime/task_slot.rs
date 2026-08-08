//! Resolve the wire task word a command payload carries to a live task slot.
//!
//! Three child commands each carry a task word in their payload and each had
//! its own private copy of the same six lines:
//!
//! ```text
//! if tasks[raw].active { raw } else { raw >> 1 }
//! ```
//!
//! `CmdExecIndirect2` (`0x35`), `CmdGetComputeInfo` (`0x3b`) and
//! `CmdHeapTextureSizeAndAlign` — one resolver spelled three times, so a reading
//! taken at one site said nothing about the other two. They are one function
//! here, and that function carries the refusal.
//!
//! # Which space each word is in
//!
//! The word is a slot id. Only `DefineTask2` (`0x38`) carries the doubled form,
//! registering under `raw >> 1` ([`crate::model::DEFINE_TASK_ID_SHIFT`]); every
//! other opcode names the slot directly, so halving one of their words names a
//! **different task**.
//!
//! The `DefineTask2` wire space is `(slot << 1) | is_kernel_task`. Enumerated
//! over a full x86/Vulkan boot for slots 0–12 it reads `0x1, 0x2, 0x4, 0x6,
//! 0x8, 0xa … 0x18`: exactly one odd word — `0x1`, the kernel task, whose slot
//! id is 0 — and then strictly even. So the discriminator is "odd **and greater
//! than one**", not merely "odd"; a looser reading of the same numbers would
//! have proved the opposite of what they say.
//!
//! `exec_indirect2` receives `0x5`, `0x7` and `0x9`, and `MapMemory2` files
//! spans under the same three. All are odd and greater than one, so that space
//! does not contain them and they are slot ids.
//!
//! This resolver therefore does not consider `raw >> 1` at all. Slots run
//! densely from 0, so `raw >> 1` is usually *also* live — that is a dense table,
//! not an ambiguity, and there is nothing for a census to decide.

use crate::model::TaskTable;

/// Which command carried the task word. Distinguishes otherwise identical
/// decodes so a per-site set difference is possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TaskWordSite {
    /// `CmdExecIndirect2` (`0x35`) — the write-carrying one.
    ExecIndirect2,
    /// `CmdGetComputeInfo` (`0x3b`).
    ComputeInfo,
    /// `CmdHeapTextureSizeAndAlign`.
    HeapTextureQuery,
}

impl TaskWordSite {
    fn name(self) -> &'static str {
        match self {
            Self::ExecIndirect2 => "exec_indirect2",
            Self::ComputeInfo => "compute_info",
            Self::HeapTextureQuery => "heap_texture_query",
        }
    }
}

/// The word named no live slot, so the resolver has no answer and the caller
/// refuses.
///
/// The only outcome worth naming. Whether `word >> 1` also happens to be live
/// used to be split out alongside this, on the reading that it might be the
/// intended task; it is not, so a live word is simply the answer and a dead one
/// is this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TaskWordDecode {
    Dead,
}

impl crate::observe::Decline for TaskWordDecode {
    fn slug(&self) -> &'static str {
        match self {
            Self::Dead => "cmd_task_dead",
        }
    }
}



/// Resolve the wire task word to the slot this crate will act on. **The word,
/// or nothing.**
///
/// This used to fall back to `raw >> 1` when the word named no live slot. That
/// could not fail safely: slots run densely from 0, so `raw >> 1` is almost
/// always some *other* live task, and returning its slot rather than refusing
/// means `CmdExecIndirect2` runs a whole command stream — including guest
/// writes — against page tables the named task does not own.
///
/// Returning `None` is what makes that case visible: each caller turns it into
/// its own always-on refusal rather than a plausible wrong answer nothing sees.
///
/// The latch is taken before the line is built. `Emit::field` renders eagerly
/// and this sits on the command path, so building and dropping the strings on
/// every decode would make the probe cost scale with the traffic it measures.
pub(crate) fn resolve_task_word(tasks: &TaskTable, site: TaskWordSite, raw: u32) -> Option<u32> {
    use crate::observe::Decline;
    if tasks.is_active(raw) {
        return Some(raw);
    }
    let decode = TaskWordDecode::Dead;
    let discriminant = ((site as u64) << 32) | u64::from(raw);
    if crate::observe::first_sight(decode.slug(), discriminant) {
        crate::observe::Emit::decline("cmd_task", &decode)
            .field("site", site.name())
            .field("raw", format!("{raw:#x}"))
            .fail();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(active: &[u32]) -> TaskTable {
        let mut tasks = TaskTable::default();
        for &id in active {
            tasks.define(
                id,
                crate::model::TaskEntry {
                    active: true,
                    ..Default::default()
                },
            );
        }
        tasks
    }

    /// Slot 5 live, slot 2 not.
    #[test]
    fn a_word_whose_shifted_slot_is_dead_resolves_directly() {
        let tasks = table(&[5]);
        assert_eq!(
            resolve_task_word(&tasks, TaskWordSite::ExecIndirect2, 5),
            Some(5)
        );
    }

    /// Slots 3 and 6 both live: the guest sent `6` and `6` is what it meant.
    /// Slot 3 being live too is a dense task table, not a second candidate —
    /// only `DefineTask2` carries the doubled form, so nothing here may read
    /// `6 >> 1` as an alternative reading of this word.
    #[test]
    fn a_live_shifted_slot_is_not_a_second_candidate() {
        let tasks = table(&[3, 6]);
        assert_eq!(
            resolve_task_word(&tasks, TaskWordSite::ExecIndirect2, 6),
            Some(6)
        );
    }

    /// Slot 6 dead, slot 3 live — the case the deleted fallback was for. It
    /// answered `3`, which is a **different task** whose page tables the guest
    /// never named. Refusing is the only answer the decoded word supports.
    #[test]
    fn a_word_naming_no_live_slot_refuses_rather_than_naming_its_neighbour() {
        let tasks = table(&[3]);
        assert_eq!(
            resolve_task_word(&tasks, TaskWordSite::ComputeInfo, 6),
            None
        );
    }

    /// Neither slot live: still nothing, and the caller emits its own refusal.
    #[test]
    fn a_word_naming_nothing_live_refuses() {
        let tasks = table(&[]);
        assert_eq!(
            resolve_task_word(&tasks, TaskWordSite::HeapTextureQuery, 6),
            None
        );
    }

    /// Slot 0 is a real task — the kernel task's id is 0 — so word 0 resolves
    /// like any other rather than reading as "unset".
    #[test]
    fn word_zero_resolves_to_the_live_slot_zero() {
        let tasks = table(&[0]);
        assert_eq!(
            resolve_task_word(&tasks, TaskWordSite::ExecIndirect2, 0),
            Some(0)
        );
    }

    /// The latch key mixes the site in, so the same word arriving at two sites
    /// is two sightings. Without this a word first seen at exec would silence
    /// the compute-info reading and the per-site set difference would be lost.
    #[test]
    fn the_latch_key_separates_the_same_word_at_different_sites() {
        let key = |site: TaskWordSite, raw: u32| ((site as u64) << 32) | u64::from(raw);
        assert_ne!(
            key(TaskWordSite::ExecIndirect2, 6),
            key(TaskWordSite::ComputeInfo, 6)
        );
        assert_ne!(
            key(TaskWordSite::ComputeInfo, 6),
            key(TaskWordSite::HeapTextureQuery, 6)
        );
    }
}
