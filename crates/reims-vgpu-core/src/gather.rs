//! Content identity state for sampled guest-resource windows.
//!
//! The witness combines the guest's resource generation with page-exact device
//! writes. Its optional byte fold audits that semantic claim without becoming
//! a second input to the guest-visible decision.

use std::collections::HashMap;

use reims_vgpu_memory::GuestRun;

use crate::{GatherKey, GatherVouch, GuestWriteReach, HostWriteVerdict, StatedGeneration};

pub const AUDIT_REBASELINE_LIMIT: u8 = 8;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AuditDensity {
    #[default]
    Disabled,
    EveryBind,
}

impl AuditDensity {
    fn stays_armed(self) -> bool {
        matches!(self, Self::EveryBind)
    }
}

/// Whether this witness is permitted to vouch at all.
///
/// [`Self::Withheld`] is a diagnostic arm, not a second contract: it can only
/// turn a vouch the contract already granted into a re-read, which costs
/// throughput and can never cost the guest work. Every other verdict is
/// untouched, so a resource the contract refuses is refused identically on both
/// arms and the difference between them isolates exactly one thing — whether a
/// bind reused the bytes the previous one read.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VouchPolicy {
    #[default]
    Contract,
    Withheld,
}

/// The witness's two diagnostic policies, resolved once at the composition edge.
///
/// They travel as one value because they have identical lifetimes — both are
/// read from the environment at device creation and both must survive a device
/// reset — and because a function taking two of them positionally is a pair of
/// same-shaped enums that a caller can silently swap. Neither reaches a decision
/// the guest's correctness depends on: one adds a read-back check, the other
/// declines an optimization.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GatherPolicies {
    pub audit: AuditDensity,
    pub vouch: VouchPolicy,
}

#[derive(Clone, Debug)]
struct Entry {
    gpas: Vec<u64>,
    span: u64,
    fold: u128,
    fold_valid: bool,
    fold_seeded: bool,
    binds_since_fold: u32,
    audit_armed: bool,
    rebaselines: u8,
    pages_epoch: u64,
    stated_gen: Option<StatedGeneration>,
    generation: u64,
}

#[derive(Debug, Default)]
pub struct GatherWitness {
    entries: HashMap<GatherKey, Entry>,
    audit: AuditDensity,
    vouch: VouchPolicy,
}

impl GatherWitness {
    pub fn with_policies(policies: GatherPolicies) -> Self {
        Self {
            entries: HashMap::new(),
            audit: policies.audit,
            vouch: policies.vouch,
        }
    }

    /// Diagnostic policies retained across a device reset.
    pub fn policies(&self) -> GatherPolicies {
        GatherPolicies {
            audit: self.audit,
            vouch: self.vouch,
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn retire_task(&mut self, task_id: u32) {
        self.entries.retain(
            |key, _| !matches!(key, GatherKey::TaskGva { task_id: seen, .. } if *seen == task_id),
        );
    }

    pub fn retire_mapping(&mut self, mapping_id: u32) {
        self.entries.retain(
            |key, _| !matches!(key, GatherKey::Mapping { mapping, .. } if mapping.get() == mapping_id),
        );
    }

    pub fn previous_pages_epoch(&self, key: &GatherKey) -> Option<u64> {
        self.entries.get(key).map(|entry| entry.pages_epoch)
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn contains(&self, key: &GatherKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Apply one bind's writer readings and optional content audit.
    ///
    /// # Safety
    /// Every run in `window` must point to its declared number of live bytes
    /// for this call. The witness reads them when its audit is due.
    pub unsafe fn observe(
        &mut self,
        key: GatherKey,
        window: GatherWindow<'_>,
        readings: GatherReadings,
        fresh_generation: u64,
    ) -> GatherObservation {
        let GatherWindow {
            gpas,
            runs,
            span,
            page_size: _,
        } = window;
        let GatherReadings {
            pages_epoch,
            pages_wrote,
            stated_gen: stated_now,
            pending,
        } = readings;

        let stale = self
            .entries
            .get(&key)
            .is_none_or(|entry| entry.gpas != gpas || entry.span != span);
        if stale {
            self.entries.insert(
                key,
                Entry {
                    gpas: gpas.to_vec(),
                    span,
                    fold: 0,
                    fold_valid: false,
                    fold_seeded: false,
                    binds_since_fold: 0,
                    audit_armed: false,
                    rebaselines: 0,
                    pages_epoch,
                    stated_gen: stated_now,
                    generation: fresh_generation,
                },
            );
            return GatherObservation {
                verdict: GatherVerdict::Rearmed,
                audit: ContentAudit::Skipped,
                generation: fresh_generation,
                vouch: GatherVouch::Fresh,
                stated: StatedGuestWrite::Unaddressed,
            };
        }

        let density = self.audit;
        let entry = self.entries.get_mut(&key).expect("existing witness entry");
        let stated = match (entry.stated_gen, stated_now) {
            (Some(before), Some(now)) if before == now => StatedGuestWrite::Quiet,
            (Some(_), Some(_)) => StatedGuestWrite::Wrote,
            _ => StatedGuestWrite::Unaddressed,
        };
        entry.stated_gen = stated_now;

        let host_quiet = pages_wrote.is_some_and(|seen| !seen.wrote());
        let verdict = match stated {
            StatedGuestWrite::Unaddressed => GatherVerdict::Unarmed,
            StatedGuestWrite::Quiet if host_quiet => GatherVerdict::Vouched,
            StatedGuestWrite::Quiet | StatedGuestWrite::Wrote => GatherVerdict::Refused {
                guest_wrote: matches!(stated, StatedGuestWrite::Wrote),
                host_wrote_pages: !host_quiet,
            },
        };
        // Withholding is applied after the contract has spoken, never instead of
        // it, so the two arms differ only where a vouch was actually granted.
        let verdict = match (self.vouch, verdict) {
            (VouchPolicy::Withheld, GatherVerdict::Vouched) => GatherVerdict::Withheld,
            (_, granted) => granted,
        };
        let vouched = matches!(verdict, GatherVerdict::Vouched);

        let audit = if density == AuditDensity::Disabled {
            ContentAudit::Skipped
        // Both of the reach answers that license the gather, spelled out. A
        // `matches!` is not exhaustiveness-checked, so a new variant does not
        // fail the build here the way a `match` arm would -- naming
        // `Quiet` and `Disjoint` together is what keeps "the audit may run"
        // meaning the same thing on both routes.
        } else if !matches!(pending, GuestWriteReach::Quiet | GuestWriteReach::Disjoint) {
            entry.audit_armed = false;
            entry.fold_valid = false;
            entry.rebaselines = 0;
            entry.binds_since_fold = 0;
            ContentAudit::Indebted
        } else if entry.audit_armed {
            if vouched {
                let fold = unsafe { fold_runs(runs, span) };
                let result = if fold == entry.fold {
                    ContentAudit::Agreed
                } else {
                    ContentAudit::Disagreed
                };
                entry.fold = fold;
                entry.fold_valid = true;
                entry.audit_armed = density.stays_armed();
                entry.rebaselines = 0;
                entry.binds_since_fold = 0;
                result
            } else if entry.rebaselines < AUDIT_REBASELINE_LIMIT {
                entry.fold = unsafe { fold_runs(runs, span) };
                entry.fold_seeded = true;
                entry.fold_valid = true;
                entry.rebaselines += 1;
                ContentAudit::Rebaselined
            } else {
                entry.audit_armed = false;
                entry.rebaselines = 0;
                entry.fold_valid = false;
                entry.binds_since_fold = 0;
                ContentAudit::Restarted
            }
        } else if entry.binds_since_fold >= 1 {
            entry.fold = unsafe { fold_runs(runs, span) };
            entry.fold_seeded = true;
            entry.fold_valid = true;
            entry.audit_armed = true;
            entry.rebaselines = 0;
            entry.binds_since_fold = 0;
            ContentAudit::Seeded
        } else {
            entry.binds_since_fold += 1;
            entry.fold_valid &= vouched;
            ContentAudit::Skipped
        };

        // The fold is an instrument over the decoded contract, not another
        // source of device behavior. A disagreement is reported by the adapter
        // but cannot silently replace the identity the contract vouched for.
        let kept = vouched;
        if !kept {
            entry.generation = fresh_generation;
        }
        entry.pages_epoch = pages_epoch;
        GatherObservation {
            verdict,
            audit,
            generation: entry.generation,
            vouch: if kept {
                GatherVouch::Vouched
            } else {
                GatherVouch::Fresh
            },
            stated,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GatherReadings {
    pub pages_epoch: u64,
    pub pages_wrote: Option<HostWriteVerdict>,
    pub stated_gen: Option<StatedGeneration>,
    pub pending: GuestWriteReach,
}

pub struct GatherWindow<'a> {
    pub gpas: &'a [u64],
    pub runs: &'a [GuestRun],
    pub span: u64,
    pub page_size: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GatherVerdict {
    Rearmed,
    Unarmed,
    Vouched,
    Refused {
        guest_wrote: bool,
        host_wrote_pages: bool,
    },
    /// The contract granted a vouch and [`VouchPolicy::Withheld`] declined to
    /// take it. Its own variant rather than a `Refused`, because a refusal names
    /// a writer this device observed and this names an operator's diagnostic
    /// choice — folding the two would put the ablation's own binds into the
    /// counter a soundness sweep reads.
    Withheld,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StatedGuestWrite {
    Unaddressed,
    Quiet,
    Wrote,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContentAudit {
    Skipped,
    Seeded,
    Restarted,
    Rebaselined,
    Agreed,
    Disagreed,
    Indebted,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GatherObservation {
    pub verdict: GatherVerdict,
    pub audit: ContentAudit,
    pub generation: u64,
    pub vouch: GatherVouch,
    pub stated: StatedGuestWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatheredIdentity {
    pub key: u64,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatherOutcome {
    pub identity: GatheredIdentity,
    pub vouch: GatherVouch,
}

/// Fold a live gathered window for the witness's diagnostic audit.
///
/// # Safety
/// Every run must point to at least `len` live bytes for the duration of this call.
pub unsafe fn fold_runs(runs: &[GuestRun], span: u64) -> u128 {
    let mut a = 0x9e37_79b9_7f4a_7c15u64;
    let mut b = 0xc2b2_ae3d_27d4_eb4fu64;
    let mut remaining = span;
    for run in runs {
        if remaining == 0 {
            break;
        }
        let n = run.len.min(remaining) as usize;
        remaining -= n as u64;
        let bytes = unsafe { std::slice::from_raw_parts(run.host_ptr as *const u8, n) };
        let (words, tail) = bytes.split_at(n & !7);
        for chunk in words.chunks_exact(8) {
            let w = u64::from_le_bytes(chunk.try_into().expect("eight-byte chunk"));
            a = (a ^ w).rotate_left(29).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            b = b.rotate_left(7).wrapping_add(w ^ a);
        }
        for (i, &byte) in tail.iter().enumerate() {
            a ^= (byte as u64) << (8 * i);
        }
        b = b.wrapping_mul(0xff51_afd7_ed55_8ccd) ^ n as u64;
    }
    ((a as u128) << 64) | b as u128
}
