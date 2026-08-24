//! Runtime reporting for the released-page write instrument.
//!
//! `reims-vgpu-core` owns the backend-independent watch and verdict. This
//! adapter supplies time, cadence, counters, and failure-channel emission.

pub use reims_vgpu_core::{ReleasedPages, ReleasedVerdict};

/// Judge watched pages and report writes that landed after guest release.
pub fn sweep(state: &mut crate::runtime::Device) {
    let now_us = crate::observe::elapsed_us();
    let crate::model::DeviceState {
        content,
        observations,
        ..
    } = &mut state.state;
    let host_writes = &content.host_writes;
    let watch = &mut observations.released_pages;
    if watch.watched() == 0 {
        return;
    }
    for (gpa, task_id, verdict) in watch.sweep(host_writes, now_us) {
        crate::runtime::drain::note_store_route(verdict.route());
        if let ReleasedVerdict::Wrote { since_us } = verdict {
            if crate::observe::first_sight("released_write_after_release", gpa) {
                crate::observe::fail(format!(
                    "released_pages reason={} task={task_id} gpa={gpa:#x} \
                     since_us={since_us} watched={} refused={} (this device wrote to a guest \
                     page after the guest released it; the guest is entitled to have given \
                     that page to something else, including its own page table)",
                    verdict.route(),
                    watch.watched(),
                    watch.refused(),
                ));
            }
        }
    }
}

/// Emit the watch population at most once per shared census interval.
pub fn note_levels(state: &crate::runtime::Device) {
    static LAST_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let watch = &state.observations.released_pages;
    if watch.watched() == 0 && watch.refused() == 0 {
        return;
    }
    if !claim_census_interval(&LAST_MS, crate::observe::elapsed_ms() as u64) {
        return;
    }
    crate::observe::off(format!(
        "released_pages_levels watching={} refused={} capacity={}",
        watch.watched(),
        watch.refused(),
        reims_vgpu_core::RELEASED_PAGE_WATCH_CAP,
    ));
}

const CENSUS_INTERVAL_MS: u64 = 1_000;

fn claim_census_interval(last_ms: &std::sync::atomic::AtomicU64, now_ms: u64) -> bool {
    use std::sync::atomic::Ordering;

    let last = last_ms.load(Ordering::Relaxed);
    if now_ms.saturating_sub(last) < CENSUS_INTERVAL_MS {
        return false;
    }
    last_ms
        .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tranche_rate_cannot_cross_the_census_gate() {
        let last = std::sync::atomic::AtomicU64::new(0);
        let base = 5_000;
        assert!(claim_census_interval(&last, base));
        assert!(
            (1..363).all(|index| { !claim_census_interval(&last, base + (1000 * index) / 363) })
        );
        assert!(claim_census_interval(&last, base + CENSUS_INTERVAL_MS));
    }

    #[test]
    fn a_stall_does_not_bank_intervals() {
        let last = std::sync::atomic::AtomicU64::new(0);
        assert!(claim_census_interval(&last, 1_000));
        assert!(claim_census_interval(&last, 11_000));
        assert!((1..=10).all(|offset| !claim_census_interval(&last, 11_000 + offset)));
    }
}
