//! rt's part of reims-vgpu's trip line (`graphics::trip`): the stages of its command buffers' trip
//! through rt that rt's metrics of the process's device counted since the part before
//! ([`Stages::take`]). A diagnostic: the metrics are rt's own, which no Metal call answers.
//!
//! A part is `rt: ...` ([`line()`]): its commits' waits for rt's submission lock, in all; its
//! `vkQueueSubmit2` calls, the `VkSubmitInfo2`s they passed, their time inside the driver and the
//! submit thread's waits for rt's queue lock before them, in all; its calibrations and their time;
//! the groups a queue held behind its submissions in flight and the GPU's time from the work
//! before each to its start; then each of rt's stage histograms as `| name n sum_us p50_us
//! p90_us [bucket:n,...]`: `commit_to_scheduled`, `scheduled_to_gpu_start` (with the part behind
//! other work, `behind`), `gpu_run` and `gpu_end_to_handler` (with its six steps' sums, `steps`),
//! each in rt's buckets (bucket 0 under 2 us, bucket `i` from 2^i us to under 2^(i+1) us, the
//! last from 2^22 us), their quantiles placed in their buckets by rank. Times are in
//! microseconds, on rt's host clock.
//!
//! A device made after the one before reads as counted from none again: each count is what it rose
//! by, or what it is where it fell.

use std::fmt::Write as _;

use metal2vulkan_runtime as rt;

/// The metrics a part read last, to tell what rose since.
#[derive(Default)]
pub struct Stages {
    last: Option<rt::Metrics>,
}

impl Stages {
    /// The part of what rt counted since the part before, or since the device was made; `None`
    /// before a device is made.
    pub fn take(&mut self) -> Option<String> {
        let now = super::device::metrics()?;
        let part = line(&self.last.unwrap_or_default(), &now);
        self.last = Some(now);
        Some(part)
    }
}

/// What a count rose by from `before` to `now`; where it fell, a new device's, what it is.
fn rose(before: u64, now: u64) -> u64 {
    now.checked_sub(before).unwrap_or(now)
}

/// [`rose`], for each of an array's counts.
fn rose_each<const N: usize>(before: &[u64; N], now: &[u64; N]) -> [u64; N] {
    std::array::from_fn(|i| rose(before[i], now[i]))
}

/// The least span of rt's stage bucket `i`, in nanoseconds (the module doc).
fn floor_ns(i: usize) -> u64 {
    if i == 0 { 0 } else { 1_000 << i }
}

/// The span at fraction `q` of `buckets`' by nearest rank, placed within its bucket by its rank
/// there, in nanoseconds; 0 with none.
fn quantile(buckets: &[u64], q: f64) -> u64 {
    let n: u64 = buckets.iter().sum();
    if n == 0 {
        return 0;
    }
    let rank = ((q * n as f64).ceil() as u64).clamp(1, n);
    let mut below = 0;
    for (i, &c) in buckets.iter().enumerate() {
        if below + c >= rank {
            let floor = floor_ns(i);
            let top = if i + 1 < buckets.len() {
                floor_ns(i + 1)
            } else {
                floor
            };
            // The rank's middle within the bucket, so the span stays in it.
            let into = ((rank - below) as f64 - 0.5) / c as f64;
            return floor + ((top - floor) as f64 * into) as u64;
        }
        below += c;
    }
    0
}

/// `ns` nanoseconds in microseconds.
fn us(ns: u64) -> f64 {
    ns as f64 / 1e3
}

/// A stage's histogram `buckets` and its sum `ns`, as the module doc's `| name ...`.
fn stage(out: &mut String, name: &str, buckets: &[u64], ns: u64) {
    let n: u64 = buckets.iter().sum();
    let _ = write!(
        out,
        " | {name} {n} {:.1} {:.1} {:.1} [",
        us(ns),
        us(quantile(buckets, 0.5)),
        us(quantile(buckets, 0.9))
    );
    let mut sep = "";
    for (i, &c) in buckets.iter().enumerate() {
        if c != 0 {
            let _ = write!(out, "{sep}{i}:{c}");
            sep = ",";
        }
    }
    out.push(']');
}

/// The part of what rose from `before` to `now` (the module doc).
pub(crate) fn line(before: &rt::Metrics, now: &rt::Metrics) -> String {
    macro_rules! rose {
        ($field:ident) => {
            rose(before.$field, now.$field)
        };
    }
    let mut out = format!(
        "rt: commit_lock {:.1}; submits {} infos {} call {:.1} queue_lock {:.1}; calibrations {} \
         {:.1}; held_starts {} gap {:.1}",
        us(rose!(commit_lock_wait_nanos)),
        rose!(submissions),
        rose!(submit_infos),
        us(rose!(submit_call_nanos)),
        us(rose!(queue_lock_wait_nanos)),
        rose!(calibrations),
        us(rose!(calibration_nanos)),
        rose!(held_starts),
        us(rose!(held_start_gap_nanos)),
    );
    stage(
        &mut out,
        "commit_to_scheduled",
        &rose_each(&before.commit_to_scheduled, &now.commit_to_scheduled),
        rose!(commit_to_scheduled_nanos),
    );
    stage(
        &mut out,
        "scheduled_to_gpu_start",
        &rose_each(&before.scheduled_to_gpu_start, &now.scheduled_to_gpu_start),
        rose!(scheduled_to_gpu_start_nanos),
    );
    let _ = write!(
        out,
        " behind {:.1}",
        us(rose!(scheduled_to_gpu_start_behind_nanos))
    );
    stage(
        &mut out,
        "gpu_run",
        &rose_each(&before.gpu_run, &now.gpu_run),
        rose!(gpu_run_nanos),
    );
    stage(
        &mut out,
        "gpu_end_to_handler",
        &rose_each(&before.gpu_end_to_handler, &now.gpu_end_to_handler),
        rose!(gpu_end_to_handler_nanos),
    );
    let steps = rose_each(
        &before.gpu_end_to_handler_steps_nanos,
        &now.gpu_end_to_handler_steps_nanos,
    );
    let steps: Vec<String> = steps.iter().map(|&ns| format!("{:.1}", us(ns))).collect();
    let _ = write!(out, " steps {}", steps.join(","));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A part gives what each stage's histogram rose by, its sum and its quantiles in its buckets
    #[test]
    fn a_part_gives_what_each_stage_rose_by() {
        let before = rt::Metrics::default();
        let mut now = rt::Metrics::default();
        now.submissions = 3;
        now.submit_call_nanos = 150_000;
        // Nine runs of 10 to 16 us (bucket 3), one of 100 us (bucket 6).
        now.gpu_run[3] = 9;
        now.gpu_run[6] = 1;
        now.gpu_run_nanos = 190_000;
        now.gpu_end_to_handler_steps_nanos[0] = 2_000;
        let line = line(&before, &now);
        assert!(
            line.starts_with("rt: commit_lock 0.0; submits 3 infos 0 call 150.0 "),
            "{line}"
        );
        assert!(line.contains(" | gpu_run 10 190.0 "), "{line}");
        assert!(line.contains("[3:9,6:1]"), "{line}");
        assert!(line.ends_with(" steps 2.0,0.0,0.0,0.0,0.0,0.0"), "{line}");
        // The median falls in bucket 3: from 8 us to under 16 us.
        let p50 = quantile(&now.gpu_run, 0.5);
        assert!((8_000..16_000).contains(&p50), "{p50}");
        // A new device's counts read as what they are.
        assert!(super::line(&now, &before).contains(" | gpu_run 0 0.0 "));
    }
}
