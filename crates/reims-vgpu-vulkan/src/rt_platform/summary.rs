//! rt's part of the host's summary of each ten seconds (reims-vgpu's `graphics::summary`): what
//! rt's metrics of the process's device counted since the part before ([`Part::take`]). A
//! diagnostic: the metrics are rt's own, which no Metal call answers, and which lines the host
//! writes is free under the contract.
//!
//! A part is `rt: ...` ([`line()`]):
//! - the `VkPipeline`s rt made;
//! - its first-use waits, their count and time in all, then those for a kernel's variant and for
//!   a content's lines among them, and the longest since the device was made (rt keeps no longest
//!   of a period); its waits for a lowering are left out;
//! - its `vkQueueSubmit2` calls, their count and time inside the driver in all, those longer than
//!   100 ms, and the longest since the device was made; then the submit thread's waits for rt's
//!   queue lock before them, their time in all and the longest since the device was made;
//! - the bytes rt copied for calls: in all, then by cause: staging (`replace_region` and
//!   `get_bytes`), attachments' images into their shadows, and the over-copies between a texture
//!   over a buffer and its image of its own, by the cause rt gives each, those of none left out.
//!
//! A device made after the one before reads as counted from none again: each count is what it rose
//! by, or what it is where it fell.

use std::time::Duration;

use metal2vulkan_rt as rt;

/// The metrics a part read last, to tell what rose since.
#[derive(Default)]
pub struct Part {
    last: Option<rt::Metrics>,
}

impl Part {
    /// The part of what rt counted since the part before, or since the device was made; `None`
    /// before a device is made.
    pub fn take(&mut self) -> Option<String> {
        let now = super::device::metrics()?;
        let part = line(&self.last.unwrap_or_default(), &now);
        self.last = Some(now);
        Some(part)
    }
}

/// The part of what rose from `before` to `now` (the module doc).
pub(crate) fn line(before: &rt::Metrics, now: &rt::Metrics) -> String {
    // What a count rose by; where it fell, a new device's, what it is.
    macro_rules! rose {
        ($field:ident) => {
            now.$field.checked_sub(before.$field).unwrap_or(now.$field)
        };
    }
    let ms = |nanos: u64| Duration::from_nanos(nanos).as_secs_f64() * 1e3;
    let causes = [
        ("imported", rose!(over_copy_bytes_imported)),
        ("host memory", rose!(over_copy_bytes_host_memory)),
        ("dma-buf", rose!(over_copy_bytes_dma_buf)),
        ("dma-bufs", rose!(over_copy_bytes_dma_bufs)),
        ("dedicated", rose!(over_copy_bytes_dedicated)),
        ("surface", rose!(over_copy_bytes_surface)),
        ("spilled", rose!(over_copy_bytes_spilled)),
        ("linear", rose!(over_copy_bytes_linear)),
        ("linear features", rose!(over_copy_bytes_linear_features)),
        ("pitch", rose!(over_copy_bytes_pitch)),
        ("render reach", rose!(over_copy_bytes_render_reach)),
        ("alignment", rose!(over_copy_bytes_alignment)),
        ("image memory", rose!(over_copy_bytes_image_memory)),
    ];
    let by_cause: Vec<String> = (causes.iter())
        .filter(|(_, bytes)| *bytes > 0)
        .map(|(cause, bytes)| format!("{cause} {bytes}"))
        .collect();
    let (copied, attachments, over) = (
        rose!(bytes_copied),
        rose!(attachment_read_copy_bytes),
        rose!(over_copy_bytes),
    );
    let staging = copied.saturating_sub(attachments).saturating_sub(over);
    format!(
        "rt: pipelines made {}; first-use waits {} in {:.1} ms (variants {} in {:.1} ms, lines {} \
         in {:.1} ms), the longest since the device was made {:.1} ms; submits {} in {:.1} ms \
         ({} over 100 ms), the longest since the device was made {:.1} ms; queue-lock waits \
         {:.1} ms, the longest since the device was made {:.1} ms; bytes copied {copied}: \
         staging {staging}, attachment reads {attachments}, over-copies {over}{}",
        rose!(pipelines_created),
        rose!(first_use_waits),
        ms(rose!(first_use_wait_nanos)),
        rose!(first_use_variant_waits),
        ms(rose!(first_use_variant_wait_nanos)),
        rose!(first_use_line_waits),
        ms(rose!(first_use_line_wait_nanos)),
        ms(now.first_use_wait_max_nanos),
        rose!(submissions),
        ms(rose!(submit_call_nanos)),
        rose!(submit_calls_over_100_ms),
        ms(now.submit_call_max_nanos),
        ms(rose!(queue_lock_wait_nanos)),
        ms(now.queue_lock_wait_max_nanos),
        if by_cause.is_empty() {
            String::new()
        } else {
            format!(" ({})", by_cause.join(", "))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A part says what rose since the metrics before, the maxima since the device was made as
    /// they are, each over-copy cause with bytes by name and none without, staging as the copies
    /// neither of attachments nor over-copies; a count that fell, a new device's, reads as it is.
    #[test]
    fn an_rt_part_says_what_rose_with_each_copy_cause_by_name() {
        let mut before = rt::Metrics::default();
        before.pipelines_created = 10;
        before.first_use_waits = 1;
        before.first_use_wait_nanos = 2_000_000;
        before.bytes_copied = 1_000;
        let mut now = before;
        now.pipelines_created = 13;
        now.first_use_waits = 4;
        now.first_use_wait_nanos = 9_500_000;
        now.first_use_variant_waits = 2;
        now.first_use_variant_wait_nanos = 6_000_000;
        now.first_use_line_waits = 1;
        now.first_use_line_wait_nanos = 1_500_000;
        now.first_use_wait_max_nanos = 4_200_000;
        before.submissions = 100;
        before.submit_call_nanos = 50_000_000;
        before.queue_lock_wait_nanos = 1_000_000;
        // The maxima are since the device was made: written as they are, not as a rise.
        before.submit_call_max_nanos = 120_000_000;
        before.queue_lock_wait_max_nanos = 500_000;
        now.submissions = 160;
        now.submit_call_nanos = 50_000_000 + 312_500_000;
        now.submit_calls_over_100_ms = 2;
        now.submit_call_max_nanos = 140_000_000;
        now.queue_lock_wait_nanos = 1_000_000 + 2_300_000;
        now.queue_lock_wait_max_nanos = 900_000;
        now.bytes_copied = 1_000 + 4_096 + 512 + 64;
        now.attachment_read_copy_bytes = 512;
        now.over_copy_bytes = 4_096;
        now.over_copy_bytes_dma_buf = 4_000;
        now.over_copy_bytes_pitch = 96;
        assert_eq!(
            line(&before, &now),
            "rt: pipelines made 3; first-use waits 3 in 7.5 ms (variants 2 in 6.0 ms, lines 1 in \
             1.5 ms), the longest since the device was made 4.2 ms; submits 60 in 312.5 ms (2 \
             over 100 ms), the longest since the device was made 140.0 ms; queue-lock waits 2.3 \
             ms, the longest since the device was made 0.9 ms; bytes copied 4672: staging 64, \
             attachment reads 512, over-copies 4096 (dma-buf 4000, pitch 96)"
        );
        let mut fresh = rt::Metrics::default();
        fresh.pipelines_created = 2;
        let part = line(&now, &fresh);
        assert!(
            part.starts_with("rt: pipelines made 2; first-use waits 0 in 0.0 ms"),
            "{part}"
        );
        assert!(part.ends_with("over-copies 0"), "{part}");
    }
}
