//! rt's part of the host's summary of each ten seconds (reims-vgpu's `graphics::summary`): what
//! rt's metrics of the process's device counted since the part before, and some of its gauges now
//! ([`Part::take`]). A diagnostic: the metrics and gauges are rt's own, which no Metal call
//! answers, and which lines the host writes is free under the contract.
//!
//! A part is `rt: ...` ([`line()`]):
//! - the `VkPipeline`s rt made;
//! - its first-use waits, their count and time in all, then those for a kernel's variant, for a
//!   content's lines and for its points among them, and the longest since the device was made (rt
//!   keeps no longest of a period); its waits for a lowering are left out;
//! - its `vkQueueSubmit2` calls, their count and time inside the driver in all, those longer than
//!   100 ms, and the longest since the device was made; then the submit thread's waits for rt's
//!   queue lock before them, their time in all and the longest since the device was made;
//! - rt's own command buffers (its uploads and Managed buffers' copy-ins) it submitted, and their
//!   GPU time in all, from their own timestamp pairs;
//! - the bytes rt copied for calls: in all, then by cause: staging (`replace_region` and
//!   `get_bytes`), attachments' images into their shadows, and the over-copies between a texture
//!   over a buffer and its image of its own, by the cause rt gives each, those of none left out;
//! - `counted`, more of what rt counted, by their fields' names: new textures' images zeroed,
//!   their bytes and the barriers that moved them home after it; the acquires of another API's
//!   writes; the barriers rt recorded; the bytes its commits' copy-ins copied from Managed
//!   buffers' bytes into their device copies; the releases of a texture or block an unsent upload
//!   held, their bytes, and the sends of uploads they started; the sends of groups a queue held
//!   behind its submissions in flight, and the GPU's nanoseconds from the end of the work submitted
//!   before each to its own start; and a pipeline's makes split in two, the nanoseconds of the
//!   driver's `vkCreate*Pipelines` calls and of rt's calls into the translator; and the command
//!   buffers whose GPU times stay 0, those that found no timestamp pair and those whose pair read
//!   as not available;
//! - `gauges`, some of rt's gauges as they are now, by their fields' names: the live textures,
//!   those held in an image of their own beside their buffer, the bytes of textures' places in
//!   texture blocks, of textures' allocations of their own and of the spills; then the bytes of
//!   the texture blocks it holds, of the sparse ones among them and their count; and the bytes of
//!   the Private allocations that hold taken blocks, of the taken blocks in them, and of the
//!   sparse allocations among them and their count.
//!
//! A device made after the one before reads as counted from none again: each count is what it rose
//! by, or what it is where it fell.

use std::time::Duration;

use metal2vulkan_runtime as rt;

/// The metrics a part read last, to tell what rose since.
#[derive(Default)]
pub struct Part {
    last: Option<rt::Metrics>,
}

impl Part {
    /// The part of what rt counted since the part before, or since the device was made, with its
    /// gauges now; `None` before a device is made.
    pub fn take(&mut self) -> Option<String> {
        let now = super::device::metrics()?;
        let gauges = super::device::gauges().unwrap_or_default();
        let part = line(&self.last.unwrap_or_default(), &now, &gauges);
        self.last = Some(now);
        Some(part)
    }
}

/// The live textures (each image and each view) and buffers of the process's latest device, as
/// rt's gauges count them now, for the summary's line of what the host keeps (reims-vgpu's
/// `graphics::summary::kept`); `None` before a device is made.
pub fn live_textures_and_buffers() -> Option<(u64, u64)> {
    let gauges = super::device::gauges()?;
    Some((gauges.textures, gauges.buffers))
}

/// The part of what rose from `before` to `now`, with `gauges` as they are (the module doc).
pub(crate) fn line(before: &rt::Metrics, now: &rt::Metrics, gauges: &rt::Gauges) -> String {
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
    // What rose, and the gauges as they are, each by its field's name.
    macro_rules! rose_by_name {
        ($($field:ident),* $(,)?) => {
            [$((stringify!($field), rose!($field))),*]
        };
    }
    macro_rules! gauges_by_name {
        ($($field:ident),* $(,)?) => {
            [$((stringify!($field), gauges.$field)),*]
        };
    }
    let counted = rose_by_name!(
        texture_clears,
        texture_clear_bytes,
        texture_clear_barriers,
        external_acquires,
        barriers,
        managed_bytes_captured,
        held_releases,
        held_release_bytes,
        held_release_starts,
        held_starts,
        held_start_gap_nanos,
        pipeline_creation_nanos,
        translation_nanos,
        untimed_command_buffers,
        stamps_not_available,
    );
    let gauged = gauges_by_name!(
        textures,
        textures_copied_over,
        texture_block_bytes,
        texture_own_bytes,
        spilled_bytes,
        texture_block_held_bytes,
        texture_block_sparse_bytes,
        texture_block_sparse_blocks,
        private_take_held_bytes,
        private_take_bytes,
        private_take_sparse_bytes,
        private_take_sparse_allocations,
    );
    format!(
        "rt: pipelines made {}; first-use waits {} in {:.1} ms (variants {} in {:.1} ms, lines {} \
         in {:.1} ms, points {} in {:.1} ms), the longest since the device was made {:.1} ms; \
         submits {} in {:.1} ms ({} over 100 ms), the longest since the device was made {:.1} \
         ms; queue-lock waits {:.1} ms, the longest since the device was made {:.1} ms; own \
         command buffers {} in {:.1} ms on the GPU; bytes copied {copied}: staging {staging}, \
         attachment reads {attachments}, over-copies {over}{}; counted {}; gauges {}",
        rose!(pipelines_created),
        rose!(first_use_waits),
        ms(rose!(first_use_wait_nanos)),
        rose!(first_use_variant_waits),
        ms(rose!(first_use_variant_wait_nanos)),
        rose!(first_use_line_waits),
        ms(rose!(first_use_line_wait_nanos)),
        rose!(first_use_point_waits),
        ms(rose!(first_use_point_wait_nanos)),
        ms(now.first_use_wait_max_nanos),
        rose!(submissions),
        ms(rose!(submit_call_nanos)),
        rose!(submit_calls_over_100_ms),
        ms(now.submit_call_max_nanos),
        ms(rose!(queue_lock_wait_nanos)),
        ms(now.queue_lock_wait_max_nanos),
        rose!(runtime_command_buffers),
        ms(rose!(runtime_command_buffer_gpu_nanos)),
        if by_cause.is_empty() {
            String::new()
        } else {
            format!(" ({})", by_cause.join(", "))
        },
        by_name(&counted),
        by_name(&gauged),
    )
}

/// `NAME VALUE` of each of `named`, in its order.
fn by_name(named: &[(&str, u64)]) -> String {
    let pairs: Vec<String> = (named.iter())
        .map(|(name, value)| format!("{name} {value}"))
        .collect();
    pairs.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A part says what rose since the metrics before, the maxima since the device was made as
    /// they are, each over-copy cause with bytes by name and none without, staging as the copies
    /// neither of attachments nor over-copies, then the other counts' rises and the gauges by name,
    /// the gauges as they are; a count that fell, a new device's, reads as it is.
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
        now.first_use_point_waits = 1;
        now.first_use_point_wait_nanos = 500_000;
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
        before.runtime_command_buffers = 30;
        before.runtime_command_buffer_gpu_nanos = 4_000_000;
        now.runtime_command_buffers = 30 + 12;
        now.runtime_command_buffer_gpu_nanos = 4_000_000 + 1_700_000;
        now.bytes_copied = 1_000 + 4_096 + 512 + 64;
        now.attachment_read_copy_bytes = 512;
        now.over_copy_bytes = 4_096;
        now.over_copy_bytes_dma_buf = 4_000;
        now.over_copy_bytes_pitch = 96;
        before.texture_clears = 5;
        before.texture_clear_bytes = 1_000_000;
        before.barriers = 40;
        before.managed_bytes_captured = 1_000_000;
        now.texture_clears = 7;
        now.texture_clear_bytes = 1_000_000 + 524_288;
        now.texture_clear_barriers = 1;
        now.external_acquires = 2;
        now.barriers = 40 + 9;
        now.managed_bytes_captured = 1_000_000 + 98_304;
        now.held_releases = 3;
        now.held_release_bytes = 16_777_216 + 65_536;
        now.held_release_starts = 1;
        before.held_starts = 4;
        before.held_start_gap_nanos = 1_000_000;
        now.held_starts = 4 + 6;
        now.held_start_gap_nanos = 1_000_000 + 250_000;
        before.pipeline_creation_nanos = 2_000_000;
        before.translation_nanos = 500_000;
        now.pipeline_creation_nanos = 2_000_000 + 7_250_000;
        now.translation_nanos = 500_000 + 3_125_000;
        now.untimed_command_buffers = 4;
        now.stamps_not_available = 1;
        let mut gauges = rt::Gauges::default();
        gauges.textures = 12;
        gauges.textures_copied_over = 2;
        gauges.texture_block_bytes = 20_971_520;
        gauges.texture_own_bytes = 33_554_432;
        gauges.spilled_bytes = 0;
        gauges.texture_block_held_bytes = 50_331_648;
        gauges.texture_block_sparse_bytes = 16_777_216;
        gauges.texture_block_sparse_blocks = 1;
        gauges.private_take_held_bytes = 33_554_432;
        gauges.private_take_bytes = 18_874_368;
        gauges.private_take_sparse_bytes = 16_777_216;
        gauges.private_take_sparse_allocations = 1;
        assert_eq!(
            line(&before, &now, &gauges),
            "rt: pipelines made 3; first-use waits 3 in 7.5 ms (variants 2 in 6.0 ms, lines 1 in \
             1.5 ms, points 1 in 0.5 ms), the longest since the device was made 4.2 ms; submits 60 \
             in 312.5 ms (2 over 100 ms), the longest since the device was made 140.0 ms; \
             queue-lock waits 2.3 ms, the longest since the device was made 0.9 ms; own command \
             buffers 12 in 1.7 ms on the GPU; bytes copied 4672: staging 64, attachment reads 512, \
             over-copies 4096 (dma-buf 4000, pitch 96); counted texture_clears 2, \
             texture_clear_bytes 524288, texture_clear_barriers 1, external_acquires 2, barriers \
             9, managed_bytes_captured 98304, held_releases 3, held_release_bytes 16842752, \
             held_release_starts 1, held_starts 6, held_start_gap_nanos 250000, \
             pipeline_creation_nanos 7250000, translation_nanos 3125000, untimed_command_buffers \
             4, stamps_not_available 1; gauges textures 12, \
             textures_copied_over 2, texture_block_bytes 20971520, texture_own_bytes 33554432, \
             spilled_bytes 0, texture_block_held_bytes 50331648, texture_block_sparse_bytes \
             16777216, texture_block_sparse_blocks 1, private_take_held_bytes 33554432, \
             private_take_bytes 18874368, private_take_sparse_bytes 16777216, \
             private_take_sparse_allocations 1"
        );
        let mut fresh = rt::Metrics::default();
        fresh.pipelines_created = 2;
        let part = line(&now, &fresh, &rt::Gauges::default());
        assert!(
            part.starts_with("rt: pipelines made 2; first-use waits 0 in 0.0 ms"),
            "{part}"
        );
        assert!(
            part.ends_with(
                "over-copies 0; counted texture_clears 0, texture_clear_bytes 0, \
                 texture_clear_barriers 0, external_acquires 0, barriers 0, \
                 managed_bytes_captured 0, held_releases 0, held_release_bytes 0, \
                 held_release_starts 0, held_starts 0, held_start_gap_nanos 0, \
                 pipeline_creation_nanos 0, translation_nanos 0, untimed_command_buffers 0, \
                 stamps_not_available 0; gauges textures 0, textures_copied_over 0, \
                 texture_block_bytes 0, texture_own_bytes 0, spilled_bytes 0, \
                 texture_block_held_bytes 0, texture_block_sparse_bytes 0, \
                 texture_block_sparse_blocks 0, private_take_held_bytes 0, private_take_bytes 0, \
                 private_take_sparse_bytes 0, private_take_sparse_allocations 0"
            ),
            "{part}"
        );
    }
}
