# DMA-BUF validation and upstream cross-check

## September 11 clean-upstream validation

The patch applies to upstream master `69a57dd6`. Tests were rerun in a separate Linux source tree containing that base plus this patch, excluding the earlier lab commits. QEMU uses base `bd88218da09b86ed9c78bf5f9354168812a7ba6b` plus the companion patch.

- Memory owner: 22 tests passed, including a fresh master-based rerun.
- Clean upstream + patch, Vulkan library without host-window: 2,166 passed, two ignored.
- Clean upstream + patch, Vulkan library with host-window: 2,231 passed, two ignored.
- Explicit ignored GPU integration test: three successive passes on RADV. Each exercises contiguous and reordered shared backing, both data-transfer directions, guards, host-pointer refusal followed by DMA-BUF success, and retirement.
- Workspace formatting passed; nonexperimental Vulkan/host-window cargo check passed; patch whitespace and build-script syntax checks passed.
- QEMU release build with experimental DMA-BUF completed and reported the PCI device and KVM accelerator available.
- Strict clippy is not clean: it stops on four pre-existing `chunks_exact_to_as_chunks` diagnostics in the unchanged observation crate. Root-only clippy completes with 25 warnings. Warnings were not suppressed or gates weakened.
- A fresh local native-oracle attempt exits 2 with `CASE device FAIL no Metal device`; this is an infrastructure limitation, not a guest result.

The first clean-source test attempt lacked the QEMU submodule and failed its cursor-source fixture (2,165 passed, one failure); supplying the matching QEMU source made the complete rerun pass. This was a test-tree setup error, not a hidden ignored failure.

## Earlier live evidence — different source lineage

The September 10/11 live binary was based on local `2368c553`, which includes additional lab fixes beyond upstream master. Its SHA-256 was `6fa59d565201a0cb461bfc962eec14499d4abc0509c4aa8f0eff24221a00e90e`. Do not attribute its complete behavior solely to this master-based patch.

The same indexed-draw executable produced byte-identical baseline/candidate outputs: eight cases, five passes and three failures (negative base vertex for 16/32-bit indices, and indexed base instance). A later screen-saver-disabled run provided approximately 7 h 56 min of monitored playback before intentional shutdown: 23,371 imports before stop request, zero export declines, zero rendering failures and zero device-error entries, with four unresolved page-guard warnings. RSS rose from about 9.16 GiB to a peak 16.54 GiB, later remaining near 16.5 GiB for a 16 GiB guest. No actual media-time, dropped-frame or audio-quality measurements were captured.

An earlier run encountered the Ventura screen saver and accumulated rendering failures. Explicitly starting that screen saver reproduced the signature on the pre-DMA-BUF baseline; see upstream issue #84. It is not a passed overnight playback test.

## What remains unproven

Page-guard warnings do not establish active page-table corruption: the observer can see a page reused between node sightings. DMA-BUF is not claimed to fix those warnings, release-ordering defects, audio skipping, boot artifacts or performance. Broad failure injection, concurrent access, pinned-resource budgets, a fresh native oracle and repeated matched guest controls remain qualification work. Feature availability stays opt-in.

## Related upstream work checked

- Historical DMA-BUF introduction: `089280a93fecc1c1397119a2ce6f055c0805e401`.
- Replacement by RAMBlock host-pointer imports: `159fde84a72458dfd3dc7e4296913a7751eda58f`; subsequent ABI v18 callback removal.
- Issue #29: prior DMA-BUF measurements and host-pointer import declines.
- Issue #70 and PR #79: guest page-release / outstanding-write correctness.
- PR #77: alias lifetime and presentation changes.
- PR #81: direct-writeback texel accounting.
- PR #83: missing-layout diagnostics.

No equivalent current fallback PR was found in the inspected master history, open/closed PR listing and import/DMA-BUF issue searches. This is not a claim to have inspected every unadvertised fork or unpublished branch. This proposal revisits an existing transport with a new narrow integration and RADV backing-specific evidence.

## Fresh-build playback observation

During the September 11 clean-master-based candidate boot, the user reported
that restored YouTube playback was very poor compared with the daytime lab
build. No matched FPS or dropped-frame benchmark was captured. This is an
adverse observation, not a performance pass, and its cause has not been
isolated between the prior local fixes, configuration/workload differences,
and the conservative serialized import path. The earlier smooth playback
must not be attributed to DMA-BUF alone.

## Final candidate and broad guest comparison

The final candidate additionally includes the earlier packed-alias retirement
fix (resource replacement/unmap/task/reset now queues owned packed aliases for
fence-safe cleanup), overlapping the lifetime portion of upstream PR #77.
Its Vulkan/host-window library tests pass: 2,235 passed, two ignored. This
extra fix is not claimed to explain the earlier smooth playback or to resolve
all conformance differences.

Using identical cloned starting disks, 16 GiB/eight virtual CPUs and the same
full-suite executable, the initial clean-master-based feature-enabled build
reported `SUMMARY cases=283 failures=18 skipped=11`; its nonexperimental
control reported `SUMMARY cases=283 failures=16 skipped=11`. The final
candidate with packed retirement reported `SUMMARY cases=283 failures=17
skipped=11`. The suite also emits 294 CASE records because skipped cases are
outside the summary's cases count. These are not clean conformance passes.

The initial candidate differed from control only on
`draw_indexed_uint16_offset_4` and `draw_indexed_uint32_offset_4`; the final
candidate differed from that control only on the latter. Crucially,
`draw_nonindexed_shared_shader_control` failed with `lit=0` in all three full
runs. Both indexed checks compare pixels to that failed reference, so the
additional FAIL labels do not by themselves prove an indexed rendering
regression; nor are they waived as harmless. Fresh repeated matched controls
and independent expected-pixel validation are required to resolve them.
The nonexperimental live control preceded the packed-retirement addition,
which is another limit on attributing the final difference.

The final candidate's separate `--indexed-draw-only` process reported eight
cases, three failures: reference draw PASS (`lit=1094`), both four-byte-offset
indexed draws PASS, and the same three negative-base-vertex/base-instance
failures as earlier testing. All outputs are included under
`docs/validation/shared-backing` without captured shaders or guest binaries.

The repository's stored Apple M4 native baseline scores the initial guest run
with 18 unexplained failures; there is no fresh working native oracle here.
The failed reference and differing full-suite results remain explicit draft
qualification blockers. No test assertion or failure inventory was weakened.

Final-source rechecks: nonexperimental Vulkan/host-window library tests also
pass (2,235 passed, one ignored), final workspace formatting passes, and the
explicit DMA-BUF GPU test passes again after the lifetime addition. All test
VMs were stopped after evidence collection.

Binary SHA-256 identities:

- Final candidate: `9baf9b527ded4666f08edf1ced81ebbf0cda162d6a3939df171ddf8e2bc83d29`.
- Initial feature-disabled control: `481c5268b7c2194a10e03338d9b8fce12ee61627e264eda9fd5e79d3c0e80ce9`.
- Initial candidate: `2c3b09305f64fcdfeb30deac9cd1d20b8506b3a6e41e489eb42ba107eff1cb2c`.
