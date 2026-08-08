//! Host GPU memory-topology classification and allocation policy.
//!
//! The engine supports **two** memory topologies as first-class targets:
//!
//! | Topology | Examples | Property that matters |
//! |---|---|---|
//! | [`MemoryTopology::Unified`] | Apple M-series, Intel/AMD iGPU, llvmpipe | GPU-local memory *is* cached CPU memory, so a staging hop is pure waste |
//! | [`MemoryTopology::Discrete`] | RTX 5080, GTX 750 Ti, Arc, RX 7900 | GPU-local memory is across PCIe, so bulk transfer must be a DMA, not a CPU map |
//!
//! ## The classification rule
//!
//! A device is `Unified` when **either** structural signal holds:
//!
//! * **A — no separate host heap:** every advertised memory heap is
//!   `DEVICE_LOCAL`. A discrete GPU always advertises the system-RAM heap
//!   without `DEVICE_LOCAL`; a single-pool device has no reason to.
//! * **B — cached direct access:** some memory type carries
//!   `DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED`. A PCIe BAR window into VRAM
//!   is mapped write-combining and is advertised `HOST_COHERENT` *without*
//!   `HOST_CACHED` (this is what distinguishes a resizable-BAR discrete GPU,
//!   which does expose a whole-VRAM host-visible type, from a real UMA part).
//!
//! Signal B exists so an AMD APU that advertises its GTT heap without
//! `DEVICE_LOCAL` is not mistaken for a discrete part.
//!
//! ## Misclassification is a performance bug, never a correctness bug
//!
//! Topology selects a *preference order* only. [`select_memory_type`] always
//! falls back to the required flags alone, so a device classified the wrong way
//! still allocates valid memory — it just pays a copy it did not need (or skips
//! one it wanted). Nothing in the engine may branch on topology in a way that
//! changes what the guest observes; that invariant is asserted by the tests at
//! the bottom of this file.

use ash::vk;

/// How host and device memory relate on the bound physical device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryTopology {
    /// One physical pool shared by CPU and GPU (Apple M-series, Intel/AMD iGPU).
    Unified,
    /// Device memory is separate from system RAM and reached over a bus.
    Discrete,
}

impl MemoryTopology {
    /// Stable slug for logs, proxy lines, and the matrix cell name.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Unified => "unified",
            Self::Discrete => "discrete",
        }
    }
}

/// What a piece of memory is *for*. The engine asks for a class; this module
/// turns the class into flags. No call site spells `MemoryPropertyFlags`
/// directly, so a topology decision is made in exactly one place.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryClass {
    /// CPU writes once, GPU reads: vertex/index/uniform/staging uploads.
    Upload,
    /// GPU writes, CPU reads: present capture and target readback.
    Readback,
    /// GPU-only working set: render targets, sampled textures, scratch buffers.
    DeviceLocal,
    /// GPU-side working memory that *wants* to be device-local but must never
    /// fail to allocate — the host-window staging image, where a
    /// host-memory placement is slower but still correct. Distinct from
    /// [`MemoryClass::DeviceLocal`], which is a hard requirement.
    DeviceLocalPreferred,
}

/// A memory-type query: the flags that MUST be present, plus a ranked list of
/// bonus flag sets, best first. [`select_memory_type`] takes the first
/// candidate matching the highest-ranked bonus it can satisfy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRequest {
    pub required: vk::MemoryPropertyFlags,
    /// Best-first. An empty list means "required flags only".
    pub preferred: Vec<vk::MemoryPropertyFlags>,
}

impl MemoryTopology {
    /// Flags to request for a [`MemoryClass`] on this topology.
    ///
    /// The only topology-dependent choices:
    ///
    /// * `Upload` on `Unified` prefers `DEVICE_LOCAL` — the same DRAM, so the
    ///   GPU reads the CPU's writes with no transfer at all. On `Discrete` a
    ///   `DEVICE_LOCAL|HOST_VISIBLE` type is a scarce BAR window (256 MiB
    ///   without resizable BAR); spending it on bulk uploads starves the paths
    ///   that genuinely need it, so plain host memory + a DMA copy is correct.
    /// * `Readback` on `Unified` prefers `DEVICE_LOCAL|HOST_CACHED` — the
    ///   render target's own pool, so the copy is a same-pool blit and the CPU
    ///   read is cached. On `Discrete` the buffer must live in system RAM
    ///   (`HOST_CACHED`); reading a BAR window from the CPU is uncached and
    ///   catastrophically slow.
    ///
    /// `Readback` requires only `HOST_VISIBLE`, and that is the whole point of
    /// the class. Adding `HOST_COHERENT` to the requirement reads as harmless —
    /// Vulkan guarantees a `HOST_VISIBLE|HOST_COHERENT` type exists, so the
    /// selection never fails — and on any driver whose cached type is *not*
    /// coherent it silently discards both preferences and lands every readback
    /// in uncached memory. Intel ANV is exactly that driver: its five types are
    /// `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` (0x07) and
    /// `DEVICE_LOCAL|HOST_VISIBLE|HOST_CACHED` (0x0b), and nothing carries both
    /// bits. Measured cost of the fallback on an Intel ARL iGPU: 460 MB/s for a
    /// full-target readback, 7-11 ms per 3.2 MB frame, 70-86 % of all draw time.
    ///
    /// So coherence is the *last* preference rather than a requirement, and a
    /// caller that gets a non-coherent type owes
    /// `vkInvalidateMappedMemoryRanges` before it reads. [`MemoryRequest`] is
    /// only a query; `ResourcePools::create_readback_buffer` records which it got.
    pub fn request(self, class: MemoryClass) -> MemoryRequest {
        use vk::MemoryPropertyFlags as F;
        let host = F::HOST_VISIBLE | F::HOST_COHERENT;
        match (class, self) {
            (MemoryClass::Upload, Self::Unified) => MemoryRequest {
                required: host,
                preferred: vec![F::DEVICE_LOCAL],
            },
            (MemoryClass::Upload, Self::Discrete) => MemoryRequest {
                required: host,
                preferred: Vec::new(),
            },
            (MemoryClass::Readback, Self::Unified) => MemoryRequest {
                required: F::HOST_VISIBLE,
                preferred: vec![
                    F::DEVICE_LOCAL | F::HOST_CACHED,
                    F::HOST_CACHED,
                    F::HOST_COHERENT,
                ],
            },
            (MemoryClass::Readback, Self::Discrete) => MemoryRequest {
                required: F::HOST_VISIBLE,
                preferred: vec![F::HOST_CACHED, F::HOST_COHERENT],
            },
            (MemoryClass::DeviceLocal, _) => MemoryRequest {
                required: F::DEVICE_LOCAL,
                preferred: Vec::new(),
            },
            (MemoryClass::DeviceLocalPreferred, _) => MemoryRequest {
                required: F::empty(),
                preferred: vec![F::DEVICE_LOCAL],
            },
        }
    }
}

/// Summary of the bound device's memory layout: the topology plus the sizes the
/// VRAM proxy reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryProfile {
    pub topology: MemoryTopology,
    /// Bytes in the largest `DEVICE_LOCAL` heap.
    pub device_local_bytes: u64,
    /// Bytes in the largest heap reachable through a `DEVICE_LOCAL|HOST_VISIBLE`
    /// memory type — 0 when the CPU cannot address device memory at all.
    pub host_visible_device_local_bytes: u64,
    /// Which structural signal decided the classification (for the one-shot
    /// selection log line; a misclassification report must say *why*).
    pub signal: TopologySignal,
}

/// Which rule fired in [`classify_memory`]. Recorded so a wrong call on an
/// unfamiliar driver can be diagnosed from one log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TopologySignal {
    /// Signal A: every heap is `DEVICE_LOCAL`.
    NoHostOnlyHeap,
    /// Signal B: a `DEVICE_LOCAL|HOST_VISIBLE|HOST_CACHED` type exists.
    CachedDeviceLocal,
    /// Neither signal fired.
    SeparateHostHeap,
}

impl TopologySignal {
    pub fn slug(self) -> &'static str {
        match self {
            Self::NoHostOnlyHeap => "no_host_only_heap",
            Self::CachedDeviceLocal => "cached_device_local",
            Self::SeparateHostHeap => "separate_host_heap",
        }
    }
}

/// Classify a device's memory layout. Pure over
/// `VkPhysicalDeviceMemoryProperties`, so every row of the support matrix is
/// testable without that GPU present.
pub fn classify_memory(props: &vk::PhysicalDeviceMemoryProperties) -> MemoryProfile {
    use vk::MemoryPropertyFlags as F;
    let heaps = &props.memory_heaps[..props.memory_heap_count as usize];
    let types = &props.memory_types[..props.memory_type_count as usize];

    // Signal A — no heap lacks DEVICE_LOCAL. An empty heap list cannot be
    // unified by omission, so require at least one heap.
    let no_host_only_heap = !heaps.is_empty()
        && heaps
            .iter()
            .all(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL));

    // Signal B — the CPU can *cache* a device-local mapping. A PCIe BAR window
    // is write-combining and never advertises HOST_CACHED.
    let cached_device_local = types.iter().any(|t| {
        t.property_flags
            .contains(F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_CACHED)
    });

    let (topology, signal) = if no_host_only_heap {
        (MemoryTopology::Unified, TopologySignal::NoHostOnlyHeap)
    } else if cached_device_local {
        (MemoryTopology::Unified, TopologySignal::CachedDeviceLocal)
    } else {
        (MemoryTopology::Discrete, TopologySignal::SeparateHostHeap)
    };

    let heap_bytes = |idx: u32| heaps.get(idx as usize).map_or(0, |h| h.size);
    let device_local_bytes = heaps
        .iter()
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size)
        .max()
        .unwrap_or(0);
    let host_visible_device_local_bytes = types
        .iter()
        .filter(|t| t.property_flags.contains(F::DEVICE_LOCAL | F::HOST_VISIBLE))
        .map(|t| heap_bytes(t.heap_index))
        .max()
        .unwrap_or(0);

    MemoryProfile {
        topology,
        device_local_bytes,
        host_visible_device_local_bytes,
        signal,
    }
}

/// Pick a memory type index satisfying `req` within `type_bits`.
///
/// Tries each `preferred` set best-first, then the required flags alone. The
/// required-only fallback is what makes a topology misclassification a
/// performance bug rather than an allocation failure.
pub fn select_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    req: &MemoryRequest,
) -> Option<u32> {
    let carries = |index: u32, flags: vk::MemoryPropertyFlags| {
        (type_bits & (1u32 << index)) != 0
            && props.memory_types[index as usize]
                .property_flags
                .contains(flags)
    };
    let find = |flags: vk::MemoryPropertyFlags| {
        (0..props.memory_type_count).find(|&index| carries(index, flags))
    };
    req.preferred
        .iter()
        .find_map(|bonus| find(req.required | *bonus))
        .or_else(|| find(req.required))
}

/// The host-side cost properties of a selected memory type, for a caller that
/// maps it.
///
/// A mapping is only free to read if it is cached, and only free of explicit
/// `vkInvalidate`/`vkFlushMappedMemoryRanges` if it is coherent. Those two are
/// independent, and [`MemoryClass::Readback`] deliberately ranks the first above
/// the second, so an allocation site cannot assume either. Reading the bits is
/// the topology policy's job (nothing outside `caps/` may name a
/// `MemoryPropertyFlags`), which is why this lives here rather than at the site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MappedMemoryKind {
    pub cached: bool,
    pub coherent: bool,
}

impl MappedMemoryKind {
    pub fn of(props: &vk::PhysicalDeviceMemoryProperties, index: u32) -> Self {
        use vk::MemoryPropertyFlags as F;
        let flags = props.memory_types[index as usize].property_flags;
        Self {
            cached: flags.contains(F::HOST_CACHED),
            coherent: flags.contains(F::HOST_COHERENT),
        }
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    //! Synthetic `VkPhysicalDeviceMemoryProperties` for every device family in
    //! the support matrix, so both memory rows are covered by unit tests on any
    //! machine. The Apple layout is transcribed from a live `vulkaninfo` on the
    //! M3 Max dev host; the rest follow each driver's documented layout.
    use ash::vk;

    const GIB: u64 = 1 << 30;

    pub fn build(
        heaps: &[(u64, vk::MemoryHeapFlags)],
        types: &[(u32, vk::MemoryPropertyFlags)],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut p = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: heaps.len() as u32,
            memory_type_count: types.len() as u32,
            ..Default::default()
        };
        for (i, (size, flags)) in heaps.iter().enumerate() {
            p.memory_heaps[i] = vk::MemoryHeap {
                size: *size,
                flags: *flags,
            };
        }
        for (i, (heap, flags)) in types.iter().enumerate() {
            p.memory_types[i] = vk::MemoryType {
                heap_index: *heap,
                property_flags: *flags,
            };
        }
        p
    }

    /// Apple M3 Max / MoltenVK 1.4.1 — transcribed from `vulkaninfo` on the
    /// arm64 dev host: one 64 GiB DEVICE_LOCAL heap, three types.
    pub fn apple_m3_max() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        build(
            &[(64 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL)],
            &[
                (0, F::DEVICE_LOCAL),
                (
                    0,
                    F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
                ),
                (0, F::DEVICE_LOCAL | F::LAZILY_ALLOCATED),
            ],
        )
    }

    /// Intel integrated (Mesa ANV): one DEVICE_LOCAL heap over system RAM.
    ///
    /// Transcribed from `vulkaninfo` on the x86 dev host (Intel ARL,
    /// 2026-07-30): one 70 GiB DEVICE_LOCAL heap and five types, `0x07` and
    /// `0x0b` each appearing twice with a PROTECTED type between them.
    ///
    /// **Nothing here carries `HOST_COHERENT` and `HOST_CACHED` together**, and
    /// that is the load-bearing property. The previous version of this fixture
    /// gave type 1 both bits — invented, not measured — which made every
    /// `MemoryClass::Readback` selection test pass while the real device fell
    /// through to uncached type 0 on every allocation. A fixture more capable
    /// than the hardware it stands for cannot fail the way the hardware does.
    pub fn intel_igpu() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        let coherent = F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT;
        let cached = F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_CACHED;
        build(
            &[(70 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL)],
            &[
                (0, coherent),
                (0, cached),
                (0, F::DEVICE_LOCAL | F::PROTECTED),
                (0, coherent),
                (0, cached),
            ],
        )
    }

    /// AMD APU (RADV) advertising its GTT heap WITHOUT `DEVICE_LOCAL`. Signal A
    /// misses here; signal B is what keeps it out of the discrete row.
    pub fn amd_apu_host_heap() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        build(
            &[
                (2 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL),
                (14 * GIB, vk::MemoryHeapFlags::empty()),
            ],
            &[
                (0, F::DEVICE_LOCAL),
                (1, F::HOST_VISIBLE | F::HOST_COHERENT),
                (
                    0,
                    F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
                ),
            ],
        )
    }

    /// NVIDIA discrete without resizable BAR: a 256 MiB host-visible window into
    /// 16 GiB of VRAM, plus the system-RAM heap.
    pub fn nvidia_discrete() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        build(
            &[
                (16 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL),
                (64 * GIB, vk::MemoryHeapFlags::empty()),
                (256 * 1024 * 1024, vk::MemoryHeapFlags::DEVICE_LOCAL),
            ],
            &[
                (0, F::DEVICE_LOCAL),
                (1, F::HOST_VISIBLE | F::HOST_COHERENT),
                (1, F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED),
                (2, F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT),
            ],
        )
    }

    /// NVIDIA discrete WITH resizable BAR: the whole 16 GiB of VRAM is
    /// host-visible, but write-combining — never `HOST_CACHED`. This is the
    /// fixture that would break a naive "has DEVICE_LOCAL|HOST_VISIBLE ⇒ UMA"
    /// rule.
    pub fn nvidia_discrete_rebar() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        build(
            &[
                (16 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL),
                (64 * GIB, vk::MemoryHeapFlags::empty()),
            ],
            &[
                (0, F::DEVICE_LOCAL),
                (1, F::HOST_VISIBLE | F::HOST_COHERENT),
                (1, F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED),
                (0, F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT),
            ],
        )
    }

    /// Software rasterizer (llvmpipe): one heap, plain host memory.
    pub fn llvmpipe() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        build(
            &[(8 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL)],
            &[(
                0,
                F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
            )],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    /// Every unified-memory device in the matrix classifies unified, and the
    /// signal that fired is recorded.
    #[test]
    fn unified_devices_classify_unified() {
        for (name, props, signal) in [
            ("apple", apple_m3_max(), TopologySignal::NoHostOnlyHeap),
            ("intel", intel_igpu(), TopologySignal::NoHostOnlyHeap),
            (
                "amd-apu",
                amd_apu_host_heap(),
                TopologySignal::CachedDeviceLocal,
            ),
            ("llvmpipe", llvmpipe(), TopologySignal::NoHostOnlyHeap),
        ] {
            let profile = classify_memory(&props);
            assert_eq!(
                profile.topology,
                MemoryTopology::Unified,
                "{name} must classify unified"
            );
            assert_eq!(profile.signal, signal, "{name} signal");
        }
    }

    /// Both discrete shapes classify discrete — including resizable BAR, where
    /// the whole of VRAM IS host-visible. That case is the reason the rule keys
    /// on `HOST_CACHED` rather than on `DEVICE_LOCAL|HOST_VISIBLE`.
    #[test]
    fn discrete_devices_classify_discrete_including_rebar() {
        for (name, props) in [
            ("nvidia", nvidia_discrete()),
            ("nvidia-rebar", nvidia_discrete_rebar()),
        ] {
            let profile = classify_memory(&props);
            assert_eq!(
                profile.topology,
                MemoryTopology::Discrete,
                "{name} must classify discrete"
            );
            assert_eq!(profile.signal, TopologySignal::SeparateHostHeap);
        }
    }

    /// The reported sizes back the VRAM proxy: device-local bytes is the largest
    /// device-local heap, and the host-visible-device-local figure exposes how
    /// small a non-resizable BAR window really is.
    #[test]
    fn profile_reports_heap_sizes() {
        let apple = classify_memory(&apple_m3_max());
        assert_eq!(apple.device_local_bytes, 64 << 30);
        assert_eq!(apple.host_visible_device_local_bytes, 64 << 30);

        let nv = classify_memory(&nvidia_discrete());
        assert_eq!(nv.device_local_bytes, 16 << 30);
        assert_eq!(nv.host_visible_device_local_bytes, 256 << 20);

        let rebar = classify_memory(&nvidia_discrete_rebar());
        assert_eq!(rebar.host_visible_device_local_bytes, 16 << 30);
    }

    /// An empty/absent memory layout must not be called unified by omission.
    #[test]
    fn empty_properties_are_not_unified() {
        let empty = vk::PhysicalDeviceMemoryProperties::default();
        let profile = classify_memory(&empty);
        assert_eq!(profile.topology, MemoryTopology::Discrete);
        assert_eq!(profile.device_local_bytes, 0);
    }

    /// Readback lands in cached memory on every device in the matrix — an
    /// uncached CPU read of a present frame is the difference between a copy and
    /// a stall (measured: 460 MB/s, 70-86 % of all draw time on an Intel iGPU).
    ///
    /// This asserts the **selected type**, not the request. The version it
    /// replaces checked that `preferred` mentioned `HOST_CACHED` and that
    /// `required` contained `HOST_VISIBLE | HOST_COHERENT` — both true, and
    /// together they guaranteed the opposite of what the test was named for on
    /// any device whose cached type is not coherent, because the preference is
    /// only ever tried *with* the requirement. A test that reads the query
    /// cannot see a preference that never matches.
    #[test]
    fn readback_lands_in_cached_memory_on_every_device() {
        let devices: [(&str, _, MemoryTopology); 6] = [
            ("apple", apple_m3_max(), MemoryTopology::Unified),
            ("intel", intel_igpu(), MemoryTopology::Unified),
            ("amd-apu", amd_apu_host_heap(), MemoryTopology::Unified),
            ("llvmpipe", llvmpipe(), MemoryTopology::Unified),
            ("nvidia", nvidia_discrete(), MemoryTopology::Discrete),
            (
                "nvidia-rebar",
                nvidia_discrete_rebar(),
                MemoryTopology::Discrete,
            ),
        ];
        for (name, props, topology) in devices {
            let req = topology.request(MemoryClass::Readback);
            let index =
                select_memory_type(&props, !0, &req).unwrap_or_else(|| panic!("{name}: no type"));
            let kind = MappedMemoryKind::of(&props, index);
            assert!(
                kind.cached,
                "{name}: readback took uncached type {index}, so every full-frame \
                 read on that host is an uncached copy"
            );
        }
    }

    /// Coherence is a preference, not a requirement, and the readback path is
    /// what owes the invalidate when it does not get it.
    ///
    /// Intel ANV is the device that forces the distinction: its cached type is
    /// not coherent and its coherent types are not cached, so requiring both
    /// silently selects the uncached one. Apple's single type carries both, so
    /// the same request needs no invalidate there — both rows are asserted
    /// because a change that "fixes" one by breaking the other would otherwise
    /// look green.
    #[test]
    fn a_readback_type_may_be_cached_without_being_coherent() {
        let intel = intel_igpu();
        let req = MemoryTopology::Unified.request(MemoryClass::Readback);
        let index = select_memory_type(&intel, !0, &req).expect("intel readback type");
        assert_eq!(index, 1, "the first cached type, not the coherent type 0");
        assert_eq!(
            MappedMemoryKind::of(&intel, index),
            MappedMemoryKind {
                cached: true,
                coherent: false,
            },
            "so `BufferSlot::coherent` is false and the reader must invalidate"
        );

        let apple = apple_m3_max();
        let index = select_memory_type(&apple, !0, &req).expect("apple readback type");
        assert_eq!(
            MappedMemoryKind::of(&apple, index),
            MappedMemoryKind {
                cached: true,
                coherent: true,
            },
            "a device with both bits still gets both, and pays no invalidate"
        );
    }

    /// A device with no cached type at all must still allocate, and the last
    /// preference is what keeps it coherent rather than leaving the choice to
    /// whatever type happens to sit at index 0.
    #[test]
    fn a_device_with_no_cached_type_falls_back_to_a_coherent_one() {
        use vk::MemoryPropertyFlags as F;
        // Type 0 is host-visible and NOT coherent — a shape no real driver is
        // required to avoid, and the one that would be picked by a bare
        // `HOST_VISIBLE` requirement with no coherence preference behind it.
        let props = build(
            &[(1 << 30, vk::MemoryHeapFlags::DEVICE_LOCAL)],
            &[
                (0, F::HOST_VISIBLE),
                (0, F::HOST_VISIBLE | F::HOST_COHERENT),
            ],
        );
        for topology in [MemoryTopology::Unified, MemoryTopology::Discrete] {
            let req = topology.request(MemoryClass::Readback);
            let index = select_memory_type(&props, !0, &req).expect("host-visible type exists");
            assert_eq!(
                MappedMemoryKind::of(&props, index),
                MappedMemoryKind {
                    cached: false,
                    coherent: true,
                },
                "{topology:?}: with nothing cached, coherence is the next best thing"
            );
        }
    }

    /// Discrete uploads must NOT prefer the scarce device-local BAR window;
    /// unified uploads should, since it is the same DRAM.
    #[test]
    fn upload_preference_follows_topology() {
        use vk::MemoryPropertyFlags as F;
        assert!(MemoryTopology::Unified
            .request(MemoryClass::Upload)
            .preferred
            .contains(&F::DEVICE_LOCAL));
        assert!(MemoryTopology::Discrete
            .request(MemoryClass::Upload)
            .preferred
            .is_empty());
    }

    /// Selection walks the preference list best-first on a unified device: the
    /// device-local cached type wins over the plain cached one.
    #[test]
    fn selection_takes_best_preference_first() {
        let props = apple_m3_max();
        let req = MemoryTopology::Unified.request(MemoryClass::Readback);
        assert_eq!(select_memory_type(&props, !0, &req), Some(1));
    }

    /// On a discrete device, readback lands in the HOST_CACHED system-RAM type
    /// (index 2), never in the write-combining BAR window (index 3).
    #[test]
    fn discrete_readback_avoids_the_bar_window() {
        let props = nvidia_discrete_rebar();
        let req = MemoryTopology::Discrete.request(MemoryClass::Readback);
        assert_eq!(select_memory_type(&props, !0, &req), Some(2));
    }

    /// THE load-bearing invariant: a topology misclassification never fails an
    /// allocation. Requesting either topology's flags against either device's
    /// memory layout always resolves to some valid type.
    #[test]
    fn misclassification_degrades_to_a_valid_type_never_a_failure() {
        let devices = [
            ("apple", apple_m3_max()),
            ("intel", intel_igpu()),
            ("amd-apu", amd_apu_host_heap()),
            ("nvidia", nvidia_discrete()),
            ("nvidia-rebar", nvidia_discrete_rebar()),
            ("llvmpipe", llvmpipe()),
        ];
        let classes = [
            MemoryClass::Upload,
            MemoryClass::Readback,
            MemoryClass::DeviceLocal,
            MemoryClass::DeviceLocalPreferred,
        ];
        for (name, props) in &devices {
            for topology in [MemoryTopology::Unified, MemoryTopology::Discrete] {
                for class in classes {
                    let req = topology.request(class);
                    let picked = select_memory_type(props, !0, &req);
                    assert!(
                        picked.is_some(),
                        "{name} under {topology:?} must resolve {class:?}"
                    );
                    let flags = props.memory_types[picked.unwrap() as usize].property_flags;
                    assert!(
                        flags.contains(req.required),
                        "{name}/{topology:?}/{class:?} must satisfy required flags"
                    );
                }
            }
        }
    }

    /// `type_bits` is honored: a driver that forbids a type for this resource
    /// must never have it selected, even when it is the best preference match.
    #[test]
    fn type_bits_mask_is_respected() {
        let props = apple_m3_max();
        let req = MemoryTopology::Unified.request(MemoryClass::Readback);
        // Mask out type 1 (the only host-visible type) → no candidate at all.
        assert_eq!(select_memory_type(&props, 0b101, &req), None);
        // Allow it again and it is chosen.
        assert_eq!(select_memory_type(&props, 0b010, &req), Some(1));
    }

    /// Every memory class resolves a type on every device family the support
    /// matrix names — including the discrete driver with no host-visible
    /// device-local heap and the software rasterizer. A class that resolves
    /// only on the dev host is a row that exists on paper: the allocation fails
    /// on the first machine that lacks the preferred layout, and nothing in the
    /// crate can see that from a single site.
    #[test]
    fn every_class_resolves_on_every_device_family() {
        for (name, props) in [
            ("apple_m3_max", apple_m3_max()),
            ("intel_igpu", intel_igpu()),
            ("amd_apu_host_heap", amd_apu_host_heap()),
            ("nvidia_discrete", nvidia_discrete()),
            ("nvidia_discrete_rebar", nvidia_discrete_rebar()),
            ("llvmpipe", llvmpipe()),
        ] {
            let profile = classify_memory(&props);
            for class in [
                MemoryClass::Upload,
                MemoryClass::Readback,
                MemoryClass::DeviceLocal,
                MemoryClass::DeviceLocalPreferred,
            ] {
                let req = profile.topology.request(class);
                assert!(
                    select_memory_type(&props, !0, &req).is_some(),
                    "{name}/{class:?} must resolve a memory type"
                );
            }
        }
    }
}
