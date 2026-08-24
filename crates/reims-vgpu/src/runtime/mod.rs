//! Behavior: turn guest work into HostActions / backend jobs.
//!
//! Drain FIFOs, parse wire through `reims-vgpu-wire` and `reims-vgpu-protocol`, resolve memory, plan
//! ops, update [`crate::model`] state. No GPU API calls here.

/// The split of [`chain_phase`]'s largest column, `binds_us`.
pub mod bind_phase;
/// Product-path blit fill/copy execution against guest GVA.
pub mod blit_exec;
/// Draw-time buffer binds, resolved once per reference and held until the
/// guest moves the addresses under them.
pub mod bound_buffers;
/// Guest-declared write generations for task-local GVA resources.
/// Always-on proxies and censuses, one per measured bug class.
pub mod census;
/// Where a draw chain's wall clock goes on the runtime side of the engine
/// boundary, which is 82% of it.
pub mod chain_phase;
/// Product-path compute bind/dispatch (pipeline + buffers + direct dispatch).
pub mod compute_exec;
/// Multi-record compute sequencing state.
pub mod compute_session;
pub mod decode;
mod device;
pub use device::Device;
pub mod drain;
/// The always-on log sink every decline and census writes to
/// (`/tmp/reims-vgpu-fail.log`); `line()` is the `REIMS_VGPU_DRAW_LOG=1`-gated tier.
/// CmdExecIndirect2 stream walk + IOSurface texture resolve.
pub mod exec;
/// Device-owned execution port and the compatibility Vulkan adapter.
pub mod executor;
/// Product-path event + encoder fence sync (event/blit/compute/render domains).
pub mod fence_exec;
/// Contract generations and exact device-write footprints for sampled gathers.
pub mod gather_witness;
/// Guest-physical control-plane writes via HostOps map_pages.
pub mod gpa_map;
/// The bound on every GPU reference to guest RAM: VM-lifetime RAMBlock imports
/// and guest-lifetime packed mapping imports share this one checked type.
pub mod guest_ram;
/// This process's imports of guest RAM, and the one place a guest physical
/// address becomes a bindable reference.
pub mod guest_ram_map;
/// Scattered guest windows → image-copy rectangles. Pure arithmetic, ungated so
/// Task GVA → guest RAM reads.
pub mod gva_mem;
pub mod gva_refusal;
/// GVA Store currency is used by Vulkan residents and retained in the decoded
/// task-resource namespace.
pub mod gva_store_witness;
/// Task-GVA HostOps views (MapMemory2 / UnmapMemory lifecycle).
pub mod gva_view;
/// CmdHeapTextureSizeAndAlign wire decode + host requirement query.
pub mod heap_query;
pub mod host;
/// Which guest pages this device has written, and when.
/// ICB descriptor materialization, host command fills, and execute writeback.
pub mod icb;

/// Draw encode and writeback when MTLBs resolve.
pub mod draw;
pub mod input;
/// IOSurface mapper capture + page-table resolve.
pub mod mapper;
/// Write host BGRA into guest mapping pages (render writeback).
pub mod mapping_write;
/// generateMipmaps for multi-mip type-2/3 linear textures.
pub mod mipmap;
pub mod mmio;
/// MTLB container → wrapped-AIR carve for metal2vulkan.
pub mod mtlb;
pub mod node_guard;
/// Object-list lookup and IOSurface texture registration.
pub mod objects;
/// A draw's pipeline and both its shaders, resolved once per pipeline object.
pub mod pipeline_resolve;
/// The resident identity an IOSurface texture guest surface renders into.
pub mod present_identity;
/// Whether a range's page-table entries are in the state the guest's own next
/// edit of them requires — the direction that is ordered is the map.
pub mod range_coverage;
pub mod released_pages;
/// Transfer a host-resident render frame into guest pages when synchronization
/// or a guest-memory reader makes the bytes observable.
pub mod render_writeback;
/// The guest's per-resource validity quad, from both of its producers.
pub mod resource_validity;
/// The split of [`chain_phase`]'s largest *undivided* column, `sampled_us`.
pub mod sampled_phase;
/// Guest surface → host BGRA8 for the QEMU console.
pub mod scanout;
/// Host surface cache (Linux/Vulkan discrete-GPU present, kb §8.5).
pub mod surface_cache;
/// The wire task word a command payload carries → a live task slot.
pub mod task_slot;
/// Texture / IOSurface texture geometry registration.
pub mod texture;
/// Track host-authoritative GVA resources and transfer them on demand.
pub mod writeback_debt;

/// The unit-test host double, gated with its definition. An ungated re-export
/// would keep it reachable and so keep it in the staticlib.
#[cfg(test)]
pub(crate) use host::FakeHost;
pub(crate) use host::{HostAction, HostOps};

/// Apply the model's child-channel range rule and report a rejected command.
///
/// The two doorbell handlers and `ensure_child_ring` all gate guest work on
/// this answer. Keeping emission here lets the model own the pure channel
/// predicate without reaching into runtime census state.
pub fn accept_child_channel(channel_id: u32, site: &'static str) -> bool {
    if crate::model::is_child_channel(channel_id) {
        return true;
    }
    drain::census::note_store_route("child_channel_out_of_range");
    if crate::observe::first_sight("channel_outside_device_range", u64::from(channel_id)) {
        crate::observe::fail(format!(
            "child_channel_out_of_range reason=channel_outside_device_range \
             site={site} channel={channel_id} max_channels={}",
            crate::model::MAX_CHANNELS
        ));
    }
    false
}

/// Emit observation-only consequences of a mapping-page mutation.
pub fn note_mapping_invalidation(effect: crate::model::MappingInvalidationEffect) {
    if effect.dropped_host_cache {
        drain::note_store_route("invalidate_dropped_host_cache");
    }
}
