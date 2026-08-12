//! The resident identity a type-11 guest surface renders into.
//!
//! This is all that survives of `import_present`, which owned three ways of
//! landing a Vulkan composite Store in guest IOSurface pages without a CPU
//! copy: a packed-contig strided DMA, a fragmented multi-run scatter DMA, and
//! an ack-fast deferred rung that pinned the resident and replayed the Store on
//! first access.
//!
//! All three needed `VK_EXT_external_memory_host` — a host pointer over the
//! guest's own pages, which is a pointer the GPU can write. Neither the
//! extension nor the two engine entry points exist any more, so type-11 Stores
//! take the CPU writeback
//! (`mapping_write::write_rgba8_image_changed`), which every one of those rails
//! already fell back to whenever an import was refused.
//!
//! What is left is the identity itself, which was never about importing: the
//! registry is keyed by it whichever way the pixels reach the guest.

#![cfg(feature = "backend-vulkan")]

use crate::backend::vulkan::engine::TargetIdentity;
use crate::model::DeviceState;
use ash::vk;

/// Build a protocol-stable resident identity for this mapping at its current
/// [`crate::model::MappingEntry::map_generation`].
///
/// One identity per mapping, always. `ResourcePools::registry` is keyed by
/// `TargetIdentity`, so two mappings with equal identities would render into and
/// capture from ONE `VkImage` — and distinct guest surfaces have independent
/// damage histories, because WindowServer redraws a buffer only where it differs
/// from what THAT buffer last held. Sharing a resident between them makes every
/// frame a fusion of damage from several buffers, which is the rubber-band
/// residue class.
/// The image format a resident for this mapping is created with.
///
/// The guest's own declaration, taken from the single place the writeback also
/// reads it. A declaration with no linear Vulkan texel — a compressed or planar
/// plane, which nothing renders into — falls back to guest scanout order, which
/// is what this namespace answered for every mapping before it could answer at
/// all; the GPU writeback rail then refuses that pair by name and the copying
/// rail converts, exactly as it did.
fn surface_format(state: &DeviceState, mapping_id: u32) -> vk::Format {
    state
        .mappings
        .get(&mapping_id)
        .map(crate::runtime::mapping_write::mapping_store_format)
        .and_then(crate::contract::pixel_format::store_texel_order)
        .map(crate::backend::vulkan::translate::pixel::vk_texel_layout)
        .unwrap_or(crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT)
}

pub fn surface_identity(
    state: &DeviceState,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> TargetIdentity {
    let gen = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.map_generation as u64)
        .unwrap_or(0);
    TargetIdentity::Surface {
        id: mapping_id,
        width,
        height,
        generation: gen,
        format: surface_format(state, mapping_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    /// Two mappings must never share an identity, and one mapping must never
    /// change identity without its `map_generation` changing. Both directions
    /// are the rubber-band residue class: the registry is keyed on this value,
    /// so a collision fuses two guest surfaces' damage histories into one
    /// `VkImage`, and a spurious change orphans a live resident.
    #[test]
    fn identity_separates_mappings_and_tracks_only_the_map_generation() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let a = surface_identity(&state, 7, 64, 32);
        let b = surface_identity(&state, 8, 64, 32);
        assert_ne!(a, b, "distinct mappings must not share a resident");
        assert_eq!(
            a,
            surface_identity(&state, 7, 64, 32),
            "identity must be a pure function of the mapping and geometry"
        );
        assert_ne!(
            a,
            surface_identity(&state, 7, 65, 32),
            "geometry is part of the resident's shape"
        );
    }

    /// A mapping's resident is created at the format the mapping declares, and
    /// redeclaring it is a different resident.
    ///
    /// Both halves matter and they fail differently. Ignoring the declaration
    /// renders a guest's half-float compositing into an eight-bit image and
    /// quantizes it with nothing to say so — the loss is invisible because every
    /// rail downstream still works, which is how the same bug survived in the
    /// `Gva` namespace until a counter on an unrelated gate exposed it. Keeping
    /// the declaration *out* of the key would be worse: one `VkImage` would be
    /// asked to be two formats at once, and `registry_ensure` would destroy and
    /// recreate it on every alternation.
    #[test]
    fn a_mappings_resident_follows_the_pixel_format_it_declares() {
        use crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT;
        use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA16_FLOAT};

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.map_surface(7));
        let declare = |state: &mut DeviceState, format: u16| {
            let m = state.mappings.get_mut(&7).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 64;
            m.height = 32;
            m.format = format;
        };

        declare(&mut state, MTL_FORMAT_BGRA8_UNORM);
        let bgra8 = surface_identity(&state, 7, 64, 32);
        assert_eq!(
            bgra8.resident_format(),
            SCANOUT_FORMAT,
            "a scanout-order plane is the format it always was"
        );

        declare(&mut state, MTL_FORMAT_RGBA16_FLOAT);
        let half = surface_identity(&state, 7, 64, 32);
        assert_eq!(
            half.resident_format(),
            vk::Format::R16G16B16A16_SFLOAT,
            "a half-float plane renders in half float, not at eight bits"
        );

        assert_ne!(
            bgra8, half,
            "two formats at one mapping are two registry keys, or one image is asked to be both"
        );
        assert!(
            bgra8.aliases(&half),
            "and still one destination, because the guest pages are the same pages"
        );
    }

    /// A declaration this device has no linear Vulkan texel for falls back to
    /// guest scanout order rather than to nothing.
    ///
    /// The resident has to be created at *some* format, and the one that
    /// preserves what this namespace did before it carried a declaration is the
    /// scanout one. The GPU writeback rail then refuses that pair by name and
    /// the copying rail converts, which is exactly the route such a mapping took
    /// when no mapping had a format at all — so an exotic plane is no worse off
    /// than it was, and the frame is never lost for want of an answer here.
    #[test]
    fn an_undeclared_or_unrepresentable_plane_keeps_guest_scanout_order() {
        use crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.map_surface(7));
        // Never declared: `has_geom` false, so there is no format to read.
        assert_eq!(
            surface_identity(&state, 7, 64, 32).resident_format(),
            SCANOUT_FORMAT
        );
        // Declared as a format with no `TexelLayout` to store into.
        {
            let m = state.mappings.get_mut(&7).unwrap();
            m.has_geom = true;
            m.width = 64;
            m.height = 32;
            m.format = crate::contract::pixel_format::MTL_FORMAT_RGBA32_FLOAT;
        }
        assert_eq!(
            surface_identity(&state, 7, 64, 32).resident_format(),
            SCANOUT_FORMAT
        );
        // And a mapping this device has never seen at all.
        assert_eq!(
            surface_identity(&state, 999, 64, 32).resident_format(),
            SCANOUT_FORMAT
        );
    }

    /// A compositor swapchain — several scanout buffers presenting at ONE
    /// geometry — must get one resident each.
    ///
    /// This is the shape that used to collapse: four buffers at 1920x1080
    /// unified onto a single geometry-keyed resident, and a held drag that
    /// reversed direction left a selection-rectangle fragment on the desktop.
    /// Interleaved on/off A/B over four boots: 5 of 12 rounds reproduced with
    /// the collapse, 1 of 12 without, and the dominant sub-class — a 15x15
    /// fragment at the press point — went 4 to 0.
    ///
    /// What holds the line now is structural rather than a check a caller has
    /// to remember: the mapping id is part of the registry key, so
    /// `registry_get` on one buffer's identity cannot return another's. That
    /// replaced an explicit `surface_mapping_id()` predicate, which the
    /// geometry-keyed resolver needed and nothing does. The pairwise case above
    /// states the same property; this one states it over the exact arity the
    /// live defect had, because "distinct in pairs" is what a reader checks and
    /// "four buffers, four residents" is what the compositor does.
    #[test]
    fn a_four_buffer_swapchain_at_one_geometry_gets_four_residents() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let ids: Vec<TargetIdentity> = [11u32, 12, 13, 14]
            .iter()
            .map(|&mid| surface_identity(&state, mid, 1920, 1080))
            .collect();
        for (i, a) in ids.iter().enumerate() {
            for (j, b) in ids.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "scanout buffers {i} and {j} share a resident");
                }
            }
        }
    }
}
