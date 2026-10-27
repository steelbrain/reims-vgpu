//! A display's gamma table: the guest's table copied from the kernel task, checked against the
//! guest's sum, uploaded into a 1D-array texture of three slices, and its end slopes into the
//! fragment properties.

use std::sync::atomic::Ordering;

use crate::device::tasks::MAX_LOCAL_TASK_LENGTH;
use crate::display::{DeathKind, DisplayCore, DisplayLocked, RaiseKind};
use crate::num::ToUsize;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::resource::{GpuResource, Texture};
use crate::platform::types::{MtlOrigin, MtlRegion, MtlSize};
use crate::platform::values::TextureDesc;
use crate::raise::{OnFault, RaiseResult};
use crate::rt::Hold;
use crate::tasks::local::LocalTask;

/// A present's gamma table: where the guest's table is in the kernel task, its entry count and
/// its sum (the present commands' and the shared page's records).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GammaTableArgs {
    /// The table's offset in the kernel task.
    pub virtual_offset: u64,
    /// The bytes the task copies.
    pub mapped_length: u64,
    /// The entries per channel.
    pub entry_count: u32,
    /// The guest's checksum of the table, compared with the one the copy computes.
    pub sum: u32,
}

/// The gamma table texture's label.
pub const GAMMA_TABLE_LABEL: &str = "reims-vgpu.gamma-table";
/// `MTLTextureType1DArray`.
const MTL_TEXTURE_TYPE_1D_ARRAY: u64 = 1;
/// `MTLPixelFormatR32Float`.
const MTL_PIXEL_FORMAT_R32_FLOAT: u64 = 55;
/// `MTLTextureUsageShaderRead`.
const MTL_TEXTURE_USAGE_SHADER_READ: u64 = 1;

/// The float at index `i` (zero-extended) of a table of bytes, or `None` past its end.
pub(crate) fn float_at(table: &[u8], i: u32) -> Option<f32> {
    word_at(table, i).map(f32::from_bits)
}

/// The `u32` at index `i` of a table of bytes, or `None` past its end.
pub(crate) fn word_at(table: &[u8], i: u32) -> Option<u32> {
    let at = i.to_usize().checked_mul(4)?;
    let b = table.get(at..at.checked_add(4)?)?;
    let b: [u8; 4] = b.try_into().ok()?;
    Some(u32::from_ne_bytes(b))
}

/// One slice's upload: `n` floats from float index `first` of `table`, into slice `slice`.
fn upload_gamma_slice<P: Platform>(
    texture: &P::Texture,
    table: &[u8],
    n: u32,
    slice: u64,
    first: u32,
    bytes_per_row: u64,
) -> RaiseResult<()> {
    let start = u64::from(first).wrapping_mul(4).to_usize();
    // The table is at least 12n bytes (padded after the copy).
    let Some(row) = start
        .checked_add(bytes_per_row.to_usize())
        .and_then(|end| table.get(start..end))
    else {
        crate::fatal!(DeathKind::GammaTableRead)
    };
    let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(u64::from(n), 1, 1));
    // Metal copies synchronously and keeps no pointer.
    let pixels = crate::platform::HostRange::of_slice(row);
    texture.replace_region(region, 0, slice, pixels, bytes_per_row, bytes_per_row)
}

impl<P: Platform> DisplayCore<P> {
    /// The gamma table, taking the display mutex (tests; the product's present holds it).
    #[cfg(test)]
    pub(crate) fn update_gamma_table(
        &self,
        hold: Hold<'_>,
        task: &LocalTask<P>,
        gamma: GammaTableArgs,
    ) -> RaiseResult<()> {
        let mut g = self.display_mutex.lock();
        self.update_gamma_table_locked(hold, &mut g, Some(task), gamma)
    }

    /// The gamma table, with the display mutex held. `task` `None` is a nil task: the copy is a
    /// send to nil and the table stays zero.
    pub(super) fn update_gamma_table_locked(
        &self,
        hold: Hold<'_>,
        g: &mut DisplayLocked<P>,
        task: Option<&LocalTask<P>>,
        gamma: GammaTableArgs,
    ) -> RaiseResult<()> {
        let n = gamma.entry_count;
        // A mapped length past the largest local task faults here, before any buffer is made: no
        // task holds such a range, so the copy would fault it anyway
        // (`tests::update_gamma_table::gamma_mapped_length_past_any_task_faults`).
        if gamma.mapped_length > MAX_LOCAL_TASK_LENGTH {
            crate::raise!(
                RaiseKind::GammaLength,
                format_args!(
                    "gamma table mapped length {:#x} is past any task",
                    gamma.mapped_length
                )
            );
        }
        let mapped = gamma.mapped_length.to_usize();
        // The table's 12n bytes are read although only `mapped` are the guest's (a heap over-read
        // past a short table): the table is padded to 12n with zeros once the copy is done.
        let twelve_n = u64::from(n).wrapping_mul(12).to_usize();
        let len = mapped.max(twelve_n);
        // Zero-allocated (lazily mapped), never filled by hand: nothing touches it before the
        // task's copy, whose range check raises first for a length the task refuses, so a
        // guest-sized table the copy rejects commits no memory. A raise out of the copy ("Address
        // (..) out of range for task") frees it.
        let mut table: Vec<u8> = vec![0u8; mapped];
        if let Some(task) = task {
            task.copy_from_virtual_offset(hold, gamma.virtual_offset, &mut table)?;
        }
        table.resize(len, 0);
        // `u32` index arithmetic and a wrapping `u32` sum.
        let mut host_sum: u32 = 0;
        let mut i: u32 = 0;
        while i < n {
            let r = word_at(&table, i);
            let gr = word_at(&table, n.wrapping_add(i));
            let b = word_at(&table, 2_u32.wrapping_mul(n).wrapping_add(i));
            let (Some(r), Some(gr), Some(b)) = (r, gr, b) else {
                crate::fatal!(DeathKind::GammaTableRead)
            };
            host_sum = host_sum.wrapping_add(r.wrapping_add(gr).wrapping_add(b));
            i = i.wrapping_add(1);
        }
        if gamma.sum != host_sum {
            let port = self.port();
            drop(table);
            crate::raise!(
                RaiseKind::GammaCorrupt,
                format_args!(
                    "guest display on port {port}: the guest's gamma table sum {:#010x} is not the \
                     table's {host_sum:#010x}",
                    gamma.sum
                )
            );
        }
        crate::fatal_unless!(n != 0, DeathKind::GammaNoEntries);
        // A nil texture's width is 0.
        if crate::nil_send!(g.gamma_table_tex, width()) != u64::from(n) {
            g.gamma_table_tex = None;
        }
        if g.gamma_table_tex.is_none() {
            let d = self.platform.texture_descriptor(&TextureDesc {
                texture_type: MTL_TEXTURE_TYPE_1D_ARRAY,
                pixel_format: MTL_PIXEL_FORMAT_R32_FLOAT,
                width: u64::from(n),
                array_length: 3,
                usage: MTL_TEXTURE_USAGE_SHADER_READ,
                ..TextureDesc::default()
            });
            let tex = match self
                .mtl_device()
                .map(|m| m.new_texture(OnFault::Return, &d))
            {
                // After the end released the pipelines, which hold the `MTLDevice`.
                None => None,
                Some(Ok(tex)) => tex,
                Some(Err(thrown)) => return Err(thrown),
            };
            let Some(tex) = tex else {
                crate::fatal!(DeathKind::GammaTextureRefused)
            };
            let tex = g.gamma_table_tex.insert(tex);
            tex.set_label(GAMMA_TABLE_LABEL)?;
            // The gamma pipeline from now on, for the display's life.
            g.pipeline.gamma = true;
            drop(d);
        }
        // `n` floats per row, in `NSUInteger`.
        let bytes_per_row = u64::from(n).wrapping_mul(4);
        if let Some(tex) = &g.gamma_table_tex {
            upload_gamma_slice::<P>(tex, &table, n, 0, 0, bytes_per_row)?;
            upload_gamma_slice::<P>(tex, &table, n, 1, n, bytes_per_row)?;
            upload_gamma_slice::<P>(tex, &table, n, 2, 2_u32.wrapping_mul(n), bytes_per_row)?;
        }
        // `f32`, with `n - 1` in `u32`.
        #[expect(clippy::as_conversions, reason = "u32 to f32 rounds to nearest")]
        let step = 1.0f32 / n.wrapping_sub(1) as f32;
        let (Some(t1), Some(t0)) = (float_at(&table, 1), float_at(&table, 0)) else {
            crate::fatal!(DeathKind::GammaTableRead)
        };
        let red_zero = (t1 - t0) / step;
        // n = 1 reads t[0xffffffff], 16 GiB past the table: it dies after the uploads and the
        // gamma pipeline's switch.
        let (Some(a), Some(b)) = (
            float_at(&table, n.wrapping_sub(1)),
            float_at(&table, n.wrapping_sub(2)),
        ) else {
            crate::fatal!(DeathKind::GammaOneEntry)
        };
        let red_one = (a - b) / step;
        let two_n = 2_u32.wrapping_mul(n);
        let (Some(a), Some(b)) = (
            float_at(&table, two_n.wrapping_sub(1)),
            float_at(&table, two_n.wrapping_sub(2)),
        ) else {
            crate::fatal!(DeathKind::GammaTableRead)
        };
        let green_one = (a - b) / step;
        let three_n = 3_u32.wrapping_mul(n);
        let (Some(a), Some(b)) = (
            float_at(&table, three_n.wrapping_sub(1)),
            float_at(&table, three_n.wrapping_sub(2)),
        ) else {
            crate::fatal!(DeathKind::GammaTableRead)
        };
        let blue_one = (a - b) / step;
        // Green and blue at zero take red's slope.
        self.update_gamma_slopes_locked(
            g,
            [red_zero, red_zero, red_zero],
            [red_one, green_one, blue_one],
        )?;
        // Whether the gamma pass maps every 8-bit level to itself through this table: for a
        // value v the shader's nearest sample takes entry floor(v n), clamped to the table, and a
        // BGRA8 target stores round(255 x) of the entry, clamped to [0, 1]. The console's CPU copy
        // relies on it (`render.rs`, `encode_current_frame_by_copy`).
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "levels and entry indices are small; the float maths is the shader's"
        )]
        let maps_identity = (0..3u32).all(|channel| {
            (0u32..=255).all(|level| {
                let v = level as f32 / 255.0;
                let index = ((v * n as f32).floor() as u32).min(n.wrapping_sub(1));
                float_at(&table, channel.wrapping_mul(n).wrapping_add(index)).is_some_and(|x| {
                    x.is_finite() && (x.clamp(0.0, 1.0) * 255.0).round() as u32 == level
                })
            })
        });
        self.gamma_identity.store(maps_identity, Ordering::SeqCst);
        drop(table);
        Ok(())
    }
}
