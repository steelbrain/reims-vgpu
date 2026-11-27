//! libva's VP9 decode interface (`va_dec_vp9.h`, and its profiles in
//! `va.h`), hand-written from libva 2.24.1's headers (VA-API 1.24.0). The
//! test at the end checks every struct's size and offsets, every
//! bit-field's position and every constant's value against those headers
//! with a C compiler, as the parent module's test does for H.264 and HEVC.
//!
//! Names keep libva's spelling; struct fields are snake-cased.

#![expect(
    non_upper_case_globals,
    reason = "libva's names, kept for grepping the headers"
)]

use super::{BitField, VA_PADDING_LOW, VA_PADDING_MEDIUM, VAProfile, VASurfaceID};

// VAProfile.
pub(crate) const VAProfileVP9Profile0: VAProfile = 19;
pub(crate) const VAProfileVP9Profile2: VAProfile = 21;

/// `VADecPictureParameterBufferVP9`: a frame's uncompressed header, as
/// the driver takes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VADecPictureParameterBufferVP9 {
    pub(crate) frame_width: u16,
    pub(crate) frame_height: u16,
    /// The eight reference slots' surfaces, `VA_INVALID_SURFACE` for an
    /// empty one.
    pub(crate) reference_frames: [VASurfaceID; 8],
    /// [`vp9_pic_fields`].
    pub(crate) pic_fields: u32,
    pub(crate) filter_level: u8,
    pub(crate) sharpness_level: u8,
    pub(crate) log2_tile_rows: u8,
    pub(crate) log2_tile_columns: u8,
    /// The uncompressed header's bytes.
    pub(crate) frame_header_length_in_bytes: u8,
    /// The compressed header's bytes (`header_size_in_bytes`).
    pub(crate) first_partition_size: u16,
    pub(crate) mb_segment_tree_probs: [u8; 7],
    pub(crate) segment_pred_probs: [u8; 3],
    pub(crate) profile: u8,
    pub(crate) bit_depth: u8,
    pub(crate) va_reserved: [u32; VA_PADDING_MEDIUM],
}

/// `VADecPictureParameterBufferVP9::pic_fields.bits`.
pub(crate) mod vp9_pic_fields {
    use super::BitField;
    pub(crate) const SUBSAMPLING_X: BitField = BitField::new(0, 1);
    pub(crate) const SUBSAMPLING_Y: BitField = BitField::new(1, 1);
    pub(crate) const FRAME_TYPE: BitField = BitField::new(2, 1);
    pub(crate) const SHOW_FRAME: BitField = BitField::new(3, 1);
    pub(crate) const ERROR_RESILIENT_MODE: BitField = BitField::new(4, 1);
    pub(crate) const INTRA_ONLY: BitField = BitField::new(5, 1);
    pub(crate) const ALLOW_HIGH_PRECISION_MV: BitField = BitField::new(6, 1);
    pub(crate) const MCOMP_FILTER_TYPE: BitField = BitField::new(7, 3);
    pub(crate) const FRAME_PARALLEL_DECODING_MODE: BitField = BitField::new(10, 1);
    pub(crate) const RESET_FRAME_CONTEXT: BitField = BitField::new(11, 2);
    pub(crate) const REFRESH_FRAME_CONTEXT: BitField = BitField::new(13, 1);
    pub(crate) const FRAME_CONTEXT_IDX: BitField = BitField::new(14, 2);
    pub(crate) const SEGMENTATION_ENABLED: BitField = BitField::new(16, 1);
    pub(crate) const SEGMENTATION_TEMPORAL_UPDATE: BitField = BitField::new(17, 1);
    pub(crate) const SEGMENTATION_UPDATE_MAP: BitField = BitField::new(18, 1);
    pub(crate) const LAST_REF_FRAME: BitField = BitField::new(19, 3);
    pub(crate) const LAST_REF_FRAME_SIGN_BIAS: BitField = BitField::new(22, 1);
    pub(crate) const GOLDEN_REF_FRAME: BitField = BitField::new(23, 3);
    pub(crate) const GOLDEN_REF_FRAME_SIGN_BIAS: BitField = BitField::new(26, 1);
    pub(crate) const ALT_REF_FRAME: BitField = BitField::new(27, 3);
    pub(crate) const ALT_REF_FRAME_SIGN_BIAS: BitField = BitField::new(30, 1);
    pub(crate) const LOSSLESS_FLAG: BitField = BitField::new(31, 1);
}

/// `VASegmentParameterVP9`: one segment's features and derived values.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VASegmentParameterVP9 {
    /// [`vp9_segment_flags`], a `uint16_t` union.
    pub(crate) segment_flags: u16,
    /// The loop filter level of each reference frame and mode.
    pub(crate) filter_level: [[u8; 2]; 4],
    pub(crate) luma_ac_quant_scale: i16,
    pub(crate) luma_dc_quant_scale: i16,
    pub(crate) chroma_ac_quant_scale: i16,
    pub(crate) chroma_dc_quant_scale: i16,
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

/// `VASegmentParameterVP9::segment_flags.fields`.
pub(crate) mod vp9_segment_flags {
    use super::BitField;
    pub(crate) const SEGMENT_REFERENCE_ENABLED: BitField = BitField::new(0, 1);
    pub(crate) const SEGMENT_REFERENCE: BitField = BitField::new(1, 2);
    pub(crate) const SEGMENT_REFERENCE_SKIPPED: BitField = BitField::new(3, 1);
}

/// `VASliceParameterBufferVP9`: the frame's bytes, and each segment's
/// parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VASliceParameterBufferVP9 {
    pub(crate) slice_data_size: u32,
    pub(crate) slice_data_offset: u32,
    pub(crate) slice_data_flag: u32,
    pub(crate) seg_param: [VASegmentParameterVP9; 8],
    pub(crate) va_reserved: [u32; VA_PADDING_LOW],
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};
    use std::io::Write;
    use std::process::{Command, Stdio};

    /// What the C side prints, line for line.
    const LAYOUT_C: &str = include_str!("vp9_layout.c");

    macro_rules! layout {
        ($out:ident, $ty:ty, $c:literal, { $($field:ident => $cfield:literal),* $(,)? }) => {
            $out.push(format!("{} size {}", $c, size_of::<$ty>()));
            $( $out.push(format!("{}.{} {}", $c, $cfield, offset_of!($ty, $field))); )*
        };
    }

    fn fields(out: &mut Vec<String>, union: &str, list: &[(&str, BitField)]) {
        for (name, field) in list {
            out.push(format!("{union}.{name} {:#x}", field.put(u32::MAX)));
        }
    }

    fn rust_layout() -> Vec<String> {
        let mut out = Vec::new();
        layout!(out, VADecPictureParameterBufferVP9, "VADecPictureParameterBufferVP9", {
            frame_width => "frame_width", frame_height => "frame_height",
            reference_frames => "reference_frames", pic_fields => "pic_fields",
            filter_level => "filter_level", sharpness_level => "sharpness_level",
            log2_tile_rows => "log2_tile_rows", log2_tile_columns => "log2_tile_columns",
            frame_header_length_in_bytes => "frame_header_length_in_bytes",
            first_partition_size => "first_partition_size",
            mb_segment_tree_probs => "mb_segment_tree_probs",
            segment_pred_probs => "segment_pred_probs", profile => "profile",
            bit_depth => "bit_depth", va_reserved => "va_reserved",
        });
        layout!(out, VASegmentParameterVP9, "VASegmentParameterVP9", {
            segment_flags => "segment_flags", filter_level => "filter_level",
            luma_ac_quant_scale => "luma_ac_quant_scale",
            luma_dc_quant_scale => "luma_dc_quant_scale",
            chroma_ac_quant_scale => "chroma_ac_quant_scale",
            chroma_dc_quant_scale => "chroma_dc_quant_scale", va_reserved => "va_reserved",
        });
        layout!(out, VASliceParameterBufferVP9, "VASliceParameterBufferVP9", {
            slice_data_size => "slice_data_size", slice_data_offset => "slice_data_offset",
            slice_data_flag => "slice_data_flag", seg_param => "seg_param",
            va_reserved => "va_reserved",
        });
        use vp9_pic_fields as pf;
        fields(
            &mut out,
            "vp9.pic_fields",
            &[
                ("subsampling_x", pf::SUBSAMPLING_X),
                ("subsampling_y", pf::SUBSAMPLING_Y),
                ("frame_type", pf::FRAME_TYPE),
                ("show_frame", pf::SHOW_FRAME),
                ("error_resilient_mode", pf::ERROR_RESILIENT_MODE),
                ("intra_only", pf::INTRA_ONLY),
                ("allow_high_precision_mv", pf::ALLOW_HIGH_PRECISION_MV),
                ("mcomp_filter_type", pf::MCOMP_FILTER_TYPE),
                (
                    "frame_parallel_decoding_mode",
                    pf::FRAME_PARALLEL_DECODING_MODE,
                ),
                ("reset_frame_context", pf::RESET_FRAME_CONTEXT),
                ("refresh_frame_context", pf::REFRESH_FRAME_CONTEXT),
                ("frame_context_idx", pf::FRAME_CONTEXT_IDX),
                ("segmentation_enabled", pf::SEGMENTATION_ENABLED),
                (
                    "segmentation_temporal_update",
                    pf::SEGMENTATION_TEMPORAL_UPDATE,
                ),
                ("segmentation_update_map", pf::SEGMENTATION_UPDATE_MAP),
                ("last_ref_frame", pf::LAST_REF_FRAME),
                ("last_ref_frame_sign_bias", pf::LAST_REF_FRAME_SIGN_BIAS),
                ("golden_ref_frame", pf::GOLDEN_REF_FRAME),
                ("golden_ref_frame_sign_bias", pf::GOLDEN_REF_FRAME_SIGN_BIAS),
                ("alt_ref_frame", pf::ALT_REF_FRAME),
                ("alt_ref_frame_sign_bias", pf::ALT_REF_FRAME_SIGN_BIAS),
                ("lossless_flag", pf::LOSSLESS_FLAG),
            ],
        );
        use vp9_segment_flags as sf;
        fields(
            &mut out,
            "vp9.segment_flags",
            &[
                ("segment_reference_enabled", sf::SEGMENT_REFERENCE_ENABLED),
                ("segment_reference", sf::SEGMENT_REFERENCE),
                ("segment_reference_skipped", sf::SEGMENT_REFERENCE_SKIPPED),
            ],
        );
        out.push(format!("VAProfileVP9Profile0 {VAProfileVP9Profile0}"));
        out.push(format!("VAProfileVP9Profile2 {VAProfileVP9Profile2}"));
        out
    }

    /// Compiles `vp9_layout.c` against the installed libva headers and
    /// compares its output with this module's layout. `CC` picks the
    /// compiler (`cc` by default).
    #[test]
    #[ignore = "needs libva's headers and a C compiler: run on the test host"]
    fn vp9_layouts_match_libva_headers() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("rvgv-va-vp9-layout-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let binary = dir.join("layout");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
        let mut compile = Command::new(cc)
            .args(["-Wno-overflow", "-x", "c", "-", "-o"])
            .arg(&binary)
            .stdin(Stdio::piped())
            .spawn()?;
        compile
            .stdin
            .take()
            .ok_or("no stdin")?
            .write_all(LAYOUT_C.as_bytes())?;
        if !compile.wait()?.success() {
            return Err("the C compiler failed".into());
        }
        let output = Command::new(&binary).output()?;
        std::fs::remove_dir_all(&dir)?;
        let c_lines: Vec<String> = String::from_utf8(output.stdout)?
            .lines()
            .map(str::to_owned)
            .collect();
        assert_eq!(rust_layout(), c_lines);
        Ok(())
    }
}
