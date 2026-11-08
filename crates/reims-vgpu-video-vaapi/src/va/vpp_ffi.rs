//! libva's video-processing interface, hand-written for the subset the
//! backend calls, from libva 2.24.1's `va_vpp.h` and `va.h` (VA-API
//! 1.24.0), apart from `ffi.rs` so that the decoders' binding and this one
//! change apart. The test at the end checks each struct's size and offsets
//! against those headers with a C compiler, as `ffi.rs`'s does.
//!
//! Names keep libva's spelling, so that each item can be found in the
//! headers; struct fields are snake-cased.

#![expect(
    non_upper_case_globals,
    reason = "libva's names, kept for grepping the headers"
)]

use core::ffi::{c_int, c_void};

use super::ffi::{VA_PADDING_LARGE, VABufferID, VABufferType, VASurfaceID};

/// `VAProcColorStandardType`, a C enum.
pub(crate) type VAProcColorStandardType = c_int;
/// `VAProcMode`, a C enum.
pub(crate) type VAProcMode = c_int;

pub(crate) const VAProcPipelineParameterBufferType: VABufferType = 41;

pub(crate) const VAProcColorStandardNone: VAProcColorStandardType = 0;
pub(crate) const VAProcColorStandardBT709: VAProcColorStandardType = 2;

pub(crate) const VA_ROTATION_NONE: u32 = 0;

pub(crate) const VA_MIRROR_NONE: u32 = 0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VARectangle {
    pub(crate) x: i16,
    pub(crate) y: i16,
    pub(crate) width: u16,
    pub(crate) height: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VAProcColorProperties {
    pub(crate) chroma_sample_location: u8,
    pub(crate) color_range: u8,
    pub(crate) colour_primaries: u8,
    pub(crate) transfer_characteristics: u8,
    pub(crate) matrix_coefficients: u8,
    pub(crate) reserved: [u8; 3],
}

/// The 64-bit layout (va_vpp.h pads it by 16 words on LP64 targets, by 13
/// elsewhere; this crate builds for x86_64 and aarch64 Linux alone).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VAProcPipelineParameterBuffer {
    pub(crate) surface: VASurfaceID,
    pub(crate) surface_region: *const VARectangle,
    pub(crate) surface_color_standard: VAProcColorStandardType,
    pub(crate) output_region: *const VARectangle,
    pub(crate) output_background_color: u32,
    pub(crate) output_color_standard: VAProcColorStandardType,
    pub(crate) pipeline_flags: u32,
    pub(crate) filter_flags: u32,
    pub(crate) filters: *mut VABufferID,
    pub(crate) num_filters: u32,
    pub(crate) forward_references: *mut VASurfaceID,
    pub(crate) num_forward_references: u32,
    pub(crate) backward_references: *mut VASurfaceID,
    pub(crate) num_backward_references: u32,
    pub(crate) rotation_state: u32,
    pub(crate) blend_state: *const c_void,
    pub(crate) mirror_state: u32,
    pub(crate) additional_outputs: *mut VASurfaceID,
    pub(crate) num_additional_outputs: u32,
    pub(crate) input_surface_flag: u32,
    pub(crate) output_surface_flag: u32,
    pub(crate) input_color_properties: VAProcColorProperties,
    pub(crate) output_color_properties: VAProcColorProperties,
    pub(crate) processing_mode: VAProcMode,
    pub(crate) output_hdr_metadata: *const c_void,
    pub(crate) va_reserved: [u32; VA_PADDING_LARGE - 16],
}

impl VAProcPipelineParameterBuffer {
    /// Every field zero and every pointer null: the whole surfaces, no
    /// filters, references or blending, progressive, default mode.
    pub(crate) fn zeroed() -> Self {
        Self {
            surface: 0,
            surface_region: core::ptr::null(),
            surface_color_standard: VAProcColorStandardNone,
            output_region: core::ptr::null(),
            output_background_color: 0,
            output_color_standard: VAProcColorStandardNone,
            pipeline_flags: 0,
            filter_flags: 0,
            filters: core::ptr::null_mut(),
            num_filters: 0,
            forward_references: core::ptr::null_mut(),
            num_forward_references: 0,
            backward_references: core::ptr::null_mut(),
            num_backward_references: 0,
            rotation_state: VA_ROTATION_NONE,
            blend_state: core::ptr::null(),
            mirror_state: VA_MIRROR_NONE,
            additional_outputs: core::ptr::null_mut(),
            num_additional_outputs: 0,
            input_surface_flag: 0,
            output_surface_flag: 0,
            input_color_properties: VAProcColorProperties::default(),
            output_color_properties: VAProcColorProperties::default(),
            processing_mode: 0,
            output_hdr_metadata: core::ptr::null(),
            va_reserved: [0; VA_PADDING_LARGE - 16],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};
    use std::io::Write;
    use std::process::{Command, Stdio};

    /// What the C side prints, line for line.
    const LAYOUT_C: &str = include_str!("vpp_ffi_layout.c");

    macro_rules! layout {
        ($out:ident, $ty:ty, $c:literal, { $($field:ident => $cfield:literal),* $(,)? }) => {
            $out.push(format!("{} size {}", $c, size_of::<$ty>()));
            $( $out.push(format!("{}.{} {}", $c, $cfield, offset_of!($ty, $field))); )*
        };
    }

    fn rust_layout() -> Vec<String> {
        let mut out = Vec::new();
        layout!(out, VARectangle, "VARectangle", {
            x => "x", y => "y", width => "width", height => "height",
        });
        layout!(out, VAProcColorProperties, "VAProcColorProperties", {
            chroma_sample_location => "chroma_sample_location", color_range => "color_range",
            colour_primaries => "colour_primaries",
            transfer_characteristics => "transfer_characteristics",
            matrix_coefficients => "matrix_coefficients", reserved => "reserved",
        });
        layout!(out, VAProcPipelineParameterBuffer, "VAProcPipelineParameterBuffer", {
            surface => "surface", surface_region => "surface_region",
            surface_color_standard => "surface_color_standard", output_region => "output_region",
            output_background_color => "output_background_color",
            output_color_standard => "output_color_standard", pipeline_flags => "pipeline_flags",
            filter_flags => "filter_flags", filters => "filters", num_filters => "num_filters",
            forward_references => "forward_references",
            num_forward_references => "num_forward_references",
            backward_references => "backward_references",
            num_backward_references => "num_backward_references",
            rotation_state => "rotation_state", blend_state => "blend_state",
            mirror_state => "mirror_state", additional_outputs => "additional_outputs",
            num_additional_outputs => "num_additional_outputs",
            input_surface_flag => "input_surface_flag", output_surface_flag => "output_surface_flag",
            input_color_properties => "input_color_properties",
            output_color_properties => "output_color_properties",
            processing_mode => "processing_mode", output_hdr_metadata => "output_hdr_metadata",
            va_reserved => "va_reserved",
        });
        let constants: [(&str, i64); 4] = [
            (
                "VAProcPipelineParameterBufferType",
                VAProcPipelineParameterBufferType.into(),
            ),
            ("VAProcColorStandardNone", VAProcColorStandardNone.into()),
            ("VAProcColorStandardBT709", VAProcColorStandardBT709.into()),
            ("VA_ROTATION_NONE", VA_ROTATION_NONE.into()),
        ];
        for (name, value) in constants {
            out.push(format!("{name} {value}"));
        }
        out
    }

    /// Compiles `vpp_ffi_layout.c` against the installed libva headers and
    /// compares its output with this module's layout. `CC` picks the
    /// compiler (`cc` by default).
    #[test]
    #[ignore = "needs libva's headers and a C compiler: run on the test host"]
    fn vpp_layouts_match_libva_headers() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("rvgv-vpp-layout-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let binary = dir.join("layout");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
        let mut compile = Command::new(cc)
            .args(["-x", "c", "-", "-o"])
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
