//! What a pipeline create's compile is handed with ([`super::PipelineJob::what`]), which its begin
//! line and its thread's record of it name (`crate::graphics::compile`, "The compile in flight"):
//! the pipeline's kind, its colour attachments' pixel formats (`MTLPixelFormat` values, as
//! `index:format`, a `+` where blending is on), its depth and stencil formats, sample count,
//! rasterization and topology, its vertex or stage-input layout, and its functions, each by a key
//! ([`function_key`]), never by its name, which is the guest's.
//!
//! A render create reads
//! `render colors [0:80+] depth 0 stencil 0 samples 1 raster 1 topology 0 vertex [b30 16/1/1, a0
//! 29@0 b30] functions vertex k1f2e3d4c5b6a7980 fragment k0123456789abcdef`: each buffer layout as
//! `b<index> <stride>/<step function>/<step rate>` and each attribute as `a<index>
//! <format>@<offset> b<buffer>`, with `linked vertex <n> fragment <n>` where it links functions.

use std::fmt::Write as _;

use reims_vgpu_vulkan::core::platform::resource::FunctionConstant;
use reims_vgpu_vulkan::core::platform::values::{
    AttributeDesc, BufferLayoutDesc, ColorAttachmentDesc,
};
use sha2::{Digest, Sha256};

use super::library::LibraryKey;
use super::{ComputeFields, MeshFields, RenderFields};

/// The bytes a function key's digest starts with ([`function_key`]).
const KEY_PREFIX: &[u8] = b"rgfx-function-v1";

/// A function's key: `k` and the first 8 bytes, in hex, of the SHA-256 of [`KEY_PREFIX`], the
/// content key the device holds its library under, its name and its constants (each one's index,
/// type and value bytes), so equal functions of equal libraries have one key in every run and on
/// every host; or, where the device holds the library under no key (`held` is `None`), `u` and
/// the digest with the library's object id in its content key's place, a key of this run alone.
pub(super) fn function_key(
    held: Option<LibraryKey>,
    library_obj_id: u64,
    name: &str,
    constants: &[FunctionConstant],
) -> String {
    let mut digest = Sha256::new();
    digest.update(KEY_PREFIX);
    let mark = match held {
        Some(key) => {
            digest.update(key);
            'k'
        }
        None => {
            digest.update(library_obj_id.to_le_bytes());
            'u'
        }
    };
    digest.update(u64::try_from(name.len()).unwrap_or(u64::MAX).to_le_bytes());
    digest.update(name.as_bytes());
    for c in constants {
        digest.update(c.index.to_le_bytes());
        digest.update(c.data_type.to_le_bytes());
        digest.update(c.value);
    }
    let digest = digest.finalize();
    let mut key = String::with_capacity(17);
    key.push(mark);
    for byte in digest.iter().take(8) {
        let _ = write!(key, "{byte:02x}");
    }
    key
}

/// A render create's description (the module docs), with its functions' keys and how many
/// functions each stage links.
pub(super) fn render(
    fields: &RenderFields,
    vertex: &str,
    fragment: Option<&str>,
    (vertex_linked, fragment_linked): (usize, usize),
) -> String {
    let mut text = String::from("render ");
    colors(&mut text, &fields.colors);
    let _ = write!(
        text,
        " depth {} stencil {} samples {} raster {} topology {}",
        fields.depth_pixel_format,
        fields.stencil_pixel_format,
        fields.raster_sample_count,
        u8::from(fields.rasterization_enabled),
        fields.input_primitive_topology,
    );
    match &fields.vertex_descriptor {
        Some(v) => layout(&mut text, " vertex", &v.layouts, &v.attributes),
        None => text.push_str(" vertex none"),
    }
    let _ = write!(
        text,
        " functions vertex {vertex} fragment {}",
        fragment.unwrap_or("none")
    );
    if vertex_linked > 0 || fragment_linked > 0 {
        let _ = write!(
            text,
            " linked vertex {vertex_linked} fragment {fragment_linked}"
        );
    }
    text
}

/// A compute create's description (the module docs), with its function's key, how many
/// functions it links and how many dynamic libraries.
pub(super) fn compute(
    fields: &ComputeFields,
    function: &str,
    linked: usize,
    dynamic: usize,
) -> String {
    let mut text = format!(
        "compute function {function} threads {}",
        fields.max_total_threads_per_threadgroup
    );
    if let Some(input) = &fields.stage_input {
        layout(&mut text, " stage input", &input.layouts, &input.attributes);
    }
    if linked > 0 {
        let _ = write!(text, " linked {linked}");
    }
    if dynamic > 0 {
        let _ = write!(text, " dynamic libraries {dynamic}");
    }
    text
}

/// A mesh render create's description (the module docs), with its functions' keys.
pub(super) fn mesh(
    fields: &MeshFields,
    object: Option<&str>,
    mesh: &str,
    fragment: Option<&str>,
) -> String {
    let mut text = String::from("mesh ");
    colors(&mut text, &fields.colors);
    let _ = write!(
        text,
        " depth {} stencil {} samples {} raster {} functions object {} mesh {mesh} fragment {}",
        fields.depth_pixel_format,
        fields.stencil_pixel_format,
        fields.raster_sample_count,
        u8::from(fields.rasterization_enabled),
        object.unwrap_or("none"),
        fragment.unwrap_or("none"),
    );
    text
}

/// Appends `colors [<index>:<format>, ...]`, a `+` after a format where blending is on.
fn colors(text: &mut String, colors: &[(u64, ColorAttachmentDesc)]) {
    text.push_str("colors [");
    for (n, (index, c)) in colors.iter().enumerate() {
        let _ = write!(
            text,
            "{}{index}:{}{}",
            if n == 0 { "" } else { ", " },
            c.pixel_format,
            if c.blending_enabled { "+" } else { "" }
        );
    }
    text.push(']');
}

/// Appends `<what> [<layouts>, <attributes>]` (the module docs).
fn layout(
    text: &mut String,
    what: &str,
    layouts: &[BufferLayoutDesc],
    attributes: &[AttributeDesc],
) {
    let _ = write!(text, "{what} [");
    let mut first = true;
    let mut sep = |text: &mut String| {
        if !first {
            text.push_str(", ");
        }
        first = false;
    };
    for l in layouts {
        sep(text);
        let _ = write!(
            text,
            "b{} {}/{}/{}",
            l.buffer_index, l.stride, l.step_function, l.step_rate
        );
    }
    for a in attributes {
        sep(text);
        let _ = write!(
            text,
            "a{} {}@{} b{}",
            a.attribute_index, a.format, a.offset, a.buffer_index
        );
    }
    text.push(']');
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A constant of index `index` whose first value byte is `byte`.
    fn constant(index: u32, byte: u8) -> FunctionConstant {
        let mut value = [0u8; FunctionConstant::MAX_BYTES];
        value[0] = byte;
        FunctionConstant {
            index,
            data_type: 53,
            value,
        }
    }

    /// A function's key names its library's content, its name and its constants, and never the
    /// name itself: equal inputs give one key, each changed input another, and a library the
    /// device holds under no key gives a `u` key of its object id.
    #[test]
    fn a_function_key_names_the_function_without_its_name() {
        let lib = [7u8; 32];
        let a = function_key(Some(lib), 0x1_0000_0003, "fragment_main", &[constant(0, 1)]);
        assert_eq!(a.len(), 17);
        assert!(a.starts_with('k'), "{a}");
        assert!(a[1..].chars().all(|c| c.is_ascii_hexdigit()), "{a}");
        assert!(!a.contains("fragment"), "{a}");
        // The object id does not count where the library is keyed.
        assert_eq!(
            a,
            function_key(Some(lib), 0x2_0000_0009, "fragment_main", &[constant(0, 1)])
        );
        let others = [
            function_key(
                Some([8; 32]),
                0x1_0000_0003,
                "fragment_main",
                &[constant(0, 1)],
            ),
            function_key(
                Some(lib),
                0x1_0000_0003,
                "fragment_mainx",
                &[constant(0, 1)],
            ),
            function_key(Some(lib), 0x1_0000_0003, "fragment_main", &[constant(1, 1)]),
            function_key(Some(lib), 0x1_0000_0003, "fragment_main", &[constant(0, 2)]),
            function_key(Some(lib), 0x1_0000_0003, "fragment_main", &[]),
        ];
        for other in &others {
            assert_ne!(&a, other);
        }
        let unkeyed = function_key(None, 0x1_0000_0003, "fragment_main", &[constant(0, 1)]);
        assert!(unkeyed.starts_with('u'), "{unkeyed}");
        assert_ne!(
            unkeyed,
            function_key(None, 0x1_0000_0004, "fragment_main", &[constant(0, 1)])
        );
    }

    /// A render create's description names its formats, layout and function keys as the module
    /// docs give them.
    #[test]
    fn a_render_description_names_formats_layout_and_keys() {
        let fields = RenderFields {
            vertex_descriptor: Some(reims_vgpu_vulkan::core::platform::values::VertexDesc {
                layouts: vec![BufferLayoutDesc {
                    buffer_index: 30,
                    stride: 16,
                    step_function: 1,
                    step_rate: 1,
                }],
                attributes: vec![AttributeDesc {
                    attribute_index: 0,
                    format: 29,
                    offset: 0,
                    buffer_index: 30,
                }],
            }),
            colors: vec![
                (
                    0,
                    ColorAttachmentDesc {
                        pixel_format: 80,
                        blending_enabled: true,
                        ..ColorAttachmentDesc::default()
                    },
                ),
                (
                    1,
                    ColorAttachmentDesc {
                        pixel_format: 65,
                        ..ColorAttachmentDesc::default()
                    },
                ),
            ],
            depth_pixel_format: 252,
            stencil_pixel_format: 0,
            raster_sample_count: 4,
            alpha_to_coverage: false,
            alpha_to_one: false,
            rasterization_enabled: true,
            input_primitive_topology: 3,
            max_vertex_amplification_count: 1,
            support_indirect_command_buffers: false,
            private_state: None,
            tessellation: Default::default(),
            max_vertex_call_stack_depth: 1,
            max_fragment_call_stack_depth: 1,
        };
        assert_eq!(
            render(&fields, "kaa", Some("kbb"), (0, 0)),
            "render colors [0:80+, 1:65] depth 252 stencil 0 samples 4 raster 1 topology 3 vertex \
             [b30 16/1/1, a0 29@0 b30] functions vertex kaa fragment kbb"
        );
        let mut bare = fields;
        bare.vertex_descriptor = None;
        bare.colors.truncate(1);
        assert_eq!(
            render(&bare, "kaa", None, (2, 0)),
            "render colors [0:80+] depth 252 stencil 0 samples 4 raster 1 topology 3 vertex none \
             functions vertex kaa fragment none linked vertex 2 fragment 0"
        );
    }
}
