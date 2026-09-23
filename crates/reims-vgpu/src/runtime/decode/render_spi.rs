//! The render-rail records the closure ledger has **not settled**.
//!
//! Thirty-one opcodes, in six families: the five patch draws, the two
//! `executeCommandsInBuffer:` forms, the six `writeDescriptor` pass-property
//! records, the three store-action-options records, the two tessellation
//! records, and the twelve vertex-amplification and tile-shader records.
//!
//! # Why they are decoded here and not in `reims-vgpu-protocol`
//!
//! An unresolved row has no established contract, so the layer that assigns
//! meaning to a wire tag may not give it a shape.
//! `reims_vgpu_protocol::render::RenderKind` names exactly the forty-five rows
//! the ledger has settled and `decode::render` lifts those; `of_opcode`
//! answering `None` is the ledger's own statement that a row is not closed, and
//! it is the routing question `runtime::exec::handle_render_record` asks.
//!
//! # Three different reasons a row is here, and they are not the same claim
//!
//! **Two rows drive real work.** `executeCommandsInBuffer:` in both its forms
//! pushes onto `StreamAccum::execute_icb` and is executed at end of stream by
//! `runtime::icb`. Those two are `Implemented` in the ledger and unsettled only
//! in the sense that no protocol record covers them yet — the compute rail's
//! own pair is in the same position, and [`super::compute_spi`] says so too.
//!
//! **Four rows are refused by contract.** `setColorStoreActionOptions:` and its
//! depth and stencil siblings, and `writeDescriptor`'s
//! `defaultRasterSampleCount`, are `Closure::Refused`. They are decoded here
//! rather than answered from `decode::no_record` because the *value* is the
//! evidence: a refusal that says which options word or which sample count the
//! guest asked for is what will settle whether the row needs implementing, and
//! a bare `RefusedByContract` says only that a record arrived.
//!
//! **The rest are declined pending evidence**, in the shape
//! [`super::blit_spi`] describes: nothing downstream tessellates, amplifies
//! vertices or runs a tile shader, so no field they carry has a consumer, and
//! the decline's count is the measurement that will settle the row. What is
//! lifted is what the *census* needs — which form, and whether the guest asked
//! for anything other than the API default — and nothing more. A field with no
//! consumer is a producer this module would have to keep correct for nobody.
//!
//! # It refuses every settled opcode by name
//!
//! No record on this rail has two readings. A settled row reaching [`decode`]
//! is [`DecodeStatus::ErrSettledElsewhere`], not a record — the routing in
//! `runtime::exec::handle_render_record` disagreeing with the ledger, and it is
//! named rather than answered a second time.

use reims_vgpu_wire::ops::render as wire;
use reims_vgpu_wire::ops::render_pass as wire_pass;
use reims_vgpu_wire::ops::tile as wire_tile;

/// Metadata for an OpenGL compatibility record this decoder does not yet
/// interpret. The PCI guest personality currently declines `supportsOpenGL`,
/// so these records should not appear in the normal x86 path. This inventory
/// is diagnostic only and does not authorize or decode their payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenGlCompatOpcode {
    pub opcode: u32,
    pub selector: &'static str,
    pub body_len: usize,
}

/// Records unlocked by the serializer's OpenGL compatibility rung. Selector
/// names and body lengths mirror the measured serializer inventory in
/// `reims-vgpu-wire::manifest`.
pub const OPENGL_COMPAT_OPCODES: &[OpenGlCompatOpcode] = &[
    OpenGlCompatOpcode {
        opcode: 0x8a,
        selector: "setAlphaTestReferenceValue:",
        body_len: 4,
    },
    OpenGlCompatOpcode {
        opcode: 0x8b,
        selector: "setPointSize:",
        body_len: 4,
    },
    OpenGlCompatOpcode {
        opcode: 0x8c,
        selector: "setClipPlane:p2:p3:p4:atIndex:",
        body_len: 20,
    },
    OpenGlCompatOpcode {
        opcode: 0x8d,
        selector: "setVertexSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:",
        body_len: 20,
    },
    OpenGlCompatOpcode {
        opcode: 0x8e,
        selector: "setFragmentSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:",
        body_len: 20,
    },
    OpenGlCompatOpcode {
        opcode: 0x8f,
        selector: "setViewportTransformEnabled:",
        body_len: 4,
    },
    OpenGlCompatOpcode {
        opcode: 0x90,
        selector: "setProvokingVertexMode:",
        body_len: 4,
    },
    OpenGlCompatOpcode {
        opcode: 0x91,
        selector: "setPrimitiveRestartEnabled:index:",
        body_len: 8,
    },
    OpenGlCompatOpcode {
        opcode: 0x92,
        selector: "setTriangleFrontFillMode:backFillMode:",
        body_len: 4,
    },
    OpenGlCompatOpcode {
        opcode: 0x93,
        selector: "setTransformFeedbackState:",
        body_len: 4,
    },
    OpenGlCompatOpcode {
        opcode: 0x94,
        selector: "setDepthCleared",
        body_len: 0,
    },
    OpenGlCompatOpcode {
        opcode: 0x95,
        selector: "setStencilCleared",
        body_len: 0,
    },
    OpenGlCompatOpcode {
        opcode: 0x96,
        selector: "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:",
        body_len: 16,
    },
    OpenGlCompatOpcode {
        opcode: 0x97,
        selector: "setDepthResolveTexture:slice:depthPlane:level:yInvert:",
        body_len: 12,
    },
    OpenGlCompatOpcode {
        opcode: 0x98,
        selector: "setStencilResolveTexture:slice:depthPlane:level:yInvert:",
        body_len: 12,
    },
];

/// Metadata for a known OpenGL compatibility opcode, if present.
#[inline]
pub fn opengl_compat_opcode(opcode: u32) -> Option<&'static OpenGlCompatOpcode> {
    OPENGL_COMPAT_OPCODES
        .iter()
        .find(|record| record.opcode == opcode)
}

/// Which unsettled record this is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Kind {
    /// Never produced by [`decode`]; the `Default` a caller builds a command
    /// from before filling it in.
    #[default]
    Unknown,
    /// `executeCommandsInBuffer:` — the only family here that reaches an
    /// executor.
    ExecuteCommands,
    /// A patch draw. Which of the six is the opcode's own answer; no field is
    /// lifted, because nothing tessellates.
    DrawPatches,
    /// `setTessellationFactorBuffer:offset:instanceStride:`.
    SetTessellationFactorBuffer,
    /// `setTessellationFactorScale:`. Shares its wire form with `setLineWidth:`
    /// and not its settlement — the line width has a `RenderKind` and this does
    /// not, which is why the two selectors are in two modules.
    SetTessellationFactorScale,
    /// `setVertexAmplificationMode:value:` or
    /// `setVertexAmplificationCount:viewMappings:`.
    SetVertexAmplification,
    /// A tile-shader bind: buffers, textures, samplers or threadgroup memory.
    TileBind,
    /// `dispatchThreadsPerTile:`, in any of its three forms.
    TileDispatch,
    /// `getTileDimensions:` — a query with a destination this device does not
    /// answer.
    TileDimensionsQuery,
    /// One of the three store-action-options records.
    SetStoreActionOptions,
    /// A `writeDescriptor` pass property: the rasterization rate map, the
    /// default raster sample count, the imageblock or threadgroup memory
    /// length, the tile size, or the sample positions.
    RenderPassProperty,
}

/// Why this decoder refused a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    ErrShort,
    ErrUnknownOpcode,
    /// A row the ledger *has* settled, which a protocol decoder owns.
    ErrSettledElsewhere,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs keep the `render_decode_` prefix the rail's decoder reported
    /// under, so a census taken across the cutover reads continuously.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::ErrShort => "render_decode_short",
            Self::ErrUnknownOpcode => "render_decode_unknown_opcode",
            Self::ErrSettledElsewhere => "render_decode_settled_elsewhere",
        })
    }
}

/// One decoded unsettled render record.
///
/// A struct rather than an enum, for the reason [`super::compute_spi::Command`]
/// is one: its consumers switch on [`Kind`] and read the fields their own arm
/// needs. The fields that are not this record's are zero, which is the flat
/// shape the settled records have left behind — and the reason to leave it here
/// rather than carry it forward: when a row is settled its record moves to the
/// protocol crate with a payload of its own.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Command {
    pub opcode: u32,
    pub kind: Kind,
    /// A bind's first slot, a store-action-options record's attachment index,
    /// or a sample-positions record's count.
    pub first: u32,
    /// How many entries a bind or a counted record carries.
    pub count: u32,
    /// The one raw ordinal or packed value a record carries: a store-action
    /// options word, an amplification mode, a raster sample count, a memory
    /// length, or a tile size packed `width | height << 16`.
    pub mode: u64,
    /// `setTessellationFactorScale:`'s float, as the guest wrote it.
    pub float_value: f32,
    /// The buffer a tessellation-factor or tile-dimensions record names.
    pub buffer_ref: u32,
    pub buffer_offset: u64,
    /// The texture a rasterization-rate-map property names.
    pub texture_ref: u32,
    /// A tile dispatch's grid. Not borrowed for anything else — a threadgroup
    /// memory length is a size and not a grid, and they were one field once.
    pub tile_threads: [u64; 3],
    /// `setVertexAmplificationMode:value:`'s second argument.
    pub amplification_value: u32,
    /// Whether any view mapping in a `setVertexAmplificationCount:` record is
    /// not the identity. The pairs stay unlifted — nothing amplifies — but
    /// "the guest asked for a count" and "the guest asked for a remapping" are
    /// different records, and only this tells them apart.
    pub amplification_offsets_views: bool,
    /// The indirect-command-buffer execution's fields. `range_*` belong to the
    /// range form and `args_*` to the indirect one; neither form writes both,
    /// and [`Command::icb_is_range`] says which.
    pub indirect_command_buffer_ref: u32,
    pub icb_is_range: bool,
    pub icb_range_location: u64,
    pub icb_range_length: u64,
    pub icb_args_buffer_ref: u32,
    pub icb_args_buffer_offset: u64,
}

/// Whether `opcode` is one of the thirty-one rows this module owns.
#[must_use]
pub fn is_unsettled(opcode: u32) -> bool {
    matches!(
        opcode,
        wire::OPCODE_EXECUTE_COMMANDS_INDIRECT
            | wire::OPCODE_EXECUTE_COMMANDS_RANGE
            | wire::OPCODE_DRAW_PATCHES
            | wire::OPCODE_DRAW_PATCHES_WIDE
            | wire::OPCODE_DRAW_INDEXED_PATCHES
            | wire::OPCODE_DRAW_PATCHES_INDIRECT
            | wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT
            | wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER
            | wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE
            | wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS
            | wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS
            | wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS
            | wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE
            | wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT
            | wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY
            | wire_tile::OPCODE_SET_TILE_BUFFER
            | wire_tile::OPCODE_SET_TILE_TEXTURE
            | wire_tile::OPCODE_SET_TILE_SAMPLER
            | wire_tile::OPCODE_SET_TILE_SAMPLER_LOD
            | wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET
            | wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE
            | wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION
            | wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX
            | wire_tile::OPCODE_GET_TILE_DIMENSIONS
            | wire_pass::OPCODE_RASTERIZATION_RATE_MAP
            | wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT
            | wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH
            | wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH
            | wire_pass::OPCODE_TILE_SIZE
            | wire_pass::OPCODE_SAMPLE_POSITIONS
    )
}

/// Bytes a bind record of `count` entries occupies past its header.
fn bind_record_len(count: u32, entry_size: usize) -> Option<usize> {
    (count as usize)
        .checked_mul(entry_size)
        .and_then(|n| n.checked_add(core::mem::size_of::<wire::BindHeader>()))
}

/// Decode one unsettled render-rail record.
///
/// # Errors
///
/// [`DecodeStatus::ErrSettledElsewhere`] for a row a protocol decoder owns,
/// [`DecodeStatus::ErrUnknownOpcode`] for an opcode no render row names, and
/// [`DecodeStatus::ErrShort`] for a record whose length is not its body's.
pub fn decode(command: &[u8]) -> Result<Command, DecodeStatus> {
    let op = reims_vgpu_wire::op(command, 0).map_err(|_| DecodeStatus::ErrShort)?;
    let opcode = op.opcode();
    if !is_unsettled(opcode) {
        // The ledger is what says which of the two this is, so neither answer
        // is this module's opinion about an opcode.
        return Err(
            if reims_vgpu_protocol::closure::find(
                reims_vgpu_protocol::closure::Rail::Render,
                opcode,
            )
            .is_some()
            {
                DecodeStatus::ErrSettledElsewhere
            } else {
                DecodeStatus::ErrUnknownOpcode
            },
        );
    }
    let command_length = op.length() as usize;
    let payload = op.payload;
    let want = |need: u32| {
        if command_length == need as usize {
            Ok(())
        } else {
            Err(DecodeStatus::ErrShort)
        }
    };
    let mut out = Command {
        opcode,
        ..Default::default()
    };
    match opcode {
        wire::OPCODE_EXECUTE_COMMANDS_INDIRECT => {
            want(wire::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN)?;
            let e = wire::execute_commands_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ExecuteCommands;
            out.icb_is_range = false;
            out.indirect_command_buffer_ref = e.icb_ref.get();
            out.icb_args_buffer_ref = e.indirect_buffer_ref.get();
            out.icb_args_buffer_offset = e.indirect_buffer_offset.get();
        }
        wire::OPCODE_EXECUTE_COMMANDS_RANGE => {
            want(wire::EXECUTE_COMMANDS_RANGE_TOTAL_LEN)?;
            let e = wire::execute_commands_range(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ExecuteCommands;
            out.icb_is_range = true;
            out.indirect_command_buffer_ref = e.icb_ref.get();
            out.icb_range_location = e.range_location.get();
            out.icb_range_length = e.range_length.get();
        }
        wire::OPCODE_DRAW_PATCHES
        | wire::OPCODE_DRAW_PATCHES_WIDE
        | wire::OPCODE_DRAW_INDEXED_PATCHES
        | wire::OPCODE_DRAW_PATCHES_INDIRECT
        | wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => {
            // Six records across five opcodes, because `0x0c` is two: the plain
            // wide draw at 56 bytes and the indexed wide draw at 68. Dispatched
            // on the length rather than guessed, and a `0x0c` at any other
            // length is refused rather than read as whichever is closer — the
            // two bodies disagree from their tenth byte on.
            //
            // Every form is length-checked exactly, so a truncated patch draw
            // is refused rather than read as a draw with invented counts.
            //
            // None of the fields is lifted. Nothing here tessellates, so a
            // `patch_count` would be a producer with no consumer; what
            // `runtime::exec` needs is that a patch draw happened and which
            // form, and the opcode carries both.
            let ok = match opcode {
                wire::OPCODE_DRAW_PATCHES => {
                    command_length == wire::DRAW_PATCHES_TOTAL_LEN as usize
                }
                wire::OPCODE_DRAW_INDEXED_PATCHES => {
                    command_length == wire::DRAW_INDEXED_PATCHES_TOTAL_LEN as usize
                }
                wire::OPCODE_DRAW_PATCHES_INDIRECT => {
                    command_length == wire::DRAW_PATCHES_INDIRECT_TOTAL_LEN as usize
                }
                wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => {
                    command_length == wire::DRAW_INDEXED_PATCHES_INDIRECT_TOTAL_LEN as usize
                }
                // `0x0c`: the two wide forms, and nothing else.
                _ => {
                    command_length == wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize
                        || command_length == wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN as usize
                }
            };
            if !ok {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::DrawPatches;
        }
        wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER => {
            // Not a bind: one buffer per encoder, so there is no slot and no
            // count — the ref and its two `u64` sit directly in the payload.
            want(wire::SET_TESSELLATION_FACTOR_BUFFER_TOTAL_LEN)?;
            let t =
                wire::set_tessellation_factor_buffer(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetTessellationFactorBuffer;
            out.buffer_ref = t.buffer_ref.get();
            out.buffer_offset = t.offset.get();
        }
        wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE => {
            let f = wire::float_state(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetTessellationFactorScale;
            out.float_value = f.value.get();
        }
        wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS
        | wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS
        | wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS => {
            // The same three-attachment split one opcode below the store
            // actions, and the widths do *not* carry over: the options are a
            // `u64` where the store action is a `u32`, so the colour form's
            // index sits at `+8` rather than `+4` and the record is 20 bytes
            // rather than 16.
            if opcode == wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS {
                want(wire::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN)?;
                let a = wire::set_color_store_action_options(&op)
                    .map_err(|_| DecodeStatus::ErrShort)?;
                out.mode = a.options.get();
                out.first = a.index.get();
            } else {
                want(wire::SET_STORE_ACTION_OPTIONS_TOTAL_LEN)?;
                let a = wire::set_store_action_options(&op).map_err(|_| DecodeStatus::ErrShort)?;
                out.mode = a.options.get();
            }
            out.kind = Kind::SetStoreActionOptions;
        }
        wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE => {
            want(wire::SET_VERTEX_AMPLIFICATION_MODE_TOTAL_LEN)?;
            let m = wire::vertex_amplification_mode(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetVertexAmplification;
            out.mode = u64::from(m.mode.get());
            out.amplification_value = m.value.get();
        }
        wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT => {
            // Four-byte count head (not `BindHeader`); mappings follow and are
            // not lifted — nothing downstream amplifies. The wire parser bounds
            // entries to the record length.
            let (head, mappings) =
                wire::vertex_amplification_count(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetVertexAmplification;
            out.count = head.count.get();
            if out.count as usize != mappings.len() {
                return Err(DecodeStatus::ErrShort);
            }
            out.amplification_offsets_views = mappings.iter().any(|m| {
                m.viewport_array_index_offset.get() != 0
                    || m.render_target_array_index_offset.get() != 0
            });
        }
        wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY => {
            want(wire_tile::SET_TILE_THREADGROUP_MEMORY_TOTAL_LEN)?;
            let m = wire_tile::tile_threadgroup_memory(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = m.index.get();
            out.count = 1;
            // The length and the offset are read by the view and then not
            // lifted, exactly as the other tile binds' entries are not: nothing
            // downstream allocates imageblock memory, so a field carrying the
            // size would have no consumer. `tile_threads` in particular is a
            // dispatch's grid and must not be borrowed for it.
        }
        wire_tile::OPCODE_SET_TILE_BUFFER
        | wire_tile::OPCODE_SET_TILE_TEXTURE
        | wire_tile::OPCODE_SET_TILE_SAMPLER
        | wire_tile::OPCODE_SET_TILE_SAMPLER_LOD => {
            // Four bind families with one shape: a `BindHeader` and `count`
            // entries whose width the opcode decides. Entries are not lifted
            // (no tile table consumer); the head and the length check are what
            // says the record is well formed.
            let (head, entry_size) = match opcode {
                wire_tile::OPCODE_SET_TILE_BUFFER => (
                    wire_tile::tile_buffer_binds(&op).map(|(h, _)| h),
                    core::mem::size_of::<wire::BufferBind>(),
                ),
                wire_tile::OPCODE_SET_TILE_TEXTURE => (
                    wire_tile::tile_texture_binds(&op).map(|(h, _)| h),
                    core::mem::size_of::<wire::RefBind>(),
                ),
                wire_tile::OPCODE_SET_TILE_SAMPLER => (
                    wire_tile::tile_sampler_binds(&op).map(|(h, _)| h),
                    core::mem::size_of::<wire::RefBind>(),
                ),
                _ => (
                    wire_tile::tile_sampler_lod_binds(&op).map(|(h, _)| h),
                    core::mem::size_of::<wire::SamplerLodBind>(),
                ),
            };
            let head = head.map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrShort);
            }
            match bind_record_len(out.count, entry_size) {
                Some(need) if payload.len() >= need => {}
                _ => return Err(DecodeStatus::ErrShort),
            }
        }
        wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET => {
            want(wire_tile::SET_TILE_BUFFER_OFFSET_TOTAL_LEN)?;
            let b = wire_tile::tile_buffer_offset(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = b.index.get();
            out.count = 1;
            out.buffer_offset = b.offset.get();
        }
        wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE => {
            want(wire_tile::DISPATCH_THREADS_PER_TILE_TOTAL_LEN)?;
            let d =
                wire_tile::dispatch_threads_per_tile(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileDispatch;
            out.tile_threads = [d.width.get(), d.height.get(), d.depth.get()];
        }
        wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION
        | wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX => {
            want(wire_tile::DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN)?;
            let d = wire_tile::dispatch_threads_per_tile_in_region(&op)
                .map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileDispatch;
            out.tile_threads = [d.width.get(), d.height.get(), d.depth.get()];
            // Region / RT index not lifted — see the wire tile module.
        }
        wire_tile::OPCODE_GET_TILE_DIMENSIONS => {
            want(wire_tile::GET_TILE_DIMENSIONS_TOTAL_LEN)?;
            let g = wire_tile::get_tile_dimensions(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileDimensionsQuery;
            out.buffer_ref = g.buffer_ref.get();
            out.buffer_offset = g.offset.get();
        }
        wire_pass::OPCODE_RASTERIZATION_RATE_MAP => {
            let r = wire_pass::pass_rate_map(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::RenderPassProperty;
            out.texture_ref = r.rate_map_ref.get();
        }
        wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT => {
            let c =
                wire_pass::default_raster_sample_count(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::RenderPassProperty;
            out.mode = u64::from(c.count.get());
        }
        wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH
        | wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH => {
            let m = wire_pass::tile_memory(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::RenderPassProperty;
            out.mode = u64::from(m.length.get());
        }
        wire_pass::OPCODE_TILE_SIZE => {
            let s = wire_pass::tile_size(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::RenderPassProperty;
            // width | height << 16 — this device's packing, not a second wire
            // layout.
            out.mode = u64::from(s.width.get()) | (u64::from(s.height.get()) << 16);
        }
        wire_pass::OPCODE_SAMPLE_POSITIONS => {
            let (head, positions) =
                wire_pass::sample_positions(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::RenderPassProperty;
            out.count = head.count.get();
            if out.count as usize != positions.len() {
                return Err(DecodeStatus::ErrShort);
            }
        }
        // `is_unsettled` gates this match, and the two are the same list.
        _ => return Err(DecodeStatus::ErrUnknownOpcode),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::closure::{Rail, LEDGER};
    use reims_vgpu_wire::OP_HEADER_LEN;

    #[test]
    fn opengl_compatibility_inventory_is_explicit_and_closed() {
        assert_eq!(OPENGL_COMPAT_OPCODES.len(), 15);
        assert!(opengl_compat_opcode(0x8a).is_some());
        assert!(opengl_compat_opcode(0x98).is_some());
        assert_eq!(opengl_compat_opcode(0x94).unwrap().body_len, 0);
        assert_eq!(opengl_compat_opcode(0x8d).unwrap().body_len, 20);
        assert!(opengl_compat_opcode(0x89).is_none());
        assert!(opengl_compat_opcode(0x99).is_none());
    }

    fn record(opcode: u32, total_len: u32) -> Vec<u8> {
        let mut v = vec![0u8; total_len as usize];
        v[0..4].copy_from_slice(&opcode.to_le_bytes());
        v[4..8].copy_from_slice(&total_len.to_le_bytes());
        v
    }

    fn body(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let total = (OP_HEADER_LEN + payload.len()) as u32;
        let mut v = record(opcode, total);
        v[OP_HEADER_LEN..].copy_from_slice(payload);
        v
    }

    /// No settled render row is decodable here.
    ///
    /// This is the claim that keeps "one record, one reading" structural rather
    /// than a property of the current routing. `runtime::exec` picks between
    /// this decoder and the protocol crate's by whether `RenderKind::of_opcode`
    /// names the row; if this one also answered a settled row, a routing
    /// mistake would be a silent second interpretation instead of a named
    /// refusal.
    ///
    /// The four `Refused` rows are *not* settled in that sense and are this
    /// module's — see the module header for why a refusal's value is the
    /// evidence — so the filter is `RenderKind::of_opcode`, which is the
    /// question the routing actually asks, rather than the ledger's closure.
    #[test]
    fn no_row_a_render_kind_names_is_a_record_this_decoder_owns() {
        let mut settled = 0;
        for op in LEDGER
            .iter()
            .filter(|o| o.rail == Rail::Render)
            .filter_map(|o| o.opcode)
            .filter(|op| reims_vgpu_protocol::render::RenderKind::of_opcode(*op).is_some())
        {
            settled += 1;
            assert_eq!(
                decode(&record(op, 256)),
                Err(DecodeStatus::ErrSettledElsewhere),
                "render {op:#x} has a RenderKind and this decoder claimed it"
            );
        }
        assert_eq!(
            settled,
            reims_vgpu_protocol::render::RenderKind::ALL.len(),
            "every RenderKind's opcode should be a render row the ledger carries"
        );
    }

    /// Every row this module claims is one the ledger carries and no
    /// `RenderKind` names.
    ///
    /// The other direction of the test above, and the one that catches an
    /// opcode invented here: a row in [`is_unsettled`] that the ledger does not
    /// record at all would be this device deciding a record exists.
    #[test]
    fn every_row_this_decoder_owns_is_an_unsettled_render_row() {
        let ledger: Vec<u32> = LEDGER
            .iter()
            .filter(|o| o.rail == Rail::Render)
            .filter_map(|o| o.opcode)
            .collect();
        let mut owned = 0;
        for op in 0..=0x400u32 {
            if !is_unsettled(op) {
                continue;
            }
            owned += 1;
            assert!(
                ledger.contains(&op),
                "render {op:#x} is claimed here and the ledger records no such row"
            );
            assert!(
                reims_vgpu_protocol::render::RenderKind::of_opcode(op).is_none(),
                "render {op:#x} is claimed here and a RenderKind names it too"
            );
        }
        assert_eq!(
            owned, 30,
            "the module header says thirty-one rows in six \
             families, of which thirty carry an opcode below the scan bound"
        );
    }

    /// An opcode no render row names is unknown rather than settled.
    ///
    /// The two refusals answer different questions — "the ledger owns this
    /// elsewhere" and "the ledger has never heard of this" — and a decoder that
    /// gave one answer for both would report a routing bug and an unknown wire
    /// tag under the same slug.
    #[test]
    fn an_opcode_no_render_row_names_is_unknown_rather_than_settled() {
        assert_eq!(
            decode(&record(0x3fff, 64)),
            Err(DecodeStatus::ErrUnknownOpcode)
        );
    }

    /// `0x0c` is two records, told apart by length and by nothing else.
    ///
    /// The plain wide patch draw is 56 bytes and the indexed wide one is 68.
    /// Their bodies disagree from the tenth byte on, so a `0x0c` at any other
    /// length is refused rather than read as whichever is closer.
    #[test]
    fn the_wide_patch_draw_opcode_is_resolved_by_length_and_refused_without_one() {
        for len in [
            wire::DRAW_PATCHES_WIDE_TOTAL_LEN,
            wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN,
        ] {
            assert_eq!(
                decode(&record(wire::OPCODE_DRAW_PATCHES_WIDE, len))
                    .expect("a wide patch draw at one of its two lengths decodes")
                    .kind,
                Kind::DrawPatches
            );
        }
        // Between the two, and one byte short of each.
        for len in [
            wire::DRAW_PATCHES_WIDE_TOTAL_LEN - 1,
            wire::DRAW_PATCHES_WIDE_TOTAL_LEN + 4,
            wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN + 1,
        ] {
            assert_eq!(
                decode(&record(wire::OPCODE_DRAW_PATCHES_WIDE, len)),
                Err(DecodeStatus::ErrShort),
                "a {len}-byte 0x0c was read as one of the two wide patch draws"
            );
        }
    }

    /// The store-action options are not wider store actions.
    ///
    /// The options word is a `u64` where the store action is a `u32`, so the
    /// colour form's attachment index sits at `+8` rather than `+4` and the
    /// record is 20 bytes rather than 16. Read at the store action's offsets,
    /// the index would come back as the options word's high half.
    #[test]
    fn the_store_action_options_are_not_wider_store_actions() {
        let mut payload = vec![0u8; 12];
        payload[0..8].copy_from_slice(&0x1111u64.to_le_bytes());
        payload[8..12].copy_from_slice(&3u32.to_le_bytes());
        let colour = decode(&body(wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS, &payload))
            .expect("the colour options record decodes");
        assert_eq!(
            (colour.kind, colour.mode, colour.first),
            (Kind::SetStoreActionOptions, 0x1111, 3)
        );

        // The depth and stencil forms name their attachment by being
        // themselves, so they carry the options word alone.
        for op in [
            wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
            wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
        ] {
            let r =
                decode(&body(op, &0x2222u64.to_le_bytes())).expect("a bare options record decodes");
            assert_eq!(
                (r.kind, r.mode, r.first),
                (Kind::SetStoreActionOptions, 0x2222, 0)
            );
        }
    }

    /// A vertex-amplification count record says whether any mapping is the
    /// identity, which is not the same question as how many there are.
    ///
    /// The pairs stay unlifted — nothing amplifies — but "the guest asked for a
    /// count" and "the guest asked for a remapping" are different records, and
    /// this flag is the only thing that tells them apart.
    #[test]
    fn an_amplification_count_reports_whether_any_mapping_moves_a_view() {
        let mapping = core::mem::size_of::<wire::ViewMapping>();
        let build = |mappings: &[(u32, u32)]| {
            let mut payload = vec![0u8; 4 + mappings.len() * mapping];
            payload[0..4].copy_from_slice(&(mappings.len() as u32).to_le_bytes());
            for (i, (viewport, target)) in mappings.iter().enumerate() {
                let at = 4 + i * mapping;
                payload[at..at + 4].copy_from_slice(&viewport.to_le_bytes());
                payload[at + 4..at + 8].copy_from_slice(&target.to_le_bytes());
            }
            body(wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT, &payload)
        };
        let identity = decode(&build(&[(0, 0), (0, 0)])).expect("two identity mappings decode");
        assert_eq!(identity.count, 2);
        assert!(
            !identity.amplification_offsets_views,
            "two identity mappings were reported as a remapping"
        );
        let moved = decode(&build(&[(0, 0), (0, 1)])).expect("a moved mapping decodes");
        assert_eq!(moved.count, 2);
        assert!(
            moved.amplification_offsets_views,
            "a mapping offsetting the render-target index was reported as the identity"
        );
    }

    /// The two `executeCommandsInBuffer:` forms name different second operands,
    /// and neither writes the other's fields.
    ///
    /// The range form carries a location and a length; the indirect form
    /// carries a buffer and an offset. `icb_is_range` is what says which, and a
    /// consumer reading the wrong pair would execute a range the guest never
    /// asked for.
    #[test]
    fn the_two_command_buffer_executions_name_different_second_operands() {
        let mut indirect =
            vec![0u8; wire::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN as usize - OP_HEADER_LEN];
        indirect[0..4].copy_from_slice(&0x4242u32.to_le_bytes());
        indirect[4..8].copy_from_slice(&0x5151u32.to_le_bytes());
        indirect[8..16].copy_from_slice(&0x900u64.to_le_bytes());
        let r = decode(&body(wire::OPCODE_EXECUTE_COMMANDS_INDIRECT, &indirect))
            .expect("the indirect form decodes");
        assert_eq!(r.kind, Kind::ExecuteCommands);
        assert!(!r.icb_is_range);
        assert_eq!(
            (
                r.indirect_command_buffer_ref,
                r.icb_args_buffer_ref,
                r.icb_args_buffer_offset
            ),
            (0x4242, 0x5151, 0x900)
        );
        assert_eq!(
            (r.icb_range_location, r.icb_range_length),
            (0, 0),
            "the indirect form wrote the range form's fields"
        );

        let mut range = vec![0u8; wire::EXECUTE_COMMANDS_RANGE_TOTAL_LEN as usize - OP_HEADER_LEN];
        range[0..4].copy_from_slice(&0x4242u32.to_le_bytes());
        range[4..12].copy_from_slice(&7u64.to_le_bytes());
        range[12..20].copy_from_slice(&11u64.to_le_bytes());
        let r = decode(&body(wire::OPCODE_EXECUTE_COMMANDS_RANGE, &range))
            .expect("the range form decodes");
        assert!(r.icb_is_range);
        assert_eq!(
            (
                r.indirect_command_buffer_ref,
                r.icb_range_location,
                r.icb_range_length
            ),
            (0x4242, 7, 11)
        );
        assert_eq!(
            (r.icb_args_buffer_ref, r.icb_args_buffer_offset),
            (0, 0),
            "the range form wrote the indirect form's fields"
        );
    }

    /// Every pass-property record reaches an arm of its own.
    ///
    /// Six records share one `Kind` and are told apart downstream by their
    /// opcode, so what this asserts is that each of the six *decodes* — a
    /// record falling through to a refusal would be counted as an unknown
    /// opcode and its property never priced.
    #[test]
    fn every_pass_property_record_reaches_an_arm_of_its_own() {
        let rate_map = decode(&body(
            wire_pass::OPCODE_RASTERIZATION_RATE_MAP,
            &0x4242u32.to_le_bytes(),
        ))
        .expect("the rate map decodes");
        assert_eq!(
            (rate_map.kind, rate_map.texture_ref),
            (Kind::RenderPassProperty, 0x4242)
        );

        let count = decode(&body(
            wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT,
            &4u32.to_le_bytes(),
        ))
        .expect("the sample count decodes");
        assert_eq!((count.kind, count.mode), (Kind::RenderPassProperty, 4));

        for op in [
            wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH,
            wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH,
        ] {
            let m = decode(&body(op, &0x1000u32.to_le_bytes())).expect("a memory length decodes");
            assert_eq!((m.kind, m.mode), (Kind::RenderPassProperty, 0x1000));
        }

        // Two `u16`, from properties declared `Q` — the width is not four
        // bytes wide, and a fixture that wrote it as one would put the height
        // where nothing reads it.
        let mut tile = vec![0u8; 4];
        tile[0..2].copy_from_slice(&16u16.to_le_bytes());
        tile[2..4].copy_from_slice(&32u16.to_le_bytes());
        let t = decode(&body(wire_pass::OPCODE_TILE_SIZE, &tile)).expect("the tile size decodes");
        // `width | height << 16` — this device's packing, asserted so a reader
        // of the counter knows which half is which.
        assert_eq!(
            (t.kind, t.mode),
            (Kind::RenderPassProperty, 16 | (32 << 16))
        );
    }

    /// A tile dispatch's grid is its own field and not the threadgroup memory
    /// length beside it.
    ///
    /// They were one field once. A threadgroup memory length is a size and a
    /// dispatch grid is three counts; borrowing one for the other would report
    /// a tile shader that never ran.
    #[test]
    fn a_tile_dispatch_grid_is_not_the_threadgroup_memory_length() {
        let mut grid = vec![0u8; 24];
        grid[0..8].copy_from_slice(&8u64.to_le_bytes());
        grid[8..16].copy_from_slice(&4u64.to_le_bytes());
        grid[16..24].copy_from_slice(&1u64.to_le_bytes());
        let d = decode(&body(wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE, &grid))
            .expect("a tile dispatch decodes");
        assert_eq!((d.kind, d.tile_threads), (Kind::TileDispatch, [8, 4, 1]));

        // `length`, `offset`, then `index` — the slot is the record's *last*
        // field, which is the field order a reader assuming "index first" (as
        // every bind record here has) would get wrong.
        let mut memory = vec![0u8; 20];
        memory[0..8].copy_from_slice(&0x800u64.to_le_bytes());
        memory[8..16].copy_from_slice(&0u64.to_le_bytes());
        memory[16..20].copy_from_slice(&2u32.to_le_bytes());
        let m = decode(&body(
            wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
            &memory,
        ))
        .expect("a threadgroup memory record decodes");
        assert_eq!((m.kind, m.first, m.count), (Kind::TileBind, 2, 1));
        assert_eq!(
            m.tile_threads,
            [0, 0, 0],
            "a memory length was lifted into a dispatch grid"
        );
    }

    /// A record short of its body is refused rather than read.
    #[test]
    fn a_record_short_of_its_body_is_refused() {
        for (op, len) in [
            (
                wire::OPCODE_EXECUTE_COMMANDS_RANGE,
                wire::EXECUTE_COMMANDS_RANGE_TOTAL_LEN,
            ),
            (
                wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
                wire::SET_TESSELLATION_FACTOR_BUFFER_TOTAL_LEN,
            ),
            (
                wire_tile::OPCODE_GET_TILE_DIMENSIONS,
                wire_tile::GET_TILE_DIMENSIONS_TOTAL_LEN,
            ),
        ] {
            assert_eq!(
                decode(&record(op, len - 1)),
                Err(DecodeStatus::ErrShort),
                "{op:#x} was decoded one byte short of its body"
            );
        }
    }
}
