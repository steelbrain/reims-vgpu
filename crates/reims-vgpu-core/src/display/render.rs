//! The pass that draws a display's frame into the host's texture: the full-screen strip, the
//! fragment properties (colour matrix and gamma slopes), and the completion that counts the
//! frame out of flight.

use std::sync::atomic::Ordering;

use crate::display::pipeline::PipelineKey;
use crate::display::{DeathKind, Display, DisplayCore, DisplayLocked};
use crate::guest::GuestPtr;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::command::{CommandBuffer, CompletedInfo};
use crate::platform::descriptor::RenderPassDescriptor;
use crate::platform::device::GpuDevice;
use crate::platform::render::{RenderCmd, RenderEncoder};
use crate::platform::resource::{Buffer, GpuResource, Texture};
use crate::platform::types::{
    ClearColor, CommandBufferStatus, MtlOrigin, MtlRegion, MtlSize, MtlViewport, NsRange,
};
use crate::platform::values::ColorAttachmentSet;
use crate::platform::{Handle, Platform};
use crate::raise::{OnFault, RaiseResult};
use crate::rt::pool;

/// `MTLTextureUsageRenderTarget`: the usage a host's encode target needs.
pub const MINIMUM_TEXTURE_USAGE: u64 = 4;
/// `MTLPixelFormatBGRA8Unorm`: the pre-warmed pipelines' format.
const MTL_PIXEL_FORMAT_BGRA8_UNORM: u64 = 80;
/// `MTLResourceStorageModeManaged`: the fragment property buffer's options.
const MTL_RESOURCE_STORAGE_MODE_MANAGED: u64 = 1 << 4;
/// `MTLLoadActionClear`.
const MTL_LOAD_ACTION_CLEAR: u64 = 2;
/// `MTLStoreActionStore`.
const MTL_STORE_ACTION_STORE: u64 = 1;
/// `MTLPrimitiveTypeTriangleStrip`.
const MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP: u64 = 4;
/// `MTLTextureType2D`.
const MTL_TEXTURE_TYPE_2D: u64 = 2;
/// `MTLStorageModePrivate`.
const MTL_STORAGE_MODE_PRIVATE: u64 = 2;
/// `MTLStorageModeMemoryless`.
const MTL_STORAGE_MODE_MEMORYLESS: u64 = 3;

/// The switch for the console's CPU copy (`encode_current_frame_by_copy`): `1` takes it where it
/// is exact, anything else or unset never (measured on an Intel UHD 630 under Linux: the gamma
/// pass over a 1920 × 1080 BGRA8 frame in guest memory took ~5 ms of the GPU per console render).
pub const CONSOLE_CPU_COPY_ENV: &str = "REIMS_VGPU_CONSOLE_CPU_COPY";

fn console_cpu_copy_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(CONSOLE_CPU_COPY_ENV).is_some_and(|value| value == "1"))
}

/// The full-screen triangle strip's positions (float4 × 4).
pub const POS_DATA: [f32; 16] = [
    -1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
];
/// Its texture coordinates (float2 × 4).
pub const TEX_DATA: [f32; 8] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
/// The identity colour matrix a new display starts with.
pub const IDENTITY_COLOR_MATRIX: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];
/// The fragment property buffer's length.
pub(crate) const FRAG_SHADER_PROPERTIES_SIZE: u64 = 96;
/// Where the gamma slopes start in it.
const SLOPE_AT_ZERO_OFFSET: u64 = 64;

/// What the fragment shader reads (96 bytes): `color_matrix` is a `simd_float4x3` (4 columns of 3
/// floats, each padded to 16 bytes). Its layout is the buffer's; the display writes the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct DisplayFragmentShaderProperties {
    /// The colour matrix's columns.
    pub color_matrix: [[f32; 4]; 4],
    /// The gamma table's slope at 0 (red, green, blue).
    pub gamma_table_slope_at_zero: [f32; 3],
    /// Its slope at 1.
    pub gamma_table_slope_at_one: [f32; 3],
    /// The tail padding.
    pub field_58: [f32; 2],
}

const _: () = assert!(size_of::<DisplayFragmentShaderProperties>() == 96);

/// A present's colour matrix, a `float[3][4]`. The guest display's presents pass the guest's page,
/// which the display reads only after the wait for frames in flight and the gamma copy: a source is
/// read element by element at that point, not copied by the caller.
pub trait ColorMatrix {
    /// `matrix[row][col]` (row < 3, col < 4), read now.
    fn at(&self, row: u8, col: u8) -> f32;

    /// `matrix[row][col]` and `matrix[row][col + 1]` (col even), read now as one 64-bit access.
    fn pair(&self, row: u8, col: u8) -> (f32, f32);
}

impl ColorMatrix for [[f32; 4]; 3] {
    fn at(&self, row: u8, col: u8) -> f32 {
        self.get(usize::from(row))
            .and_then(|r| r.get(usize::from(col)))
            .copied()
            .unwrap_or(0.0)
    }

    fn pair(&self, row: u8, col: u8) -> (f32, f32) {
        (self.at(row, col), self.at(row, col.wrapping_add(1)))
    }
}

/// The native-endian bytes of `floats`.
pub(crate) fn f32_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

impl<P: Platform> Display<P> {
    /// The initializer's render state, under the display mutex: both BGRA8 pipelines pre-warmed,
    /// the strip's two buffers filled, the property buffer, and the identity matrix. Metal
    /// refusing a buffer dies; every Metal call passes `OnFault::Die`, so a raise out of Metal
    /// dies at its send and this never answers `Err`.
    pub(super) fn build_render_state(&self) -> RaiseResult<()> {
        let Some(cache) = &self.render_pipeline_cache else {
            return Ok(());
        };
        let mtl_device = &cache.mtl_device;
        let mut g = self.display_mutex.lock();
        // Both pipelines, released once both are made; a nil one is only not cached.
        g.pipeline = PipelineKey {
            format: MTL_PIXEL_FORMAT_BGRA8_UNORM,
            gamma: true,
        };
        let with_gamma = cache.pipeline_for(g.pipeline);
        g.pipeline.gamma = false;
        let without_gamma = cache.pipeline_for(g.pipeline);
        drop((with_gamma, without_gamma));
        let pos = mtl_device.new_buffer_of_length(OnFault::Die, 64, 0)?;
        let Some(pos) = pos else {
            crate::fatal!(DeathKind::PositionBufferRefused)
        };
        GuestPtr::new(pos.contents()).copy_from(0, &f32_bytes(&POS_DATA), DeathKind::NullContents);
        g.pos_buf = Some(pos);
        g.vtx_count = 4;
        let tex = mtl_device.new_buffer_of_length(OnFault::Die, 32, 0)?;
        let Some(tex) = tex else {
            crate::fatal!(DeathKind::TexCoordBufferRefused)
        };
        GuestPtr::new(tex.contents()).copy_from(0, &f32_bytes(&TEX_DATA), DeathKind::NullContents);
        g.tex_buf = Some(tex);
        let frag = mtl_device.new_buffer_of_length(
            OnFault::Die,
            FRAG_SHADER_PROPERTIES_SIZE,
            MTL_RESOURCE_STORAGE_MODE_MANAGED,
        )?;
        let Some(frag) = frag else {
            crate::fatal!(DeathKind::PropertyBufferRefused)
        };
        g.frag_shader_property_buf = Some(frag);
        // The identity; the slopes stay the new buffer's zeros.
        self.update_color_matrix_locked(OnFault::Die, &g, &IDENTITY_COLOR_MATRIX)?;
        Ok(())
    }
}

impl<P: Platform> DisplayCore<P> {
    /// `false`, with nothing encoded, for a command buffer or texture of another device, or a
    /// texture without render-target usage. Counts a host present when the current frame is
    /// new.
    pub fn encode_current_frame(
        &self,
        command_buffer: &P::CommandBuffer,
        texture: &P::Texture,
        region: MtlRegion,
    ) -> RaiseResult<bool> {
        if !self.encodable(command_buffer, texture) {
            return Ok(false);
        }
        let mut g = self.display_mutex.lock();
        self.internal_encode_locked(&mut g, command_buffer, texture, region)?;
        self.count_host_present_locked(&g);
        Ok(true)
    }

    /// Whether `command_buffer` and `texture` are this display's device's and `texture` has the
    /// usage the display's frames need: an encode takes no other.
    fn encodable(&self, command_buffer: &P::CommandBuffer, texture: &P::Texture) -> bool {
        let dev = self.mtl_device().map(Handle::id);
        // Pointer identity, short-circuit.
        command_buffer.device().map(|d| d.id()) == dev
            && texture.device().map(|d| d.id()) == dev
            && (texture.usage() & self.minimum_texture_usage()) == self.minimum_texture_usage()
    }

    /// A host present counted for the first encode of a new current frame, with the display
    /// mutex held.
    fn count_host_present_locked(&self, g: &DisplayLocked<P>) {
        if g.current_frame.is_some() && !self.current_frame_encoded.load(Ordering::SeqCst) {
            // A load and a store of the count.
            let host = self.host_present_count().wrapping_add(1);
            self.props.set_host_present_count(host);
            self.current_frame_encoded.store(true, Ordering::SeqCst);
        }
    }

    /// The encode of the current frame, taking the display mutex (tests; the product holds it).
    #[cfg(test)]
    pub(crate) fn internal_encode_current_frame(
        &self,
        command_buffer: &P::CommandBuffer,
        texture: &P::Texture,
        region: MtlRegion,
    ) -> RaiseResult<()> {
        let mut g = self.display_mutex.lock();
        self.internal_encode_locked(&mut g, command_buffer, texture, region)
    }

    /// Under `REIMS_VGPU_CONSOLE_CPU_COPY=1`, where the gamma pass would leave every byte of the
    /// current frame as it is, the frame itself (a new reference) for the host to read on the CPU
    /// in place of the pass's output: a BGRA8 2D frame the CPU reaches (not private, one sample,
    /// no protection), `texture` BGRA8 of its size, `region` the whole of it, and either no gamma
    /// table or one mapping every 8-bit level to itself with the colour matrix the identity. Into
    /// `command_buffer` goes only a copy of the frame's first texel into `texture`, so that it
    /// completes after the guest's writes of the frame; the frame counts in flight until then,
    /// as the pass's does, and the first encode after a new guest frame counts a host present.
    /// `None`, with nothing encoded, otherwise. The frame's alpha is the guest's, where the gamma
    /// pass writes 1 (a host reading x8r8g8b8 ignores it).
    pub fn encode_current_frame_by_copy(
        &self,
        command_buffer: &P::CommandBuffer,
        texture: &P::Texture,
        region: MtlRegion,
    ) -> RaiseResult<Option<P::Texture>> {
        if !console_cpu_copy_requested() || !self.encodable(command_buffer, texture) {
            return Ok(None);
        }
        let dev = self.mtl_device().map(Handle::id);
        let g = self.display_mutex.lock();
        let Some(frame) = g.current_frame.as_ref() else {
            return Ok(None);
        };
        let whole = region.origin.x == 0
            && region.origin.y == 0
            && region.size.width == texture.width()
            && region.size.height == texture.height();
        // Without a gamma table the pass samples the frame as it is (the colour matrix is the
        // gamma pipeline's alone).
        let exact = !g.pipeline.gamma
            || (self.color_matrix_identity.load(Ordering::SeqCst)
                && self.gamma_identity.load(Ordering::SeqCst));
        if !whole
            || !exact
            || g.protection_requirements != 0
            || frame.device().map(|d| d.id()) != dev
            || frame.pixel_format() != MTL_PIXEL_FORMAT_BGRA8_UNORM
            || texture.pixel_format() != MTL_PIXEL_FORMAT_BGRA8_UNORM
            || frame.width() != texture.width()
            || frame.height() != texture.height()
            || frame.texture_type() != MTL_TEXTURE_TYPE_2D
            || frame.sample_count() != 1
            || frame.storage_mode() == MTL_STORAGE_MODE_PRIVATE
            || frame.storage_mode() == MTL_STORAGE_MODE_MEMORYLESS
        {
            return Ok(None);
        }
        let frame = frame.clone_ref();
        // Still under the display mutex, in a pool (Metal's `+0` encoder), as the pass is.
        let encoded = pool::around(&*self.platform, || {
            let Some(blit) = command_buffer.blit_encoder(OnFault::Die)? else {
                return Ok(false);
            };
            blit.encode(
                OnFault::Die,
                BlitCmd::CopyTextureRegion {
                    source_texture: &frame,
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::new(0, 0, 0),
                    source_size: MtlSize::new(1, 1, 1),
                    destination_texture: texture,
                    destination_slice: 0,
                    destination_level: 0,
                    destination_origin: MtlOrigin::new(0, 0, 0),
                },
            )?;
            blit.encode(OnFault::Die, BlitCmd::EndEncoding)?;
            Ok(true)
        })?;
        if !encoded {
            return Ok(None);
        }
        self.count_frame_in_flight(command_buffer)?;
        self.count_host_present_locked(&g);
        drop(g);
        Ok(Some(frame))
    }

    /// The current frame into `texture` over `region`, with the display mutex held.
    fn internal_encode_locked(
        &self,
        g: &mut DisplayLocked<P>,
        command_buffer: &P::CommandBuffer,
        texture: &P::Texture,
        region: MtlRegion,
    ) -> RaiseResult<()> {
        // The region's depth and z are dropped; znear = zfar = 0.
        #[expect(clippy::as_conversions, reason = "u64 to f64 rounds to nearest")]
        let viewport = MtlViewport {
            origin_x: region.origin.x as f64,
            origin_y: region.origin.y as f64,
            width: region.size.width as f64,
            height: region.size.height as f64,
            znear: 0.0,
            zfar: 0.0,
        };
        self.encode_render_frame_locked(g, None, texture, command_buffer, viewport)
    }

    /// An encode of `frame`, taking the display mutex (tests; the product holds it).
    #[cfg(test)]
    pub(crate) fn encode_render_frame(
        &self,
        frame: &P::Texture,
        dst: &P::Texture,
        command_buffer: &P::CommandBuffer,
        viewport: MtlViewport,
    ) -> RaiseResult<()> {
        let mut g = self.display_mutex.lock();
        self.encode_render_frame_locked(&mut g, Some(frame), dst, command_buffer, viewport)
    }

    /// The pass that draws `frame` (`None`: the current frame) into `dst`, with the display mutex
    /// held, and its completion, which counts the frame out of flight. Even with no frame and no
    /// guest display, one frame is counted in. Only the host's encode, an API entry with no
    /// handler, runs it, so a raise in a send dies there.
    fn encode_render_frame_locked(
        &self,
        g: &mut DisplayLocked<P>,
        frame: Option<&P::Texture>,
        dst: &P::Texture,
        command_buffer: &P::CommandBuffer,
        viewport: MtlViewport,
    ) -> RaiseResult<()> {
        // Still under the display mutex, in a pool (the pass descriptor and the encoder are
        // Metal's `+0`), which a raise leaves pushed.
        pool::around(&*self.platform, || {
            let pass = self.platform.render_pass_descriptor();
            // The whole destination is cleared, not only the region.
            pass.set_color_attachment(
                OnFault::Die,
                0,
                &ColorAttachmentSet {
                    texture: Some(Some(dst)),
                    load_action: Some(MTL_LOAD_ACTION_CLEAR),
                    store_action: Some(MTL_STORE_ACTION_STORE),
                    clear_color: Some(ClearColor {
                        red: 0.0,
                        green: 0.0,
                        blue: 0.0,
                        alpha: 1.0,
                    }),
                    ..ColorAttachmentSet::default()
                },
            )?;
            let frame = match frame {
                Some(f) => Some(f),
                None => g.current_frame.as_ref(),
            };
            let encoder;
            if let Some(frame) = frame {
                if g.protection_requirements != 0 {
                    command_buffer
                        .set_protection_options(OnFault::Die, g.protection_requirements)?;
                }
                encoder = command_buffer.render_encoder(OnFault::Die, &pass)?;
                g.pipeline.format = dst.pixel_format();
                let pipeline = match &self.render_pipeline_cache {
                    Some(cache) => cache.pipeline_for(g.pipeline),
                    None => None,
                };
                // A nil pipeline (an integer render target) is set as nil, and Metal crashes at
                // the draw (testing.md 7.4).
                crate::nil_send!(
                    encoder,
                    encode(
                        OnFault::Die,
                        RenderCmd::SetRenderPipelineState {
                            pipeline_state: pipeline.as_ref()
                        }
                    )
                )?;
                crate::nil_send!(
                    encoder,
                    encode(OnFault::Die, RenderCmd::SetViewport { viewport })
                )?;
                crate::nil_send!(
                    encoder,
                    encode(
                        OnFault::Die,
                        RenderCmd::SetVertexBuffer {
                            buffer: g.pos_buf.as_ref(),
                            offset: 0,
                            index: 0
                        }
                    )
                )?;
                crate::nil_send!(
                    encoder,
                    encode(
                        OnFault::Die,
                        RenderCmd::SetVertexBuffer {
                            buffer: g.tex_buf.as_ref(),
                            offset: 0,
                            index: 1
                        }
                    )
                )?;
                crate::nil_send!(
                    encoder,
                    encode(
                        OnFault::Die,
                        RenderCmd::SetFragmentTexture {
                            texture: g.gamma_table_tex.as_ref(),
                            index: 1
                        }
                    )
                )?;
                crate::nil_send!(
                    encoder,
                    encode(
                        OnFault::Die,
                        RenderCmd::SetFragmentTexture {
                            texture: Some(frame),
                            index: 0
                        }
                    )
                )?;
                crate::nil_send!(
                    encoder,
                    encode(
                        OnFault::Die,
                        RenderCmd::SetFragmentBuffer {
                            buffer: g.frag_shader_property_buf.as_ref(),
                            offset: 0,
                            index: 0
                        }
                    )
                )?;
                crate::nil_send!(
                    encoder,
                    encode(
                        OnFault::Die,
                        RenderCmd::DrawPrimitives {
                            primitive_type: MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP,
                            vertex_start: 0,
                            vertex_count: g.vtx_count
                        }
                    )
                )?;
            } else {
                encoder = command_buffer.render_encoder(OnFault::Die, &pass)?;
            }
            crate::nil_send!(encoder, encode(OnFault::Die, RenderCmd::EndEncoding))?;
            // The encoder and the pass go before the pool pops.
            drop(encoder);
            drop(pass);
            Ok(())
        })?;
        self.count_frame_in_flight(command_buffer)
    }

    /// One frame counted in flight until `command_buffer` completes: its completion counts it out
    /// and wakes the presents waiting for frames in flight.
    fn count_frame_in_flight(&self, command_buffer: &P::CommandBuffer) -> RaiseResult<()> {
        self.in_flight_frame_count.fetch_add(1, Ordering::SeqCst);
        // The block holds the display (a display whose last release has begun: its core), and the
        // command buffer lets it go when it has run.
        let this = self.block_self();
        let block = move |info: CompletedInfo| {
            if info.status == CommandBufferStatus::ERROR {
                match &info.error {
                    Some(error) => log::error!("a frame's command buffer failed: {error}"),
                    None => log::error!("a frame's command buffer failed, with no error"),
                }
            }
            if let Some(this) = this {
                let _g = this.display_mutex.lock();
                let left = this
                    .in_flight_frame_count
                    .fetch_sub(1, Ordering::SeqCst)
                    .wrapping_sub(1);
                if left == 0 {
                    this.in_flight_condition.notify_all();
                }
            }
        };
        command_buffer.add_completed_handler(OnFault::Die, Box::new(block))
    }

    /// The colour matrix into the property buffer, taking the display mutex (tests; the product
    /// holds it).
    #[cfg(test)]
    pub(crate) fn update_color_matrix(
        &self,
        on: OnFault,
        matrix: &[[f32; 4]; 3],
    ) -> RaiseResult<()> {
        let g = self.display_mutex.lock();
        self.update_color_matrix_locked(on, &g, matrix)
    }

    /// The colour matrix into the property buffer, with the display mutex held: column j of the
    /// `simd_float4x3` is `{m[0][j], m[1][j], m[2][j]}` (16 bytes each; the padding lane is
    /// written 0, which the shader never reads). `didModifyRange:`'s raise is as `on` says: the
    /// initializer writes the identity with no handler, and a guest display's present its matrix
    /// under one.
    pub(super) fn update_color_matrix_locked(
        &self,
        on: OnFault,
        g: &DisplayLocked<P>,
        matrix: &dyn ColorMatrix,
    ) -> RaiseResult<()> {
        let contents = crate::nil_send!(g.frag_shader_property_buf, contents());
        // The reads in order, here (a guest display's matrix is still in the guest's page): m[0][0]
        // and m[0][1] together (one 64-bit access: the matrix is 16-byte aligned in the page),
        // m[1][0], m[2][0], m[1][1], m[2][1], then m[0][2] and m[0][3] together, m[1][2], m[2][2],
        // m[1][3], m[2][3], each of those single lanes a 32-bit load.
        let mut columns = [[0f32; 4]; 4];
        for (col, two) in (0u8..).step_by(2).zip(columns.as_chunks_mut::<2>().0) {
            let (a, b) = matrix.pair(0, col);
            for (c, (lane, top)) in (col..).zip(two.iter_mut().zip([a, b])) {
                let mut rows = lane.iter_mut();
                if let Some(v) = rows.next() {
                    *v = top;
                }
                for (row, v) in (1u8..3).zip(rows) {
                    *v = matrix.at(row, c);
                }
            }
        }
        GuestPtr::new(contents).copy_from(
            0,
            &f32_bytes(columns.as_flattened()),
            DeathKind::NullContents,
        );
        // Exactly the identity (-0.0 counts as 0.0; a NaN does not match): the gamma pass then
        // changes no colour the table leaves alone, which the console's CPU copy relies on
        // (`encode_current_frame_by_copy`).
        const IDENTITY_COLUMNS: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        #[expect(
            clippy::float_cmp,
            reason = "exact equality with the identity's entries"
        )]
        let identity = columns
            .as_flattened()
            .iter()
            .zip(IDENTITY_COLUMNS.as_flattened())
            .all(|(a, b)| a == b);
        self.color_matrix_identity.store(identity, Ordering::SeqCst);
        Ok(crate::nil_send!(
            g.frag_shader_property_buf,
            did_modify_range(on, NsRange::new(0, FRAG_SHADER_PROPERTIES_SIZE))
        )?)
    }

    /// The gamma slopes into the property buffer, taking the display mutex (tests; the product
    /// holds it).
    #[cfg(test)]
    pub(crate) fn update_gamma_slopes(
        &self,
        at_zero: [f32; 3],
        at_one: [f32; 3],
    ) -> RaiseResult<()> {
        let g = self.display_mutex.lock();
        self.update_gamma_slopes_locked(&g, at_zero, at_one)
    }

    /// The six slopes at bytes 64..88 of the property buffer, with the display mutex held. Only
    /// the gamma table's update, under a guest display present's handler, writes them.
    pub(super) fn update_gamma_slopes_locked(
        &self,
        g: &DisplayLocked<P>,
        at_zero: [f32; 3],
        at_one: [f32; 3],
    ) -> RaiseResult<()> {
        let contents = crate::nil_send!(g.frag_shader_property_buf, contents());
        let [zr, zg, zb] = at_zero;
        let [or, og, ob] = at_one;
        GuestPtr::new(contents).copy_from(
            SLOPE_AT_ZERO_OFFSET,
            &f32_bytes(&[zr, zg, zb, or, og, ob]),
            DeathKind::NullContents,
        );
        Ok(crate::nil_send!(
            g.frag_shader_property_buf,
            did_modify_range(
                OnFault::Return,
                NsRange::new(0, FRAG_SHADER_PROPERTIES_SIZE)
            )
        )?)
    }
}
