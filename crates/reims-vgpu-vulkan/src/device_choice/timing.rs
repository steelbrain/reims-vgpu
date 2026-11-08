//! The workload the measured pick times on each GPU it compares ([`time_on`]), on a device made
//! for it alone and torn down after: the library's display composite at a fixed size, and one
//! pass over textures in host memory, where the guest's RAM lives.
//!
//! **A frame** is one command buffer of [`PASSES`] render passes over a [`WIDTH`] x [`HEIGHT`]
//! BGRA8 target, each loading what the one before stored (the first clears it) and drawing
//! [`DRAWS`] quads of a quarter of the target, through the display's own present pipeline
//! (`displayPresentVertex` and `displayPresentFragment` of `shaders.metal`, the metallib the
//! display loads), each sampling a frame texture of the same size: about half the cut of
//! WindowServer's composite the reims-graphics device's compositor bench takes (reims-vgpu's
//! `composite_cost`, from a lane's count of 63 render encoders and about 170 draws a frame),
//! without its reads of the target. Half: with 60 passes the measurement took 293 ms in all on
//! um790, and with 30, 197 ms (`device_choice`'s doc). Then one pass loads a whole texture in
//! host memory and draws over it, sampling another there, so the GPU reads two textures' worth of
//! host memory each frame and writes one, as it does the guest's surfaces: an integrated GPU in
//! the memory it shares with the host, a discrete one across its bus. The host memory is a mapping
//! of a file in memory, as the guest's RAM is (the library's `RegionMemory`); where the host has
//! none (macOS), the frame has no such pass.
//!
//! **The time** is the wall time from the first of [`TIMED_FRAMES`] frames' commit to the last
//! one's completion, after one frame not counted, which makes the pipeline and the imports. The
//! device's making, the pipeline's and the teardown are logged, not compared.
//!
//! **Unheard.** What the runtime announces on the timing thread is heard by no one
//! (`crate::answer::unheard`): neither counted nor relayed, nor dying in strict mode. The devices
//! are made with the process's options (`crate::platform::device_options`), so they share its
//! instance, and are dropped before the process's own device is made, which then announces what
//! it meets as before.

use std::ptr::NonNull;
use std::time::Duration;

use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime as rt;

use super::{Seen, TARGET};
use crate::core::platform::os::Clock as _;

/// The target's width and height, in texels.
const WIDTH: u64 = 1920;
const HEIGHT: u64 = 1200;

/// The render passes of a frame's composite, and the quads each draws.
const PASSES: usize = 30;
const DRAWS: usize = 3;

/// The frames timed, after one that is not.
const TIMED_FRAMES: usize = 2;

/// The texture coordinates of a quad's four vertices, a strip: `float2` each.
const TEX_COORDS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

/// The time the workload took on `device` (the module doc), logged with what its device's making
/// and teardown took; `None` where the runtime refused any of it, which is logged.
pub(super) fn time_on(at: usize, device: &Seen) -> Option<Duration> {
    let id = device.registry_id()?;
    let clock = crate::os::OsClock::new();
    let start = clock.now_mono();
    let timed = crate::answer::unheard(|| run(id, &clock));
    let all = clock.now_mono().saturating_duration_since(start);
    match &timed {
        Ok(time) => log::info!(
            target: TARGET,
            "Vulkan device {at}, {}: the choice's workload took {:.2} ms, {:.1} ms with its \
             device's making and teardown",
            device.name,
            millis(*time),
            millis(all)
        ),
        Err(e) => log::warn!(
            target: TARGET,
            "Vulkan device {at}, {}: the runtime refused the choice's workload ({e})",
            device.name
        ),
    }
    timed.ok()
}

/// `time` in milliseconds, for a log line.
pub(super) fn millis(time: Duration) -> f64 {
    time.as_secs_f64() * 1e3
}

/// The workload on a device of registry ID `id`, made for it and dropped at the end: the timed
/// frames' wall time on `clock`.
fn run(id: u64, clock: &crate::os::OsClock) -> Result<Duration, rt::Error> {
    let device = rt::Device::with_registry_id(id, crate::platform::device_options())?;
    let library = device.new_library_with_data(crate::gpu::shaders::METALLIB)?;
    let vertex = library.new_function_with_name("displayPresentVertex")?;
    let fragment = library.new_function_with_name("displayPresentFragment")?;
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    let state = device.new_render_pipeline_state(&descriptor, &rt::PipelineOptions::default())?;
    let private = |usage: TextureUsage| {
        let mut texture =
            rt::TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
        texture.usage = usage;
        texture.resource_options = rt::ResourceOptions::private();
        device.new_texture(&texture)
    };
    let work = Work {
        frame: private(TextureUsage::SHADER_READ)?,
        target: private(TextureUsage::RENDER_TARGET)?,
        host: host_textures(&device)?,
        state,
    };
    let queue = device.new_command_queue()?;
    let warm = work.encode(&queue)?;
    warm.commit()?;
    warm.wait_until_completed()?;
    let frames = (0..TIMED_FRAMES)
        .map(|_| work.encode(&queue))
        .collect::<Result<Vec<_>, _>>()?;
    let start = clock.now_mono();
    for frame in &frames {
        frame.commit()?;
    }
    for frame in &frames {
        frame.wait_until_completed()?;
    }
    Ok(clock.now_mono().saturating_duration_since(start))
}

/// What a frame draws with.
struct Work {
    /// The texture the composite samples.
    frame: rt::Texture,
    /// The composite's target.
    target: rt::Texture,
    /// The textures in host memory, the one sampled and the one written, with the buffer they
    /// are made over; none where the host has no such memory.
    host: Option<(rt::Texture, rt::Texture, rt::Buffer)>,
    /// The display's present pipeline, to a BGRA8 target.
    state: rt::RenderPipelineState,
}

impl Work {
    /// One frame's command buffer, encoded and not committed (the module doc).
    fn encode(&self, queue: &rt::CommandQueue) -> Result<rt::CommandBuffer, rt::Error> {
        let buffer = queue.command_buffer()?;
        for pass in 0..PASSES {
            let load = match pass {
                0 => rt::LoadAction::Clear,
                _ => rt::LoadAction::Load,
            };
            let quads = (0..DRAWS).map(|draw| quarter(pass * DRAWS + draw));
            self.pass(&buffer, &self.target, load, &self.frame, quads)?;
        }
        if let Some((sampled, written, _)) = &self.host {
            let whole = [[-1.0, 1.0], [1.0, -1.0]];
            self.pass(&buffer, written, rt::LoadAction::Load, sampled, [whole])?;
        }
        Ok(buffer)
    }

    /// A render pass of `buffer` over `target`, loaded by `load` and stored, drawing each quad of
    /// `quads` (its top-left and bottom-right corners in clip space) sampling `sampled`.
    fn pass(
        &self,
        buffer: &rt::CommandBuffer,
        target: &rt::Texture,
        load: rt::LoadAction,
        sampled: &rt::Texture,
        quads: impl IntoIterator<Item = [[f32; 2]; 2]>,
    ) -> Result<(), rt::Error> {
        let mut descriptor = rt::RenderPassDescriptor::new();
        let color = &mut descriptor.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = load;
        color.store_action = rt::StoreAction::Store;
        let mut encoder = buffer.render_command_encoder(&descriptor)?;
        encoder.set_render_pipeline_state(&self.state)?;
        encoder.set_fragment_texture(Some(sampled), 0)?;
        encoder.set_vertex_bytes(&floats(TEX_COORDS.as_flattened()), 1)?;
        for [[left, top], [right, bottom]] in quads {
            let strip = [
                [left, top, 0.0, 1.0],
                [right, top, 0.0, 1.0],
                [left, bottom, 0.0, 1.0],
                [right, bottom, 0.0, 1.0],
            ];
            encoder.set_vertex_bytes(&floats(strip.as_flattened()), 0)?;
            encoder.draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)?;
        }
        encoder.end_encoding()
    }
}

/// Quad `n`'s corners: a quarter of the target, the quarters in turn.
fn quarter(n: usize) -> [[f32; 2]; 2] {
    let left = if n % 2 == 0 { -1.0 } else { 0.0 };
    let top = if n % 4 < 2 { 1.0 } else { 0.0 };
    [[left, top], [left + 1.0, top - 1.0]]
}

/// `values`' bytes, little-endian, as a shader reads them.
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Two textures of the target's size in host memory, the first to sample and the second to
/// write, over one no-copy buffer of a mapping of a file in memory, with the buffer; none where
/// the host gives no such mapping (macOS, the module doc).
fn host_textures(
    device: &rt::Device,
) -> Result<Option<(rt::Texture, rt::Texture, rt::Buffer)>, rt::Error> {
    let plane = WIDTH * HEIGHT * 4;
    let Some(memory) = crate::os::RegionMemory::new(2 * plane) else {
        return Ok(None);
    };
    let Some(base) = NonNull::new(memory.base()) else {
        return Ok(None);
    };
    // SAFETY: the mapping stays while the runtime holds its import: the keep-alive owns it, and
    // the runtime drops the keep-alive only once it freed the import.
    let host = unsafe { crate::os::host_memory(base) };
    let contents = rt::BufferContents::NoCopy {
        memory: host,
        deallocator: None,
        keep_alive: Some(Box::new(memory)),
    };
    let descriptor = rt::BufferDescriptor::new(2 * plane, rt::ResourceOptions::shared(), contents);
    let buffer = device.new_buffer_with_descriptor(descriptor)?;
    let mut texture =
        rt::TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    texture.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    texture.resource_options = rt::ResourceOptions::shared();
    let sampled = buffer.new_texture(&texture, 0, WIDTH * 4)?;
    let written = buffer.new_texture(&texture, plane, WIDTH * 4)?;
    Ok(Some((sampled, written, buffer)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quads go round the target's four quarters, each a quarter of it
    #[test]
    fn the_quads_cover_each_quarter_in_turn() {
        let quads: Vec<_> = (0..4).map(quarter).collect();
        assert_eq!(
            quads,
            [
                [[-1.0, 1.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[-1.0, 0.0], [0.0, -1.0]],
                [[0.0, 0.0], [1.0, -1.0]],
            ]
        );
        assert_eq!(quarter(4), quarter(0));
    }
}
