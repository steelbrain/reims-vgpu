//! The rt platform's GPU copies of the VA-API arm's pictures on Windows (`super::open_on_rt`):
//! none yet. A picture there is decoded into a shared Direct3D 12 texture of the VA crate's pool
//! (`Frame::texture`), which rt imports no image of, so [`RtCopies`]' copier declines each
//! picture at the GPU's copy, the module doc's second way, and counts it ([`DECLINED_CODE`]); the
//! picture then lands by the CPU's copy, the third way: read back by `vaGetImage` and copied into
//! the session's staging, which is copied into the output's device cache.
//!
//! **Measured** (2026-11-19, Intel UHD 630, Vulkan driver 101.2145, Mesa 26.2.7's `vaon12`): the
//! Vulkan platform's GPU copy of a 1080p NV12 picture took 1.47 ms at the mean, a 4K NV12
//! picture's 4.18 ms and a 4K P010 picture's 10.36 ms; the CPU's copy took 18.77 ms, 68.35 ms
//! and 79.28 ms, of which the read back took 15.09 ms, 57.00 ms and 60.49 ms.
//!
//! **The draws.** The copies also make the video's draws into a cache's planes over Direct3D 11
//! textures (`graphics::drawn`), at the video's open, so that no picture waits for their devices
//! and pipelines ([`RtCopies::new`]). Where they make none (a device answers no
//! `ExtendedResourceSharing`, or a call fails), the open logs why and the caches keep their
//! planes in images of their own, where the CPU's copy lands each picture.
//!
//! **The draws' check**, a diagnostic: where the draws are made, the open draws a picture of
//! known samples into a pair of targets of its own, imported as a cache's are, and writes one
//! line of what the Direct3D 11 device and the device each read of each plane, and what the
//! driver answers of rt's import (`graphics::drawn_check`, [`check`]). Nothing decides by it.

use std::sync::Arc;

use reims_vgpu_video::core::backend::DeviceWrite;
use reims_vgpu_video::vaapi::{Frame, GuestWrites, PlaneDraw};
use reims_vgpu_vulkan::RtPlatform;
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_vulkan::core::platform::resource::{Library as _, Texture as _};
use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;
use reims_vgpu_vulkan::core::platform::values::{
    ColorAttachmentSet, RenderPipelineDesc, TextureDesc,
};
use reims_vgpu_vulkan::core::platform::{Handle as _, Platform};
use reims_vgpu_vulkan::core::raise::OnFault;

use crate::graphics::drawn_check;
use crate::platform::ApiPlatform;

use super::super::video_host::{BackendError, OutputPlane};
use super::{GpuCopier, GpuCopies, GpuFiller};

/// The code of a picture the rt platform's GPU copy declined on Windows, which the CPU copied
/// (docs/conventions.md, "Each degradation is logged once per boot, then counted").
pub(crate) const DECLINED_CODE: &str = "rt-video-gpu-copy-declined";

/// Why the rt platform's GPU copy declines a picture on Windows.
const WHY: &str = "the rt platform imports no shared Direct3D 12 texture to copy from";

/// The rt platform's GPU copies on Windows: a copier that declines each picture, no encoder's
/// fills, as the arm answers no encoder there, and the draws into a cache's planes over Direct3D
/// 11 textures where they were made (module doc).
pub(super) struct RtCopies {
    /// The draws, made at the video's open.
    draw: Option<Arc<PlaneDraw>>,
}

impl RtCopies {
    /// The copies of the video of `device` on the DXGI adapter of `luid`, with its draws made
    /// there and checked (module doc's "The draws' check"); where none are made, the line says
    /// why (module doc's "The draws").
    pub(super) fn new(device: &super::RtDevice, luid: [u8; 8]) -> RtCopies {
        let draw = match PlaneDraw::new(Some(luid)) {
            Ok(draw) => Some(Arc::new(draw)),
            Err(why) => {
                log::info!(
                    "hardware video: no draws into the caches' planes over Direct3D 11 textures \
                     ({why}): each picture lands by the CPU's copy"
                );
                None
            }
        };
        if let Some(draw) = &draw {
            log::info!(
                target: "reims_vgpu::graphics",
                "hardware video: the draws' check at the video's open (a diagnostic; the draws \
                 stay as they are): {}",
                check(device, draw)
            );
        }
        RtCopies { draw }
    }
}

impl GpuCopies<RtPlatform> for RtCopies {
    fn copier(&self) -> Result<Box<dyn GpuCopier<RtPlatform>>, String> {
        Ok(Box::new(RtCopier))
    }

    fn filler(&self) -> Result<Box<dyn GpuFiller<RtPlatform>>, String> {
        Err(WHY.to_owned())
    }

    fn plane_draws(&self) -> Option<Arc<PlaneDraw>> {
        self.draw.clone()
    }
}

/// One session's copier: each picture declined and counted (module doc).
struct RtCopier;

impl GpuCopier<RtPlatform> for RtCopier {
    fn copy(
        &mut self,
        _frame: &Frame,
        _imports: &GuestWrites,
        _luma: &OutputPlane<RtPlatform>,
        _chroma: &OutputPlane<RtPlatform>,
    ) -> Result<DeviceWrite, BackendError> {
        Ok(declined())
    }
}

/// A picture's GPU copy declined, and counted (module doc).
fn declined() -> DeviceWrite {
    reims_vgpu_vulkan::answer::count_kind(DECLINED_CODE);
    DeviceWrite::Declined(WHY.to_owned())
}

/// The line of the draws' check (module doc's "The draws' check"): 8-bit and then 10-bit, each
/// case's read of each plane by the Direct3D 11 device and by `device`, and the driver's answers
/// of rt's import of the targets.
fn check(device: &super::RtDevice, draw: &PlaneDraw) -> String {
    let reader = Reader::new(device);
    let depths: Vec<String> = [false, true]
        .into_iter()
        .map(|ten_bit| {
            let bits = if ten_bit { "10-bit" } else { "8-bit" };
            check_depth(device, draw, &reader, ten_bit)
                .unwrap_or_else(|why| format!("{bits}: not checked ({why})"))
        })
        .collect();
    depths.join("; ")
}

/// [`check`]'s part of 10-bit samples where `ten_bit`, else 8-bit; `Err` says why it made no
/// reads.
fn check_depth(
    device: &super::RtDevice,
    draw: &PlaneDraw,
    reader: &Result<Reader, String>,
    ten_bit: bool,
) -> Result<String, String> {
    let bits = if ten_bit { "10-bit" } else { "8-bit" };
    let targets = draw.targets(drawn_check::SIZE, ten_bit)?;
    let planes = crate::graphics::decode::drawn_planes(&targets);
    let import = reims_vgpu_vulkan::rt_platform::d3d11_surface::import_answers(device, &planes);
    let made = (device.import_d3d11_surface(OnFault::Return, &planes))
        .map_err(|raise| format!("the import: {raise}"))?
        .ok_or("rt imports no surface of the targets")?;
    let [luma, chroma] = drawn_check::picture(ten_bit);
    let texture = draw.picture(drawn_check::SIZE, ten_bit, [&luma, &chroma])?;
    let mut parts = Vec::new();
    for case in &drawn_check::CASES {
        if case.owe_zeros {
            targets.owe_zeros();
        }
        let read = match draw.draw(&texture, case.origin, &targets, case.extent) {
            Err(why) => format!("not drawn ({why})"),
            Ok(()) => {
                let sampled = (made.memory.external_writes_done()).and_then(|()| match reader {
                    Ok(reader) => reader.read(&made.planes),
                    Err(why) => Err(why.clone()),
                });
                let back = draw.read_back(&targets);
                let side = |plane: usize| {
                    let want = drawn_check::expected(ten_bit, case, plane);
                    let d3d = match &back {
                        Ok(bytes) => {
                            let got = drawn_check::from_bytes(ten_bit, plane, &bytes[plane]);
                            drawn_check::compare(plane, &want, &got)
                        }
                        Err(why) => format!("not read ({why})"),
                    };
                    let rt = match &sampled {
                        Ok(read) => {
                            let got = drawn_check::from_floats(ten_bit, plane, &read[plane]);
                            drawn_check::compare(plane, &want, &got)
                        }
                        Err(why) => format!("not read ({why})"),
                    };
                    let name = if plane == 0 { "luma" } else { "chroma" };
                    format!("{name} Direct3D {d3d}, rt {rt}")
                };
                format!("{}, {}", side(0), side(1))
            }
        };
        parts.push(format!("{bits} {}: {read}", case.name()));
    }
    parts.push(format!("{bits} import: {import}"));
    Ok(parts.join("; "))
}

/// `MTLPixelFormatRGBA32Float`: the reader's target, which holds each sample read exactly.
const READ_FORMAT: u64 = 125;

/// `MTLTextureUsageRenderTarget`.
const READ_USAGE: u64 = 0x4;

/// `MTLPrimitiveTypeTriangle`.
const TRIANGLE: u64 = 3;

/// `MTLLoadActionClear` and `MTLStoreActionStore`.
const LOAD_CLEAR: u64 = 2;
const STORE_STORE: u64 = 1;

/// The device's reads of a plane through its texture, sampled as the guest's shaders sample it:
/// a draw of each texel into a texture of [`READ_FORMAT`] that the host reads back.
struct Reader {
    platform: Arc<RtPlatform>,
    device: super::RtDevice,
    pipeline: <RtPlatform as Platform>::RenderPipeline,
    queue: <RtPlatform as Platform>::CommandQueue,
}

impl Reader {
    /// The reader on `device`; `Err` says why it makes none.
    fn new(device: &super::RtDevice) -> Result<Reader, String> {
        let library = match device.new_shader_library() {
            Ok(Some(library)) => library,
            Ok(None) => return Err("the platform's library is not made".to_owned()),
            Err(error) => return Err(format!("the platform's library: {error}")),
        };
        let function = |name: &str| match library.new_function(OnFault::Return, name) {
            Ok(Some(function)) => Ok(function),
            Ok(None) => Err(format!("the platform's library holds no {name}")),
            Err(raise) => Err(format!("{name}: {raise}")),
        };
        let vertex = function("videoPlaneCheckVertex")?;
        let fragment = function("videoPlaneCheckFragment")?;
        let pipeline = match device.new_render_pipeline_state(&RenderPipelineDesc {
            label: Some("video plane check"),
            vertex: &vertex,
            fragment: &fragment,
            color0_pixel_format: READ_FORMAT,
            resource_index: None,
        }) {
            Ok(Some(pipeline)) => pipeline,
            Ok(None) => return Err("the check's pipeline is refused".to_owned()),
            Err(error) => return Err(format!("the check's pipeline: {error}")),
        };
        let queue = device.new_command_queue().ok_or("no command queue")?;
        Ok(Reader {
            platform: RtPlatform::process(),
            device: device.clone_ref(),
            pipeline,
            queue,
        })
    }

    /// What the device samples of each of `planes`, texel by texel, row after row; `Err` says
    /// why it read none.
    fn read(
        &self,
        planes: &[<RtPlatform as Platform>::Texture],
    ) -> Result<[Vec<[f32; 4]>; 2], String> {
        let [luma, chroma] = planes else {
            return Err(format!("a surface of {} planes", planes.len()));
        };
        Ok([self.read_plane(luma)?, self.read_plane(chroma)?])
    }

    /// What the device samples of `plane` (the type's doc).
    fn read_plane(
        &self,
        plane: &<RtPlatform as Platform>::Texture,
    ) -> Result<Vec<[f32; 4]>, String> {
        let (width, height) = (plane.width(), plane.height());
        let desc = TextureDesc {
            pixel_format: READ_FORMAT,
            width,
            height,
            usage: READ_USAGE,
            resource_options: Some(0),
            ..TextureDesc::default()
        };
        let out =
            (self.platform.own_texture(&self.device, &desc)).ok_or("no texture to read into")?;
        let raised = |what: &str, raise: &dyn std::fmt::Display| format!("{what}: {raise}");
        let cb = (self.queue.command_buffer(OnFault::Return))
            .map_err(|raise| raised("the command buffer", &raise))?
            .ok_or("no command buffer")?;
        let pass = self.platform.render_pass_descriptor();
        pass.set_color_attachment(
            OnFault::Return,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&out)),
                load_action: Some(LOAD_CLEAR),
                store_action: Some(STORE_STORE),
                ..ColorAttachmentSet::default()
            },
        )
        .map_err(|raise| raised("the pass", &raise))?;
        let encoder = (cb.render_encoder(OnFault::Return, &pass))
            .map_err(|raise| raised("the encoder", &raise))?
            .ok_or("no render encoder")?;
        let cmds = [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&self.pipeline),
            },
            RenderCmd::SetFragmentTexture {
                texture: Some(plane),
                index: 0,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: TRIANGLE,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ];
        for cmd in cmds {
            (encoder.encode(OnFault::Return, cmd)).map_err(|raise| raised("the draw", &raise))?;
        }
        (cb.commit(OnFault::Return)).map_err(|raise| raised("the commit", &raise))?;
        (cb.wait_until_completed(OnFault::Return)).map_err(|raise| raised("the wait", &raise))?;
        if cb.status() != CommandBufferStatus::COMPLETED {
            return Err(format!("the read's command buffer ended {:?}", cb.status()));
        }
        let row = usize::try_from(width * 16).map_err(|_| "a plane too wide")?;
        let rows = usize::try_from(height).map_err(|_| "a plane too tall")?;
        let mut bytes = vec![0u8; row * rows];
        self.platform.read_pixels(&out, &mut bytes, row);
        let (texels, _) = bytes.as_chunks::<16>();
        Ok((texels.iter())
            .map(|texel| {
                let (words, _) = texel.as_chunks::<4>();
                std::array::from_fn(|c| words.get(c).map_or(f32::NAN, |w| f32::from_le_bytes(*w)))
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A session has a copier, whose every picture is declined with why and counted by its code,
    /// so that the CPU's copy lands each; and its fills are refused with why (module doc).
    #[test]
    fn each_picture_is_declined_with_why_and_counted_and_the_fills_are_refused() {
        let counted = || {
            (reims_vgpu_vulkan::answer::counted_kinds().into_iter())
                .find(|&(code, _)| code == DECLINED_CODE)
                .map_or(0, |(_, n)| n)
        };
        let copies = RtCopies { draw: None };
        assert!(copies.copier().is_ok());
        assert_eq!(copies.filler().err().as_deref(), Some(WHY));
        assert!(copies.plane_draws().is_none());
        let before = counted();
        for n in 1..=2 {
            assert_eq!(declined(), DeviceWrite::Declined(WHY.to_owned()));
            assert_eq!(counted(), before + n);
        }
    }
}
