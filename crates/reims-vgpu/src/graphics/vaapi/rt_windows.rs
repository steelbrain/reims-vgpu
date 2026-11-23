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
//! driver answers of rt's import (`graphics::drawn_check`, [`check`]). The device reads each
//! plane two ways: sampled by a draw from a view of the plane in its own format, as the guest's
//! textures over a plane are views of it, and copied by a blit. rt makes each plane at no index,
//! to be reached through its views alone, so it refuses a draw's bind of the plane itself
//! (`Refusal::TextureWithoutIndex`), which a slot would read as nil. The draw's target is
//! cleared to -1 and the host's bytes set to all ones first, so that a read that wrote nothing
//! shows. Nothing decides by it.
//!
//! **The reads of real pictures**, a diagnostic beside the check, removed with it: a session's
//! copier reads its first picture drawn and its 30th ([`RtCopier`]'s `drawn`), once a boot for
//! each of the first four sizes and depths of the sessions' pictures and each of the two
//! (`graphics::drawn_check`'s `takes`), so at most eight a boot. A picture is read once its draw's
//! writes are said done, four ways in order: its decode by the CPU (`Frame::read_back`, the CPU's
//! copy's `vaGetImage`), the surface's planes by the device's blit, the targets by the Direct3D 11
//! device that made them (`PlaneDraw::read_back`), and the planes by the blit again. One line says
//! how each read of each plane differs from the next, and what the decoded luma holds. A read not
//! made says why in its place, and the picture lands as it would have: nothing decides by it.
//! Direct3D's acquire and copy may resolve the targets in place, which the second blit shows, so a
//! picture read so may show right to the guest where its neighbours do not.

use std::sync::Arc;

use reims_vgpu_video::core::backend::DeviceWrite;
use reims_vgpu_video::vaapi::{Frame, GuestWrites, PlaneDraw, PlaneRows, Profile};
use reims_vgpu_vulkan::RtPlatform;
use reims_vgpu_vulkan::core::num::ToUsize as _;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, Library as _, Texture as _};
use reims_vgpu_vulkan::core::platform::types::{
    ClearColor, CommandBufferStatus, MtlOrigin, MtlSize,
};
use reims_vgpu_vulkan::core::platform::values::{
    ColorAttachmentSet, RenderPipelineDesc, TextureDesc,
};
use reims_vgpu_vulkan::core::platform::{Handle as _, Platform};
use reims_vgpu_vulkan::core::raise::OnFault;
use reims_vgpu_vulkan::core::sync::Mutex;

use crate::graphics::drawn::DrawnTarget;
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
    /// The device's reads of the planes, for the check and the reads of real pictures (module
    /// doc), made with the draws; `Err` says why none were made.
    reader: Arc<Result<Reader, String>>,
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
        let reader = match &draw {
            Some(_) => Reader::new(device),
            None => Err("no draws".to_owned()),
        };
        if let Some(draw) = &draw {
            log::info!(
                target: "reims_vgpu::graphics",
                "hardware video: the draws' check at the video's open (a diagnostic; the draws \
                 stay as they are): {}",
                check(device, draw, &reader)
            );
        }
        RtCopies {
            draw,
            reader: Arc::new(reader),
        }
    }
}

impl GpuCopies<RtPlatform> for RtCopies {
    fn copier(&self) -> Result<Box<dyn GpuCopier<RtPlatform>>, String> {
        Ok(Box::new(RtCopier {
            draw: self.draw.clone(),
            reader: Arc::clone(&self.reader),
            drawn: 0,
        }))
    }

    fn filler(&self) -> Result<Box<dyn GpuFiller<RtPlatform>>, String> {
        Err(WHY.to_owned())
    }

    fn plane_draws(&self) -> Option<Arc<PlaneDraw>> {
        self.draw.clone()
    }
}

/// One session's copier: each picture declined and counted, and the reads of its real pictures
/// (module doc).
struct RtCopier {
    /// The draws, which read their targets back.
    draw: Option<Arc<PlaneDraw>>,
    /// The device's reads of the planes.
    reader: Arc<Result<Reader, String>>,
    /// The session's pictures drawn so far.
    drawn: u64,
}

/// The boot's reads of real pictures so far, by what the sessions' pictures are
/// (`drawn_check::takes`).
static READ: Mutex<Vec<(drawn_check::ReadKey, [bool; 2])>> = Mutex::new(Vec::new());

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

    /// Reads the picture and writes their line where it is one the reads of real pictures read
    /// (module doc's "The reads of real pictures").
    fn drawn(
        &mut self,
        frame: &Frame,
        profile: Option<Profile>,
        luma: &DrawnTarget<RtPlatform>,
        chroma: &DrawnTarget<RtPlatform>,
    ) {
        self.drawn += 1;
        let Some(which) = (drawn_check::READ_PICTURES.iter()).position(|&n| n == self.drawn) else {
            return;
        };
        let key = ((frame.width(), frame.height()), frame.bit_depth() > 8);
        if !drawn_check::takes(&mut READ.lock(), key, which) {
            return;
        }
        log::info!(
            target: "reims_vgpu::graphics",
            "hardware video: the reads of a real picture (a diagnostic; the picture lands as it \
             would have): {}",
            self.read(frame, profile, luma, chroma)
        );
    }
}

impl RtCopier {
    /// The reads of `frame`'s picture of `profile`, drawn into `luma` and `chroma`'s surface, in
    /// order, in their line (module doc's "The reads of real pictures").
    fn read(
        &self,
        frame: &Frame,
        profile: Option<Profile>,
        luma: &DrawnTarget<RtPlatform>,
        chroma: &DrawnTarget<RtPlatform>,
    ) -> String {
        let sample = if frame.bit_depth() > 8 { 2 } else { 1 };
        let decoded = read_decode(frame, sample);
        self.read_drawn(&decoded, frame, profile, luma, chroma)
    }

    /// [`Self::read`]'s reads after the decode's, `decoded`, in order: the planes by the device's
    /// blit, the targets by the Direct3D 11 device and the planes by the blit again, in their line
    /// with `decoded`.
    fn read_drawn(
        &self,
        decoded: &drawn_check::Read,
        frame: &Frame,
        profile: Option<Profile>,
        luma: &DrawnTarget<RtPlatform>,
        chroma: &DrawnTarget<RtPlatform>,
    ) -> String {
        let ten_bit = frame.bit_depth() > 8;
        let sample: u64 = if ten_bit { 2 } else { 1 };
        let blit = || -> drawn_check::Read {
            let reader = (*self.reader).as_ref().map_err(Clone::clone)?;
            let plane = |target: &DrawnTarget<RtPlatform>, texel: u64| {
                let width = target.texture.width().to_usize();
                Ok::<_, String>((reader.copy_plane(&target.texture, texel)?, width))
            };
            Ok([plane(luma, sample)?, plane(chroma, 2 * sample)?])
        };
        let before = blit();
        let targets = luma.planes.targets();
        let (width, height) = targets.size();
        let d3d = match &self.draw {
            Some(draw) => (draw.read_back(targets))
                .map(|[y, uv]| [(y, width.to_usize()), (uv, width.div_ceil(2).to_usize())]),
            None => Err("no draws".to_owned()),
        };
        let after = blit();
        let (x, y) = frame.crop_origin();
        let (coded_width, coded_height) = frame.coded_size();
        let profile = profile.and_then(Profile::name).unwrap_or("unnamed");
        let picture = format!(
            "picture {} of a session of {}x{} {}-bit {profile} pictures (cropped at ({x}, {y}) \
             of {coded_width}x{coded_height}), into targets of {width}x{height}",
            self.drawn,
            frame.width(),
            frame.height(),
            frame.bit_depth(),
        );
        drawn_check::reads_line(&picture, ten_bit, [decoded, &d3d, &before, &after])
    }
}

/// `frame`'s crop as the CPU reads its decode back (`Frame::read_back`, as the CPU's copy does):
/// the luma's samples and the chroma's pairs, of `sample` bytes each, rows tight.
fn read_decode(frame: &Frame, sample: usize) -> drawn_check::Read {
    let (width, height) = (frame.width().to_usize(), frame.height().to_usize());
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let (luma_row, chroma_row) = (width * sample, chroma_width * 2 * sample);
    let mut luma = vec![0u8; luma_row * height];
    let mut chroma = vec![0u8; chroma_row * chroma_height];
    let y = PlaneRows {
        bytes: &mut luma,
        bytes_per_row: luma_row,
        rows: height,
        row_bytes: luma_row,
    };
    let uv = PlaneRows {
        bytes: &mut chroma,
        bytes_per_row: chroma_row,
        rows: chroma_height,
        row_bytes: chroma_row,
    };
    let back = frame.read_back().map_err(|error| error.to_string())?;
    back.copy_into(y, uv).map_err(|error| error.to_string())?;
    Ok([(luma, width), (chroma, chroma_width)])
}

/// A picture's GPU copy declined, and counted (module doc).
fn declined() -> DeviceWrite {
    reims_vgpu_vulkan::answer::count_kind(DECLINED_CODE);
    DeviceWrite::Declined(WHY.to_owned())
}

/// The line of the draws' check (module doc's "The draws' check"): 8-bit and then 10-bit, each
/// case's read of each plane by the Direct3D 11 device and by `device`, and the driver's answers
/// of rt's import of the targets.
fn check(device: &super::RtDevice, draw: &PlaneDraw, reader: &Result<Reader, String>) -> String {
    let depths: Vec<String> = [false, true]
        .into_iter()
        .map(|ten_bit| {
            let bits = if ten_bit { "10-bit" } else { "8-bit" };
            check_depth(device, draw, reader, ten_bit)
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
                let done = made.memory.external_writes_done();
                let sampled = (done.clone()).and_then(|()| match reader {
                    Ok(reader) => reader.read(&made.planes),
                    Err(why) => Err(why.clone()),
                });
                let copied = done.and_then(|()| match reader {
                    Ok(reader) => reader.copy(&made.planes, ten_bit),
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
                    let blit = match &copied {
                        Ok(bytes) => {
                            let got = drawn_check::from_bytes(ten_bit, plane, &bytes[plane]);
                            drawn_check::compare(plane, &want, &got)
                        }
                        Err(why) => format!("not copied ({why})"),
                    };
                    let name = if plane == 0 { "luma" } else { "chroma" };
                    format!("{name} Direct3D {d3d}, rt sampled {rt}, rt copied {blit}")
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

/// The reader's clear colour, which no sample reads as.
const UNREAD: ClearColor = ClearColor {
    red: -1.0,
    green: -1.0,
    blue: -1.0,
    alpha: -1.0,
};

/// `MTLLoadActionClear` and `MTLStoreActionStore`.
const LOAD_CLEAR: u64 = 2;
const STORE_STORE: u64 = 1;

/// The device's reads of a plane: sampled through a view of it as the guest's shaders sample
/// one, a draw of each texel into a texture of [`READ_FORMAT`] cleared to [`UNREAD`] that the
/// host reads back; and copied by a blit into a buffer (module doc's "The draws' check").
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

    /// What the device samples of each of `planes`, texel by texel, row after row, through a
    /// view of each in its own format (module doc's "The draws' check"); `Err` says why it read
    /// none.
    fn read(
        &self,
        planes: &[<RtPlatform as Platform>::Texture],
    ) -> Result<[Vec<[f32; 4]>; 2], String> {
        let [luma, chroma] = planes else {
            return Err(format!("a surface of {} planes", planes.len()));
        };
        let read = |plane: &<RtPlatform as Platform>::Texture| match plane
            .new_texture_view(plane.pixel_format())
        {
            Ok(Some(view)) => self.read_plane(&view),
            Ok(None) => Err("no view of the plane".to_owned()),
            Err(raise) => Err(format!("the plane's view: {raise}")),
        };
        Ok([read(luma)?, read(chroma)?])
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
                clear_color: Some(UNREAD),
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
        let mut bytes = vec![0xffu8; row * rows];
        self.platform.read_pixels(&out, &mut bytes, row);
        let (texels, _) = bytes.as_chunks::<16>();
        Ok((texels.iter())
            .map(|texel| {
                let (words, _) = texel.as_chunks::<4>();
                std::array::from_fn(|c| words.get(c).map_or(f32::NAN, |w| f32::from_le_bytes(*w)))
            })
            .collect())
    }

    /// What a blit copies of each of `planes` into a buffer, rows tight, of 16-bit samples where
    /// `ten_bit`; `Err` says why it copied none.
    fn copy(
        &self,
        planes: &[<RtPlatform as Platform>::Texture],
        ten_bit: bool,
    ) -> Result<[Vec<u8>; 2], String> {
        let [luma, chroma] = planes else {
            return Err(format!("a surface of {} planes", planes.len()));
        };
        let sample = if ten_bit { 2 } else { 1 };
        Ok([
            self.copy_plane(luma, sample)?,
            self.copy_plane(chroma, 2 * sample)?,
        ])
    }

    /// What a blit copies of `plane`, of `texel` bytes a texel ([`Self::copy`]); the buffer is
    /// set to all ones first.
    fn copy_plane(
        &self,
        plane: &<RtPlatform as Platform>::Texture,
        texel: u64,
    ) -> Result<Vec<u8>, String> {
        let (width, height) = (plane.width(), plane.height());
        let row = width * texel;
        let length = row * height;
        let raised = |what: &str, raise: &dyn std::fmt::Display| format!("{what}: {raise}");
        let buffer = (self.device.new_buffer_of_length(OnFault::Return, length, 0))
            .map_err(|raise| raised("the buffer", &raise))?
            .ok_or("no buffer to copy into")?;
        let size = usize::try_from(length).map_err(|_| "a plane too large")?;
        let contents: *mut u8 = buffer.contents().as_ptr();
        // SAFETY: the Shared buffer's contents are mapped, `length` bytes, and no work of the
        // device's uses it yet.
        unsafe { contents.write_bytes(0xff, size) };
        let cb = (self.queue.command_buffer(OnFault::Return))
            .map_err(|raise| raised("the command buffer", &raise))?
            .ok_or("no command buffer")?;
        let encoder = (cb.blit_encoder(OnFault::Return))
            .map_err(|raise| raised("the encoder", &raise))?
            .ok_or("no blit encoder")?;
        let cmds = [
            BlitCmd::CopyTextureToBuffer {
                source_texture: plane,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin { x: 0, y: 0, z: 0 },
                source_size: MtlSize {
                    width,
                    height,
                    depth: 1,
                },
                destination_buffer: &buffer,
                destination_offset: 0,
                destination_bytes_per_row: row,
                destination_bytes_per_image: length,
            },
            BlitCmd::EndEncoding,
        ];
        for cmd in cmds {
            (encoder.encode(OnFault::Return, cmd)).map_err(|raise| raised("the copy", &raise))?;
        }
        (cb.commit(OnFault::Return)).map_err(|raise| raised("the commit", &raise))?;
        (cb.wait_until_completed(OnFault::Return)).map_err(|raise| raised("the wait", &raise))?;
        if cb.status() != CommandBufferStatus::COMPLETED {
            return Err(format!("the copy's command buffer ended {:?}", cb.status()));
        }
        // SAFETY: as above; the copy's work is done.
        Ok(unsafe { std::slice::from_raw_parts(contents, size) }.to_vec())
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
        let copies = RtCopies {
            draw: None,
            reader: Arc::new(Err("no draws".to_owned())),
        };
        assert!(copies.copier().is_ok());
        assert_eq!(copies.filler().err().as_deref(), Some(WHY));
        assert!(copies.plane_draws().is_none());
        let before = counted();
        for n in 1..=2 {
            assert_eq!(declined(), DeviceWrite::Declined(WHY.to_owned()));
            assert_eq!(counted(), before + n);
        }
    }

    /// The reads of a real picture (module doc) show a texel that differs between two of them:
    /// the first picture of the H.264 stream `RVG_VA_H264` names, drawn into targets of its crop
    /// imported as a cache's are, read by the CPU (`read_decode`), and then one luma texel planted
    /// in the targets, by a draw of its block of the picture's own samples with that one changed,
    /// before the other reads (`RtCopier::read_drawn`). The line counts that texel, its box and
    /// both its values in the decode against Direct3D, and no texel in the other pairs or in the
    /// chroma. Run holding the GPU, on vaon12 on the rt device's adapter.
    #[test]
    #[ignore = "needs vaon12 on the rt device's GPU and RVG_VA_H264: run on the Windows host"]
    fn a_texel_planted_after_the_decodes_read_differs_between_the_decode_and_direct3d_alone() {
        use reims_vgpu_video::codec::h264::nal;
        use reims_vgpu_video::core::device_memory::MemoryLife;
        use reims_vgpu_video::vaapi::{Display, H264Decoder};

        use crate::graphics::drawn::DrawnPlanes;

        /// The plant's block, from the targets' first texel, in luma samples each way.
        const BLOCK: usize = 16;
        /// The planted luma texel's column and row.
        const AT: (usize, usize) = (9, 6);

        let platform = Arc::new(RtPlatform::new());
        crate::entry::run(|_| {
            let device = platform.system_default_device().expect("an rt device");
            let luid = device.host_gpu().luid.expect("the device's DXGI adapter");
            let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
            let display = Display::open_adapter(Some(luid), runtime).expect("a VA display");
            let path = std::env::var_os("RVG_VA_H264").expect("RVG_VA_H264: an Annex B stream");
            let file = std::fs::read(&path).expect("the stream");
            let mut decoder = H264Decoder::new_annex_b(&display);
            let mut frames = Vec::new();
            for (id, unit) in (0u64..).zip(nal::access_units(&file)) {
                frames.extend(decoder.decode(unit, id).expect("a decode"));
                if !frames.is_empty() {
                    break;
                }
            }
            if frames.is_empty() {
                frames.extend(decoder.flush().expect("the flush"));
            }
            let frame = frames.first().expect("a picture");
            let ten_bit = frame.bit_depth() > 8;
            let sample = if ten_bit { 2 } else { 1 };
            let (width, height) = (frame.width().to_usize(), frame.height().to_usize());
            assert!(width >= BLOCK && height >= BLOCK, "a picture of {width}x{height}");

            let draw = Arc::new(PlaneDraw::new(Some(luid)).expect("the plane draws"));
            let size = (frame.width(), frame.height());
            let planes = Arc::new(DrawnPlanes::new(&draw, size, ten_bit).expect("the targets"));
            let descs = crate::graphics::decode::drawn_planes(planes.targets());
            let made = (device.import_d3d11_surface(OnFault::Return, &descs))
                .expect("no raise")
                .expect("a surface over the targets");
            let life = MemoryLife::new();
            let target = |n: usize| DrawnTarget::<RtPlatform> {
                memory: Arc::clone(&made.memory),
                planes: Arc::clone(&planes),
                identity: life.identity(),
                texture: made.planes[n].clone_ref(),
            };
            let (luma, chroma) = (target(0), target(1));
            (planes.draw(frame, &mut || {})).expect("the picture's draw");

            let decoded = read_decode(frame, sample);
            let [(y_bytes, _), (uv_bytes, _)] = decoded.as_ref().expect("the decode's read");
            let rows = |bytes: &[u8], row: usize, take: usize, n: usize| -> Vec<u8> {
                (0..n)
                    .flat_map(|y| bytes[y * row..y * row + take].iter().copied())
                    .collect()
            };
            let block_luma = rows(y_bytes, width * sample, BLOCK * sample, BLOCK);
            let block_chroma = rows(
                uv_bytes,
                width.div_ceil(2) * 2 * sample,
                BLOCK * sample,
                BLOCK / 2,
            );
            let mut planted = block_luma.clone();
            let at = (AT.1 * BLOCK + AT.0) * sample;
            // The sample's top bit, which a P010 sample holds too.
            planted[at + sample - 1] ^= 0x80;
            let texel = |bytes: &[u8]| {
                let mut word = [0; 4];
                word[..sample].copy_from_slice(&bytes[at..at + sample]);
                u32::from_le_bytes(word)
            };
            let (was, now) = (texel(&block_luma), texel(&planted));
            let block = u32::try_from(BLOCK).expect("a block");
            let picture = (draw.picture((block, block), ten_bit, [&planted, &block_chroma]))
                .expect("the plant's picture");
            (draw.draw(&picture, (0, 0), planes.targets(), (block, block)))
                .expect("the plant's draw");
            (made.memory.external_writes_done()).expect("the writes said done");

            let copier = RtCopier {
                draw: Some(Arc::clone(&draw)),
                reader: Arc::new(Reader::new(&device)),
                drawn: 1,
            };
            let line = copier.read_drawn(&decoded, frame, decoder.profile(), &luma, &chroma);
            eprintln!("rt_windows: the reads' line: {line}");
            let n = width * height;
            let c = width.div_ceil(2) * height.div_ceil(2);
            let (x, y) = AT;
            let want = format!(
                "; luma: decode against Direct3D: 1 of {n} differ, in ({x}, {y})-({x}, {y}), \
                 first ({x}, {y}) {was:#x}/{now:#x}, Direct3D against rt: 0 of {n} differ, rt \
                 against rt again: 0 of {n} differ; chroma: decode against Direct3D: 0 of {c} \
                 differ, Direct3D against rt: 0 of {c} differ, rt against rt again: 0 of {c} \
                 differ; the decoded luma: "
            );
            assert!(line.contains(&want), "the line: {line}\nwanted in it: {want}");
        });
    }
}
