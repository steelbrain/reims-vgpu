//! The XR10 layout ([`DualPlaneCompute`], object type 12 in formats 550 and 551): a single-level,
//! non-array, uncompressed, single-sample texture, paged by the XR10 compute kernels through two
//! plane views over the transfer buffer.

use std::sync::Arc;

use crate::nil_send;
use crate::platform::Platform;
use crate::platform::compute::{ComputeCmd, ComputeEncoder};
use crate::platform::descriptor::TextureDescriptor;
use crate::platform::resource::{Buffer, GpuResource as _, Texture};
use crate::platform::types::MtlSize;
use crate::platform::values::TextureDesc;
use crate::raise::{Kept, keep_forever};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::resource::{Resource, ResourceState, TaskRef, features_of, platform_for};
use crate::resources::texture::dimension::{
    TextureDimensionRecord, TexturePagingInfo, validate_texture_dimension,
};
use crate::resources::texture::pixel_format::{PIXEL_FORMAT_550, PIXEL_FORMAT_551};
use crate::resources::texture::{
    DIMENSION_SIZE, DeathKind, MIP_DIMENSION_SIZE, PagedTexture, RaiseKind, TextureLayout,
    into_resource, mip_at, raise, record_at,
};
use crate::rt::Hold;
use crate::sync::Mutex;

/// `MTLPixelFormatBGR10_XR`: plane 0's view format for 550.
const PIXEL_FORMAT_BGR10_XR: u64 = 554;
/// `MTLPixelFormatBGR10_XR_sRGB`: plane 0's view format for 551.
const PIXEL_FORMAT_BGR10_XR_SRGB: u64 = 555;
/// Plane 1's view format (8-bit; not in the public headers).
const PIXEL_FORMAT_PLANE1: u64 = 84;
/// `MTLTextureType2D`.
const TEXTURE_TYPE_2D: u64 = 2;
/// The threads per threadgroup of both dispatches.
const THREADGROUP: MtlSize = MtlSize::new(32, 32, 1);

/// Two planes, paged by the XR10 kernels: plane views over the transfer buffer that the kernels
/// read (up) and write (down).
pub struct DualPlaneCompute<P: Platform> {
    /// The two planes' validated dimensions, one mip each.
    pub(crate) planes: [TextureDimensionRecord; 2],
    /// The plane views over the transfer buffer, made when the pages are bound. Declared first
    /// of the handles: the last release drops them, 0 then 1, before the kernels.
    pub(crate) plane_textures: Mutex<[Option<P::Texture>; 2]>,
    /// The views' pixel formats.
    pub(crate) plane_texture_formats: [u64; 2],
    /// The task's kernel that pages up (guest to host).
    pub(crate) blit_up_state: Option<P::ComputePipeline>,
    /// The task's kernel that pages down.
    pub(crate) blit_down_state: Option<P::ComputePipeline>,
}

/// The view formats and kernels of an XR10 format.
struct Kernels<P: Platform> {
    formats: [u64; 2],
    up: Option<P::ComputePipeline>,
    down: Option<P::ComputePipeline>,
}

impl<P: Platform> DualPlaneCompute<P> {
    /// An XR10 texture of object type 12. `None` when validation refuses the record, or with no
    /// platform scope. It starts guest-valid, so a synchronize after a host-only fill writes
    /// nothing back. `texture` is the caller's reference (see
    /// [`create_texture_resource`](super::create_texture_resource)).
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub(crate) fn new(
        task: TaskRef<P>,
        paging_info: &TexturePagingInfo,
        dimension: &TextureDimensionRecord,
        dimension_length: u64,
        texture: P::Texture,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let Some(platform) = platform_for(&task) else {
            drop(texture);
            return Ok(None);
        };
        let mut paged = PagedTexture::new(&platform, task, &texture);
        if dimension.header.mip_levels() != 1 {
            drop(paged);
            // The caller's texture is kept for good: its resource index and GPU VA stay taken,
            // and later placements are numbered after them.
            keep_forever(texture, Kept::MetalNumbering);
            return raise(RaiseKind::MippedMultiPlane);
        }
        if dimension_length < 2_u64.wrapping_mul(DIMENSION_SIZE) {
            // The half-made texture is released, then the caller's texture. Unreachable: the
            // manager sizes the copy.
            drop(paged);
            drop(texture);
            return raise(RaiseKind::DimensionLength);
        }
        // Plane 1 after one mip.
        let plane1 = record_at(
            dimension,
            DIMENSION_SIZE.wrapping_add(MIP_DIMENSION_SIZE),
            1,
        );
        let mut dest = [
            TextureDimensionRecord::default(),
            TextureDimensionRecord::default(),
        ];
        paged.page = paging_info.page;
        paged.length = paging_info.length;
        let features = features_of(&paged.task);
        let [d0, d1] = &mut dest;
        let valid = validate_texture_dimension::<P, 2>(
            [dimension, &plane1],
            [d0, d1],
            1,
            false,
            paging_info.length,
            &texture,
            &mut paged.is_pvrtc,
            &features,
        );
        if !valid {
            drop(paged);
            drop(texture);
            return Ok(None);
        }
        let [p0, _] = &dest;
        let h = p0.header;
        // The sample count is asked only when the rest pass.
        if h.is_cube()
            || h.is_compressed()
            || h.array_length > 1
            || paged.texture.sample_count() > 1
        {
            drop(paged);
            // The caller's texture is kept for good: its resource index and GPU VA stay taken,
            // and later placements are numbered after them.
            keep_forever(texture, Kept::MetalNumbering);
            return raise(RaiseKind::ComputeShape);
        }
        let Some(kernels) = Self::kernels(&paged) else {
            drop(paged);
            // The caller's texture is released. Unreachable: the factory picks this layout for
            // 550 and 551 only.
            drop(texture);
            return raise(RaiseKind::ComputeFormat);
        };
        drop(texture);
        Ok(Some(into_resource(
            paged,
            TextureLayout::DualPlaneCompute(DualPlaneCompute {
                planes: dest,
                plane_textures: Mutex::new([None, None]),
                plane_texture_formats: kernels.formats,
                blit_up_state: kernels.up,
                blit_down_state: kernels.down,
            }),
        )))
    }

    /// The view formats for the texture's format, then the task's two XR10 kernels (up, then
    /// down); `None` for a format that is not XR10.
    fn kernels(paged: &PagedTexture<P>) -> Option<Kernels<P>> {
        let color = match paged.texture.pixel_format() {
            PIXEL_FORMAT_550 => PIXEL_FORMAT_BGR10_XR,
            PIXEL_FORMAT_551 => PIXEL_FORMAT_BGR10_XR_SRGB,
            _ => return None,
        };
        let Some(task) = paged.task.upgrade() else {
            crate::fatal!(DeathKind::GoneTaskKernels)
        };
        Some(Kernels {
            formats: [color, PIXEL_FORMAT_PLANE1],
            up: task.blit_in_rgb_2p_xr10_a8_pipeline(),
            down: task.blit_out_rgb_2p_xr10_a8_pipeline(),
        })
    }

    /// Binds the guest pages unless bound, then makes a view per plane over them from one
    /// descriptor. A refused view releases the transfer buffer (the next call binds again) and
    /// keeps the views made so far.
    pub(crate) fn ensure_physical(
        &self,
        hold: Hold<'_>,
        paged: &PagedTexture<P>,
    ) -> RaiseResult<()> {
        if paged.transfer_buffer.lock().is_some() {
            return Ok(());
        }
        let Some(task) = paged.task.upgrade() else {
            crate::fatal!(DeathKind::GoneTaskPages)
        };
        // Trace `tobj`: the guest object whose pages the next `tbuf` imports (5, a compute
        // texture), its page and its length.
        crate::trace::event("tobj", &[5, u64::from(paged.page), paged.length]);
        // The task raises rather than answer nothing.
        let buffer = task.new_buffer_for_virtual_page(hold, paged.page, paged.length)?;
        drop(task);
        let mut transfer = paged.transfer_buffer.lock();
        // A buffer a racing caller stored between the check above and this store is let go
        // here, but stays live for good: the racer's views over it, which the loop below
        // overwrites without a release, hold it.
        let transfer_buffer = &*transfer.insert(buffer);
        // One descriptor for both planes: made from plane 0's value, then plane 1's size,
        // type and pixel format set on it.
        let mut descriptor: Option<P::TextureDescriptor> = None;
        let mut views = self.plane_textures.lock();
        for ((dimension, format), slot) in self
            .planes
            .iter()
            .zip(self.plane_texture_formats)
            .zip(views.iter_mut())
        {
            let mip0 = mip_at(dimension, 0);
            let (width, height, depth) = (
                u64::from(mip0.width),
                u64::from(mip0.height),
                u64::from(mip0.depth),
            );
            let td = match &descriptor {
                Some(d) => {
                    d.set_width(width);
                    d.set_height(height);
                    d.set_depth(depth);
                    d.set_texture_type(TEXTURE_TYPE_2D);
                    d.set_pixel_format(format);
                    d
                }
                None => descriptor.insert(paged.platform.texture_descriptor(&TextureDesc {
                    texture_type: TEXTURE_TYPE_2D,
                    pixel_format: format,
                    width,
                    height,
                    depth,
                    ..TextureDesc::default()
                })),
            };
            // The view takes the buffer's resource options, as Metal's texture over a buffer
            // must (`display/efi.rs`): the descriptor's default storage is Managed.
            td.set_resource_options(transfer_buffer.resource_options());
            // Metal validates the view here: a format-550 texture 63 texels wide (plane 0's
            // rows not 256-aligned) dies in MTLReportFailure
            // (iso-b:textures_dual_plane_xr10_bad_row).
            let view = transfer_buffer.new_texture(
                OnFault::Return,
                td,
                mip0.offset.wrapping_add(dimension.header.base_offset),
                mip0.bytes_per_row,
            )?;
            // A view made again overwrites the previous one without releasing it: the previous
            // view is kept for good, with its resource index.
            if let Some(previous) = std::mem::replace(slot, view) {
                keep_forever(previous, Kept::MetalNumbering);
            }
            if slot.is_none() {
                drop(transfer.take());
                return raise(RaiseKind::PlaneViewRefused);
            }
        }
        Ok(())
    }

    /// Drops the plane views (0, then 1), then discards as every paged texture does.
    pub(crate) fn discard(
        &self,
        paged: &PagedTexture<P>,
        state: &ResourceState,
    ) -> RaiseResult<()> {
        {
            let mut views = self.plane_textures.lock();
            let [v0, v1] = &mut *views;
            drop(v0.take());
            drop(v1.take());
        }
        paged.discard(state)
    }

    /// One dispatch per direction: the kernel, the three textures in index order, then the grid
    /// of the texture's size (its getters asked even for a nil encoder).
    pub(crate) fn page(
        &self,
        paged: &PagedTexture<P>,
        encoder: Option<&P::ComputeEncoder>,
        is_download: bool,
    ) -> RaiseResult<()> {
        let views = self.plane_textures.lock();
        let [v0, v1] = &*views;
        let texture = Some(&paged.texture);
        if is_download {
            nil_send!(
                encoder,
                encode(
                    OnFault::Return,
                    ComputeCmd::SetComputePipelineState {
                        state: self.blit_down_state.as_ref()
                    }
                )
            )?;
            nil_send!(
                encoder,
                encode(
                    OnFault::Return,
                    ComputeCmd::SetTexture { texture, index: 0 }
                )
            )?;
            nil_send!(
                encoder,
                encode(
                    OnFault::Return,
                    ComputeCmd::SetTexture {
                        texture: v0.as_ref(),
                        index: 1
                    }
                )
            )?;
            nil_send!(
                encoder,
                encode(
                    OnFault::Return,
                    ComputeCmd::SetTexture {
                        texture: v1.as_ref(),
                        index: 2
                    }
                )
            )?;
        } else {
            nil_send!(
                encoder,
                encode(
                    OnFault::Return,
                    ComputeCmd::SetComputePipelineState {
                        state: self.blit_up_state.as_ref()
                    }
                )
            )?;
            nil_send!(
                encoder,
                encode(
                    OnFault::Return,
                    ComputeCmd::SetTexture {
                        texture: v0.as_ref(),
                        index: 0
                    }
                )
            )?;
            nil_send!(
                encoder,
                encode(
                    OnFault::Return,
                    ComputeCmd::SetTexture {
                        texture: v1.as_ref(),
                        index: 1
                    }
                )
            )?;
            nil_send!(
                encoder,
                encode(
                    OnFault::Return,
                    ComputeCmd::SetTexture { texture, index: 2 }
                )
            )?;
        }
        let grid = MtlSize::new(paged.texture.width(), paged.texture.height(), 1);
        nil_send!(
            encoder,
            encode(
                OnFault::Return,
                ComputeCmd::DispatchThreads {
                    threads_per_grid: grid,
                    threads_per_threadgroup: THREADGROUP
                }
            )
        )?;
        Ok(())
    }

    /// Guest to host through the up kernel.
    pub(crate) fn prepare(
        &self,
        hold: Hold<'_>,
        paged: &PagedTexture<P>,
        state: &ResourceState,
        encoder: Option<&P::ComputeEncoder>,
    ) -> RaiseResult<()> {
        paged.prepare(
            state,
            || self.ensure_physical(hold, paged),
            || self.page(paged, encoder, false),
        )
    }

    /// Host to guest through the down kernel.
    pub(crate) fn synchronize(
        &self,
        hold: Hold<'_>,
        paged: &PagedTexture<P>,
        state: &ResourceState,
        encoder: Option<&P::ComputeEncoder>,
    ) -> RaiseResult<()> {
        paged.synchronize(
            state,
            || self.ensure_physical(hold, paged),
            || self.page(paged, encoder, true),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::Handle;
    use crate::resources::resource::ResourceKind;
    use crate::resources::texture::TextureResource;
    use crate::resources::texture::test_support::*;
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::Model;
    use crate::testkit::{Arg, FakePlatform, Fk, Step, kind};

    type Pf = FakePlatform;

    /// A 4x4 550 record: plane 0 at 4 bytes per pixel from 0x10, plane 1 at 4 from 0x80.
    fn xr10() -> TextureDimensionRecord {
        let mut h0 = header(0, 4, 1);
        h0.base_offset = 0x10;
        rec2(
            h0,
            vec![mip(0, 0x40, 16, 4, 4, 1)],
            header(0, 4, 1),
            vec![mip(0x80, 0x40, 16, 4, 4, 1)],
        )
    }

    const LEN: u64 = 0x78;

    fn env() -> (Env, Fk<kind::ComputePipeline>, Fk<kind::ComputePipeline>) {
        let e = Env::new();
        let up: Fk<kind::ComputePipeline> = e.p.world().new_obj(Model::None);
        let down: Fk<kind::ComputePipeline> = e.p.world().new_obj(Model::None);
        e.rec.provide(0, &up);
        e.rec.provide(1, &down);
        (e, up, down)
    }

    fn make(
        e: &Env,
        t: Tex,
        r: &TextureDimensionRecord,
        l: u64,
    ) -> RaiseResult<Option<Arc<Resource<Pf>>>> {
        e.run(|| DualPlaneCompute::new(e.task(), &paging(9, 0x100), r, l, t))
    }

    fn parts(r: &Resource<Pf>) -> (&TextureResource<Pf>, &DualPlaneCompute<Pf>) {
        match &r.kind {
            ResourceKind::Texture(t) => match &t.layout {
                TextureLayout::DualPlaneCompute(s) => (t, s),
                _ => panic!("not compute"),
            },
            _ => panic!("not a texture"),
        }
    }

    fn made(e: &Env, pf: u64) -> Arc<Resource<Pf>> {
        let t = e.tex(desc(pf, 4, 4, 1, 1), false);
        make(e, t, &xr10(), LEN).unwrap().unwrap()
    }

    mod new {
        use super::*;

        #[test]
        fn formats_and_pipelines() {
            for (pf, color) in [(550u64, 554u64), (551, 555)] {
                let (e, up, down) = env();
                let t = e.tex(desc(pf, 4, 4, 1, 1), false);
                let m = e.p.call_log().mark();
                let r = make(&e, t.clone_ref(), &xr10(), LEN).unwrap().unwrap();
                let (tr, s) = parts(&r);
                assert_eq!(s.plane_texture_formats, [color, 84]);
                assert_eq!(s.blit_up_state.as_ref().unwrap().id(), up.id());
                assert_eq!(s.blit_down_state.as_ref().unwrap().id(), down.id());
                assert_eq!({ s.planes[1].mips[0].offset }, 0x80);
                assert_eq!(tr.paged.page, 9);
                // Guest-valid from the start.
                assert!(r.state.guest_valid());
                let sels = e.sels(m);
                let tail = &sels[sels.len() - 2..];
                assert_eq!(
                    tail,
                    [
                        "blit_in_rgb_2p_xr10_a8_pipeline",
                        "blit_out_rgb_2p_xr10_a8_pipeline"
                    ]
                );
                assert_eq!(t.retain_count(), 2);
            }
        }

        /// Mipped: raise before anything is sent
        #[test]
        fn mipped_raises() {
            let (e, _, _) = env();
            let t = e.tex(desc(550, 4, 4, 1, 2), false);
            let mut r = xr10();
            r.header.set_mip_levels(2);
            let m = e.p.call_log().mark();
            let err = make(&e, t.clone_ref(), &r, LEN).err().unwrap();
            assert_eq!(raised(&err), Some(RaiseKind::MippedMultiPlane));
            assert!(e.sels(m).is_empty());
            assert_eq!(t.retain_count(), 2);
        }

        #[test]
        fn short_length() {
            let (e, _, _) = env();
            let t = e.tex(desc(550, 4, 4, 1, 1), false);
            let err = make(&e, t.clone_ref(), &xr10(), 0x2f).err().unwrap();
            assert_eq!(raised(&err), Some(RaiseKind::DimensionLength));
            // The half-made texture and the caller's reference are released.
            assert_eq!(t.retain_count(), 1);
        }

        #[test]
        fn invalid_nil() {
            let (e, _, _) = env();
            // The device lacks extended XR10: flags 0, not two-plane.
            let d = e.p.device();
            let mut caps = crate::platform::caps::DeviceCaps::clone(
                &crate::platform::device::GpuDevice::caps(&d),
            );
            caps.flags.supports_extended_xr10_formats = false;
            d.set_caps(caps);
            let t = e.tex(desc(550, 4, 4, 1, 1), false);
            let r = make(&e, t.clone_ref(), &xr10(), LEN).unwrap();
            assert!(r.is_none());
            assert_eq!(t.retain_count(), 1);
        }

        /// Multisample refused after the other checks
        #[test]
        fn multisample_raises() {
            let (e, _, _) = env();
            let mut d = desc(550, 4, 4, 1, 1);
            d.sample_count = 4;
            let t = e.tex(d, false);
            // Rows of 4 pixels x 4 bytes x 4 samples.
            let r = rec2(
                header(0, 4, 1),
                vec![mip(0, 0x100, 64, 4, 4, 1)],
                header(0, 4, 1),
                vec![mip(0x100, 0x100, 64, 4, 4, 1)],
            );
            let err = make(&e, t.clone_ref(), &r, LEN).err().unwrap();
            assert_eq!(raised(&err), Some(RaiseKind::ComputeShape));
            assert_eq!(t.retain_count(), 2);
        }

        /// An array is refused
        #[test]
        fn array_raises_short_circuit() {
            let (e, _, _) = env();
            let mut d = desc(550, 4, 4, 1, 1);
            d.array_length = 2;
            let t = e.tex(d, false);
            let mut h = header(0, 4, 1);
            h.array_length = 2;
            let r = rec2(
                h,
                vec![mip(0, 0x40, 16, 4, 4, 1)],
                h,
                vec![mip(0x80, 0x40, 16, 4, 4, 1)],
            );
            let err = make(&e, t, &r, LEN).err().unwrap();
            assert_eq!(raised(&err), Some(RaiseKind::ComputeShape));
        }

        /// Another format is refused
        #[test]
        fn nope() {
            let (e, _, _) = env();
            // Plane checks pass with a scripted two-plane Metal format of 4-byte planes.
            e.p.script().always(
                "MTLPixelFormatGetInfoForDevice()",
                Step::value(crate::platform::types::PixelFormatInfo {
                    name: None,
                    flags: 0x80_0000,
                    cast_class: 0,
                    block_bytes: 4,
                    block_width: 1,
                    block_height: 1,
                    block_depth: 1,
                }),
            );
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            // Metal's plane 1 info is zero: a zero-byte plane with zero-byte rows.
            let r = rec2(
                header(0, 4, 1),
                vec![mip(0, 0x40, 16, 4, 4, 1)],
                header(0, 0, 1),
                vec![mip(0x80, 0, 0, 4, 4, 1)],
            );
            let m = e.p.call_log().mark();
            let err = make(&e, t.clone_ref(), &r, LEN).err().unwrap();
            assert_eq!(raised(&err), Some(RaiseKind::ComputeFormat));
            assert!(e.calls(m, "blit_in_rgb_2p_xr10_a8_pipeline").is_empty());
            // The caller's reference is released.
            assert_eq!(t.retain_count(), 1);
        }
    }

    mod ensure_physical {
        use super::*;

        /// One descriptor, a view per plane
        #[test]
        fn views() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let (tr, s) = parts(&r);
            let m = e.p.call_log().mark();
            s.ensure_physical(Hold::NONE, &tr.paged).unwrap();
            let ordered = e.p.call_log().since(m);
            assert_eq!(
                ordered.iter().map(|c| c.sel).collect::<Vec<_>>(),
                vec![
                    "new_buffer_for_virtual_page",
                    "setPixelFormat:",
                    "setResourceOptions:",
                    "newTextureWithDescriptor:offset:bytesPerRow:",
                    "setPixelFormat:",
                    "setResourceOptions:",
                    "newTextureWithDescriptor:offset:bytesPerRow:",
                ]
            );
            // Each plane's pixel format, and the transfer buffer's resource options, are set on
            // the one descriptor its view is made from.
            let (pf, ro, c) = (
                [&ordered[1], &ordered[4]],
                [&ordered[2], &ordered[5]],
                [&ordered[3], &ordered[6]],
            );
            let descriptor = Arg::Obj(Some(pf[0].obj));
            assert!(pf.iter().all(|x| Arg::Obj(Some(x.obj)) == descriptor));
            assert!(ro.iter().all(|x| Arg::Obj(Some(x.obj)) == descriptor));
            let options = (tr.paged.transfer_buffer.lock().as_ref())
                .map(crate::platform::resource::GpuResource::resource_options)
                .unwrap();
            assert!(ro.iter().all(|x| u(x, 0) == options));
            assert!(c.iter().all(|x| x.args[0] == descriptor));
            assert_eq!((u(c[0], 1), u(c[0], 2)), (0x10, 16));
            assert_eq!((u(c[1], 1), u(c[1], 2)), (0x80, 16));
            assert_eq!((u(pf[0], 0), u(pf[1], 0)), (554, 84));
            assert!(s.plane_textures.lock().iter().all(Option::is_some));
            // Bound: nothing more.
            let m = e.p.call_log().mark();
            s.ensure_physical(Hold::NONE, &tr.paged).unwrap();
            assert!(e.sels(m).is_empty());
        }

        /// Plane 1 fails: buffer released, plane 0 kept; a retry leaks plane 0's first view
        #[test]
        fn plane_one_fails_then_retry_leaks() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let (tr, s) = parts(&r);
            let sel = "newTextureWithDescriptor:offset:bytesPerRow:";
            e.p.script().after(sel, 1, Step::Nil);
            let hold = Hold::NONE;
            let err = s.ensure_physical(hold, &tr.paged).err().unwrap();
            assert_eq!(raised(&err), Some(RaiseKind::PlaneViewRefused));
            assert!(tr.paged.transfer_buffer.lock().is_none());
            let first = {
                let v = s.plane_textures.lock();
                assert!(v[1].is_none());
                v[0].as_ref().unwrap().clone_ref()
            };
            assert_eq!(first.retain_count(), 2);
            s.ensure_physical(hold, &tr.paged).unwrap();
            // The first view was overwritten without a release.
            assert_eq!(first.retain_count(), 2);
            assert_ne!(
                s.plane_textures.lock()[0].as_ref().unwrap().id(),
                first.id()
            );
        }

        /// A transfer buffer a racing caller stored between this call's check and its store
        /// stays live, with its GPU VA and its guest pages' mapping, for good: the racer's views
        /// over it, overwritten without a release, hold it
        #[test]
        fn a_transfer_buffer_a_racing_caller_stored_is_kept() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let racer = Arc::clone(&r);
            let raced = Arc::new(Mutex::new(None));
            let seen = Arc::clone(&raced);
            e.rec.during_next("new_buffer_for_virtual_page", move || {
                let (tr, s) = parts(&racer);
                s.ensure_physical(Hold::NONE, &tr.paged).unwrap();
                *seen.lock() = tr.paged.transfer_buffer.lock().as_ref().map(Handle::id);
            });
            let (tr, s) = parts(&r);
            s.ensure_physical(Hold::NONE, &tr.paged).unwrap();
            let raced = raced.lock().take().expect("the racer stored a buffer");
            let stored = tr.paged.transfer_buffer.lock().as_ref().map(Handle::id);
            assert_ne!(stored, Some(raced));
            assert!(
                e.p.world().find(raced).is_some(),
                "the raced buffer is kept"
            );
        }

        /// The descriptor goes with the call, on a refusal and on a normal exit
        #[test]
        fn a_refusal_lets_the_descriptor_go() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let (tr, s) = parts(&r);
            e.p.script()
                .push("newTextureWithDescriptor:offset:bytesPerRow:", Step::Nil);
            let hold = Hold::NONE;
            let err = s.ensure_physical(hold, &tr.paged).err().unwrap();
            assert_eq!(raised(&err), Some(RaiseKind::PlaneViewRefused));
            assert!(e.p.world().live("MTLTextureDescriptor").is_empty());
            s.ensure_physical(hold, &tr.paged).unwrap();
            assert!(e.p.world().live("MTLTextureDescriptor").is_empty());
        }

        /// A Metal raise making a view also lets the descriptor go
        #[test]
        fn a_metal_raise_lets_the_descriptor_go() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let (tr, s) = parts(&r);
            e.p.script().push(
                "newTextureWithDescriptor:offset:bytesPerRow:",
                Step::raise("NSInvalidArgumentException", "bad"),
            );
            let hold = Hold::NONE;
            assert!(s.ensure_physical(hold, &tr.paged).is_err());
            assert!(e.p.world().live("MTLTextureDescriptor").is_empty());
        }

        /// The task's raise: nothing made
        #[test]
        fn task_raise() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let (tr, s) = parts(&r);
            e.p.script().push(
                "new_buffer_for_virtual_page",
                Step::raise("NSGenericException", "x"),
            );
            let hold = Hold::NONE;
            let m = e.p.call_log().mark();
            assert!(s.ensure_physical(hold, &tr.paged).is_err());
            assert_eq!(e.sels(m), vec!["new_buffer_for_virtual_page"]);
        }

        #[test]
        fn dead_task() {
            let (e, _, _) = env();
            let t = e.tex(desc(550, 4, 4, 1, 1), false);
            let mut base = PagedTexture::new(&e.p, e.task(), &t);
            base.task = {
                let d: Arc<dyn crate::resources::host::ResourceHost<Pf>> =
                    crate::testkit::resource_host::FakeResourceHost::new(e.p.clone());
                Arc::downgrade(&d)
            };
            let s = DualPlaneCompute::<Pf> {
                planes: [xr10(), xr10()],
                plane_textures: Mutex::new([None, None]),
                plane_texture_formats: [554, 84],
                blit_up_state: None,
                blit_down_state: None,
            };
            let got = catch_death(|| s.ensure_physical(Hold::NONE, &base));
            assert_eq!(got.err(), Some(died(DeathKind::GoneTaskPages)));
        }
    }

    mod discard {
        use super::*;

        /// Views released, then the superclass's
        #[test]
        fn releases_views_then_super() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let (tr, s) = parts(&r);
            s.ensure_physical(Hold::NONE, &tr.paged).unwrap();
            let v0 = s.plane_textures.lock()[0].as_ref().unwrap().clone_ref();
            let m = e.p.call_log().mark();
            r.discard().unwrap();
            assert_eq!(v0.retain_count(), 1);
            assert!(s.plane_textures.lock().iter().all(Option::is_none));
            assert!(tr.paged.transfer_buffer.lock().is_none());
            assert_eq!(e.sels(m), vec!["setPurgeableState:"]);
            assert!(r.state.is_volatile());
        }
    }

    #[test]
    fn the_last_release_releases_the_views() {
        let (e, _, _) = env();
        let t = e.tex(desc(550, 4, 4, 1, 1), false);
        let r = make(&e, t.clone_ref(), &xr10(), LEN).unwrap().unwrap();
        let (tr, s) = parts(&r);
        s.ensure_physical(Hold::NONE, &tr.paged).unwrap();
        let v1 = s.plane_textures.lock()[1].as_ref().unwrap().clone_ref();
        drop(r);
        assert_eq!(v1.retain_count(), 1);
        assert_eq!(t.retain_count(), 1);
    }

    mod paging {
        use super::*;

        /// Down: pipeline, texture 0, planes 1 and 2, dispatch
        #[test]
        fn down_and_up() {
            let (e, up, down) = env();
            let r = made(&e, 550);
            let (tr, s) = parts(&r);
            s.ensure_physical(Hold::NONE, &tr.paged).unwrap();
            let (v0, v1) = {
                let v = s.plane_textures.lock();
                (v[0].as_ref().unwrap().id(), v[1].as_ref().unwrap().id())
            };
            let t = tr.paged.texture.id();
            let enc = e.compute();
            let m = e.p.call_log().mark();
            s.page(&tr.paged, Some(&enc), true).unwrap();
            let calls = e.p.call_log().since(m);
            let sels: Vec<_> = calls.iter().map(|c| c.sel).collect();
            assert_eq!(
                sels,
                vec![
                    "setComputePipelineState:",
                    "setTexture:atIndex:",
                    "setTexture:atIndex:",
                    "setTexture:atIndex:",
                    "dispatchThreads:threadsPerThreadgroup:"
                ]
            );
            assert_eq!(calls[0].args, vec![Arg::Obj(Some(down.id()))]);
            assert_eq!(calls[1].args, vec![Arg::Obj(Some(t)), Arg::U64(0)]);
            assert_eq!(calls[2].args, vec![Arg::Obj(Some(v0)), Arg::U64(1)]);
            assert_eq!(calls[3].args, vec![Arg::Obj(Some(v1)), Arg::U64(2)]);
            assert_eq!(
                calls[4].args,
                vec![
                    Arg::val(&MtlSize::new(4, 4, 1)),
                    Arg::val(&MtlSize::new(32, 32, 1))
                ]
            );
            let m = e.p.call_log().mark();
            s.page(&tr.paged, Some(&enc), false).unwrap();
            let calls = e.p.call_log().since(m);
            assert_eq!(calls[0].args, vec![Arg::Obj(Some(up.id()))]);
            assert_eq!(calls[1].args, vec![Arg::Obj(Some(v0)), Arg::U64(0)]);
            assert_eq!(calls[2].args, vec![Arg::Obj(Some(v1)), Arg::U64(1)]);
            assert_eq!(calls[3].args, vec![Arg::Obj(Some(t)), Arg::U64(2)]);
        }

        /// A nil encoder encodes nothing
        #[test]
        fn nil_encoder_encodes_nothing() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let (tr, s) = parts(&r);
            let m = e.p.call_log().mark();
            s.page(&tr.paged, None, false).unwrap();
            assert!(e.sels(m).is_empty());
        }
    }

    mod prepare_synchronize {
        use super::*;

        #[test]
        fn prepare() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let enc = e.compute();
            let m = e.p.call_log().mark();
            r.prepare_in_compute_encoder(Hold::NONE, Some(&enc))
                .unwrap();
            let sels = e.sels(m);
            assert_eq!(sels[0], "new_buffer_for_virtual_page");
            assert_eq!(
                *sels.last().unwrap(),
                "dispatchThreads:threadsPerThreadgroup:"
            );
            assert!(r.state.host_valid());
            let m = e.p.call_log().mark();
            r.prepare_in_compute_encoder(Hold::NONE, Some(&enc))
                .unwrap();
            assert!(e.sels(m).is_empty());
        }

        /// A failed view: host-invalid, buffer released
        #[test]
        fn prepare_view_failure() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let (tr, _) = parts(&r);
            e.p.script()
                .push("newTextureWithDescriptor:offset:bytesPerRow:", Step::Nil);
            let hold = Hold::NONE;
            assert!(r.prepare_in_compute_encoder(hold, None).is_err());
            assert!(!r.state.host_valid());
            assert!(tr.paged.transfer_buffer.lock().is_none());
        }

        /// Iso-b:textures_dual_plane_xr10_bad_row: a format-550 texture 63 texels wide reaches
        /// Metal's view creation with plane 0's unaligned rows, and Metal's death there comes
        /// before any compute state is set
        #[test]
        fn a_bad_row_dies_in_metal() {
            let (e, _, _) = env();
            let t = e.tex(desc(550, 63, 4, 1, 1), false);
            let mut h0 = header(0, 4, 1);
            h0.base_offset = 0x10;
            let rec = rec2(
                h0,
                vec![mip(0, 252 * 4, 252, 63, 4, 1)],
                header(0, 4, 1),
                vec![mip(0x400, 252 * 4, 252, 63, 4, 1)],
            );
            let r = e
                .run(|| DualPlaneCompute::new(e.task(), &paging(9, 0x1000), &rec, LEN, t))
                .unwrap()
                .unwrap();
            let enc = e.compute();
            let sel = "newTextureWithDescriptor:offset:bytesPerRow:";
            // Metal's MTLReportFailure for rows that are not 256-aligned (the fake's marker).
            e.p.script().push(sel, Step::Die(None));
            let m = e.p.call_log().mark();
            let got = catch_death(|| r.prepare_in_compute_encoder(Hold::NONE, Some(&enc)));
            assert_eq!(
                got.err(),
                Some(crate::testkit::script::metal_crash_site(sel))
            );
            let c = e.calls(m, sel);
            assert_eq!(c.len(), 1);
            assert_eq!((u(&c[0], 1), u(&c[0], 2)), (0x10, 252));
            let sels = e.sels(m);
            assert!(!sels.contains(&"setComputePipelineState:"));
            assert!(!sels.contains(&"dispatchThreads:threadsPerThreadgroup:"));
        }

        /// Guest-valid from init, nothing written back
        #[test]
        fn synchronize() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let enc = e.compute();
            r.state.set_host_valid(true);
            let m = e.p.call_log().mark();
            r.synchronize_in_compute_encoder(Hold::NONE, Some(&enc))
                .unwrap();
            assert!(e.sels(m).is_empty());
            r.state.set_guest_valid(false);
            r.synchronize_in_compute_encoder(Hold::NONE, Some(&enc))
                .unwrap();
            assert_eq!(
                e.calls(m, "dispatchThreads:threadsPerThreadgroup:").len(),
                1
            );
            assert!(r.state.guest_valid());
        }

        /// The invalid-state logs are this class's own lines; protected: never paged
        #[test]
        fn invalid_state_logs() {
            let (e, _, _) = env();
            let r = made(&e, 550);
            let enc = e.compute();
            parts(&r).0.paged.texture.set_protection_options(1);
            r.state.set_guest_valid(false);
            let m = e.p.call_log().mark();
            let ((), lines) = logged(|| {
                r.prepare_in_compute_encoder(Hold::NONE, Some(&enc))
                    .unwrap();
                assert!(r.state.host_valid());
                r.state.set_host_valid(false);
                r.state.set_guest_valid(false);
                r.synchronize_in_compute_encoder(Hold::NONE, Some(&enc))
                    .unwrap();
            });
            assert!(r.state.guest_valid());
            assert!(e.sels(m).is_empty());
            assert_eq!(lines, vec![PREPARE_INVALID, SYNCHRONIZE_INVALID]);
        }

        #[test]
        fn blit_operations_do_nothing() {
            let (e, _, _) = env();
            let r = made(&e, 551);
            r.state.set_guest_valid(false);
            let enc = e.blit();
            let m = e.p.call_log().mark();
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert!(e.sels(m).is_empty());
            assert!(!r.state.host_valid());
            assert!(!r.state.guest_valid());
            assert!(r.needs_compute_paging());
        }
    }
}
