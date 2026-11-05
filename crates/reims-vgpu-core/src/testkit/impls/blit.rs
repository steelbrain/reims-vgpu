//! Fake: the `platform/blit.rs` traits over the fake models (architecture 13.2).
//!
//! Every blit command is ordered: each logs its selector and arguments (`fake::calllog`) and
//! consults the script (`fake::script`); see `fake::impls` for the outcome channels.

use std::ops::Range;

use super::super::calllog::Arg;
use super::super::objects::{Fk, kind};
use super::super::platform::FakePlatform;
use super::res_id;
use crate::platform::blit::*;
use crate::raise::{OnFault, RaiseResult};
use crate::testkit::script::Policy;

/// The name the fake logs and scripts [`BlitEncoder::copy_from_guest_ram`] under, with the runs
/// (each run's start and end, in order), the destination and its offset: no selector of Metal's
/// names it. It copies where the device copies the guest's RAM
/// ([`World::copy_guest_ram`](super::super::objects::World::copy_guest_ram)), unless a scripted
/// value says otherwise.
pub const GUEST_RAM_TO_BUFFER: &str = "copyFromGuestRam:toBuffer:destinationOffset: (no Metal)";

/// The name the fake logs and scripts [`BlitEncoder::copy_to_guest_ram`] under, with the source,
/// its offset and the runs, as [`GUEST_RAM_TO_BUFFER`].
pub const BUFFER_TO_GUEST_RAM: &str = "copyFromBuffer:sourceOffset:toGuestRam: (no Metal)";

/// The runs as the fake logs them: each run's start and end, in order.
fn runs_arg(runs: &[Range<u64>]) -> Arg {
    Arg::U64s(runs.iter().flat_map(|r| [r.start, r.end]).collect())
}

impl BlitEncoder<FakePlatform> for Fk<kind::BlitEncoder> {
    fn copy_from_guest_ram(
        &self,
        on: OnFault,
        runs: &[Range<u64>],
        destination: &<FakePlatform as crate::platform::Platform>::Buffer,
        destination_offset: u64,
    ) -> RaiseResult<bool> {
        let copies = self.world().copies_guest_ram();
        let args = vec![
            runs_arg(runs),
            Arg::obj(destination),
            Arg::U64(destination_offset),
        ];
        self.thr(Policy::On(on), GUEST_RAM_TO_BUFFER, args, || copies)
    }

    fn copy_to_guest_ram(
        &self,
        on: OnFault,
        source: &<FakePlatform as crate::platform::Platform>::Buffer,
        source_offset: u64,
        runs: &[Range<u64>],
    ) -> RaiseResult<bool> {
        let copies = self.world().copies_guest_ram();
        let args = vec![Arg::obj(source), Arg::U64(source_offset), runs_arg(runs)];
        self.thr(Policy::On(on), BUFFER_TO_GUEST_RAM, args, || copies)
    }

    fn encode(&self, on: OnFault, cmd: BlitCmd<'_, FakePlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        match cmd {
            BlitCmd::CopyBufferToTexture {
                source_buffer,
                source_offset,
                source_bytes_per_row,
                source_bytes_per_image,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::obj(source_buffer),
                    Arg::U64(source_offset),
                    Arg::U64(source_bytes_per_row),
                    Arg::U64(source_bytes_per_image),
                    Arg::val(&source_size),
                    Arg::obj(destination_texture),
                    Arg::U64(destination_slice),
                    Arg::U64(destination_level),
                    Arg::val(&destination_origin),
                ],
            ),
            BlitCmd::CopyBufferToTextureOptions {
                source_buffer,
                source_offset,
                source_bytes_per_row,
                source_bytes_per_image,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
                options,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::opt(source_buffer),
                    Arg::U64(source_offset),
                    Arg::U64(source_bytes_per_row),
                    Arg::U64(source_bytes_per_image),
                    Arg::val(&source_size),
                    Arg::opt(destination_texture),
                    Arg::U64(destination_slice),
                    Arg::U64(destination_level),
                    Arg::val(&destination_origin),
                    Arg::U64(options),
                ],
            ),
            BlitCmd::CopyBufferToBuffer {
                source_buffer,
                source_offset,
                destination_buffer,
                destination_offset,
                size,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::opt(source_buffer),
                    Arg::U64(source_offset),
                    Arg::opt(destination_buffer),
                    Arg::U64(destination_offset),
                    Arg::U64(size),
                ],
            ),
            BlitCmd::CopyTextureToBuffer {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_buffer,
                destination_offset,
                destination_bytes_per_row,
                destination_bytes_per_image,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::obj(source_texture),
                    Arg::U64(source_slice),
                    Arg::U64(source_level),
                    Arg::val(&source_origin),
                    Arg::val(&source_size),
                    Arg::obj(destination_buffer),
                    Arg::U64(destination_offset),
                    Arg::U64(destination_bytes_per_row),
                    Arg::U64(destination_bytes_per_image),
                ],
            ),
            BlitCmd::CopyTextureToBufferOptions {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_buffer,
                destination_offset,
                destination_bytes_per_row,
                destination_bytes_per_image,
                options,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::opt(source_texture),
                    Arg::U64(source_slice),
                    Arg::U64(source_level),
                    Arg::val(&source_origin),
                    Arg::val(&source_size),
                    Arg::opt(destination_buffer),
                    Arg::U64(destination_offset),
                    Arg::U64(destination_bytes_per_row),
                    Arg::U64(destination_bytes_per_image),
                    Arg::U64(options),
                ],
            ),
            BlitCmd::CopyTextureRegion {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::obj(source_texture),
                    Arg::U64(source_slice),
                    Arg::U64(source_level),
                    Arg::val(&source_origin),
                    Arg::val(&source_size),
                    Arg::obj(destination_texture),
                    Arg::U64(destination_slice),
                    Arg::U64(destination_level),
                    Arg::val(&destination_origin),
                ],
            ),
            BlitCmd::CopyTextureRegionOptions {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
                options,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::obj(source_texture),
                    Arg::U64(source_slice),
                    Arg::U64(source_level),
                    Arg::val(&source_origin),
                    Arg::val(&source_size),
                    Arg::obj(destination_texture),
                    Arg::U64(destination_slice),
                    Arg::U64(destination_level),
                    Arg::val(&destination_origin),
                    Arg::U64(options),
                ],
            ),
            BlitCmd::CopyTextureSlices {
                source_texture,
                source_slice,
                source_level,
                destination_texture,
                destination_slice,
                destination_level,
                slice_count,
                level_count,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::obj(source_texture),
                    Arg::U64(source_slice),
                    Arg::U64(source_level),
                    Arg::obj(destination_texture),
                    Arg::U64(destination_slice),
                    Arg::U64(destination_level),
                    Arg::U64(slice_count),
                    Arg::U64(level_count),
                ],
            ),
            BlitCmd::CopyIndirectCommandBuffer {
                source,
                source_range,
                destination,
                destination_index,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::obj(source),
                    Arg::val(&source_range),
                    Arg::obj(destination),
                    Arg::U64(destination_index),
                ],
            ),
            BlitCmd::EndEncoding => self.act(Policy::On(on), sel, vec![]),
            BlitCmd::FillBufferPattern4 {
                buffer,
                range,
                pattern,
            } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::obj(buffer), Arg::val(&range), Arg::U32(pattern)],
            ),
            BlitCmd::FillBuffer {
                buffer,
                range,
                value,
            } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::obj(buffer), Arg::val(&range), Arg::U8(value)],
            ),
            BlitCmd::FillTexture {
                texture,
                level,
                slice,
                region,
                bytes,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::obj(texture),
                    Arg::U64(level),
                    Arg::U64(slice),
                    Arg::val(&region),
                    Arg::Ptr(bytes.ptr().addr()),
                    Arg::U64(bytes.len()),
                ],
            ),
            BlitCmd::FillTextureColor {
                texture,
                level,
                slice,
                region,
                color,
                pixel_format,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::obj(texture),
                    Arg::U64(level),
                    Arg::U64(slice),
                    Arg::val(&region),
                    Arg::val(&color),
                    Arg::U64(pixel_format),
                ],
            ),
            BlitCmd::GenerateMipmaps { texture } => {
                self.act(Policy::Caught, sel, vec![Arg::opt(texture)])
            }
            BlitCmd::InvalidateCompressedTexture { texture } => {
                self.act(Policy::Caught, sel, vec![Arg::obj(texture)])
            }
            BlitCmd::InvalidateCompressedTextureSlice {
                texture,
                slice,
                level,
            } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::obj(texture), Arg::U64(slice), Arg::U64(level)],
            ),
            BlitCmd::OptimizeForCpu { texture } => {
                self.act(Policy::Caught, sel, vec![Arg::obj(texture)])
            }
            BlitCmd::OptimizeForCpuSlice {
                texture,
                slice,
                level,
            } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::obj(texture), Arg::U64(slice), Arg::U64(level)],
            ),
            BlitCmd::OptimizeForGpu { texture } => {
                self.act(Policy::Caught, sel, vec![Arg::obj(texture)])
            }
            BlitCmd::OptimizeForGpuSlice {
                texture,
                slice,
                level,
            } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::obj(texture), Arg::U64(slice), Arg::U64(level)],
            ),
            BlitCmd::OptimizeIndirectCommandBuffer {
                indirect_command_buffer,
                range,
            } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::obj(indirect_command_buffer), Arg::val(&range)],
            ),
            BlitCmd::ResetCommandsInBuffer { buffer, range } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::obj(buffer), Arg::val(&range)],
            ),
            BlitCmd::SynchronizeResource { resource } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::Obj(resource.as_ref().map(res_id))],
            ),
            BlitCmd::SynchronizeTexture {
                texture,
                slice,
                level,
            } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::obj(texture), Arg::U64(slice), Arg::U64(level)],
            ),
            BlitCmd::UpdateFence { fence } => {
                self.act(Policy::Caught, "updateFence:", vec![Arg::obj(fence)])
            }
            BlitCmd::WaitForFence { fence } => {
                self.act(Policy::Caught, "waitForFence:", vec![Arg::obj(fence)])
            }
        }
    }
}
