//! NVDEC for a host device that hands its decoder whole Annex B access
//! units, one picture each, in decode order, and says where each picture
//! goes (the reims-graphics device's video, reims-vgpu's
//! `graphics::nvidia`): [`AnnexBDecoder`] and its [`AnnexBPicture`]s.
//!
//! - **The access unit is sent in place.** The codec crate's decoder parses
//!   the access unit and hands NVDEC each picture's slices, units of the
//!   access unit's own bytes. Where they lie in it as one run, after start
//!   codes with nothing but zeros between them, NVDEC is given that run of
//!   the access unit, with no copy; otherwise the slices are gathered into
//!   one stream (`nvdec::decoder::one_run`). NVDEC reads the bitstream within
//!   `cuvidDecodePicture`. [`AnnexBDecoder::bitstreams`] counts both.
//! - **A picture is written by the GPU** into memory the host's device
//!   reaches in place ([`MemoryPlane`]): the memory's view registered with
//!   CUDA once per mapping (`crate::nvidia::zero_copy`), and one `cuMemcpy2D` a
//!   plane from the mapped picture into it ([`AnnexBPicture::write_into`]).
//!   Where the memory is not to be had so, the picture is read back into
//!   the decoder's staging memory for the CPU to copy
//!   ([`AnnexBPicture::read_back`]). Into the GPU's own memory, another
//!   API's buffer imported on the picture's GPU (`crate::nvidia::external`), the
//!   copy is within the GPU ([`AnnexBPicture::write_into_device`]).
//! - **A picture NVDEC fails** fails its write or read back, as bad data
//!   ([`Error::DecodeFailed`]), and the decoder then fails each picture
//!   after it until an I picture (H.264) or an IRAP picture (HEVC) decodes
//!   (`crate::nvidia::decode`'s `Failures`); or, where the caller asks
//!   ([`AnnexBDecoder::copy_failed_pictures`]), is written or read back as
//!   NVDEC left it, telling its status, and the pictures after it decode, as
//!   a Mac's hardware decoders hand them on.
//! - **A picture holds its decode surface** until it drops: the decoder has
//!   [`AnnexBDecoder::SPARE_PICTURES`] surfaces beyond its references and
//!   the picture being decoded, and a decode with none free is
//!   [`Error::OutOfSurfaces`].

use std::any::Any;
use std::sync::Arc;

use crate::codec::h264::{Decoder as H264Decoder, nal::NalFormat};
use crate::codec::hevc::Decoder as HevcDecoder;
use crate::core::device_memory::DeviceMemory;
use crate::core::runtime::Runtime;
use crate::core::surfaces::crop_within;

use crate::nvidia::cuda::Gpu;
use crate::nvidia::decode::Failures;
use crate::nvidia::error::Error;
use crate::nvidia::external::ImportedBuffer;
use crate::nvidia::nvdec::decoder::{
    Crop, Destination, InPlace, Memory, OnFailed, Picture, PictureReadBack, SPARE_SURFACES,
};
use crate::nvidia::nvdec::h264::NvdecH264;
use crate::nvidia::nvdec::hevc::NvdecHevc;
use crate::nvidia::wait::Deadline;

/// A codec an [`AnnexBDecoder`] decodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnnexBCodec {
    /// H.264, of the profiles the codec crate decodes.
    H264,
    /// HEVC, Main and Main 10.
    Hevc,
}

/// One stream's decoder of whole Annex B access units on NVDEC (module
/// docs).
pub struct AnnexBDecoder {
    codec: Codec,
    failures: Failures,
    /// The id the next sample is decoded with.
    next_id: u64,
    /// What its pictures' writes and read backs do with a decode NVDEC
    /// answers failed ([`Self::copy_failed_pictures`]).
    failed: OnFailed,
}

/// The codec crate's decoder of the stream, with NVDEC as its accelerator.
enum Codec {
    H264(H264Decoder<NvdecH264>),
    Hevc(HevcDecoder<NvdecHevc>),
}

/// How an [`AnnexBDecoder`]'s pictures reached NVDEC: as a run of their
/// access unit, or gathered into a stream of the decoder's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bitstreams {
    /// Pictures sent in place.
    pub in_place: u64,
    /// Pictures whose slices were gathered.
    pub gathered: u64,
}

impl AnnexBDecoder {
    /// The pictures a decoder's caller may hold beside its references and
    /// the one being decoded before a decode finds no free surface.
    pub const SPARE_PICTURES: usize = SPARE_SURFACES;

    /// A decoder of `codec` on `gpu`, with no parameter sets yet. The
    /// process's registrations of memory pictures are written into drop on
    /// a thread of `runtime`'s, and the crate's waits on the GPU run on its
    /// clock, unless the crate was handed a runtime before ([`crate::nvidia::use_runtime`],
    /// which the backend's opening calls, or a decoder made before).
    pub fn new(gpu: &Arc<Gpu>, codec: AnnexBCodec, runtime: &Arc<dyn Runtime>) -> Self {
        crate::nvidia::wait::use_runtime(runtime);
        crate::nvidia::zero_copy::make_registrations(runtime);
        let codec = match codec {
            AnnexBCodec::H264 => Codec::H264(H264Decoder::new(
                NvdecH264::new(Arc::clone(gpu)),
                NalFormat::AnnexB,
            )),
            AnnexBCodec::Hevc => Codec::Hevc(HevcDecoder::new(
                NvdecHevc::new(Arc::clone(gpu)),
                NalFormat::AnnexB,
            )),
        };
        Self {
            codec,
            failures: Failures::default(),
            next_id: 0,
            failed: OnFailed::Refuse,
        }
    }

    /// Has each picture NVDEC answers failed written or read back as NVDEC
    /// left it, rather than failed, and the pictures after it decoded as any,
    /// predicting from it, rather than refused until an I (H.264) or IRAP
    /// (HEVC) picture: as a Mac's hardware decoders hand their client a
    /// picture they decoded with an error, concealed, its status 0, and decode
    /// on from it. **Measured** on a MacBook Pro of 2019, macOS 26.7.1,
    /// VideoToolbox's hardware decode: of an H.264 stream of 90 pictures whose
    /// 20th, a P picture, was cut to half its slice or had 64 bytes of it
    /// changed, every picture came back with status 0, the 20th and those
    /// after it 26.5 to 29.7 dB from the undamaged stream's luma and none all
    /// zeros, until the IDR picture at 45, which came back as the undamaged
    /// stream's; of an HEVC stream at 1080p on the UHD 630 whose P picture at
    /// 10 had its last slice segment cut to half, that picture and
    /// each after it came back with status 0 and an image, their luma 19.6 to
    /// 27.7 from the undamaged stream's on average (of 255, over a 64 by 36
    /// thumbnail), until the IDR picture at 30, which came back as the
    /// undamaged stream's. Each such picture tells its status
    /// ([`AnnexBPicture::copied_failed`]).
    pub fn copy_failed_pictures(&mut self) {
        self.failed = OnFailed::Copy;
    }

    /// Takes parameter sets (H.264's SPS and PPS, HEVC's VPS, SPS and PPS),
    /// units in Annex B form; units that make a picture are an error.
    pub fn parameter_sets(&mut self, sets: &[u8]) -> Result<(), Error> {
        match self.decode_sample(sets, false)? {
            None => Ok(()),
            Some(_) => Err(Error::Bitstream(
                "parameter sets that hold a picture".to_owned(),
            )),
        }
    }

    /// Decodes `au`, an access unit in Annex B form, and answers its
    /// picture once NVDEC has its decode; `None` for a RASL picture the
    /// stream's random access skips. An access unit of more than one
    /// picture is an error. The picture's decode is done when it is
    /// written or read back.
    pub fn decode(&mut self, au: &[u8]) -> Result<Option<AnnexBPicture>, Error> {
        match self.decode_sample(au, true) {
            Err(Error::SkippedLeadingPicture(_)) => Ok(None),
            decoded => decoded,
        }
    }

    /// How its pictures reached NVDEC so far.
    pub fn bitstreams(&self) -> Bitstreams {
        let in_place = match &self.codec {
            Codec::H264(decoder) => &decoder.accelerator().in_place,
            Codec::Hevc(decoder) => &decoder.accelerator().in_place,
        };
        Bitstreams {
            in_place: in_place.in_place,
            gathered: in_place.gathered,
        }
    }

    /// Decodes `sample`, its slices sent in place where `in_place` and they
    /// lie in one run of it; its one picture, or none.
    fn decode_sample(
        &mut self,
        sample: &[u8],
        in_place: bool,
    ) -> Result<Option<AnnexBPicture>, Error> {
        let id = self.next_id;
        self.next_id += 1;
        let failed = self.failures.take();
        let on_failed = self.failed;
        let outputs = match &mut self.codec {
            Codec::H264(decoder) => {
                if failed {
                    decoder.mark_failed();
                }
                let decoded = with_sample(
                    sample,
                    in_place,
                    |d: &mut H264Decoder<NvdecH264>| &mut d.accelerator_mut().in_place,
                    decoder,
                    |d| d.decode(sample, id),
                )?;
                (decoded.into_iter())
                    .map(|out| (out.picture, out.crop))
                    .collect::<Vec<_>>()
            }
            Codec::Hevc(decoder) => {
                if failed {
                    decoder.mark_failed();
                }
                let decoded = with_sample(
                    sample,
                    in_place,
                    |d: &mut HevcDecoder<NvdecHevc>| &mut d.accelerator_mut().in_place,
                    decoder,
                    |d| d.decode(sample, id),
                )?;
                (decoded.into_iter())
                    .map(|out| (out.picture, out.crop))
                    .collect::<Vec<_>>()
            }
        };
        let mut outputs = outputs.into_iter();
        let picture = outputs.next();
        if outputs.next().is_some() {
            return Err(Error::Bitstream(
                "an access unit of more than one picture".to_owned(),
            ));
        }
        Ok(
            picture.map(|(picture, (x, y, width, height))| AnnexBPicture {
                picture,
                crop: Crop {
                    x,
                    y,
                    width,
                    height,
                },
                failures: self.failures.clone(),
                on_failed,
            }),
        )
    }
}

/// `decode` run on `decoder` with `sample` set as the one its slices are
/// sent from where `in_place`, and cleared after.
fn with_sample<D, T>(
    sample: &[u8],
    in_place: bool,
    state: impl Fn(&mut D) -> &mut InPlace,
    decoder: &mut D,
    decode: impl FnOnce(&mut D) -> T,
) -> T {
    if in_place {
        // SAFETY: `sample` is borrowed, unchanged, through the decode, and
        // cleared from the state before it returns.
        unsafe { state(decoder).set(sample) };
    }
    let decoded = decode(decoder);
    state(decoder).clear();
    decoded
}

/// A plane of memory a picture is written into by the GPU
/// ([`AnnexBPicture::write_into`]): `rows` rows of `row_bytes`,
/// `bytes_per_row` apart, from `offset` into `memory`.
pub struct MemoryPlane<'a> {
    /// The memory the plane lies in.
    pub memory: &'a dyn DeviceMemory,
    /// The plane's first byte in the memory.
    pub offset: usize,
    /// Bytes from one of its rows to the next.
    pub bytes_per_row: usize,
    /// The bytes of one of its rows that may be written.
    pub row_bytes: usize,
    /// Its rows.
    pub rows: u32,
}

/// A plane of the GPU's own memory a picture is written into by a copy within the GPU
/// ([`AnnexBPicture::write_into_device`]): `rows` rows of `row_bytes`, `bytes_per_row` apart,
/// from `offset` into `buffer`.
pub struct DevicePlane<'a> {
    /// The buffer the plane lies in, imported on the picture's GPU.
    pub buffer: &'a Arc<ImportedBuffer>,
    /// The plane's first byte in the buffer.
    pub offset: usize,
    /// Bytes from one of its rows to the next.
    pub bytes_per_row: usize,
    /// The bytes of one of its rows that may be written.
    pub row_bytes: usize,
    /// Its rows.
    pub rows: u32,
}

/// The end of a plane's rows, `(offset, bytes_per_row, row_bytes, rows)`, in memory of `length`
/// bytes: its last row's last byte's next; `Err` says why there is none.
fn plane_end(
    (offset, bytes_per_row, row_bytes, rows): (usize, usize, usize, u32),
    length: usize,
) -> Result<usize, String> {
    let rows = usize::try_from(rows).unwrap_or(usize::MAX);
    let end = match rows.checked_sub(1) {
        None => offset,
        Some(past) => past
            .checked_mul(bytes_per_row)
            .and_then(|bytes| bytes.checked_add(offset))
            .and_then(|bytes| bytes.checked_add(row_bytes))
            .ok_or_else(|| "a plane past the address space".to_owned())?,
    };
    if end > length {
        return Err(format!(
            "a plane at {offset} to {end} past its memory's {length} bytes"
        ));
    }
    Ok(end)
}

/// A plane of a picture read back ([`ReadBack::planes`]): `rows` rows of
/// `bytes_per_row` each, packed.
pub struct ReadPlane<'a> {
    /// Its rows' bytes.
    pub bytes: &'a [u8],
    /// The bytes of one of its rows.
    pub bytes_per_row: usize,
    /// Its rows.
    pub rows: u32,
}

/// A picture read back into its decoder's staging memory, held there until
/// dropped ([`AnnexBPicture::read_back`]).
pub struct ReadBack<'a>(PictureReadBack<'a>);

impl ReadBack<'_> {
    /// The luma and the chroma (interleaved Cb and Cr, half as tall).
    pub fn planes(&self) -> [ReadPlane<'_>; 2] {
        self.0
            .planes()
            .map(|(bytes, bytes_per_row, rows)| ReadPlane {
                bytes,
                bytes_per_row,
                rows,
            })
    }
}

/// A picture an [`AnnexBDecoder`] decoded, which holds its decode surface
/// until it drops.
pub struct AnnexBPicture {
    picture: Arc<Picture>,
    /// The stream's crop of it.
    crop: Crop,
    /// Its decoder's, which a failed decode is noted in.
    failures: Failures,
    /// Whether its writes and read backs fail a decode NVDEC answers failed, or
    /// copy it as NVDEC left it ([`AnnexBDecoder::copy_failed_pictures`]).
    on_failed: OnFailed,
}

impl AnnexBPicture {
    /// Bytes a sample: 1 for NV12, 2 for P016 (10-bit samples in each
    /// sample's high bits, as P010's).
    pub fn bytes_per_sample(&self) -> usize {
        self.picture.bytes_per_sample()
    }

    /// NVDEC's status of its decode, where NVDEC answered it failed and a
    /// write or read back of it copied it as NVDEC left it
    /// ([`AnnexBDecoder::copy_failed_pictures`]): 8
    /// (`cuvidDecodeStatus_Error`) or 9 (`cuvidDecodeStatus_Error_Concealed`).
    /// Such a picture is not noted as failed, and the pictures after it decode.
    pub fn copied_failed(&self) -> Option<u32> {
        self.picture.copied_failed()
    }

    /// The stream's crop of it: x, y, width and height in luma samples.
    pub fn crop(&self) -> (u32, u32, u32, u32) {
        let Crop {
            x,
            y,
            width,
            height,
        } = self.crop;
        (x, y, width, height)
    }

    /// Narrows the crop to the `width` by `height` samples from (`x`, `y`) of it: what its writes
    /// and read back take from then on, as they take an odd crop. `x` and `y` are even and the
    /// rect lies inside the crop ([`crop_within`]); `Err` says why not, the crop left as it was.
    pub fn crop_within(&mut self, x: u32, y: u32, width: u32, height: u32) -> Result<(), String> {
        let Crop {
            x: crop_x,
            y: crop_y,
            width: crop_width,
            height: crop_height,
        } = self.crop;
        let [x, y, width, height] = crop_within(
            [crop_x, crop_y, crop_width, crop_height],
            [x, y, width, height],
        )
        .ok_or_else(|| {
            format!(
                "a crop of {width}x{height} at ({x}, {y}) of a picture's {crop_width}x{crop_height}"
            )
        })?;
        self.crop = Crop {
            x,
            y,
            width,
            height,
        };
        Ok(())
    }

    /// The crop at `size`: the stream's, from its corner, `size` at most.
    fn crop_at(&self, (width, height): (u32, u32)) -> Crop {
        Crop {
            width: width.min(self.crop.width),
            height: height.min(self.crop.height),
            ..self.crop
        }
    }

    /// Waits for the picture's decode and writes `size` of its crop into
    /// `luma` and `chroma` by the GPU's copies, each plane's rows alone, as
    /// much as fits: one `cuMemcpy2D` a plane from the mapped picture into
    /// the memory's view, registered with CUDA once per mapping (module
    /// docs). The outer `Err` is the picture's: a decode NVDEC failed (but
    /// for one copied as NVDEC left it, [`Self::copied_failed`]), or work not
    /// done by its deadline, which keeps `hold` and the registrations until
    /// the GPU is done. The inner `Err` says why the
    /// memory could not be written so: a view CUDA does not register, a
    /// plane past its memory, or a copy that failed, after which the planes
    /// may hold part of the picture.
    ///
    /// # Safety
    ///
    /// Each plane's rows lie apart from the other's, and are the caller's
    /// for the GPU to write from the call on, while `hold` lives: nothing
    /// else writes them meanwhile, and the memory's view stays mapped.
    pub unsafe fn write_into(
        &self,
        size: (u32, u32),
        luma: MemoryPlane<'_>,
        chroma: MemoryPlane<'_>,
        hold: Box<dyn Any + Send>,
    ) -> Result<Result<(), String>, Error> {
        let gpu = self.picture.gpu();
        let mut written: Vec<Box<dyn Any + Send>> = vec![hold];
        let mut destinations = Vec::with_capacity(2);
        for plane in [luma, chroma] {
            let registration = match crate::nvidia::zero_copy::registered(gpu, plane.memory) {
                Ok(registration) => registration,
                Err(why) => return Ok(Err(why)),
            };
            let place = (
                plane.offset,
                plane.bytes_per_row,
                plane.row_bytes,
                plane.rows,
            );
            let end = match plane_end(place, plane.memory.length()) {
                Ok(end) => end,
                Err(why) => return Ok(Err(why)),
            };
            destinations.push(Destination {
                // SAFETY: the offset is inside the memory (just checked),
                // which lies inside the registered view from its first byte.
                at: Memory::Host(unsafe { registration.first().as_ptr().add(plane.offset) }),
                len: end - plane.offset,
                bytes_per_row: plane.bytes_per_row,
                rows: plane.rows,
            });
            written.push(Box::new(registration));
        }
        let [luma, chroma] = destinations[..] else {
            return Ok(Err("not two planes".to_owned()));
        };
        let deadline = Deadline::default();
        // SAFETY: each destination is a plane's rows alone, in a view its
        // registration keeps mapped and pinned, which the caller lets the
        // GPU write and nothing else, for as long as `written` (its hold and
        // the registrations) lives.
        let copied = unsafe {
            (self.picture).write_into(
                self.crop_at(size),
                luma,
                chroma,
                written,
                &deadline,
                self.on_failed,
            )
        };
        match copied {
            Ok(Ok(())) => Ok(Ok(())),
            Ok(Err(error)) => Ok(Err(format!("the GPU's copy failed: {error}"))),
            Err(error) => Err(self.failures.noted(error)),
        }
    }

    /// [`Self::write_into`] into the GPU's own memory: `size` of the crop into `luma` and
    /// `chroma`, each plane's rows alone, as much as fits, by one `cuMemcpy2D` a plane within
    /// the GPU, into buffers of another API's memory imported on the picture's GPU
    /// ([`ImportedBuffer`]). The outer `Err` is as `write_into`'s, keeping `hold` and the imports
    /// until the GPU is done; the inner says why the planes could not be written so: an import on
    /// another GPU, a plane past its buffer, or a copy that failed, after which the planes may
    /// hold part of the picture.
    ///
    /// # Safety
    ///
    /// Each plane's rows lie apart from the other's, and are the caller's for the GPU to write
    /// from the call on, while `hold` lives: nothing else writes them meanwhile, and the memory
    /// the buffers were imported from stays allocated.
    pub unsafe fn write_into_device(
        &self,
        size: (u32, u32),
        luma: DevicePlane<'_>,
        chroma: DevicePlane<'_>,
        hold: Box<dyn Any + Send>,
    ) -> Result<Result<(), String>, Error> {
        let gpu = self.picture.gpu();
        let mut written: Vec<Box<dyn Any + Send>> = vec![hold];
        let mut destinations = Vec::with_capacity(2);
        for plane in [luma, chroma] {
            // One GPU's imports share its primary context, which its decoders' copies run in.
            if plane.buffer.gpu().info().pci_bus_id != gpu.info().pci_bus_id {
                return Ok(Err("a buffer imported on another GPU".to_owned()));
            }
            let place = (
                plane.offset,
                plane.bytes_per_row,
                plane.row_bytes,
                plane.rows,
            );
            let end = match plane_end(place, plane.buffer.length()) {
                Ok(end) => end,
                Err(why) => return Ok(Err(why)),
            };
            destinations.push(Destination {
                at: Memory::Device(plane.buffer.address_at(plane.offset)),
                len: end - plane.offset,
                bytes_per_row: plane.bytes_per_row,
                rows: plane.rows,
            });
            written.push(Box::new(Arc::clone(plane.buffer)));
        }
        let [luma, chroma] = destinations[..] else {
            return Ok(Err("not two planes".to_owned()));
        };
        let deadline = Deadline::default();
        // SAFETY: each destination is a plane's rows alone, in a buffer imported on the
        // picture's GPU and mapped in its context, which the caller lets the GPU write and
        // nothing else, for as long as `written` (its hold and the imports) lives.
        let copied = unsafe {
            (self.picture).write_into(
                self.crop_at(size),
                luma,
                chroma,
                written,
                &deadline,
                self.on_failed,
            )
        };
        match copied {
            Ok(Ok(())) => Ok(Ok(())),
            Ok(Err(error)) => Ok(Err(format!("the GPU's copy failed: {error}"))),
            Err(error) => Err(self.failures.noted(error)),
        }
    }

    /// Waits for the picture's decode and reads `size` of its crop back
    /// into the decoder's staging memory, each plane's rows packed, for the
    /// CPU to copy. `Err` as [`Self::write_into`]'s outer one.
    pub fn read_back(&self, size: (u32, u32)) -> Result<ReadBack<'_>, Error> {
        let deadline = Deadline::default();
        (self
            .picture
            .read_back(self.crop_at(size), &deadline, self.on_failed))
        .map(ReadBack)
        .map_err(|error| self.failures.noted(error))
    }
}

#[cfg(test)]
mod tests;
