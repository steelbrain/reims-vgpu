//! A probe, for tests: whether the driver decodes into render targets imported over memory of
//! ours, laid out as a device cache's planes are, as a decode straight into a cache would; and
//! whether a picture's later references read their targets, which a decode into a guest's output
//! cannot have, since the output holds the picture alone (the wire's `DECODE_H264`).
//!
//! Each conformance stream is decoded three ways: into the driver's own surfaces, read back as the
//! reference; into imported targets, the picture read from our memory itself; and into imported
//! targets each overwritten once its picture is read, while the decoder may still reference it.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hasher as _};

use super::display::{Display, Error, check};
use super::ffi;
use super::frame::{Frame, PlaneRows};
use super::host_memory::HostMemory;
use super::surface::{MAKE_SURFACE, SurfaceFormat};
use super::{H264Decoder, HevcDecoder};

const PAGE: usize = 4096;

/// The most pictures of a stream the probe decodes.
const MOST_PICTURES: usize = 120;

/// How the probe lays an imported target out: its rows a multiple of `row_align` bytes, its luma
/// plane `luma_at` bytes into the memory, its chroma plane right after the luma's rows, on the next
/// 256 bytes.
#[derive(Clone, Copy, Debug)]
struct Laid {
    row_align: usize,
    luma_at: usize,
}

/// An imported target: its memory, and where its planes lie in it.
struct Target {
    memory: HostMemory,
    pitch: usize,
    luma_at: usize,
    chroma_at: usize,
}

thread_local! {
    /// Each imported target, by its surface.
    static TARGETS: RefCell<HashMap<ffi::VASurfaceID, Target>> = RefCell::new(HashMap::new());
}

/// A surface of `format`, `width` by `height`, over memory of the probe's own laid out as `laid`.
fn import(
    display: &Display,
    laid: Laid,
    format: SurfaceFormat,
    width: u32,
    height: u32,
) -> Result<ffi::VASurfaceID, Error> {
    let (w, h) = (width as usize, height as usize);
    let pitch = (w * format.bytes_per_sample()).next_multiple_of(laid.row_align);
    let chroma_at = (laid.luma_at + pitch * h).next_multiple_of(256);
    // Room past the planes, as a driver may take the buffer for more rows than the surface's
    // (a CPU mapping of the import past its end faults).
    let len = (chroma_at + pitch * (h.div_ceil(2) + 64) + 16 * PAGE).next_multiple_of(PAGE);
    let memory = HostMemory::new(len)?;
    let mut descriptor = ffi::VADRMPRIMESurfaceDescriptor {
        fourcc: format.fourcc(),
        width,
        height,
        num_objects: 1,
        num_layers: 1,
        ..Default::default()
    };
    descriptor.objects[0] = ffi::VADRMPRIMESurfaceObject {
        fd: memory.dma_buf(),
        size: len as u32,
        drm_format_modifier: ffi::DRM_FORMAT_MOD_LINEAR,
    };
    descriptor.layers[0] = ffi::VADRMPRIMESurfaceLayer {
        drm_format: match format {
            SurfaceFormat::Nv12 => ffi::DRM_FORMAT_NV12,
            SurfaceFormat::P010 => ffi::DRM_FORMAT_P010,
        },
        num_planes: 2,
        object_index: [0, 0, 0, 0],
        offset: [laid.luma_at as u32, chroma_at as u32, 0, 0],
        pitch: [pitch as u32, pitch as u32, 0, 0],
    };
    let mut attributes = [
        ffi::VASurfaceAttrib::zeroed(),
        ffi::VASurfaceAttrib::zeroed(),
    ];
    attributes[0].type_ = ffi::VASurfaceAttribMemoryType;
    attributes[0].flags = ffi::VA_SURFACE_ATTRIB_SETTABLE;
    attributes[0].value.type_ = ffi::VAGenericValueTypeInteger;
    attributes[0].value.value.i =
        i32::from_ne_bytes(ffi::VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2.to_ne_bytes());
    attributes[1].type_ = ffi::VASurfaceAttribExternalBufferDescriptor;
    attributes[1].flags = ffi::VA_SURFACE_ATTRIB_SETTABLE;
    attributes[1].value.type_ = ffi::VAGenericValueTypePointer;
    attributes[1].value.value.p = (&raw mut descriptor).cast();
    let mut surface = ffi::VA_INVALID_SURFACE;
    // SAFETY: one surface out; the attributes and the descriptor they point at live for the call,
    // and the memory's dma-buf for the surface's life (the target keeps it).
    let status = display.call(|raw| unsafe {
        ffi::vaCreateSurfaces(
            raw,
            format.rt_format(),
            width,
            height,
            &mut surface,
            1,
            attributes.as_mut_ptr(),
            2,
        )
    })?;
    check("vaCreateSurfaces (a probe's target)", status)?;
    let target = Target {
        memory,
        pitch,
        luma_at: laid.luma_at,
        chroma_at,
    };
    TARGETS.with(|targets| targets.borrow_mut().insert(surface, target));
    Ok(surface)
}

/// The crop's rows of each plane: the luma's `height` rows of `width` samples from `(x, y)`, and
/// the chroma's rows of their pairs.
fn crop(frame: &Frame) -> (usize, usize, usize, usize, usize) {
    let sample = frame.surface.format().bytes_per_sample();
    let r = frame.rect;
    let (x, y) = (r.x as usize, r.y as usize);
    let luma_bytes = r.width as usize * sample;
    let chroma_bytes = (r.width as usize).div_ceil(2) * 2 * sample;
    (x * sample, y, r.height as usize, luma_bytes, chroma_bytes)
}

/// The picture's crop as the read back gives it, hashed.
fn read_hash(frame: &Frame) -> Result<u64, Error> {
    let (_, _, rows, luma_bytes, chroma_bytes) = crop(frame);
    let mut y = vec![0u8; rows * luma_bytes];
    let mut uv = vec![0u8; rows.div_ceil(2) * chroma_bytes];
    frame.read_back()?.copy_into(
        PlaneRows {
            bytes: &mut y,
            bytes_per_row: luma_bytes,
            rows,
            row_bytes: luma_bytes,
        },
        PlaneRows {
            bytes: &mut uv,
            bytes_per_row: chroma_bytes,
            rows: rows.div_ceil(2),
            row_bytes: chroma_bytes,
        },
    )?;
    let mut hash = DefaultHasher::new();
    hash.write(&y);
    hash.write(&uv);
    Ok(hash.finish())
}

/// The picture's crop as the decode wrote it into its imported target, hashed; then, where
/// `scribble`, every byte of the target overwritten.
fn target_hash(frame: &Frame, scribble: bool) -> Result<Option<u64>, Error> {
    frame.sync()?;
    let (x, y, rows, luma_bytes, chroma_bytes) = crop(frame);
    let id = frame.surface.id();
    Ok(TARGETS.with(|targets| {
        let mut targets = targets.borrow_mut();
        let target = targets.get_mut(&id)?;
        let bytes = target.memory.bytes();
        let mut hash = DefaultHasher::new();
        for row in y..y + rows {
            let at = target.luma_at + row * target.pitch + x;
            hash.write(&bytes[at..at + luma_bytes]);
        }
        for row in y / 2..(y + rows).div_ceil(2) {
            let at = target.chroma_at + row * target.pitch + x;
            hash.write(&bytes[at..at + chroma_bytes]);
        }
        if scribble {
            target.memory.bytes_mut().fill(0x5a);
        }
        Some(hash.finish())
    }))
}

enum Decoder {
    H264(H264Decoder),
    Hevc(HevcDecoder),
}

/// How a stream's pictures came out one way: their hashes in output order, and why it ended
/// early, where it did.
struct Run {
    hashes: Vec<Option<u64>>,
    ended: Option<String>,
}

/// One way through `stream`: into the driver's surfaces, read back (`laid` none), or into targets
/// laid out as `laid`, read from our memory, each overwritten after its read where `scribble`.
fn run(display: &Display, stream: &std::path::Path, laid: Option<Laid>, scribble: bool) -> Run {
    use crate::codec::{h264, hevc};

    let file = std::fs::read(stream).expect("a stream");
    let hevc_stream = stream.to_string_lossy().ends_with(".265");
    let (units, mut decoder) = if hevc_stream {
        (
            hevc::nal::access_units(&file),
            Decoder::Hevc(HevcDecoder::new_annex_b(display)),
        )
    } else {
        (
            h264::nal::access_units(&file),
            Decoder::H264(H264Decoder::new_annex_b(display)),
        )
    };
    let make = laid.map(|laid| -> super::surface::MakeSurface {
        Box::new(move |display: &Display, format, width, height| {
            import(display, laid, format, width, height)
        })
    });
    MAKE_SURFACE.with(|hook| *hook.borrow_mut() = make);
    let mut out = Run {
        hashes: Vec::new(),
        ended: None,
    };
    let take = |frames: Vec<Frame>, out: &mut Run| {
        for frame in &frames {
            let hash = match laid {
                None => read_hash(frame).map(Some),
                Some(_) => target_hash(frame, scribble),
            };
            match hash {
                Ok(hash) => out.hashes.push(hash),
                Err(error) => {
                    out.ended.get_or_insert(format!("a read: {error}"));
                    return false;
                }
            }
        }
        true
    };
    for (id, unit) in units.iter().enumerate() {
        if out.hashes.len() >= MOST_PICTURES {
            break;
        }
        let decoded = match &mut decoder {
            Decoder::H264(d) => d.decode(unit, id as u64),
            Decoder::Hevc(d) => d.decode(unit, id as u64),
        };
        match decoded {
            Ok(frames) => {
                if !take(frames, &mut out) {
                    break;
                }
            }
            Err(Error::HevcBitstream(hevc::Error::SkippedLeadingPicture)) => {}
            Err(error) => {
                out.ended = Some(error.to_string());
                break;
            }
        }
    }
    if out.ended.is_none() {
        let flushed = match &mut decoder {
            Decoder::H264(d) => d.flush(),
            Decoder::Hevc(d) => d.flush(),
        };
        match flushed {
            Ok(frames) => {
                take(frames, &mut out);
            }
            Err(error) => out.ended = Some(error.to_string()),
        }
    }
    drop(decoder);
    MAKE_SURFACE.with(|hook| *hook.borrow_mut() = None);
    // The driver holds its own reference to each imported dma-buf, so our mappings may go.
    TARGETS.with(|targets| targets.borrow_mut().clear());
    out
}

/// The probe over the conformance streams in `RVG_VA_STREAMS` (`:`-separated directories), on the
/// display of `RVG_VAAPI_NODE`: for each way, how many of the reference's pictures came out the
/// same, and the first stream each way refused or came out other at. Prints its answers; asserts
/// only that it decoded something.
#[test]
#[ignore = "needs a VA-API GPU and the conformance suite's streams: run on the VA-API host"]
fn imported_targets_and_their_references() {
    use crate::core::runtime::StdRuntime;

    let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
    let display = Display::open(&node, std::sync::Arc::new(StdRuntime)).expect("a display");
    let dirs = std::env::var("RVG_VA_STREAMS").expect("RVG_VA_STREAMS: the streams' directories");
    let mut streams: Vec<std::path::PathBuf> = (dirs.split(':'))
        .flat_map(|dir| std::fs::read_dir(dir).expect("a streams directory"))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| {
            let name = path.to_string_lossy();
            (name.ends_with(".264") || name.ends_with(".265")) && !name.contains(".part.")
        })
        .collect();
    streams.sort();
    let ways = [
        (
            "rows of 256, luma at 256",
            Laid {
                row_align: 256,
                luma_at: 256,
            },
            false,
        ),
        (
            "rows of 64, luma at 256",
            Laid {
                row_align: 64,
                luma_at: 256,
            },
            false,
        ),
        (
            "rows of 256, luma at 256, each overwritten once read",
            Laid {
                row_align: 256,
                luma_at: 256,
            },
            true,
        ),
    ];
    // Per way: pictures the same, pictures other, streams refused or ended early, first other.
    let mut tally: Vec<(usize, usize, Vec<String>, Vec<String>)> = ways
        .iter()
        .map(|_| (0, 0, Vec::new(), Vec::new()))
        .collect();
    let mut reference_pictures = 0;
    for stream in &streams {
        let name = stream
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        eprintln!("START {name}: the driver's surfaces");
        let reference = run(&display, stream, None, false);
        let known: Vec<u64> = reference.hashes.iter().map_while(|h| *h).collect();
        reference_pictures += known.len();
        let mut line = format!("{name}: {} pictures", known.len());
        // A stream of dropped pictures has the decoder conceal each missing reference with a copy
        // of another through vaGetImage and vaPutImage, which on radeonsi (Mesa 26.2.6) map an
        // imported target on the CPU and fault (SIGBUS): a landing in a cache would keep those
        // stand-ins in the driver's own surfaces, so the probe leaves such streams out.
        if name.starts_with("drop") {
            eprintln!("{line}; left out: its concealment copies map the targets on the CPU");
            continue;
        }
        for (i, &(what, laid, scribble)) in ways.iter().enumerate() {
            eprintln!("START {name}: {what}");
            let got = run(&display, stream, Some(laid), scribble);
            let (same, other, ended, first) = &mut tally[i];
            let (mut stream_same, mut stream_other, mut first_other) = (0, 0, None);
            for (at, (want, hash)) in known.iter().zip(&got.hashes).enumerate() {
                if *hash == Some(*want) {
                    stream_same += 1;
                } else {
                    stream_other += 1;
                    first_other.get_or_insert(at);
                }
            }
            *same += stream_same;
            *other += stream_other;
            if let Some(at) = first_other {
                first.push(format!("{name} at {at}"));
            }
            let came = stream_same + stream_other;
            if let Some(why) = &got.ended
                && came < known.len()
            {
                ended.push(format!("{name}: {why}"));
            }
            let why = got.ended.as_deref().map(|why| format!(" (ended: {why})"));
            line += &format!(
                "; {what}: {came} came, {stream_other} other{}",
                why.unwrap_or_default()
            );
        }
        eprintln!("{line}");
    }
    for (i, &(what, _, _)) in ways.iter().enumerate() {
        let (same, other, ended, first) = &tally[i];
        eprintln!(
            "PROBE {node} {what}: {same} of {reference_pictures} pictures the same, {other} other; \
             first other in {} streams: {:?}; ended early in {}: {:?}",
            first.len(),
            first.iter().take(12).collect::<Vec<_>>(),
            ended.len(),
            ended.iter().take(8).collect::<Vec<_>>()
        );
    }
    assert!(reference_pictures > 0, "no pictures in {dirs}");
}
