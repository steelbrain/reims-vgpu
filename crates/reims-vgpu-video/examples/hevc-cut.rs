//! Cuts an Annex B HEVC stream at a random access point, as a seek or a
//! splice would, for tools/hevc-check.sh: a stream that starts at a CRA
//! picture whose leading RASL pictures cannot be decoded, or that turns
//! the CRA into a BLA picture.
//!
//! ```sh
//! hevc-cut list stream.265             # each IRAP access unit: index, kind
//! hevc-cut from stream.265 out.265 N   # the parameter sets, then AU N on
//! hevc-cut bla stream.265 out.265 N    # the stream, AU N's CRA made a BLA
//! hevc-cut truncate stream.265 out.265 N PCT   # AU N's last unit cut short
//! hevc-cut forbid stream.265 out.265 N  # AU N with a forbidden unit after it
//! ```
//!
//! Access units are counted in decode order from 0. `from` writes the
//! first access unit's VPS, SPS and PPS, then access units `N` on,
//! unchanged. `bla` writes the whole stream with the slice segments of
//! access unit `N` turned from `CRA_NUT` to `BLA_W_LP`, whose syntax is the
//! same (H.265 7.3.6.1), as a splice at that picture would have it.
//! `truncate` writes the whole stream with access unit `N`'s last NAL unit
//! cut to `PCT` per cent of its bytes, 16 at least: one damaged frame of a
//! valid stream.
//! `forbid` writes the whole stream with a NAL unit of `TRAIL_R` with
//! `forbidden_zero_bit` set after access unit `N`'s, in the same access
//! unit: one the codec crate's parser refuses before any of the access unit
//! reaches an accelerator, so that what a GPU sees is the stream without
//! that frame. Runs on any host: it needs no GPU.

use reims_vgpu_video::codec::h264::nal::AnnexB;
use reims_vgpu_video::codec::hevc::nal::{self, NalUnit, nal_type};

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hevc-cut: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "usage: hevc-cut list <stream.265> | hevc-cut from|bla <stream.265> <out.265> <index> \
     | hevc-cut truncate <stream.265> <out.265> <index> <percent> \
     | hevc-cut forbid <stream.265> <out.265> <index>";

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let read = |path: &str| std::fs::read(path).map_err(|e| format!("{path}: {e}"));
    match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["list", input] => {
            let data = read(input)?;
            for (index, unit) in nal::access_units(&data).into_iter().enumerate() {
                if let Some(kind) = irap_kind(unit) {
                    println!("{index} {kind}");
                }
            }
            Ok(())
        }
        [mode @ ("from" | "bla"), input, output, index] => {
            let index: usize = index.parse().map_err(|_| USAGE)?;
            let data = read(input)?;
            let units = nal::access_units(&data);
            let cut = units.get(index).ok_or("no such access unit")?;
            let mut out = Vec::new();
            if mode == "from" {
                out.extend_from_slice(&parameter_sets(units[0]));
                out.extend_from_slice(cut);
            } else {
                if irap_kind(cut) != Some("CRA") {
                    return Err(format!("access unit {index} is not a CRA picture"));
                }
                for unit in &units[..index] {
                    out.extend_from_slice(unit);
                }
                out.extend_from_slice(&as_bla(cut));
            }
            for unit in &units[index + 1..] {
                out.extend_from_slice(unit);
            }
            std::fs::write(output, out).map_err(|e| format!("{output}: {e}"))
        }
        ["truncate", input, output, index, percent] => {
            let index: usize = index.parse().map_err(|_| USAGE)?;
            let percent: usize = percent.parse().map_err(|_| USAGE)?;
            let data = read(input)?;
            let units = nal::access_units(&data);
            let cut = units.get(index).ok_or("no such access unit")?;
            let mut out = Vec::new();
            for unit in &units[..index] {
                out.extend_from_slice(unit);
            }
            out.extend_from_slice(&truncated(cut, percent.min(100))?);
            for unit in &units[index + 1..] {
                out.extend_from_slice(unit);
            }
            std::fs::write(output, out).map_err(|e| format!("{output}: {e}"))
        }
        ["forbid", input, output, index] => {
            let index: usize = index.parse().map_err(|_| USAGE)?;
            let data = read(input)?;
            let units = nal::access_units(&data);
            if index >= units.len() {
                return Err("no such access unit".into());
            }
            let mut out = Vec::new();
            for (i, unit) in units.iter().enumerate() {
                out.extend_from_slice(unit);
                if i == index {
                    // TRAIL_R, layer 0, TemporalId 0, with the forbidden
                    // bit, and first_slice_segment_in_pic_flag 0, so that a
                    // muxer keeps it in access unit N.
                    out.extend_from_slice(&[
                        0,
                        0,
                        0,
                        1,
                        0x80 | (nal_type::TRAIL_R << 1),
                        1,
                        0x2f,
                        0x55,
                    ]);
                }
            }
            std::fs::write(output, out).map_err(|e| format!("{output}: {e}"))
        }
        _ => Err(USAGE.into()),
    }
}

/// The access unit with its last NAL unit cut to `percent` per cent of its
/// bytes, 16 at least.
fn truncated(unit: &[u8], percent: usize) -> Result<Vec<u8>, String> {
    let last = AnnexB::new(unit)
        .last()
        .ok_or("an access unit with no NAL unit")?;
    let start = last.as_ptr() as usize - unit.as_ptr() as usize;
    let keep = (last.len() * percent / 100).max(16).min(last.len());
    Ok(unit[..start + keep].to_vec())
}

/// An access unit's VPS, SPS and PPS NAL units, each after a start code.
fn parameter_sets(unit: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for bytes in AnnexB::new(unit) {
        if NalUnit::new(bytes).is_ok_and(|nal| {
            matches!(
                nal.unit_type(),
                nal_type::VPS | nal_type::SPS | nal_type::PPS
            )
        }) {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(bytes);
        }
    }
    out
}

/// The kind of IRAP picture an access unit holds, if it holds one.
fn irap_kind(unit: &[u8]) -> Option<&'static str> {
    AnnexB::new(unit).find_map(|bytes| {
        let nal = NalUnit::new(bytes).ok()?;
        let t = nal.unit_type();
        (nal.layer_id() == 0 && nal::is_irap(t)).then_some(match t {
            _ if nal::is_idr(t) => "IDR",
            _ if nal::is_bla(t) => "BLA",
            nal_type::CRA_NUT => "CRA",
            _ => "IRAP",
        })
    })
}

/// The access unit with each `CRA_NUT` NAL unit's type set to `BLA_W_LP`.
fn as_bla(unit: &[u8]) -> Vec<u8> {
    let mut out = unit.to_vec();
    let base = unit.as_ptr() as usize;
    for bytes in AnnexB::new(unit) {
        let Ok(nal) = NalUnit::new(bytes) else {
            continue;
        };
        if nal.unit_type() == nal_type::CRA_NUT {
            let at = bytes.as_ptr() as usize - base;
            out[at] = (out[at] & 0x81) | (nal_type::BLA_W_LP << 1);
        }
    }
    out
}
