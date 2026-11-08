//! Says what an H.264 stream exercises, from the crate's own parser:
//! profile, order count type, interlacing (MBAFF, field pictures), frame
//! numbering and its gaps, slices, picture types, memory management
//! operations, list modifications, weights and scaling lists. The test
//! script prints it beside each stream's result, as the evidence of what
//! the stream covered.
//!
//! Its last part is the stream's colour: the first SPS's VUI (range,
//! primaries/transfer/matrix, chroma location), and each distinct colour
//! SEI message with its count. A mastering display prints its three
//! primaries in coded order, its white point, then its maximum and minimum
//! luminance, in the SEI's units.
//!
//! ```sh
//! cargo run --example h264-info -- stream.264
//! ```
//!
//! Takes an Annex B stream. Runs on any host: it needs no GPU.

use std::collections::BTreeMap;
use std::sync::Arc;

use reims_vgpu_video::codec::h264::nal::{AnnexB, NalUnit, nal_type};
use reims_vgpu_video::codec::h264::pps::Pps;
use reims_vgpu_video::codec::h264::sei::{self, ColourMessage};
use reims_vgpu_video::codec::h264::slice::{MemoryManagement, SliceHeader, SliceType};
use reims_vgpu_video::codec::h264::sps::{Sps, Vui};

fn main() -> std::process::ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: h264-info <stream.264>");
        return std::process::ExitCode::FAILURE;
    };
    match info(&path) {
        Ok(line) => {
            println!("{line}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("h264-info: {path}: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[derive(Default)]
struct Picture {
    slices: usize,
    idr: bool,
    reference: bool,
    all_intra: bool,
}

fn info(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let mut sps: Vec<Option<Arc<Sps>>> = vec![None; 32];
    let mut pps: Vec<Option<Arc<Pps>>> = vec![None; 256];
    let mut pictures: Vec<Picture> = Vec::new();
    let mut slice_types: BTreeMap<&str, usize> = BTreeMap::new();
    let mut mmco: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut modifications, mut weights, mut long_term_idr) = (0usize, 0usize, 0usize);
    let mut field_pictures = 0usize;
    let mut last: Option<SliceHeader> = None;
    let mut colour_sei: BTreeMap<String, usize> = BTreeMap::new();
    for unit in AnnexB::new(&data) {
        let nal = NalUnit::new(unit)?;
        match nal.unit_type() {
            nal_type::SEI => match sei::colour_messages(unit, 1) {
                Ok(messages) => {
                    for message in messages {
                        *colour_sei.entry(describe(&message)).or_default() += 1;
                    }
                }
                Err(e) => *colour_sei.entry(format!("unreadable({e})")).or_default() += 1,
            },
            nal_type::SPS => {
                let s = Sps::parse(unit)?;
                let id = usize::from(s.seq_parameter_set_id);
                sps[id] = Some(Arc::new(s));
            }
            nal_type::PPS => {
                let p = Pps::parse(unit, |id| sps[usize::from(id)].clone())?;
                let id = usize::from(p.pic_parameter_set_id);
                pps[id] = Some(Arc::new(p));
            }
            nal_type::SLICE | nal_type::IDR => {
                let h = SliceHeader::parse(
                    nal,
                    |id| pps[usize::from(id)].clone(),
                    |id| sps[usize::from(id)].clone(),
                )?;
                let new = last.as_ref().is_none_or(|l| h.starts_new_picture(l));
                if new {
                    pictures.push(Picture {
                        idr: h.idr,
                        reference: h.nal_ref_idc != 0,
                        all_intra: true,
                        ..Picture::default()
                    });
                    for op in h.ref_pic_marking.adaptive.iter().flatten() {
                        let name = match op {
                            MemoryManagement::UnmarkShortTerm(_) => "1",
                            MemoryManagement::UnmarkLongTerm(_) => "2",
                            MemoryManagement::ShortTermToLongTerm(..) => "3",
                            MemoryManagement::MaxLongTermFrameIdx(_) => "4",
                            MemoryManagement::UnmarkAll => "5",
                            MemoryManagement::CurrentToLongTerm(_) => "6",
                        };
                        *mmco.entry(name).or_default() += 1;
                    }
                    if h.ref_pic_marking.long_term_reference_flag {
                        long_term_idr += 1;
                    }
                    if h.field_pic_flag {
                        field_pictures += 1;
                    }
                }
                if let Some(p) = pictures.last_mut() {
                    p.slices += 1;
                    p.all_intra &= h.slice_type == SliceType::I;
                }
                let kind = match h.slice_type {
                    SliceType::P => "P",
                    SliceType::B => "B",
                    SliceType::I => "I",
                    SliceType::Sp => "SP",
                    SliceType::Si => "SI",
                };
                *slice_types.entry(kind).or_default() += 1;
                modifications += h.list_modifications.iter().map(Vec::len).sum::<usize>();
                weights += usize::from(h.pred_weight_table.is_some());
                last = Some(h);
            }
            _ => {}
        }
    }
    let s = sps.iter().flatten().next().ok_or("no SPS")?;
    let p = pps.iter().flatten().next().ok_or("no PPS")?;
    let (_, _, width, height) = s.crop_rect();
    let slices: Vec<usize> = pictures.iter().map(|p| p.slices).collect();
    let fmt = |m: &BTreeMap<&str, usize>| {
        m.iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    Ok(format!(
        "profile_idc {} level {} {}x{} poc_type {} frame_mbs_only {} mbaff {} max_frame_num {} \
         gaps_allowed {} refs {} dpb {} reorder {} \
         cabac {} 8x8 {} sps_matrix {} pps_matrix {} | pictures {} field-pictures {} idr {} \
         non-idr-I {} non-ref {} slices/picture {}..{} slice types {} | mmco {} lt-idr {} \
         list-mods {} weight-tables {} | colour {} sei {}",
        s.profile_idc,
        s.level_idc,
        width,
        height,
        s.pic_order_cnt_type,
        u8::from(s.frame_mbs_only_flag),
        u8::from(s.mb_adaptive_frame_field_flag),
        s.max_frame_num(),
        u8::from(s.gaps_in_frame_num_value_allowed_flag),
        s.max_num_ref_frames,
        s.dpb_frames(),
        s.max_num_reorder_frames(),
        u8::from(p.entropy_coding_mode_flag),
        u8::from(p.transform_8x8_mode_flag),
        u8::from(s.seq_scaling_matrix_present_flag),
        u8::from(p.pic_scaling_matrix_present_flag()),
        pictures.len(),
        field_pictures,
        pictures.iter().filter(|p| p.idr).count(),
        pictures.iter().filter(|p| !p.idr && p.all_intra).count(),
        pictures.iter().filter(|p| !p.reference).count(),
        slices.iter().min().unwrap_or(&0),
        slices.iter().max().unwrap_or(&0),
        fmt(&slice_types),
        if mmco.is_empty() {
            "none".into()
        } else {
            fmt(&mmco)
        },
        long_term_idr,
        modifications,
        weights,
        colour(s.vui.as_ref()),
        if colour_sei.is_empty() {
            "none".into()
        } else {
            colour_sei
                .iter()
                .map(|(k, v)| format!("{k}:{v}"))
                .collect::<Vec<_>>()
                .join(" ")
        },
    ))
}

/// The VUI's colour: `range`, `cp/tc/mc` as coded or `none`, and
/// `chroma-loc top,bottom` or `none`.
fn colour(vui: Option<&Vui>) -> String {
    let Some(vui) = vui else {
        return "no-vui".into();
    };
    format!(
        "range {} cp/tc/mc {} chroma-loc {}",
        if vui.video_full_range_flag {
            "full"
        } else {
            "limited"
        },
        vui.colour_description
            .map_or("none".into(), |(p, t, m)| format!("{p}/{t}/{m}")),
        vui.chroma_sample_loc
            .map_or("none".into(), |(top, bottom)| format!("{top},{bottom}")),
    )
}

/// One colour SEI message, as one word.
fn describe(message: &ColourMessage) -> String {
    match message {
        ColourMessage::MasteringDisplay(d) => {
            let [a, b, c] = d.display_primaries;
            format!(
                "mdcv={},{}/{},{}/{},{}/{},{}/{},{}",
                a.0,
                a.1,
                b.0,
                b.1,
                c.0,
                c.1,
                d.white_point.0,
                d.white_point.1,
                d.max_luminance,
                d.min_luminance
            )
        }
        ColourMessage::ContentLightLevel(l) => format!(
            "cll={},{}",
            l.max_content_light_level, l.max_pic_average_light_level
        ),
        ColourMessage::AlternativeTransfer(t) => format!("atc={t}"),
    }
}
