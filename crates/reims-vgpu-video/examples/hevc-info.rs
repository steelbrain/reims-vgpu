//! Says what an HEVC stream exercises, from the crate's own parser:
//! profile, bit depth, size, scaling lists, tiles, wavefronts, slices and
//! dependent slice segments, picture types (IDR, CRA, BLA, RASL, RADL),
//! short-term sets the slice headers code themselves (`st-own`, beside the
//! SPS's count of sets, `st-sets`: a header's own set can be predicted only
//! from one of those), long-term pictures, list modifications, weights and
//! whether the VUI says the pictures are fields. The test script
//! prints it beside each stream's result, as the evidence of what the
//! stream covered.
//!
//! Its last part is the stream's colour: the first SPS's VUI (range,
//! primaries/transfer/matrix, chroma location), and each distinct colour
//! SEI message, prefix or suffix, with its count. A mastering display
//! prints its three primaries in coded order, its white point, then its
//! maximum and minimum luminance, in the SEI's units.
//!
//! ```sh
//! cargo run --example hevc-info -- stream.265 [--pictures]
//! ```
//!
//! `--pictures` first prints a line per picture in decode order: its NAL
//! unit type, its independent slice segments' types and its
//! `slice_pic_order_cnt_lsb`, so that where a stream's IRAP, leading and
//! intra pictures fall can be read.
//!
//! Takes an Annex B stream. Runs on any host: it needs no GPU.

use std::collections::BTreeMap;
use std::sync::Arc;

use reims_vgpu_video::codec::h264::nal::AnnexB;
use reims_vgpu_video::codec::h264::sei::{self, ColourMessage};
use reims_vgpu_video::codec::hevc::nal::{NalUnit, is_bla, is_idr, is_radl, is_rasl, nal_type};
use reims_vgpu_video::codec::hevc::pps::Pps;
use reims_vgpu_video::codec::hevc::slice::{SliceHeader, SliceType};
use reims_vgpu_video::codec::hevc::sps::{Sps, Vui};

/// `SUFFIX_SEI_NUT` (HEVC Table 7-1).
const SUFFIX_SEI: u8 = 40;

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pictures = args.iter().any(|a| a == "--pictures");
    let Some(path) = args.iter().find(|a| !a.starts_with("--")) else {
        eprintln!("usage: hevc-info <stream.265> [--pictures]");
        return std::process::ExitCode::FAILURE;
    };
    match info(path, pictures) {
        Ok(line) => {
            println!("{line}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("hevc-info: {path}: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// What the stream's parameter sets turn on, over every set it carries.
#[derive(Default)]
struct Tools {
    sps_lists: bool,
    pps_lists: bool,
    tiles: BTreeMap<String, usize>,
    wpp: bool,
    dependent: bool,
    weighted_p: bool,
    weighted_b: bool,
    list_modification: bool,
    long_term_present: bool,
    sign_hiding: bool,
    transform_skip: bool,
    transquant_bypass: bool,
    pcm: bool,
    /// The most `num_short_term_ref_pic_sets` of any SPS.
    sps_sets: usize,
}

fn info(path: &str, list_pictures: bool) -> Result<String, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let mut sps: Vec<Option<Arc<Sps>>> = vec![None; 16];
    let mut pps: Vec<Option<Arc<Pps>>> = vec![None; 64];
    let mut tools = Tools::default();
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    let mut slice_types: BTreeMap<&str, usize> = BTreeMap::new();
    let mut segments_per_picture: Vec<usize> = Vec::new();
    let (mut dependent, mut entry_points, mut long_term, mut used_long_term) = (0, 0, 0, 0);
    let (mut modifications, mut weights, mut hidden, mut sub_layers) = (0, 0, 0, 0);
    let mut own_sets = 0;
    let mut independent: Option<SliceHeader> = None;
    let mut colour_sei: BTreeMap<String, usize> = BTreeMap::new();
    // Each picture's NAL unit type, slice types and order count LSB.
    let mut pictures: Vec<(u8, String, u32)> = Vec::new();
    for unit in AnnexB::new(&data) {
        let nal = NalUnit::new(unit)?;
        if nal.layer_id() != 0 {
            continue;
        }
        match nal.unit_type() {
            nal_type::PREFIX_SEI | SUFFIX_SEI => match sei::colour_messages(unit, 2) {
                Ok(messages) => {
                    for message in messages {
                        *colour_sei.entry(describe(&message)).or_default() += 1;
                    }
                }
                Err(e) => *colour_sei.entry(format!("unreadable({e})")).or_default() += 1,
            },
            nal_type::SPS => {
                let s = Sps::parse(unit)?;
                tools.sps_lists |= s.scaling_lists.is_some();
                tools.long_term_present |= s.long_term_ref_pics_present_flag;
                tools.pcm |= s.pcm.is_some();
                tools.sps_sets = tools.sps_sets.max(s.short_term_ref_pic_sets.len());
                let id = usize::from(s.sps_seq_parameter_set_id);
                sps[id] = Some(Arc::new(s));
            }
            nal_type::PPS => {
                let p = Pps::parse(unit)?;
                tools.pps_lists |= p.scaling_lists.is_some();
                if p.tiles_enabled_flag {
                    let layout = format!(
                        "{}x{}{}",
                        p.num_tile_columns_minus1 + 1,
                        p.num_tile_rows_minus1 + 1,
                        if p.uniform_spacing_flag {
                            ""
                        } else {
                            "-explicit"
                        }
                    );
                    *tools.tiles.entry(layout).or_default() += 1;
                }
                tools.wpp |= p.entropy_coding_sync_enabled_flag;
                tools.dependent |= p.dependent_slice_segments_enabled_flag;
                tools.weighted_p |= p.weighted_pred_flag;
                tools.weighted_b |= p.weighted_bipred_flag;
                tools.list_modification |= p.lists_modification_present_flag;
                tools.sign_hiding |= p.sign_data_hiding_enabled_flag;
                tools.transform_skip |= p.transform_skip_enabled_flag;
                tools.transquant_bypass |= p.transquant_bypass_enabled_flag;
                let id = usize::from(p.pps_pic_parameter_set_id);
                pps[id] = Some(Arc::new(p));
            }
            t if t <= nal_type::RSV_VCL31 => {
                let h = SliceHeader::parse(
                    nal,
                    |id| pps[usize::from(id)].clone(),
                    |id| sps[usize::from(id)].clone(),
                    independent.as_ref(),
                )?;
                if h.first_slice_segment_in_pic_flag {
                    segments_per_picture.push(0);
                    let kind = match t {
                        _ if is_idr(t) => "IDR",
                        nal_type::CRA_NUT => "CRA",
                        _ if is_bla(t) => "BLA",
                        _ if is_rasl(t) => "RASL",
                        _ if is_radl(t) => "RADL",
                        _ => "TRAIL",
                    };
                    *kinds.entry(kind).or_default() += 1;
                    hidden += usize::from(!h.pic_output_flag);
                    sub_layers += usize::from(nal.temporal_id() > 0);
                    pictures.push((t, String::new(), h.slice_pic_order_cnt_lsb));
                }
                if let Some(n) = segments_per_picture.last_mut() {
                    *n += 1;
                }
                entry_points += h.num_entry_point_offsets as usize;
                if h.dependent_slice_segment_flag {
                    dependent += 1;
                } else {
                    let kind = match h.slice_type {
                        SliceType::B => "B",
                        SliceType::P => "P",
                        SliceType::I => "I",
                    };
                    *slice_types.entry(kind).or_default() += 1;
                    if let Some((_, types, _)) = pictures.last_mut() {
                        types.push_str(kind);
                    }
                    own_sets += usize::from(!is_idr(t) && !h.short_term_ref_pic_set_sps_flag);
                    long_term += h.long_term.len();
                    used_long_term += h.long_term.iter().filter(|l| l.used_by_current).count();
                    modifications += h.list_entries.iter().flatten().count();
                    weights += usize::from(h.pred_weight_table.is_some());
                    independent = Some(h);
                }
            }
            _ => {}
        }
    }
    if list_pictures {
        for (index, (unit_type, types, lsb)) in pictures.iter().enumerate() {
            println!(
                "picture {index}: {} slices {types} poc-lsb {lsb}",
                unit_type_name(*unit_type)
            );
        }
    }
    let s = sps.iter().flatten().next().ok_or("no SPS")?;
    let (_, _, width, height) = s.crop_rect();
    let flag = |b: bool| u8::from(b);
    Ok(format!(
        "profile {} level {} {}x{} (coded {}x{}, CTB {}) {}-bit dpb {} reorder {} | \
         scaling sps {} pps {} enabled {} tiles {} wpp {} dependent {} weighted p {} b {} \
         list-mods {} lt-present {} sdh {} ts {} bypass {} pcm {} | pictures {} kinds {} \
         hidden {} temporal-id>0 {} \
         segments/picture {}..{} dependent {} entry-points {} slice types {} | \
         st-sets {} st-own {} lt {} lt-used {} list-mods {} weight-tables {} | fields {} \
         colour {} sei {}",
        s.profile_tier_level.general_profile_idc,
        s.profile_tier_level.general_level_idc,
        width,
        height,
        s.pic_width_in_luma_samples,
        s.pic_height_in_luma_samples,
        1u32 << s.ctb_log2_size(),
        u32::from(s.bit_depth_luma_minus8) + 8,
        s.dpb_size(),
        s.highest_ordering().max_num_reorder_pics,
        flag(tools.sps_lists),
        flag(tools.pps_lists),
        flag(s.scaling_list_enabled_flag),
        fmt(&tools.tiles),
        flag(tools.wpp),
        flag(tools.dependent),
        flag(tools.weighted_p),
        flag(tools.weighted_b),
        flag(tools.list_modification),
        flag(tools.long_term_present),
        flag(tools.sign_hiding),
        flag(tools.transform_skip),
        flag(tools.transquant_bypass),
        flag(tools.pcm),
        segments_per_picture.len(),
        fmt(&kinds),
        hidden,
        sub_layers,
        segments_per_picture.iter().min().unwrap_or(&0),
        segments_per_picture.iter().max().unwrap_or(&0),
        dependent,
        entry_points,
        fmt(&slice_types),
        tools.sps_sets,
        own_sets,
        long_term,
        used_long_term,
        modifications,
        weights,
        flag(s.vui.as_ref().is_some_and(|v| v.field_seq_flag)),
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

/// A VCL NAL unit type's name (Table 7-1).
fn unit_type_name(unit_type: u8) -> String {
    let name = match unit_type {
        0 => "TRAIL_N",
        1 => "TRAIL_R",
        2 => "TSA_N",
        3 => "TSA_R",
        4 => "STSA_N",
        5 => "STSA_R",
        6 => "RADL_N",
        7 => "RADL_R",
        8 => "RASL_N",
        9 => "RASL_R",
        16 => "BLA_W_LP",
        17 => "BLA_W_RADL",
        18 => "BLA_N_LP",
        19 => "IDR_W_RADL",
        20 => "IDR_N_LP",
        21 => "CRA_NUT",
        other => return format!("type-{other}"),
    };
    name.into()
}

/// `key:count,...`, or `none`.
fn fmt<K: std::fmt::Display>(counts: &BTreeMap<K, usize>) -> String {
    if counts.is_empty() {
        return "none".into();
    }
    counts
        .iter()
        .map(|(k, v)| format!("{k}:{v}"))
        .collect::<Vec<_>>()
        .join(",")
}
