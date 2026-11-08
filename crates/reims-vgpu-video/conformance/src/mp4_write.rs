//! Just enough of the ISO base media file format (ISO/IEC 14496-12) to
//! write one video track of encoded samples, so that a reader such as
//! ffmpeg sees the configuration record, the decode and presentation
//! times, and the sync samples the encoder gave them.

/// One sample, as the track stores it.
pub struct Sample {
    /// Its bytes, as the configuration record's NAL unit lengths frame
    /// them.
    pub data: Vec<u8>,
    /// Its presentation time, in the track's timescale.
    pub pts: i64,
    /// Its decode time.
    pub dts: i64,
    /// Its duration.
    pub duration: i64,
    /// Whether a decoder can start at it.
    pub sync: bool,
}

fn put_box(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    let size = u32::try_from(body.len() + 8).unwrap_or(u32::MAX);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
}

fn full_box(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(body.len() + 12);
    let mut inner = Vec::with_capacity(body.len() + 4);
    inner.extend_from_slice(&((u32::from(version) << 24) | flags).to_be_bytes());
    inner.extend_from_slice(body);
    put_box(&mut b, kind, &inner);
    b
}

fn plain_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(body.len() + 8);
    put_box(&mut b, kind, body);
    b
}

fn u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

/// The whole file: `ftyp`, `mdat`, then `moov`. `entry` is the sample
/// entry's type (`avc1` or `hvc1`) and `record_box` its configuration box
/// (`avcC` or `hvcC`).
pub fn write(
    entry: &[u8; 4],
    record_box: &[u8; 4],
    record: &[u8],
    width: u32,
    height: u32,
    timescale: u32,
    samples: &[Sample],
) -> Vec<u8> {
    let mut out = Vec::new();
    put_box(
        &mut out,
        b"ftyp",
        &[b"isom".as_slice(), &0x200u32.to_be_bytes(), b"isomiso2mp41"].concat(),
    );
    let mdat_body: usize = samples.iter().map(|s| s.data.len()).sum();
    out.extend_from_slice(&u32::try_from(mdat_body + 8).unwrap_or(0).to_be_bytes());
    out.extend_from_slice(b"mdat");
    let mut offsets = Vec::with_capacity(samples.len());
    for s in samples {
        offsets.push(u32::try_from(out.len()).unwrap_or(0));
        out.extend_from_slice(&s.data);
    }
    let duration: i64 = samples.iter().map(|s| s.duration.max(0)).sum();
    let duration = u32::try_from(duration).unwrap_or(0);
    let (w16, h16) = (
        u16::try_from(width).unwrap_or(0),
        u16::try_from(height).unwrap_or(0),
    );

    // stts: each sample's decode duration, the gap to the next DTS.
    let mut deltas: Vec<u32> = samples
        .windows(2)
        .map(|w| u32::try_from(w[1].dts - w[0].dts).unwrap_or(0))
        .collect();
    if let Some(last) = samples.last() {
        deltas.push(u32::try_from(last.duration.max(1)).unwrap_or(1));
    }
    let mut stts = Vec::new();
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for d in deltas {
        match runs.last_mut() {
            Some((count, delta)) if *delta == d => *count += 1,
            _ => runs.push((1, d)),
        }
    }
    stts.extend_from_slice(&u32::try_from(runs.len()).unwrap_or(0).to_be_bytes());
    for (count, delta) in runs {
        stts.extend_from_slice(&u32s(&[count, delta]));
    }
    // ctts: PTS less DTS. A B picture's DTS is after its PTS where the
    // encoder gives Apple's decode times, so its offset is negative, which
    // takes the box's version 1, whose offsets are signed (ISO/IEC
    // 14496-12, 8.6.1.3).
    let signed = samples.iter().any(|s| s.pts < s.dts);
    let mut ctts = u32s(&[u32::try_from(samples.len()).unwrap_or(0)]);
    for s in samples {
        let offset = s.pts.saturating_sub(s.dts);
        let offset = if signed {
            i32::try_from(offset).unwrap_or(0).cast_unsigned()
        } else {
            u32::try_from(offset).unwrap_or(0)
        };
        ctts.extend_from_slice(&u32s(&[1, offset]));
    }
    let syncs: Vec<u32> = samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.sync)
        .map(|(i, _)| u32::try_from(i + 1).unwrap_or(0))
        .collect();
    let mut stss = u32s(&[u32::try_from(syncs.len()).unwrap_or(0)]);
    stss.extend_from_slice(&u32s(&syncs));
    let mut stsz = u32s(&[0, u32::try_from(samples.len()).unwrap_or(0)]);
    for s in samples {
        stsz.extend_from_slice(&u32s(&[u32::try_from(s.data.len()).unwrap_or(0)]));
    }
    let mut stco = u32s(&[u32::try_from(offsets.len()).unwrap_or(0)]);
    stco.extend_from_slice(&u32s(&offsets));
    let stsc = u32s(&[1, 1, 1, 1]);

    let mut sample_entry = Vec::new();
    sample_entry.extend_from_slice(&[0; 6]);
    sample_entry.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    sample_entry.extend_from_slice(&[0; 16]);
    sample_entry.extend_from_slice(&w16.to_be_bytes());
    sample_entry.extend_from_slice(&h16.to_be_bytes());
    sample_entry.extend_from_slice(&u32s(&[0x0048_0000, 0x0048_0000, 0]));
    sample_entry.extend_from_slice(&1u16.to_be_bytes()); // frame_count
    sample_entry.extend_from_slice(&[0; 32]); // compressorname
    sample_entry.extend_from_slice(&0x18u16.to_be_bytes());
    sample_entry.extend_from_slice(&(-1i16).to_be_bytes());
    sample_entry.extend_from_slice(&plain_box(record_box, record));
    let stsd = full_box(
        b"stsd",
        0,
        0,
        &[u32s(&[1]), plain_box(entry, &sample_entry)].concat(),
    );
    let stbl = plain_box(
        b"stbl",
        &[
            stsd,
            full_box(b"stts", 0, 0, &stts),
            full_box(b"ctts", u8::from(signed), 0, &ctts),
            full_box(b"stss", 0, 0, &stss),
            full_box(b"stsc", 0, 0, &stsc),
            full_box(b"stsz", 0, 0, &stsz),
            full_box(b"stco", 0, 0, &stco),
        ]
        .concat(),
    );
    let dref = full_box(
        b"dref",
        0,
        0,
        &[u32s(&[1]), full_box(b"url ", 0, 1, &[])].concat(),
    );
    let minf = plain_box(
        b"minf",
        &[
            full_box(b"vmhd", 0, 1, &[0; 8]),
            plain_box(b"dinf", &dref),
            stbl,
        ]
        .concat(),
    );
    let mdhd = full_box(
        b"mdhd",
        0,
        0,
        &[u32s(&[0, 0, timescale, duration]), vec![0x55, 0xc4, 0, 0]].concat(),
    );
    let hdlr = full_box(
        b"hdlr",
        0,
        0,
        &[
            u32s(&[0]),
            b"vide".to_vec(),
            vec![0; 12],
            b"video\0".to_vec(),
        ]
        .concat(),
    );
    let mdia = plain_box(b"mdia", &[mdhd, hdlr, minf].concat());
    let matrix = u32s(&[0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000]);
    let tkhd = full_box(
        b"tkhd",
        0,
        3,
        &[
            u32s(&[0, 0, 1, 0, duration, 0, 0]),
            vec![0; 4], // layer, alternate_group
            vec![0; 4], // volume, reserved
            matrix.clone(),
            u32s(&[width << 16, height << 16]),
        ]
        .concat(),
    );
    let trak = plain_box(b"trak", &[tkhd, mdia].concat());
    let mvhd = full_box(
        b"mvhd",
        0,
        0,
        &[
            u32s(&[0, 0, timescale, duration, 0x0001_0000]),
            vec![0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            matrix,
            vec![0; 24],
            u32s(&[2]),
        ]
        .concat(),
    );
    put_box(&mut out, b"moov", &[mvhd, trak].concat());
    out
}
