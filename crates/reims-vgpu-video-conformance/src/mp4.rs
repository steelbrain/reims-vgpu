//! Just enough of the ISO base media file format (ISO/IEC 14496-12) to
//! take an H.264 or HEVC track's decoder configuration record (`avcC`,
//! `hvcC`) and its samples out of an MP4, the way
//! VideoToolbox would be handed them: the record from the sample
//! description, each sample as stored.

/// One track: its decoder configuration record and where its samples are.
pub struct Track {
    /// The decoder configuration record: `avcC` or `hvcC`.
    pub avcc: Vec<u8>,
    /// Each sample's offset in the file and size, in decode order.
    pub samples: Vec<(usize, usize)>,
    /// The sample entry's width and height, which CoreMedia's format
    /// description takes.
    pub size: (u32, u32),
}

fn boxes(mut data: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8])> {
    core::iter::from_fn(move || {
        if data.len() < 8 {
            return None;
        }
        let size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let kind = [data[4], data[5], data[6], data[7]];
        let (header, size) = match size {
            0 => (8, data.len()),
            1 => {
                let large = u64::from_be_bytes(data.get(8..16)?.try_into().ok()?);
                (16, usize::try_from(large).ok()?)
            }
            n => (8, n),
        };
        if size < header || size > data.len() {
            return None;
        }
        let body = &data[header..size];
        data = &data[size..];
        Some((kind, body))
    })
}

fn child<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(data).find(|(k, _)| k == kind).map(|(_, b)| b)
}

fn be32(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

fn be64(data: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(data.get(at..at + 8)?.try_into().ok()?))
}

/// The first track whose sample description is `avc1` or `avc3`.
pub fn h264_track(file: &[u8]) -> Result<Track, String> {
    track(file, &[b"avc1", b"avc3"], b"avcC").map_err(|e| e.replace("{}", "H.264"))
}

/// The first track whose sample description is `hvc1` or `hev1`, its
/// `hvcC` record in the place of `avcC`.
pub fn hevc_track(file: &[u8]) -> Result<Track, String> {
    track(file, &[b"hvc1", b"hev1"], b"hvcC").map_err(|e| e.replace("{}", "HEVC"))
}

/// The first track whose sample description is one of `kinds`, with its
/// `record` box.
fn track(file: &[u8], kinds: &[&[u8; 4]], record: &[u8; 4]) -> Result<Track, String> {
    let moov = child(file, b"moov").ok_or("no moov box")?;
    for (kind, trak) in boxes(moov) {
        if &kind != b"trak" {
            continue;
        }
        let Some(stbl) = child(trak, b"mdia")
            .and_then(|m| child(m, b"minf"))
            .and_then(|m| child(m, b"stbl"))
        else {
            continue;
        };
        let Some(stsd) = child(stbl, b"stsd") else {
            continue;
        };
        // FullBox header and entry_count, then the entries.
        let Some((entry_kind, entry)) = stsd.get(8..).and_then(|e| boxes(e).next()) else {
            continue;
        };
        if !kinds.contains(&&entry_kind) {
            continue;
        }
        // VisualSampleEntry's fixed fields are 78 bytes; boxes follow.
        let avcc = entry
            .get(78..)
            .and_then(|e| child(e, record))
            .ok_or_else(|| format!("no {} box", String::from_utf8_lossy(record)))?
            .to_vec();
        let samples = sample_table(stbl).ok_or("bad sample table")?;
        // VisualSampleEntry: 8 bytes of SampleEntry, 16 reserved, then the
        // width and height, 16 bits each.
        let field = |at: usize| {
            entry
                .get(at..at + 2)
                .map(|b| u32::from(u16::from_be_bytes([b[0], b[1]])))
        };
        let size = (field(24).unwrap_or(0), field(26).unwrap_or(0));
        return Ok(Track {
            avcc,
            samples,
            size,
        });
    }
    Err("no {} track".into())
}

fn sample_table(stbl: &[u8]) -> Option<Vec<(usize, usize)>> {
    let stsz = child(stbl, b"stsz")?;
    let uniform = be32(stsz, 4)?;
    let count = be32(stsz, 8)? as usize;
    let sizes: Vec<usize> = (0..count)
        .map(|i| {
            if uniform != 0 {
                Some(uniform as usize)
            } else {
                be32(stsz, 12 + 4 * i).map(|s| s as usize)
            }
        })
        .collect::<Option<_>>()?;
    let chunks: Vec<u64> = if let Some(stco) = child(stbl, b"stco") {
        let n = be32(stco, 4)? as usize;
        (0..n)
            .map(|i| be32(stco, 8 + 4 * i).map(u64::from))
            .collect::<Option<_>>()?
    } else {
        let co64 = child(stbl, b"co64")?;
        let n = be32(co64, 4)? as usize;
        (0..n)
            .map(|i| be64(co64, 8 + 8 * i))
            .collect::<Option<_>>()?
    };
    let stsc = child(stbl, b"stsc")?;
    let runs = be32(stsc, 4)? as usize;
    let mut table = Vec::with_capacity(runs);
    for i in 0..runs {
        let at = 8 + 12 * i;
        table.push((be32(stsc, at)? as usize, be32(stsc, at + 4)? as usize));
    }
    let mut samples = Vec::with_capacity(count);
    let mut next = 0usize;
    for (chunk_index, &offset) in chunks.iter().enumerate() {
        let chunk = chunk_index + 1;
        let per_chunk = table
            .iter()
            .rev()
            .find(|&&(first, _)| first <= chunk)
            .map(|&(_, n)| n)?;
        let mut at = usize::try_from(offset).ok()?;
        for _ in 0..per_chunk {
            let size = *sizes.get(next)?;
            samples.push((at, size));
            at += size;
            next += 1;
        }
    }
    (next == count).then_some(samples)
}
