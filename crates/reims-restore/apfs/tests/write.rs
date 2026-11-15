//! Writing containers and reading them back: a volume copied record for record from a System
//! volume ferrosys formats (every compressed layout, resource forks among them, links and many
//! files), with entries replaced by links and a file added, as a restore makes the Data volume.

use std::io::{Seek, SeekFrom, Write};

use ferrosys::EntryKind;
use reims_apfs::fs::{self, key_header};
use reims_apfs::ondisk::{apfs, rec};
use reims_apfs::write::{self, VolumeDef, copy, fsbuild};

mod common;
use common::*;

/// The copy's container: 128000 blocks, the iBoot system container's size.
const BLOCKS: u64 = 128_000;
const T: u64 = 1_793_000_000_000_000_000;
const ADDED: &[u8] = b"an added file\n";

/// Copy `image`'s System volume into a Data volume, `/many` and `/empty` made links and
/// `System/Library/added.txt` added, and write it into a sparse disk.
fn copied(image: &Sparse) -> Sparse {
    let c = reims_apfs::open(source(image.clone())).expect("open the source");
    let sys = c.system_volume().expect("the source's System volume");
    let records = sys.fs_tree_records().expect("read its records");
    let root = sys.lookup("/").expect("its root").id;
    let added = [fsbuild::Item {
        path: "System/Library/added.txt".into(),
        kind: fsbuild::Kind::File {
            len: ADDED.len() as u64,
        },
        mode: 0o644,
        uid: 0,
        gid: 0,
        bsd_flags: 0,
        xattrs: Vec::new(),
    }];
    let out = copy::copy_tree(
        &records,
        root,
        true,
        T,
        &mut |p, is_dir| match (p, is_dir) {
            ("many", true) => copy::Replace::Link(b"../elsewhere/many/".to_vec()),
            ("empty", false) => copy::Replace::Link(b"gone".to_vec()),
            _ => copy::Replace::Keep,
        },
        &added,
    )
    .expect("copy the tree");
    let def = VolumeDef {
        name: "Data",
        uuid: [0x44; 16],
        role: apfs::ROLE_DATA,
        system: false,
        group: [0; 16],
        records: out.records,
        streams: out.streams.clone(),
        counts: out.counts,
        next_obj_id: out.next_obj_id,
        seal: None,
    };
    let formed = write::container(BLOCKS, [0x55; 16], T, vec![def]).expect("form the container");
    let mut disk = Sparse::new(BLOCKS * 4096);
    for (addr, b) in &formed.blocks {
        disk.seek(SeekFrom::Start(addr * 4096)).unwrap();
        disk.write_all(b).unwrap();
    }
    for ((first, src), st) in formed.stream_blocks[0]
        .iter()
        .zip(&out.sources)
        .zip(&out.streams)
    {
        let bytes = match src {
            copy::StreamSource::Runs(runs) => {
                let mut b = Vec::new();
                for r in runs {
                    match r.phys {
                        Some(p) => b.extend(image.get(p * 4096, r.len as usize)),
                        None => b.resize(b.len() + r.len as usize, 0),
                    }
                }
                b
            }
            copy::StreamSource::Added(_) => ADDED.to_vec(),
        };
        assert_eq!(bytes.len() as u64, st.len, "a stream's length");
        disk.seek(SeekFrom::Start(first * 4096)).unwrap();
        disk.write_all(&bytes).unwrap();
    }
    disk
}

#[test]
fn a_copied_tree_reads_back_as_its_source() {
    let image = container();
    let disk = copied(&image);

    let src = reims_apfs::open(source(image)).unwrap();
    let sys = src.system_volume().unwrap();
    let c = reims_apfs::open(source(disk)).expect("open the copy");
    let vols = c.volumes().unwrap();
    let data = &vols[0];

    // Every file the source holds reads the same from the copy, compressed ones decoded the same
    // way, but those under or at the paths made links.
    let mut files = 0;
    for e in system_entries() {
        let path = String::from_utf8(e.path.clone()).unwrap();
        if path.starts_with("/many") || path == "/empty" {
            continue;
        }
        if let EntryKind::File(_) = e.kind {
            let want = sys.read_file(&path).unwrap();
            let got = data
                .read_file(&path)
                .unwrap_or_else(|err| panic!("{path}: {err}"));
            assert!(
                got == want,
                "{path}: {} bytes, {} in the source",
                got.len(),
                want.len()
            );
            files += 1;
        }
    }
    assert!(
        files > 5,
        "the compressed files and the booter were compared"
    );
    // The links the source holds are copied: the booter reads through them.
    assert_eq!(
        data.read_file(
            "/System/Library/Frameworks/Virtualization.framework/Resources/AVPBooter.vmapple2.bin"
        )
        .unwrap(),
        sys.read_file(
            "/System/Library/Frameworks/Virtualization.framework/Resources/AVPBooter.vmapple2.bin"
        )
        .unwrap()
    );
    assert_eq!(data.read_file("/System/Library/added.txt").unwrap(), ADDED);

    // `/many` and `/empty` are links to the targets given, and nothing below `/many` was copied.
    let recs = data.fs_tree_records().unwrap();
    let child = |name: &[u8]| {
        recs.iter()
            .find(|(k, _)| key_header(k) == (2, rec::DIR_REC) && fs::drec_name(k) == name)
            .map(|(_, v)| u64::from_le_bytes(v[0..8].try_into().unwrap()))
            .unwrap()
    };
    for (name, target) in [
        (&b"many"[..], &b"../elsewhere/many/"[..]),
        (b"empty", b"gone"),
    ] {
        let id = child(name);
        let inode = recs
            .iter()
            .find(|(k, _)| key_header(k) == (id, rec::INODE))
            .and_then(|(_, v)| fs::Inode::parse(v))
            .unwrap();
        // The link keeps the replaced entry's permissions, owner and flags, but compression.
        let was = sys
            .lookup(&format!("/{}", String::from_utf8_lossy(name)))
            .unwrap()
            .inode;
        assert_eq!(
            inode.mode,
            0o120000 | (was.mode & 0o7777),
            "{}",
            String::from_utf8_lossy(name)
        );
        assert_eq!(
            (inode.owner, inode.group, inode.bsd_flags),
            (was.owner, was.group, was.bsd_flags & !0x20)
        );
        let attr = recs
            .iter()
            .find(|(k, _)| {
                key_header(k) == (id, rec::XATTR) && fs::xattr_name(k) == b"com.apple.fs.symlink"
            })
            .and_then(|(_, v)| fs::parse_xattr(v))
            .unwrap();
        let mut want = target.to_vec();
        want.push(0);
        assert_eq!(attr.1, fs::XattrData::Embedded(want));
    }
    let inodes = recs
        .iter()
        .filter(|(k, _)| key_header(k).1 == rec::INODE)
        .count();
    assert!(
        inodes < MANY,
        "nothing below /many was copied ({inodes} inodes)"
    );
}
