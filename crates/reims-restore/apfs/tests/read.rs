//! Reading containers ferrosys formats: a System volume found among a volume group, files by
//! path through symbolic links and folded case, every compressed layout, a GPT disk, and the
//! refusals.

use std::sync::Arc;

use reims_apfs::ondisk::apfs;
use reims_apfs::{Error, open};

mod common;
use common::*;

#[test]
fn reads_files_by_path_from_the_system_volume() {
    let image = container();
    let c = open(source(image.clone())).unwrap();
    assert_eq!(c.base(), 0);
    let v = c.system_volume().unwrap();
    assert_eq!(
        (v.index, v.name(), v.role()),
        (1, "Macintosh HD", apfs::ROLE_SYSTEM)
    );
    let want = noise(300_000, 1);
    assert_eq!(v.read_file(BOOTER).unwrap(), want);
    // Through the framework's links, an absolute link, `..`, and folded case.
    assert_eq!(
        v.read_file(&format!("{FRAMEWORK}/Resources/AVPBooter.vmapple2.bin"))
            .unwrap(),
        want
    );
    assert_eq!(
        v.read_file(
            "/abs/Frameworks/Virtualization.framework/Versions/A/Resources/AVPBooter.vmapple2.bin"
        )
        .unwrap(),
        want
    );
    assert_eq!(v.read_file(&format!("/c/../{BOOTER}")).unwrap(), want);
    assert_eq!(v.read_file(&BOOTER.to_uppercase()).unwrap(), want);
    // Its extents, read from the source, are its content.
    let e = v.lookup(BOOTER).unwrap();
    assert!(e.is_file() && !e.is_compressed());
    let mut got = Vec::new();
    for x in v.extents(&e).unwrap() {
        assert_eq!(x.logical, got.len() as u64);
        got.extend_from_slice(&image.get(x.offset.unwrap(), x.len as usize));
    }
    assert_eq!(got, want);
    assert_eq!(v.read_file("/empty").unwrap(), b"");
    // The other volume.
    let d = c.volume(0).unwrap();
    assert_eq!(d.read_file("/marker").unwrap(), b"data");
    assert!(d.lookup("/System").is_err());
}

#[test]
fn finds_every_name_of_a_directory_spanning_many_nodes() {
    let c = open(source(container())).unwrap();
    let v = c.system_volume().unwrap();
    for i in [0, 1, 777, 2500, MANY - 2, MANY - 1] {
        assert_eq!(
            v.read_file(&format!("/many/f{i:05}")).unwrap(),
            format!("{i}").into_bytes(),
            "f{i:05}"
        );
    }
    assert!(matches!(
        v.lookup(&format!("/many/f{MANY:05}")),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn decodes_every_compressed_layout() {
    let c = open(source(container())).unwrap();
    let v = c.system_volume().unwrap();
    for (path, content, _) in compressed() {
        let e = v.lookup(path).unwrap();
        assert!(e.is_compressed(), "{path}");
        assert_eq!(v.read_file(path).unwrap(), content, "{path}");
    }
}

#[test]
fn opens_the_apfs_partition_of_a_gpt_disk() {
    let ss = 512u64;
    let first = 2048u64;
    let mut disk = Sparse::new((first + CONTAINER / ss + 2048) * ss);
    let mut g = gptman::GPT::new_from(&mut disk, ss, [0x44; 16]).unwrap();
    g[1] = gptman::GPTPartitionEntry {
        // 7C3457EF-0000-11AA-AA11-00306543ECAC, on disk.
        partition_type_guid: [
            0xef, 0x57, 0x34, 0x7c, 0, 0, 0xaa, 0x11, 0xaa, 0x11, 0x00, 0x30, 0x65, 0x43, 0xec,
            0xac,
        ],
        unique_partition_guid: [0x55; 16],
        starting_lba: first,
        ending_lba: first + CONTAINER / ss - 1,
        attribute_bits: 0,
        partition_name: "Container".into(),
    };
    gptman::GPT::write_protective_mbr_into(&mut disk, ss).unwrap();
    g.write_into(&mut disk).unwrap();
    disk.base = first * ss;
    format_into(&mut disk);
    disk.base = 0;
    let c = open(source(disk)).unwrap();
    assert_eq!(c.base(), first * ss);
    let v = c.system_volume().unwrap();
    assert_eq!(v.read_file(BOOTER).unwrap(), noise(300_000, 1));
    let e = v.lookup(BOOTER).unwrap();
    assert!(
        v.extents(&e)
            .unwrap()
            .iter()
            .all(|x| x.offset.unwrap() >= first * ss)
    );
}

#[test]
fn refuses_what_is_not_there_or_does_not_check() {
    let image = container();
    let c = open(source(image.clone())).unwrap();
    let v = c.system_volume().unwrap();
    assert!(matches!(
        v.lookup("/System/nothing"),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        v.lookup(&format!("{BOOTER}/below")),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(v.read_file("/System"), Err(Error::Invalid(_))));
    assert!(matches!(v.lookup("/loop"), Err(Error::Invalid(_))));
    assert!(open(Arc::new(vec![0u8; 1 << 20])).is_err());
    // Every file-system tree root with one bit flipped: the lookup's first read fails its
    // checksum.
    let mut bad = image;
    for page in bad.pages.values_mut() {
        let kind = u32::from_le_bytes(page[24..28].try_into().unwrap()) & 0xffff;
        let subtype = u32::from_le_bytes(page[28..32].try_into().unwrap());
        if kind == 2 && subtype == 0x0e {
            page[PAGE as usize - 100] ^= 1;
        }
    }
    let c = open(source(bad)).unwrap();
    let v = c.system_volume().unwrap();
    assert!(
        matches!(v.lookup(BOOTER), Err(Error::Checksum { .. })),
        "{:?}",
        v.lookup(BOOTER)
    );
}
