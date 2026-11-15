//! Read one file out of an APFS container or a GPT disk image holding one, checking every object
//! on the way: `apfs-cat IMAGE PATH [OUT]`. Without OUT, it prints the file's size, how it is
//! stored and its extents; with OUT, it writes the file's content there. The System volume is
//! read, or the container's only volume.

use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [image, path, rest @ ..] = args.as_slice() else {
        return Err("usage: apfs-cat IMAGE PATH [OUT]".into());
    };
    let c = reims_apfs::open(Arc::new(std::fs::File::open(image)?))?;
    for v in c.volumes()? {
        eprintln!("volume {}: {:?}, role {:#x}", v.index, v.name(), v.role());
    }
    let v = c.system_volume()?;
    let e = v.lookup(path)?;
    let data = v.read(&e)?;
    eprintln!(
        "{path}: inode {}, mode {:o}, {} bytes{}",
        e.id,
        e.inode.mode,
        data.len(),
        if e.is_compressed() {
            ", compressed"
        } else {
            ""
        }
    );
    for x in v.extents(&e)? {
        eprintln!(
            "  extent at {}: {} bytes at {:?}",
            x.logical, x.len, x.offset
        );
    }
    if let [out] = rest {
        std::fs::write(out, &data)?;
    }
    Ok(())
}
