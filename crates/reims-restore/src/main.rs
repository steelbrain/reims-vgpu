//! reims-restore: restores macOS from an IPSW onto an arm64 VM's disk (docs/design.md).

use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use reims_restore::aea::AppleKeyService;
use reims_restore::http::Client;
use reims_restore::image;
use reims_restore::ipsw::Ipsw;
use reims_restore::manifest::{self, BuildManifest};
use reims_restore::source::ReadAt;
use reims_restore::{catalog, download};

#[derive(Parser)]
#[command(
    version,
    about = "Restore macOS from an IPSW onto an arm64 VM's disk, without Apple's tools"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show an IPSW's build, its virtual Mac identities and how its images are stored.
    Inspect {
        /// The IPSW.
        ipsw: PathBuf,
    },
    /// Write one of an IPSW's images out as a raw, sparse file: decrypted, unpacked, and with
    /// `--apfs` only its APFS container.
    Extract {
        /// The IPSW.
        ipsw: PathBuf,
        /// The build identity's component, such as `OS` or `Cryptex1,SystemOS`.
        component: String,
        /// Where to write it; must not exist.
        out: PathBuf,
        /// Write only the image's APFS container.
        #[arg(long)]
        apfs: bool,
    },
    /// Restore an IPSW into a new VM directory: disk.img, aux.img, aux.img.trimmed,
    /// machine-identifier.bin and config.cpio, Ambrosia's bundle of the patches that swap the
    /// guest's LocalPolicy anchors for the project's, for QEMU's vmapple machine on Ambrosia.
    Restore {
        /// The IPSW: a UniversalMac IPSW of macOS 14 or later.
        ipsw: PathBuf,
        /// The VM's directory, made if missing; it must not hold a restore's files.
        dir: PathBuf,
        /// The disk's size in GiB.
        #[arg(long, default_value_t = 64)]
        disk_gib: u64,
        /// The VM's ECID, in hex; random by default.
        #[arg(long, value_parser = parse_hex)]
        ecid: Option<u64>,
    },
    /// After a macOS 14 VM's first boot: macOS 14 makes its Data volume at that boot and resets
    /// the VM, which then doesn't start; this makes it bootable again, for Setup Assistant.
    FinishEacs {
        /// The VM's directory.
        dir: PathBuf,
    },
    /// Download the IPSW of the current macOS from Apple, or find it downloaded, and print its
    /// path. Apple offers only the current release: another is restored from an IPSW you have.
    /// A download that stops is continued by the next.
    Download {
        /// latest, or the current release's version (27.2.1) or build (26B131); also any
        /// release of an IPSW already in the directory.
        version: String,
        /// The directory IPSWs are kept in (default: ipsw in the cache).
        #[arg(long)]
        cache: Option<PathBuf>,
    },
    /// List the IPSW Apple offers to download, and the IPSWs downloaded.
    List {
        /// The directory IPSWs are kept in (default: ipsw in the cache).
        #[arg(long)]
        cache: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect { ipsw } => inspect(&ipsw),
        Command::Extract {
            ipsw,
            component,
            out,
            apfs,
        } => extract(&ipsw, &component, &out, apfs),
        Command::Restore {
            ipsw,
            dir,
            disk_gib,
            ecid,
        } => restore(ipsw, dir, disk_gib, ecid),
        Command::FinishEacs { dir } => {
            reims_restore::pipeline::finish_eacs(&dir)?;
            println!(
                "{} is ready for its next boot, to Setup Assistant",
                dir.display()
            );
            Ok(())
        }
        Command::Download { version, cache } => download(&version, cache),
        Command::List { cache } => list(cache),
    }
}

fn parse_hex(s: &str) -> Result<u64, String> {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|e| e.to_string())
}

fn restore(ipsw: PathBuf, dir: PathBuf, disk_gib: u64, ecid: Option<u64>) -> Result<()> {
    use reims_restore::pipeline::{self, Event};

    let disk_size = disk_gib
        .checked_mul(1 << 30)
        .with_context(|| format!("a disk of {disk_gib} GiB is too large"))?;
    std::fs::create_dir_all(&dir).with_context(|| format!("make {}", dir.display()))?;
    let job = pipeline::Restore {
        ipsw,
        dir,
        disk_size,
        ecid,
    };
    on_interrupt_cancel();
    let mut last = (String::new(), u64::MAX);
    let done = pipeline::run(
        &job,
        &mut |e| match e {
            Event::Step(s) => eprintln!("{s}"),
            Event::Note(s) => eprintln!("  {s}"),
            Event::Progress { what, done, total } => {
                let pct = (done * 100).checked_div(total).unwrap_or(0);
                if last.0 != what || last.1 / 10 != pct / 10 {
                    eprintln!("  {what}: {pct}%");
                    last = (what, pct);
                }
            }
        },
        &CANCEL,
    )?;
    println!(
        "Restored macOS {} ({}) into {}, ECID {:#x}",
        done.version,
        done.build,
        job.dir.display(),
        done.ecid
    );
    if done.first_boot == pipeline::FirstBoot::Eacs {
        println!(
            "macOS {} makes its Data volume at the VM's first boot, then resets the VM, which \
             then doesn't start. Once it has, run `reims-restore finish-eacs {}`; the next boot \
             goes to Setup Assistant.",
            done.version,
            job.dir.display()
        );
    }
    Ok(())
}

/// The cache's default: `$XDG_CACHE_HOME/reims-restore`, else `~/.cache/reims-restore`.
fn default_cache() -> Result<PathBuf> {
    if let Some(x) = std::env::var_os("XDG_CACHE_HOME").filter(|x| !x.is_empty()) {
        return Ok(PathBuf::from(x).join("reims-restore"));
    }
    let home = std::env::var_os("HOME").context("HOME is not set: pass --cache")?;
    Ok(PathBuf::from(home).join(".cache/reims-restore"))
}

/// The directory IPSWs are kept in: `cache`, else the cache's `ipsw`.
fn ipsw_dir(cache: Option<PathBuf>) -> Result<PathBuf> {
    match cache {
        Some(c) => Ok(c),
        None => Ok(default_cache()?.join("ipsw")),
    }
}

/// Apple's catalog's entries for the virtual Mac, or why it couldn't be read.
fn catalog() -> (Vec<catalog::Entry>, Option<anyhow::Error>) {
    match catalog::fetch(&Client::apple(), catalog::URL) {
        Ok(e) => (e, None),
        Err(e) => (Vec::new(), Some(e)),
    }
}

fn download(version: &str, cache: Option<PathBuf>) -> Result<()> {
    use reims_restore::pipeline::Event;
    let dir = ipsw_dir(cache)?;
    let (listed, unread) = catalog();
    let found = catalog::pick(
        version,
        &catalog::on_disk(std::slice::from_ref(&dir)),
        &listed,
    );
    let found = match (found, unread) {
        (Ok(f), _) => f,
        (Err(e), Some(why)) => return Err(e.context(format!("{why:#}"))),
        (Err(e), None) => return Err(e),
    };
    let path = match found {
        catalog::Found::OnDisk(o) => o.path,
        catalog::Found::Download(e) => {
            on_interrupt_cancel();
            let mut last = u64::MAX;
            download::download(
                &Client::apple(),
                &e,
                &dir,
                &mut |ev| match ev {
                    Event::Step(s) => eprintln!("{s}"),
                    Event::Note(s) => eprintln!("  {s}"),
                    Event::Progress { done, total, .. } => {
                        let pct = (done * 100).checked_div(total).unwrap_or(0);
                        if pct != last {
                            eprint!("\r  {pct}% of {:.1} GB", total as f64 / 1e9);
                            last = pct;
                        }
                        if done == total {
                            eprintln!();
                        }
                    }
                },
                &CANCEL,
            )?
        }
    };
    println!("{}", path.display());
    Ok(())
}

fn list(cache: Option<PathBuf>) -> Result<()> {
    let dir = ipsw_dir(cache)?;
    let (listed, unread) = catalog();
    match unread {
        Some(why) => println!("Apple's catalog couldn't be read: {why:#}"),
        None => {
            for e in &listed {
                println!("Apple offers macOS {} ({}): {}", e.version, e.build, e.url);
            }
        }
    }
    let have = catalog::on_disk(std::slice::from_ref(&dir));
    if have.is_empty() {
        println!("No IPSW downloaded in {}", dir.display());
    } else {
        println!("Downloaded, in {}:", dir.display());
        for o in have {
            println!(
                "  macOS {} ({})  {:.1} GB  {}",
                o.version,
                o.build,
                o.size as f64 / 1e9,
                o.path.file_name().unwrap_or_default().to_string_lossy()
            );
        }
    }
    Ok(())
}

/// Set by SIGINT or SIGTERM: a restore stops and its files are removed; a download stops and
/// keeps its part for the next to continue.
static CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The first SIGINT or SIGTERM cancels the restore; a second ends the process at once.
fn on_interrupt_cancel() {
    extern "C" fn handle(_: libc::c_int) {
        if CANCEL.swap(true, std::sync::atomic::Ordering::SeqCst) {
            // SAFETY: _exit is async-signal-safe.
            unsafe { libc::_exit(130) };
        }
    }
    for sig in [libc::SIGINT, libc::SIGTERM] {
        // SAFETY: the handler only touches an atomic and calls _exit, both async-signal-safe.
        unsafe { libc::signal(sig, handle as *const () as libc::sighandler_t) };
    }
}

fn inspect(path: &std::path::Path) -> Result<()> {
    let ipsw = Ipsw::open(path)?;
    let bm = BuildManifest::parse(&ipsw.read("BuildManifest.plist")?)?;
    println!("macOS {} ({})", bm.product_version()?, bm.build_version()?);
    for variant in [manifest::VARIANT_INSTALL, manifest::VARIANT_ERASE] {
        let id = bm.vma2_identity(variant)?;
        println!("{variant}:");
        let mut comps: Vec<_> = id.components().collect();
        comps.sort_by_key(|(n, _)| *n);
        for (name, c) in comps {
            let Some(p) = c.path() else { continue };
            let how = match ipsw.member(p) {
                Ok(m) => format!("{} bytes, {:?}", m.size, m.method),
                Err(_) => "missing".into(),
            };
            println!("  {name:40} {p} ({how})");
        }
    }
    Ok(())
}

fn extract(
    path: &std::path::Path,
    component: &str,
    out: &std::path::Path,
    apfs: bool,
) -> Result<()> {
    let ipsw = Ipsw::open(path)?;
    let bm = BuildManifest::parse(&ipsw.read("BuildManifest.plist")?)?;
    let id = bm.vma2_identity(manifest::VARIANT_INSTALL)?;
    let member = id
        .component(component)?
        .path()
        .context("the component has no path")?
        .to_owned();
    let keys = AppleKeyService::default();
    let (src, layers) = if apfs {
        image::open_apfs(&ipsw, &member, &keys)?
    } else {
        image::open_disk(&ipsw, &member, &keys)?
    };
    eprintln!("{member}: {layers:?}, {} bytes", src.len());
    let mut f = File::options()
        .write(true)
        .create_new(true)
        .open(out)
        .with_context(|| format!("create {}", out.display()))?;
    f.set_len(src.len())?;
    let mut buf = vec![0u8; 8 << 20];
    let mut off = 0u64;
    let mut next = 0u64;
    while off < src.len() {
        let n = buf.len().min((src.len() - off) as usize);
        src.read_exact_at(off, &mut buf[..n])?;
        if buf[..n].iter().any(|&b| b != 0) {
            use std::os::unix::fs::FileExt as _;
            f.write_all_at(&buf[..n], off)?;
        }
        off += n as u64;
        if off >= next {
            eprintln!("{} MiB", off >> 20);
            next += 1 << 30;
        }
    }
    f.flush()?;
    f.sync_all()?;
    Ok(())
}
