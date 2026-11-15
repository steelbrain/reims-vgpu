//! A whole restore, as reims-ctl and the CLI run it: from an IPSW to a VM's directory holding
//! `disk.img`, `aux.img`, `aux.img.trimmed` and `machine-identifier.bin` (docs/design.md). The
//! restore writes the disk itself, on any host: it reads the IPSW where it lies, has Apple's TSS
//! sign the VM's tickets, and lays out the GPT, the three APFS containers and NVRAM as a fresh
//! restore by Virtualization.framework leaves them. No VM runs, and none of Apple's code.

use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use anyhow::{Context, Result, bail, ensure};
use plist::{Dictionary, Value};
use uuid::Uuid;

use crate::aea::AppleKeyService;
use crate::aux::{self, Nvram};
use crate::gpt::Table;
use crate::ipsw::Ipsw;
use crate::manifest::{BuildManifest, VARIANT_ERASE, VARIANT_INSTALL};
use crate::tss::{self, ApNonces, ApTicket, TssClient};
use crate::{anchors, containers, image, img4, kernelcache, plan, policy};

/// A restore to run.
#[derive(Debug, Clone)]
pub struct Restore {
    /// The IPSW, a local file.
    pub ipsw: PathBuf,
    /// The VM's directory: it exists and holds none of the files a restore writes.
    pub dir: PathBuf,
    /// The disk's size in bytes.
    pub disk_size: u64,
    /// The VM's ECID; random when `None`.
    pub ecid: Option<u64>,
    /// The directory of the project's LocalPolicy keys (`root.key`, `int.key`, `leaf.key`):
    /// with them the restore swaps the guest's trust anchors for the project's chain and writes
    /// Ambrosia's bundle of the patches, [`CONFIG`] ([`crate::anchors`]); without them it signs
    /// with the built-in chain and writes no bundle.
    pub lp_keys: Option<PathBuf>,
}

/// What a restore reports as it goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// A step began: a short sentence.
    Step(String),
    /// Progress through a long step.
    Progress {
        /// What is progressing.
        what: String,
        /// How far it is.
        done: u64,
        /// Where it ends.
        total: u64,
    },
    /// A line for the log.
    Note(String),
}

/// A finished restore.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Restored {
    /// The VM's ECID.
    pub ecid: u64,
    /// The macOS version restored, such as `15.6.1`.
    pub version: String,
    /// Its build, such as `24G90`.
    pub build: String,
    /// What the VM's first boot does.
    pub first_boot: FirstBoot,
}

/// What a restored VM's first boot does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstBoot {
    /// It goes to Setup Assistant.
    SetupAssistant,
    /// macOS 14 ([`data_by_eacs`]): it makes the Data volume by EACS and resets the VM, and then
    /// Ambrosia finds no LocalPolicy and stops. [`finish_eacs`] on the VM's directory then makes
    /// it bootable, and the next boot goes to Setup Assistant.
    Eacs,
}

/// Whether macOS `version`'s first boot makes its Data volume itself, by EACS, with the restore
/// leaving it out: macOS 14. Its kernel refuses an unencrypted Data volume ("unencrypted data
/// volume is not allowed") **[measured]**, and Virtualization.framework's 14.6.1 restore leaves
/// Data encrypted where those of 15.6.1 and later leave it unencrypted **[vz]**; only the guest's
/// keystore can make an encrypted one (docs/design.md, "macOS 14's Data volume").
pub fn data_by_eacs(version: &str) -> bool {
    version.split('.').next() == Some("14")
}

/// The files a restore leaves in the VM's directory.
pub const DISK: &str = "disk.img";
/// The auxiliary storage in Virtualization.framework's form, with its 16 KiB header.
pub const AUX: &str = "aux.img";
/// The auxiliary storage as QEMU attaches it.
pub const AUX_TRIMMED: &str = "aux.img.trimmed";
/// The ECID in Virtualization.framework's form, a binary plist `{ECID: n}`.
pub const MACHINE_IDENTIFIER: &str = "machine-identifier.bin";
/// Ambrosia's configuration bundle of the patches that swap the guest's LocalPolicy anchors for
/// the project's, where the restore was given the project's keys.
pub const CONFIG: &str = "config.cpio";
/// For [`FirstBoot::Eacs`], until [`finish_eacs`]: the iBoot system container's partition as the
/// restore wrote it, sparse.
pub const EACS_ISC: &str = "isc.pre-eacs";
/// For [`FirstBoot::Eacs`], until [`finish_eacs`]: the auxiliary storage as the restore wrote it.
pub const EACS_AUX: &str = "aux.pre-eacs";

/// The auxiliary storage's size as QEMU attaches it.
const AUX_SIZE: u64 = aux::SIZE as u64;

/// The smallest disk a restore makes. A restored system takes 25.1 GiB of 27.2.1's 64 GiB disk
/// (Virtualization.framework) **[measured]**; below 32 GiB the system has little room to run in.
pub const MIN_DISK: u64 = 32 << 30;

/// The values a restore draws at random, which bind the VM's tickets and policies to it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Secrets {
    /// The VM's ECID.
    pub ecid: u64,
    /// The AP nonce the boot manifest and the Cryptex1 ticket carry (`BNCH`, `cnch`).
    pub nonces: ApNonces,
    /// NVRAM's policy nonces and the disk's GUIDs.
    pub nvram: Nvram,
}

/// Runs a restore. `events` hears its steps; setting `cancel` stops it between steps. On
/// success the VM's files are in `r.dir`; otherwise none of them is.
pub fn run(r: &Restore, events: &mut dyn FnMut(Event), cancel: &AtomicBool) -> Result<Restored> {
    let step = |events: &mut dyn FnMut(Event), s: &str| events(Event::Step(s.to_owned()));

    step(events, "Reading the IPSW");
    let ipsw = Ipsw::open(&r.ipsw)?;
    let bm = BuildManifest::parse(&ipsw.read("BuildManifest.plist")?)
        .with_context(|| format!("read {}'s BuildManifest.plist", r.ipsw.display()))?;
    let version = bm.product_version()?.to_owned();
    let build = bm.build_version()?.to_owned();
    let (install, erase) = match (
        bm.vma2_identity(VARIANT_INSTALL),
        bm.vma2_identity(VARIANT_ERASE),
    ) {
        (Ok(i), Ok(e)) => (i, e),
        _ => bail!(
            "{} (macOS {version}, {build}) has no virtual Mac build identity: a restore needs a \
             UniversalMac IPSW of macOS 14 or later",
            r.ipsw.display()
        ),
    };
    events(Event::Note(format!("macOS {version} ({build})")));
    let eacs = data_by_eacs(&version);
    if eacs {
        events(Event::Note(
            "macOS 14 makes its Data volume at its first boot: the restore leaves it out".into(),
        ));
    }

    check_dir(&r.dir)?;
    ensure!(
        r.disk_size >= MIN_DISK,
        "a {} disk is too small for macOS: give it at least {}",
        gib(r.disk_size),
        gib(MIN_DISK)
    );
    let table = Table::layout(r.disk_size)?;
    let secrets = Secrets::generate(r.ecid, &table)?;
    cancelled(cancel)?;

    step(events, "Reading the system image's data template");
    let keys = AppleKeyService::default();
    let os = install
        .component("OS")?
        .path()
        .context("the manifest's OS has no path")?
        .to_owned();
    let container = image::open_container(&ipsw, &os, &keys)?;
    let system = container.system_volume()?;
    let mut images = 0;
    for m in plan::copied_images(&install)? {
        images += ipsw.member(&m)?.size;
    }
    let need = space_needed(
        system.sb.alloc_count * u64::from(container.block_size()),
        images,
    );
    check_space(&r.dir, free_space(&r.dir)?, need)?;
    let patterns = system
        .read_file(DATA_PATTERNS)
        .with_context(|| format!("read {DATA_PATTERNS} from {os}"))?;
    let receipt = plan::receipt(
        &version,
        &build,
        &String::from_utf8_lossy(&patterns),
        SystemTime::now(),
    );
    cancelled(cancel)?;

    step(events, "Asking Apple to sign the VM's tickets");
    let tss = TssClient::new();
    let boot_manifest: Arc<[u8]> = tss
        .ap_ticket(&install, ApTicket::Boot, secrets.ecid, &secrets.nonces)?
        .into();
    let sfr_manifest: Arc<[u8]> = tss
        .ap_ticket(&erase, ApTicket::Sfr, secrets.ecid, &secrets.nonces)?
        .into();
    let cryptex1: Arc<[u8]> = tss
        .cryptex1_ticket(&install, secrets.ecid, &secrets.nonces.ap_nonce)?
        .into();
    let ronh = img4::sha384(secrets.nvram.recovery_policy_nonce());
    let mut args = Dictionary::new();
    args.insert(
        "Ap,NextStageIM4MHash".into(),
        Value::Data(img4::sha384(&sfr_manifest).to_vec()),
    );
    args.insert(
        "Ap,RecoveryOSPolicyNonceHash".into(),
        Value::Data(ronh.to_vec()),
    );
    args.insert(
        "Ap,VolumeUUID".into(),
        Value::String(plan::RECOVERY_GROUP.to_string().to_uppercase()),
    );
    let recovery_os_policy = img4::stitch_img4(
        &img4::LPOL_IM4P,
        &tss.local_policy_ticket(&erase, &args, secrets.ecid)?,
    );
    cancelled(cancel)?;

    step(events, "Signing the VM's LocalPolicies");
    let (signer, bundle) = match &r.lp_keys {
        Some(dir) => {
            let kernel = install
                .component("KernelCache")?
                .path()
                .context("the manifest's KernelCache has no path")?;
            let kc = kernelcache::decompress(&ipsw.read(kernel)?)
                .with_context(|| format!("read {kernel}"))?;
            let guest = anchors::guest(&kernelcache::FileSet::parse(&kc)?)
                .with_context(|| format!("find the LocalPolicy anchors in {kernel}"))?;
            drop(kc);
            let keys = anchors::Keys::load(dir)?;
            let chain = anchors::chain(&guest, &keys)?;
            let signer = policy::Signer::with_key(keys.leaf, &[&chain.intermediate, &chain.leaf])?;
            let bundle = anchors::config_bundle(&anchors::config(&guest, &chain))?;
            (signer, Some(bundle))
        }
        None => {
            events(Event::Note(
                "no LocalPolicy keys given: the policies carry the built-in chain, and no \
                 anchor bundle is written"
                    .into(),
            ));
            (policy::Signer::project(), None)
        }
    };
    let boot = img4::parse_im4m(&boot_manifest).context("read the boot manifest")?;
    let prop = |k: &str| {
        boot.props
            .get(k)
            .with_context(|| format!("the boot manifest has no {k}"))
    };
    let binding = policy::Binding {
        ecid: secrets.ecid,
        chip: prop("CHIP")?.as_int().context("CHIP")?,
        board: prop("BORD")?.as_int().context("BORD")?,
        security_domain: prop("SDOM")?.as_int().context("SDOM")?,
        love: prop("love")?.as_data().context("love")?.to_vec(),
        boot_manifest_hash: img4::sha384(&boot_manifest),
        cryptex1_ticket_hash: img4::sha384(&cryptex1),
        local_nonce: *secrets.nvram.local_policy_nonce(),
        remote_nonce: *secrets.nvram.remote_policy_nonce(),
        group: secrets.nvram.group,
    };
    let names: Vec<String> = ipsw.names().map(str::to_owned).collect();
    let has = |n: &str| ipsw.has(n);
    let the_plan = plan::plan(&plan::Inputs {
        manifest: &bm,
        has: &has,
        names: &names,
        install,
        erase,
        ecid: secrets.ecid,
        group: secrets.nvram.group,
        boot_manifest,
        sfr_manifest: sfr_manifest.clone(),
        cryptex1_ticket: cryptex1,
        policy: (
            binding.file_name(policy::Kind::MacOs),
            binding.sign(policy::Kind::MacOs, &signer).into(),
        ),
        policy_recovery: (
            binding.file_name(policy::Kind::Recovery),
            binding.sign(policy::Kind::Recovery, &signer).into(),
        ),
        recovery_os_policy: recovery_os_policy.into(),
        ronh: ronh.iter().map(|b| format!("{b:02X}")).collect(),
        receipt,
        eacs_store: eacs.then_some(table.main().guid),
    })?;
    cancelled(cancel)?;

    let aux = aux::trimmed(&secrets.nvram, &sfr_manifest)?;
    step(events, "Making the VM's disks");
    let files = Files::new(&r.dir);
    if let Err(e) = files.make(r.disk_size) {
        files.discard();
        return Err(e);
    }
    let src = containers::Sources {
        ipsw: &ipsw,
        keys: &keys,
        root_hash: None,
        t: nanos(SystemTime::now()),
    };
    let written = write(&files, &table, &aux, &the_plan, src, events, cancel).and_then(|()| {
        if eacs {
            files.keep_for_eacs(&table)
        } else {
            Ok(())
        }
    });
    match written.and_then(|()| files.finish(secrets.ecid, bundle.as_deref(), eacs)) {
        Ok(()) => Ok(Restored {
            ecid: secrets.ecid,
            version,
            build,
            first_boot: if eacs {
                FirstBoot::Eacs
            } else {
                FirstBoot::SetupAssistant
            },
        }),
        Err(e) => {
            files.discard();
            Err(e)
        }
    }
}

/// Where the system image keeps the patterns of its data template's paths that the Data volume
/// links to instead of copying.
const DATA_PATTERNS: &str = "/System/Library/Templates/Data.symlink-patterns";

impl Secrets {
    /// Draws them, with `ecid` when given, for a disk of `table`.
    pub fn generate(ecid: Option<u64>, table: &Table) -> Result<Self> {
        let ecid = match ecid {
            Some(e) => e,
            None => tss::random_ecid()?,
        };
        let mut policy_nonces = [[0u8; 16]; 7];
        for n in &mut policy_nonces {
            getrandom::fill(n).map_err(|e| anyhow::anyhow!("no random bytes for a nonce: {e}"))?;
        }
        Ok(Self {
            ecid,
            nonces: ApNonces::generate()?,
            nvram: Nvram {
                policy_nonces,
                container: table.main().guid,
                group: Uuid::new_v4(),
            },
        })
    }
}

/// Writes the disk's table, its containers as `plan` lays them out (the main container's system
/// volume sealed first, so its boot directories carry the seal's root hash), and the auxiliary
/// storage `aux` into the restore's files.
fn write(
    files: &Files,
    table: &Table,
    aux: &[u8],
    plan: &plan::Plan,
    mut src: containers::Sources,
    events: &mut dyn FnMut(Event),
    cancel: &AtomicBool,
) -> Result<()> {
    let mut disk = OpenOptions::new()
        .write(true)
        .open(files.part(DISK))
        .context("open the disk")?;
    table
        .write(&mut disk)
        .context("write the disk's partition table")?;
    for (what, part, volumes) in [
        (
            "Sealing the system volume and writing the main container",
            table.main(),
            &plan.main,
        ),
        ("Writing the iBoot system container", table.isc(), &plan.isc),
        (
            "Writing the recoveryOS container",
            table.recovery(),
            &plan.recovery,
        ),
    ] {
        cancelled(cancel)?;
        events(Event::Step(what.to_owned()));
        containers::write_container(&mut disk, part, volumes, &mut src)?;
    }
    disk.sync_all().context("flush the disk")?;
    let mut trimmed = OpenOptions::new()
        .write(true)
        .open(files.part(AUX_TRIMMED))
        .context("open the auxiliary storage")?;
    trimmed
        .write_all(aux)
        .context("write the auxiliary storage")?;
    Ok(())
}

/// `t` in nanoseconds since 1970.
fn nanos(t: SystemTime) -> u64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

/// Fails once `cancel` is set, between the steps that don't watch it themselves.
fn cancelled(cancel: &AtomicBool) -> Result<()> {
    ensure!(!cancel.load(Ordering::SeqCst), "the restore was cancelled");
    Ok(())
}

/// What a restore writes beyond the system volume and the images it copies: the plan's own files
/// (86 MB on 14.6.1), the Data volume's copy of the data template (the Data volume a restore of
/// 14.6.1 by Apple's restore OS made took 207 MB) and APFS's metadata, with room to spare.
const SLACK: u64 = 2 << 30;

/// The space a restore writes: the system volume as the OS image allocates it (`system` bytes),
/// the images it copies whole (`images` bytes, as the IPSW stores them, which an encrypted
/// archive's plaintext is within), the rest within [`SLACK`], and the auxiliary storage twice
/// over (`aux.img.trimmed` and `aux.img`). Every file is written uncompressed.
pub fn space_needed(system: u64, images: u64) -> u64 {
    system + images + SLACK + 2 * AUX_SIZE
}

/// The space free to this user on the file system `path` is on, or would be made on: that of
/// the nearest directory above it that exists.
pub fn free_space(path: &Path) -> Result<u64> {
    let mut p = path;
    while !p.exists() {
        p = p.parent().context("no directory above it exists")?;
    }
    let c = std::ffi::CString::new(p.as_os_str().as_encoded_bytes())?;
    // SAFETY: statvfs fills the struct it is given, from a NUL-terminated path.
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("statvfs {}", p.display()));
    }
    #[allow(clippy::useless_conversion)]
    Ok(u64::from(s.f_bavail) * u64::from(s.f_frsize))
}

/// Refuses a restore into `dir` where `free` (the space free there) is short of `need`, saying
/// how much it needs.
pub fn check_space(dir: &Path, free: u64, need: u64) -> Result<()> {
    ensure!(
        free >= need,
        "the restore needs about {} free in {}, and there are {}: free some space, or restore \
         onto another disk",
        gib(need),
        dir.display(),
        gib(free)
    );
    Ok(())
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// Refuses a directory that doesn't exist or already holds a restore's files.
fn check_dir(dir: &Path) -> Result<()> {
    ensure!(
        dir.is_dir(),
        "{} is not a directory: make the VM's directory first",
        dir.display()
    );
    for f in [
        DISK,
        AUX,
        AUX_TRIMMED,
        CONFIG,
        MACHINE_IDENTIFIER,
        EACS_ISC,
        EACS_AUX,
    ] {
        ensure!(
            !dir.join(f).exists(),
            "{} already holds {f}: restore into an empty VM directory",
            dir.display()
        );
    }
    Ok(())
}

/// The restore's files in the VM's directory, under temporary names until it succeeds.
struct Files {
    dir: PathBuf,
}

impl Files {
    fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_owned(),
        }
    }

    fn part(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{name}.part"))
    }

    fn make(&self, disk_size: u64) -> Result<()> {
        for (name, size) in [(DISK, disk_size), (AUX_TRIMMED, AUX_SIZE)] {
            let p = self.part(name);
            let f = File::create(&p).with_context(|| format!("make {}", p.display()))?;
            f.set_len(size)
                .with_context(|| format!("size {} to {size} bytes", p.display()))?;
        }
        Ok(())
    }

    /// For [`FirstBoot::Eacs`]: keeps the iBoot system container's partition of the disk and the
    /// auxiliary storage as written, for [`finish_eacs`].
    fn keep_for_eacs(&self, table: &Table) -> Result<()> {
        let disk = File::open(self.part(DISK)).context("open the disk")?;
        let isc = table.isc();
        copy_sparse(&disk, isc.offset(), isc.len(), &self.part(EACS_ISC), 0)?;
        let aux = File::open(self.part(AUX_TRIMMED)).context("open the auxiliary storage")?;
        copy_sparse(&aux, 0, AUX_SIZE, &self.part(EACS_AUX), 0)
    }

    /// Writes `aux.img`, the machine identifier and the anchor bundle (if any), and moves the
    /// disks into place, and, for `eacs`, the copies [`finish_eacs`] puts back.
    fn finish(&self, ecid: u64, bundle: Option<&[u8]>, eacs: bool) -> Result<()> {
        if let Some(b) = bundle {
            let p = self.part(CONFIG);
            std::fs::write(&p, b).with_context(|| format!("write {}", p.display()))?;
        }
        let aux = self.part(AUX);
        {
            let mut out = File::create(&aux).with_context(|| format!("make {}", aux.display()))?;
            out.write_all(&aux::vz_header())?;
            let mut trimmed = File::open(self.part(AUX_TRIMMED))?;
            std::io::copy(&mut trimmed, &mut out)
                .with_context(|| format!("write {}", aux.display()))?;
            out.sync_all()?;
        }
        let id = self.part(MACHINE_IDENTIFIER);
        machine_identifier(ecid)
            .to_file_binary(&id)
            .with_context(|| format!("write {}", id.display()))?;
        for name in [DISK, AUX_TRIMMED] {
            let p = self.part(name);
            File::open(&p)
                .and_then(|f| f.sync_all())
                .with_context(|| format!("flush {}", p.display()))?;
        }
        // The machine identifier goes last; a failed move takes back those made before it.
        let mut moved = Vec::new();
        let mut names = vec![DISK, AUX_TRIMMED, AUX];
        if bundle.is_some() {
            names.push(CONFIG);
        }
        if eacs {
            names.extend([EACS_ISC, EACS_AUX]);
        }
        names.push(MACHINE_IDENTIFIER);
        for name in names {
            let to = self.dir.join(name);
            if let Err(e) = std::fs::rename(self.part(name), &to) {
                for m in moved {
                    let _ = std::fs::remove_file(m);
                }
                return Err(e).with_context(|| format!("move {} into place", to.display()));
            }
            moved.push(to);
        }
        if let Ok(d) = File::open(&self.dir) {
            let _ = d.sync_all();
        }
        Ok(())
    }

    fn discard(&self) {
        for name in [
            DISK,
            AUX_TRIMMED,
            AUX,
            CONFIG,
            MACHINE_IDENTIFIER,
            EACS_ISC,
            EACS_AUX,
        ] {
            let _ = std::fs::remove_file(self.part(name));
        }
    }
}

/// Copies `len` bytes of `from` at `at` into a new file `to` at `to_at`, writing only the 4 KiB
/// blocks that aren't all zeroes, so the copy stays sparse.
fn copy_sparse(from: &File, at: u64, len: u64, to: &Path, to_at: u64) -> Result<()> {
    use std::os::unix::fs::FileExt as _;
    let out = File::create(to).with_context(|| format!("make {}", to.display()))?;
    out.set_len(to_at + len)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0;
    while done < len {
        let n = buf
            .len()
            .min(usize::try_from(len - done).unwrap_or(usize::MAX));
        from.read_exact_at(&mut buf[..n], at + done)
            .with_context(|| format!("read {n} bytes at {}", at + done))?;
        for (i, b) in buf[..n].chunks(4096).enumerate() {
            if b.iter().any(|x| *x != 0) {
                out.write_all_at(b, to_at + done + (i * 4096) as u64)
                    .with_context(|| format!("write {}", to.display()))?;
            }
        }
        done += n as u64;
    }
    out.sync_all()?;
    Ok(())
}

/// Makes a VM whose restore left its Data volume to EACS ([`FirstBoot::Eacs`]) bootable, after
/// its first boot made the Data volume and reset: puts the iBoot system container back as the
/// restore wrote it (EACS deleted the macOS LocalPolicy, which only the restore can sign) and the
/// policy nonces back into NVRAM ([`aux::put_back_policy_entries`]), keeping what the guest
/// wrote in the auxiliary storage, its keys in the ean storage among it; then removes the copies.
/// Refuses before the guest's keystore has made its keys. A run cut short can run again.
pub fn finish_eacs(dir: &Path) -> Result<()> {
    use std::os::unix::fs::FileExt as _;
    let (isc, pre_aux) = (dir.join(EACS_ISC), dir.join(EACS_AUX));
    ensure!(
        isc.is_file() && pre_aux.is_file(),
        "{} holds no restore waiting for its first boot's EACS",
        dir.display()
    );
    let mut a = std::fs::read(dir.join(AUX_TRIMMED))
        .with_context(|| format!("read {}", dir.join(AUX_TRIMMED).display()))?;
    ensure!(
        a.len() == aux::SIZE,
        "{AUX_TRIMMED} is not {} bytes",
        aux::SIZE
    );
    ensure!(
        aux::ean_entries(&a[aux::EAN_AT..]).is_some_and(|n| n > 1),
        "the VM's first boot hasn't made its keys yet: boot it once, until macOS has made its \
         Data volume and reset, then finish"
    );
    let restored =
        std::fs::read(&pre_aux).with_context(|| format!("read {}", pre_aux.display()))?;
    aux::put_back_policy_entries(&mut a, &restored)?;

    let disk = OpenOptions::new()
        .read(true)
        .write(true)
        .open(dir.join(DISK))
        .context("open the disk")?;
    let t = reims_apfs::gpt::read(&disk).context("read the disk's partition table")?;
    let p = t
        .partitions
        .iter()
        .find(|p| p.index == 1)
        .context("the disk has no first partition")?;
    let (at, len) = (
        p.first_lba * t.sector_size,
        (p.last_lba + 1 - p.first_lba) * t.sector_size,
    );
    let from = File::open(&isc).with_context(|| format!("open {}", isc.display()))?;
    ensure!(
        from.metadata()?.len() == len,
        "{EACS_ISC} is not the size of the disk's first partition"
    );
    let (mut was, mut now) = (vec![0u8; 1 << 20], vec![0u8; 1 << 20]);
    let mut done = 0;
    while done < len {
        let n = was
            .len()
            .min(usize::try_from(len - done).unwrap_or(usize::MAX));
        from.read_exact_at(&mut was[..n], done)?;
        disk.read_exact_at(&mut now[..n], at + done)?;
        if was[..n] != now[..n] {
            disk.write_all_at(&was[..n], at + done)
                .context("write the iBoot system container back")?;
        }
        done += n as u64;
    }
    disk.sync_all().context("flush the disk")?;
    let mut h = aux::vz_header();
    h.extend_from_slice(&a);
    for (name, bytes) in [(AUX_TRIMMED, &a), (AUX, &h)] {
        let p = dir.join(name);
        let tmp = dir.join(format!("{name}.part"));
        std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
        File::open(&tmp)?.sync_all()?;
        std::fs::rename(&tmp, &p).with_context(|| format!("move {} into place", p.display()))?;
    }
    for f in [&isc, &pre_aux] {
        std::fs::remove_file(f).with_context(|| format!("remove {}", f.display()))?;
    }
    Ok(())
}

/// Virtualization.framework's machine identifier: `{ECID: n}`.
fn machine_identifier(ecid: u64) -> Value {
    let mut d = Dictionary::new();
    d.insert("ECID".into(), Value::Integer(ecid.into()));
    Value::Dictionary(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let mut b = [0u8; 6];
        getrandom::fill(&mut b).unwrap();
        let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
        let p = std::env::temp_dir().join(format!("reims-restore-test-{hex}"));
        std::fs::create_dir(&p).unwrap();
        p
    }

    #[test]
    fn a_finished_restore_leaves_vzs_files() {
        let d = temp_dir();
        let f = Files::new(&d);
        f.make(1 << 20).unwrap();
        std::fs::write(f.part(AUX_TRIMMED), b"nvram").unwrap();
        f.finish(0x1234_5678_9abc_def0, None, false).unwrap();
        assert!(!d.join(CONFIG).exists(), "no bundle without the keys");
        for n in [DISK, AUX, AUX_TRIMMED, MACHINE_IDENTIFIER] {
            assert!(d.join(n).is_file(), "{n}");
            assert!(!f.part(n).exists(), "{n}.part");
        }
        assert_eq!(std::fs::metadata(d.join(DISK)).unwrap().len(), 1 << 20);
        let aux = std::fs::read(d.join(AUX)).unwrap();
        assert_eq!(&aux[..aux::HEADER_LEN], &aux::vz_header()[..]);
        assert_eq!(&aux[aux::HEADER_LEN..], b"nvram");
        let id = Value::from_file(d.join(MACHINE_IDENTIFIER)).unwrap();
        let ecid = id
            .as_dictionary()
            .and_then(|d| d.get("ECID"))
            .and_then(Value::as_unsigned_integer);
        assert_eq!(ecid, Some(0x1234_5678_9abc_def0));
        assert!(check_dir(&d).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn only_macos_14_makes_its_data_volume_by_eacs() {
        for v in ["14.0", "14.6.1", "14"] {
            assert!(data_by_eacs(v), "{v}");
        }
        for v in [
            "15.0", "15.6.1", "26.6.2", "27.0.1", "27.2.1", "140.1", "4.1",
        ] {
            assert!(!data_by_eacs(v), "{v}");
        }
    }

    fn write_at(p: &Path, at: u64, b: &[u8]) {
        use std::os::unix::fs::FileExt as _;
        OpenOptions::new()
            .write(true)
            .open(p)
            .unwrap()
            .write_all_at(b, at)
            .unwrap();
    }

    fn read_at(p: &Path, at: u64, n: usize) -> Vec<u8> {
        use std::os::unix::fs::FileExt as _;
        let mut b = vec![0u8; n];
        File::open(p).unwrap().read_exact_at(&mut b, at).unwrap();
        b
    }

    /// A finished restore for EACS, of the tests' own bytes: the disk's table, two marks in the
    /// iBoot system container and one in the main container, and the auxiliary storage.
    fn restored_for_eacs() -> (PathBuf, Table, Nvram, Vec<u8>) {
        let d = temp_dir();
        let table = Table::layout(MIN_DISK).unwrap();
        let f = Files::new(&d);
        f.make(MIN_DISK).unwrap();
        table
            .write(&mut OpenOptions::new().write(true).open(f.part(DISK)).unwrap())
            .unwrap();
        let isc = table.isc();
        write_at(&f.part(DISK), isc.offset() + 0x1000, b"the restore's iSC");
        write_at(&f.part(DISK), isc.offset() + isc.len() - 0x1000, b"its end");
        write_at(&f.part(DISK), table.main().offset(), b"the main container");
        let n = Nvram {
            policy_nonces: [
                [0x11; 16], [0x22; 16], [3; 16], [4; 16], [5; 16], [6; 16], [7; 16],
            ],
            container: table.main().guid,
            group: Uuid::from_u128(0xbf30_4653),
        };
        let a = aux::trimmed(&n, b"a ticket of the tests' own").unwrap();
        std::fs::write(f.part(AUX_TRIMMED), &a).unwrap();
        f.keep_for_eacs(&table).unwrap();
        f.finish(7, None, true).unwrap();
        (d, table, n, a)
    }

    /// What a guest's EACS boot leaves, as far as finishing reads it: other bytes in the iBoot
    /// system container and the main container, the keystore's entries counted in the ean
    /// storage, and NVRAM's policy nonces not the restore's.
    fn boot_by_eacs(d: &Path, t: &Table, n: &Nvram) {
        let isc = t.isc();
        let disk = d.join(DISK);
        write_at(&disk, isc.offset() + 0x1000, &[0; 17]);
        write_at(&disk, isc.offset() + 0x20_0000, b"the guest's iSC");
        write_at(&disk, t.main().offset(), b"the guest's Data");
        let mut other = n.clone();
        other.policy_nonces[1] = [0x99; 16];
        let mut a = aux::trimmed(&other, b"a ticket of the tests' own").unwrap();
        let e = &mut a[aux::EAN_AT..aux::EAN_AT + 0x1000];
        e[8..12].copy_from_slice(&3u32.to_le_bytes());
        let crc = crc32fast::hash(&e[..0xffc]);
        e[0xffc..].copy_from_slice(&crc.to_le_bytes());
        std::fs::write(d.join(AUX_TRIMMED), &a).unwrap();
    }

    #[test]
    fn a_restore_for_eacs_keeps_what_finishing_puts_back() {
        let (d, t, _, a) = restored_for_eacs();
        for n in [
            DISK,
            AUX,
            AUX_TRIMMED,
            MACHINE_IDENTIFIER,
            EACS_ISC,
            EACS_AUX,
        ] {
            assert!(d.join(n).is_file(), "{n}");
        }
        let isc = std::fs::metadata(d.join(EACS_ISC)).unwrap();
        assert_eq!(isc.len(), t.isc().len());
        assert_eq!(read_at(&d.join(EACS_ISC), 0x1000, 17), b"the restore's iSC");
        assert_eq!(std::fs::read(d.join(EACS_AUX)).unwrap(), a);
        assert!(check_dir(&d).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn finishing_eacs_puts_the_restores_isc_and_nonces_back() {
        let (d, t, n, _) = restored_for_eacs();
        boot_by_eacs(&d, &t, &n);
        finish_eacs(&d).unwrap();
        let disk = d.join(DISK);
        let isc = t.isc();
        assert_eq!(
            read_at(&disk, isc.offset() + 0x1000, 17),
            b"the restore's iSC"
        );
        assert_eq!(read_at(&disk, isc.offset() + 0x20_0000, 15), [0; 15]);
        assert_eq!(
            read_at(&disk, isc.offset() + isc.len() - 0x1000, 7),
            b"its end"
        );
        assert_eq!(read_at(&disk, t.main().offset(), 16), b"the guest's Data");
        let a = std::fs::read(d.join(AUX_TRIMMED)).unwrap();
        assert_eq!(
            aux::ean_entries(&a[aux::EAN_AT..]),
            Some(3),
            "the guest's keys stay"
        );
        for i in 0..2 {
            let at = aux::NVRAM_AT + i * 0x8_0000;
            let parts = aux::read_bank(&a[at..at + 0x8_0000]).unwrap();
            let system = &parts.iter().find(|(p, _)| p == "system").unwrap().1;
            let nonces = &system
                .iter()
                .find(|(k, _)| k == "boop-storage-nonces")
                .unwrap()
                .1;
            assert_eq!(nonces, &n.policy_nonces.concat());
        }
        let vz = std::fs::read(d.join(AUX)).unwrap();
        assert_eq!(&vz[..aux::HEADER_LEN], &aux::vz_header()[..]);
        assert_eq!(&vz[aux::HEADER_LEN..], &a[..]);
        assert!(!d.join(EACS_ISC).exists() && !d.join(EACS_AUX).exists());
        assert!(finish_eacs(&d).is_err(), "nothing left to finish");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn finishing_eacs_waits_for_the_guests_keys() {
        let (d, t, _, a) = restored_for_eacs();
        let e = finish_eacs(&d).unwrap_err().to_string();
        assert!(e.contains("hasn't made its keys"), "{e}");
        assert_eq!(std::fs::read(d.join(AUX_TRIMMED)).unwrap(), a);
        assert_eq!(
            read_at(&d.join(DISK), t.isc().offset() + 0x1000, 17),
            b"the restore's iSC"
        );
        assert!(d.join(EACS_ISC).exists() && d.join(EACS_AUX).exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_restore_with_the_keys_leaves_the_bundle() {
        let d = temp_dir();
        let f = Files::new(&d);
        f.make(4096).unwrap();
        f.finish(1, Some(b"070701 a bundle"), false).unwrap();
        assert_eq!(std::fs::read(d.join(CONFIG)).unwrap(), b"070701 a bundle");
        assert!(!f.part(CONFIG).exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_failed_restore_leaves_nothing() {
        let d = temp_dir();
        let f = Files::new(&d);
        f.make(4096).unwrap();
        f.discard();
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
        check_dir(&d).unwrap();
        std::fs::remove_dir(&d).unwrap();
    }

    #[test]
    fn a_failed_move_leaves_nothing_that_looks_finished() {
        let d = temp_dir();
        let f = Files::new(&d);
        f.make(4096).unwrap();
        // The machine identifier's place is taken by a directory with a file in it.
        std::fs::create_dir(d.join(MACHINE_IDENTIFIER)).unwrap();
        std::fs::write(d.join(MACHINE_IDENTIFIER).join("x"), b"x").unwrap();
        assert!(f.finish(1, Some(b"a bundle"), false).is_err());
        for n in [DISK, AUX, AUX_TRIMMED, CONFIG] {
            assert!(!d.join(n).exists(), "{n} was left in place");
        }
        std::fs::remove_dir_all(d.join(MACHINE_IDENTIFIER)).unwrap();
        f.discard();
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
        std::fs::remove_dir(&d).unwrap();
    }

    #[test]
    fn a_restore_needs_the_space_it_writes() {
        let d = temp_dir();
        let need = space_needed(10 << 30, 8 << 30);
        assert_eq!(need, (20 << 30) + (64 << 20));
        assert!(check_space(&d, need, need).is_ok());
        let e = check_space(&d, need - 1, need).unwrap_err().to_string();
        assert!(e.contains("20.1 GiB"), "{e}");
        assert!(free_space(&d.join("not/made/yet")).unwrap() > 0);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn secrets_bind_nvram_to_the_disk() {
        let t = Table::layout(64 << 30).unwrap();
        let s = Secrets::generate(Some(0xabc), &t).unwrap();
        assert_eq!(s.ecid, 0xabc);
        assert_eq!(s.nvram.container, t.main().guid);
        assert_eq!(s.nonces.sep_nonce, tss::VZ_SEP_NONCE);
        let other = Secrets::generate(None, &t).unwrap();
        assert_ne!(other.nvram.policy_nonces, s.nvram.policy_nonces);
        assert_ne!(other.nonces.ap_nonce, s.nonces.ap_nonce);
    }

    #[test]
    fn a_cancelled_restore_stops_between_steps() {
        let c = AtomicBool::new(false);
        assert!(cancelled(&c).is_ok());
        c.store(true, Ordering::SeqCst);
        assert!(cancelled(&c).is_err());
    }
}
