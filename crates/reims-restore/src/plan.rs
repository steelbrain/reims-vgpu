//! What a restore writes onto the disk, as data: each container's volumes, and each volume's
//! entries with where every file's bytes come from (docs/design.md, "The disk"). The pipeline
//! builds the plan from the IPSW's manifest and the VM's tickets and policies; reims-apfs writes
//! it. Paths, modes and owners are those of a fresh restore by Virtualization.framework of
//! 27.2.1 **[vz]**.

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use plist::{Dictionary, Value};
use uuid::Uuid;

use crate::img4;
use crate::manifest::{BuildManifest, Identity};
use crate::xml_plist;

/// The recoveryOS container's volume group, the same on every Apple silicon Mac and VM **[vz]**.
pub const RECOVERY_GROUP: Uuid = Uuid::from_u128(0x3d3287de_280d_4619_aaab_d97469ca9c71);

/// A volume's role, as APFS records it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    Preboot,
    Recovery,
    Update,
    Xart,
    Hardware,
}

impl Role {
    /// The superblock's `apfs_role`.
    pub fn apfs(self) -> u16 {
        match self {
            Role::System => 0x1,
            Role::Recovery => 0x4,
            Role::Preboot => 0x10,
            Role::Update => 0xc0,
            Role::Xart => 0x100,
            Role::Hardware => 0x140,
        }
    }
}

/// Where a file's bytes come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Data {
    /// Bytes the restore makes: a ticket, a policy, a plist.
    Bytes(Arc<[u8]>),
    /// An IPSW member as stored, streamed.
    Member(String),
    /// An IPSW member's plaintext, streamed (`image::plain_stream`): an Apple Encrypted Archive
    /// decrypted, a deflated member inflated, any other member as stored.
    Plain(String),
    /// A member's IM4P stitched to a manifest, as `component` (`img4::personalize`).
    Stitched {
        component: String,
        member: String,
        manifest: Arc<[u8]>,
    },
    /// The system volume's root hash IM4P (`img4::RootHash::system_im4p`) stitched to a
    /// manifest: its hashes are those of the seal reims-apfs gives the system volume, so the
    /// writer makes this once the system volume is sealed, and the system volume comes first. The
    /// manifest's `isys` digest names the IPSW's root hash IM4P, not this one; nothing checks it
    /// under Ambrosia (docs/design.md, "The system volume's seal").
    SealedRootHash { manifest: Arc<[u8]> },
}

/// What an entry is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    Dir,
    File(Data),
    /// A symbolic link to the target given.
    Symlink(String),
    /// A clone of the file at the path given, in the same volume, made before it.
    Clone(String),
}

/// One file, directory or link.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Its path from the volume's root, without a leading `/`.
    pub path: String,
    pub node: Node,
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
}

/// What a volume holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Content {
    /// These entries, each after its parent.
    Entries(Vec<Entry>),
    /// The system volume: the volume of the IPSW's OS image, member `image`, with the canonical
    /// metadata of member `mtree` (`Ap,SystemVolumeCanonicalMetadata`, IM4P `msys`), sealed by
    /// reims-apfs to a root hash of its own. Member `root_hash` (`SystemVolume`, IM4P `isys`) is
    /// the IPSW's signed root hash, of Apple's seal, which ours need not equal; the boot
    /// directories' `root_hash.img4` carry ours ([`Data::SealedRootHash`]).
    SealedSystem {
        image: String,
        mtree: String,
        root_hash: String,
    },
}

/// One volume.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Volume {
    pub name: &'static str,
    pub role: Role,
    pub uuid: Uuid,
    /// The volume group it belongs to, if any.
    pub group: Option<Uuid>,
    pub content: Content,
}

/// The three containers' volumes, in the order they are made: the iBoot system container's, the
/// main container's, the recoveryOS container's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub isc: Vec<Volume>,
    pub main: Vec<Volume>,
    pub recovery: Vec<Volume>,
}

/// What the plan is made from.
pub struct Inputs<'a> {
    /// The IPSW's build manifest, and whether it has a member.
    pub manifest: &'a BuildManifest,
    pub has: &'a dyn Fn(&str) -> bool,
    /// Every member's name, for the BootabilityBundle's.
    pub names: &'a [String],
    pub install: Identity<'a>,
    pub erase: Identity<'a>,
    pub ecid: u64,
    /// The volume group: the Data volume's UUID.
    pub group: Uuid,
    pub boot_manifest: Arc<[u8]>,
    pub sfr_manifest: Arc<[u8]>,
    pub cryptex1_ticket: Arc<[u8]>,
    /// The macOS LocalPolicy and its recoveryOS's, and their file names.
    pub policy: (String, Arc<[u8]>),
    pub policy_recovery: (String, Arc<[u8]>),
    /// The recoveryOS container's LocalPolicy, from TSS, and its `ronh` in upper-case hex.
    pub recovery_os_policy: Arc<[u8]>,
    pub ronh: String,
    /// The main container's partition GUID, which EACS's cookie names: the guest's first boot
    /// makes the Data volume itself ([`eacs_cookie`]).
    pub eacs_store: Uuid,
}

/// Where a restore that leaves the Data volume out puts EACS's cookie, under the volume group's
/// directory in the main container's Preboot **[measured]**.
pub const EACS_COOKIE: &str = "var/db/.com.apple.eacs";

/// The cookie that has the guest's first boot erase and make the Data volume (EACS, erase all
/// content and settings) of `group`, on the main container's partition `store`. At a boot that
/// finds it, BaseSystem's `migrationModeSwitcher` reads `MigrationMode` (2 starts EACS) and
/// `obliterationKickStarter` hands `MobileObliterationOptions` to `mobile_obliterator`, which,
/// for `ObliterateDataPartition`, makes the Data volume of `PreservedVolumeGroupUUID`, encrypted,
/// and the keystore's keys for it, then resets the VM **[measured]** (docs/design.md, "The Data
/// volume, by EACS").
pub fn eacs_cookie(group: Uuid, store: Uuid) -> Arc<[u8]> {
    let mut o = Dictionary::new();
    o.insert(
        "ObliterationType".into(),
        Value::String("ObliterateDataPartition".into()),
    );
    o.insert(
        "PreservedVolumeGroupUUID".into(),
        Value::String(group.to_string().to_uppercase()),
    );
    o.insert(
        "boot-physical-store-UUID".into(),
        Value::String(store.to_string().to_uppercase()),
    );
    o.insert("DisplayProgressBar".into(), Value::Boolean(true));
    let mut d = Dictionary::new();
    d.insert("MigrationMode".into(), Value::Integer(2.into()));
    d.insert("MobileObliterationOptions".into(), Value::Dictionary(o));
    xml_plist::to_xml(&Value::Dictionary(d)).into()
}

/// SHA-384 in upper-case hex: the name of a boot directory, from its manifest.
pub fn hash_name(manifest: &[u8]) -> String {
    img4::sha384(manifest)
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect()
}

/// A volume's entries as they are added.
#[derive(Default)]
struct Tree {
    entries: Vec<Entry>,
    seen: BTreeSet<String>,
}

impl Tree {
    fn add(&mut self, path: &str, node: Node, mode: u16, owner: (u32, u32)) {
        if self.seen.insert(path.to_owned()) {
            self.entries.push(Entry {
                path: path.to_owned(),
                node,
                mode,
                uid: owner.0,
                gid: owner.1,
            });
        }
    }

    /// `path` and every directory above it, each of `mode`, those already made kept as they are.
    fn dirs(&mut self, path: &str, mode: u16) {
        let mut at = String::new();
        for part in path.split('/') {
            if !at.is_empty() {
                at.push('/');
            }
            at.push_str(part);
            self.add(&at.clone(), Node::Dir, mode, ROOT);
        }
    }

    fn file(&mut self, path: &str, data: Data, mode: u16) {
        self.add(path, Node::File(data), mode, ROOT);
    }

    fn bytes(&mut self, path: &str, b: &Arc<[u8]>, mode: u16) {
        self.file(path, Data::Bytes(b.clone()), mode);
    }
}

const ROOT: (u32, u32) = (0, 0);
/// The owner the BootabilityBundle's files carry, as Virtualization.framework's host packs them
/// **[vz]**.
const BOOTABILITY_OWNER: (u32, u32) = (501, 20);

/// The boot objects a boot directory holds, by the install identity's component, and where.
const BOOT_OBJECTS: &[(&str, &str)] = &[
    (
        "KernelCache",
        "System/Library/Caches/com.apple.kernelcaches/kernelcache",
    ),
    ("DeviceTree", "usr/standalone/firmware/devicetree.img4"),
    ("SystemVolume", "usr/standalone/firmware/root_hash.img4"),
    (
        "BaseSystemVolume",
        "usr/standalone/firmware/base_system_root_hash.img4",
    ),
    ("iBoot", "usr/standalone/firmware/iBoot.img4"),
    (
        "StaticTrustCache",
        "usr/standalone/firmware/FUD/StaticTrustCache.img4",
    ),
    (
        "Ap,BaseSystemTrustCache",
        "usr/standalone/firmware/FUD/BaseSystemTrustCache.img4",
    ),
];

/// The Cryptex1 images and their files in `cryptex1/current/`: component prefix, name.
const CRYPTEXES: &[(&str, &str)] = &[("System", "os"), ("App", "app"), ("Rosetta", "rosetta")];

impl Inputs<'_> {
    fn path(&self, id: &Identity<'_>, component: &str) -> Result<String> {
        Ok(id
            .component(component)?
            .path()
            .with_context(|| format!("the manifest's {component} has no path"))?
            .to_owned())
    }

    /// A boot directory's objects under `dir`, stitched to `manifest`, files of `mode` and
    /// directories of `dir_mode`.
    fn boot_dir(
        &self,
        t: &mut Tree,
        dir: &str,
        id: &Identity<'_>,
        manifest: &Arc<[u8]>,
        (dir_mode, mode): (u16, u16),
    ) -> Result<()> {
        t.dirs(&format!("{dir}/System/Library/Caches"), dir_mode);
        t.bytes(
            &format!("{dir}/System/Library/Caches/apticket.der"),
            manifest,
            mode,
        );
        for (component, at) in BOOT_OBJECTS {
            let member = self.path(id, component)?;
            let path = format!("{dir}/{at}");
            t.dirs(path.rsplit_once('/').map_or("", |p| p.0), dir_mode);
            let data = if *component == "SystemVolume" {
                Data::SealedRootHash {
                    manifest: manifest.clone(),
                }
            } else {
                Data::Stitched {
                    component: (*component).to_owned(),
                    member,
                    manifest: manifest.clone(),
                }
            };
            t.file(&path, data, mode);
        }
        Ok(())
    }

    /// The install identity alone, as `cryptex1/current/` and `restore/` keep the manifest.
    fn install_manifest(&self) -> Arc<[u8]> {
        let mut d = Dictionary::new();
        d.insert(
            "BuildIdentities".into(),
            Value::Array(vec![Value::Dictionary(self.install.dict().clone())]),
        );
        xml_plist::to_xml(&Value::Dictionary(d)).into()
    }

    /// The erase identity with the manifest's version keys, as the recoveryOS container keeps it.
    fn erase_manifest(&self) -> Arc<[u8]> {
        let mut d = Dictionary::new();
        d.insert(
            "BuildIdentities".into(),
            Value::Array(vec![Value::Dictionary(self.erase.dict().clone())]),
        );
        for k in ["ManifestVersion", "ProductBuildVersion", "ProductVersion"] {
            if let Some(v) = self.manifest.get(k) {
                d.insert(k.into(), v.clone());
            }
        }
        d.insert(
            "SupportedProductTypes".into(),
            Value::Array(vec![Value::String("VirtualMac2,1".into())]),
        );
        xml_plist::to_xml(&Value::Dictionary(d)).into()
    }
}

/// The members of the images a restore of `install` copies whole, as the plan places them: the
/// BaseSystem twice (the main container's Recovery and the recoveryOS container's), then each
/// Cryptex1 image the identity has.
pub fn copied_images(install: &Identity<'_>) -> Result<Vec<String>> {
    let path = |c: &str| -> Result<String> {
        Ok(install
            .component(c)?
            .path()
            .with_context(|| format!("the manifest's {c} has no path"))?
            .to_owned())
    };
    let base_system = path("BaseSystem")?;
    let mut out = vec![base_system.clone(), base_system];
    for (component, _) in CRYPTEXES {
        let image = format!("Cryptex1,{component}OS");
        if install.component(&image).is_ok() {
            out.push(path(&image)?);
        }
    }
    Ok(out)
}

/// Builds the plan.
pub fn plan(i: &Inputs<'_>) -> Result<Plan> {
    let g = i.group.to_string().to_uppercase();
    let r = RECOVERY_GROUP.to_string().to_uppercase();
    let boot_dir = hash_name(&i.boot_manifest);
    let sfr_dir = hash_name(&i.sfr_manifest);
    let os = i.path(&i.install, "OS")?;
    let base_system = i.path(&i.install, "BaseSystem")?;
    for m in ["SystemVersion.plist", "usr/standalone/bootcaches.plist"] {
        ensure!((i.has)(m), "the IPSW has no {m}");
    }
    let restore_version =
        "BootabilityBundle/Restore/Bootability/System/Library/CoreServices/RestoreVersion.plist";
    ensure!(
        (i.has)(restore_version),
        "the IPSW has no {restore_version}"
    );
    let member = |m: &str| Data::Member(m.to_owned());

    // The iBoot system container.
    let mut isc = Tree::default();
    isc.dirs(&format!("{g}/LocalPolicy"), 0o777);
    isc.bytes(
        &format!("{g}/LocalPolicy/{}", i.policy.0),
        &i.policy.1,
        0o666,
    );
    isc.bytes(
        &format!("{g}/LocalPolicy/{}", i.policy_recovery.0),
        &i.policy_recovery.1,
        0o666,
    );
    isc.dirs(&format!("{r}/LocalPolicy"), 0o777);
    isc.bytes(
        &format!("{r}/LocalPolicy/{}.recovery.img4", i.ronh),
        &i.recovery_os_policy,
        0o666,
    );
    isc.dirs("SFR/current", 0o755);
    isc.dirs(&format!("SystemRecovery/{sfr_dir}"), 0o755);
    for dir in [
        "SFR/current".to_owned(),
        format!("SystemRecovery/{sfr_dir}"),
    ] {
        isc.file(
            &format!("{dir}/RestoreVersion.plist"),
            member(restore_version),
            0o666,
        );
        isc.file(
            &format!("{dir}/SystemVersion.plist"),
            member("SystemVersion.plist"),
            0o666,
        );
        let ticket_mode = if dir == "SFR/current" { 0o644 } else { 0o666 };
        isc.bytes(&format!("{dir}/apticket.der"), &i.sfr_manifest, ticket_mode);
        isc.file(
            &format!("{dir}/bootcaches.plist"),
            member("usr/standalone/bootcaches.plist"),
            0o644,
        );
    }
    isc.add(
        &format!("SystemRecovery/{}", i.ronh),
        Node::Symlink(sfr_dir.clone()),
        0o777,
        ROOT,
    );

    let mut hardware = Tree::default();
    hardware.dirs(
        "FactoryData/System/Library/Caches/com.apple.factorydata",
        0o755,
    );
    hardware.dirs("MobileActivation/logs", 0o755);
    hardware.dirs("MobileActivation/splunk", 0o755);

    // The main container's Preboot.
    let mut pb = Tree::default();
    pb.dirs(
        &format!("{g}/Library/Preferences/SystemConfiguration"),
        0o777,
    );
    for d in ["CoreServices", "KernelCollections", "PrelinkedKernels"] {
        pb.dirs(&format!("{g}/System/Library/{d}"), 0o777);
    }
    pb.file(
        &format!("{g}/System/Library/CoreServices/boot.efi"),
        Data::Bytes(Arc::from(&[][..])),
        0o666,
    );
    pb.dirs(&format!("{g}/boot/System/Library/KernelCollections"), 0o777);
    pb.dirs(&format!("{g}/boot/{boot_dir}"), 0o755);
    i.boot_dir(
        &mut pb,
        &format!("{g}/boot/{boot_dir}"),
        &i.install,
        &i.boot_manifest,
        (0o755, 0o644),
    )?;
    pb.file(
        &format!("{g}/boot/active"),
        Data::Bytes(boot_dir.clone().into_bytes().into()),
        0o644,
    );
    let cx = format!("{g}/cryptex1/current");
    pb.dirs(&cx, 0o777);
    let install_manifest = i.install_manifest();
    pb.bytes(
        &format!("{cx}/BuildManifest.plist"),
        &install_manifest,
        0o666,
    );
    pb.file(
        &format!("{cx}/RestoreVersion.plist"),
        member(restore_version),
        0o644,
    );
    pb.file(
        &format!("{cx}/SystemVersion.plist"),
        member("SystemVersion.plist"),
        0o444,
    );
    for (component, name) in CRYPTEXES {
        let image = format!("Cryptex1,{component}OS");
        if i.install.component(&image).is_err() {
            continue;
        }
        pb.file(
            &format!("{cx}/{name}.dmg"),
            Data::Plain(i.path(&i.install, &image)?),
            0o666,
        );
        if *name == "os" {
            pb.add(
                &format!("{cx}/os.clone.dmg"),
                Node::Clone(format!("{cx}/os.dmg")),
                0o666,
                ROOT,
            );
        }
        for (part, ext) in [("Volume", "root_hash"), ("TrustCache", "trustcache")] {
            let m = i.path(&i.install, &format!("Cryptex1,{component}{part}"))?;
            pb.file(&format!("{cx}/{name}.dmg.{ext}"), member(&m), 0o666);
        }
    }
    pb.bytes(
        &format!("{cx}/apticket.vma2macosap.{:016X}.im4m", i.ecid),
        &i.cryptex1_ticket,
        0o666,
    );
    let global = "Firmware/Manifests/restore/cryptex1/macOS Customer/apticket.vma2macosap.im4m";
    ensure!((i.has)(global), "the IPSW has no {global}");
    pb.file(
        &format!("{cx}/apticket.vma2macosap.im4m"),
        member(global),
        0o666,
    );

    let rs = format!("{g}/restore");
    pb.dirs(&format!("{rs}/Firmware/all_flash"), 0o777);
    pb.dirs(&format!("{rs}/usr/standalone"), 0o777);
    pb.bytes(
        &format!("{rs}/BuildManifest.plist"),
        &install_manifest,
        0o666,
    );
    pb.file(
        &format!("{rs}/RestoreVersion.plist"),
        member(restore_version),
        0o666,
    );
    pb.file(
        &format!("{rs}/SystemVersion.plist"),
        member("SystemVersion.plist"),
        0o666,
    );
    pb.bytes(
        &format!("{rs}/apticket.vma2macosap.{:016X}.im4m", i.ecid),
        &i.boot_manifest,
        0o666,
    );
    let restore_ticket = "Firmware/Manifests/restore/macOS Customer/apticket.vma2macosap.im4m";
    ensure!((i.has)(restore_ticket), "the IPSW has no {restore_ticket}");
    pb.file(
        &format!("{rs}/apticket.vma2macosap.im4m"),
        member(restore_ticket),
        0o666,
    );
    let kernel = i.path(&i.install, "KernelCache")?;
    pb.file(&format!("{rs}/{kernel}"), member(&kernel), 0o666);
    for c in ["DeviceTree", "iBoot"] {
        let m = i.path(&i.install, c)?;
        pb.file(&format!("{rs}/{m}"), member(&m), 0o666);
    }
    for c in [
        "BaseSystemVolume",
        "Ap,BaseSystemTrustCache",
        "SystemVolume",
        "StaticTrustCache",
    ] {
        let m = i.path(&i.install, c)?;
        pb.file(&format!("{rs}/{m}"), member(&m), 0o666);
    }
    pb.file(
        &format!("{rs}/usr/standalone/bootcaches.plist"),
        member("usr/standalone/bootcaches.plist"),
        0o644,
    );
    bootability(&mut pb, &format!("{rs}/Bootability"), i.names);
    for d in [
        "App",
        "ExclaveOS",
        "Incoming/App",
        "Incoming/OS",
        "OS",
        "Rosetta",
    ] {
        pb.dirs(&format!("Cryptexes/{d}"), 0o777);
    }
    pb.dirs(&format!("staging/{g}/boot/{boot_dir}"), 0o755);
    pb.dirs(&format!("{g}/var/db"), 0o755);
    pb.bytes(
        &format!("{g}/{EACS_COOKIE}"),
        &eacs_cookie(i.group, i.eacs_store),
        0o644,
    );

    // The main container's Recovery.
    let mut rec = Tree::default();
    rec.dirs(&format!("{g}/boot/{boot_dir}"), 0o755);
    i.boot_dir(
        &mut rec,
        &format!("{g}/boot/{boot_dir}"),
        &i.install,
        &i.boot_manifest,
        (0o755, 0o644),
    )?;
    rec.add(
        &format!("{g}/boot/{boot_dir}/usr/standalone/firmware/arm64eBaseSystem.dmg"),
        Node::Symlink("../../../../../usr/standalone/firmware/arm64eBaseSystem.dmg".into()),
        0o777,
        ROOT,
    );
    rec.dirs(&format!("{g}/usr/standalone/firmware"), 0o777);
    rec.file(
        &format!("{g}/usr/standalone/firmware/arm64eBaseSystem.dmg"),
        Data::Plain(base_system.clone()),
        0o666,
    );
    rec.dirs(&format!("staging/{g}/boot/{boot_dir}"), 0o755);

    let mut update = Tree::default();
    for (d, mode) in [
        ("Controller", 0o711),
        ("Firmware", 0o755),
        ("Hardware", 0o500),
        ("db", 0o777),
    ] {
        update.dirs(d, mode);
    }

    // The recoveryOS container.
    let mut recos = Tree::default();
    let dir = format!("{r}/boot/{sfr_dir}");
    recos.dirs(&dir, 0o777);
    recos.bytes(
        &format!("{dir}/BuildManifest.plist"),
        &i.erase_manifest(),
        0o666,
    );
    i.boot_dir(&mut recos, &dir, &i.erase, &i.sfr_manifest, (0o777, 0o666))?;
    recos.file(
        &format!("{dir}/usr/standalone/firmware/arm64eBaseSystem.dmg"),
        Data::Plain(base_system),
        0o666,
    );

    let entries = |t: Tree| Content::Entries(t.entries);
    let vol = |name, role, group, content| Volume {
        name,
        role,
        uuid: Uuid::new_v4(),
        group,
        content,
    };
    let main = vec![
        vol(
            "Macintosh HD",
            Role::System,
            Some(i.group),
            Content::SealedSystem {
                image: os.clone(),
                mtree: i.path(&i.install, "Ap,SystemVolumeCanonicalMetadata")?,
                root_hash: i.path(&i.install, "SystemVolume")?,
            },
        ),
        vol("Preboot", Role::Preboot, None, entries(pb)),
        vol("Recovery", Role::Recovery, None, entries(rec)),
        vol("Update", Role::Update, None, entries(update)),
    ];
    Ok(Plan {
        isc: vec![
            vol("iSCPreboot", Role::Preboot, None, entries(isc)),
            vol("xART", Role::Xart, None, Content::Entries(Vec::new())),
            vol("Hardware", Role::Hardware, None, entries(hardware)),
            vol(
                "Recovery",
                Role::Recovery,
                None,
                Content::Entries(Vec::new()),
            ),
        ],
        main,
        recovery: vec![vol("Recovery", Role::Recovery, None, entries(recos))],
    })
}

/// The BootabilityBundle, unpacked under `at` as Virtualization.framework's restore leaves it
/// **[vz]**: `BootabilityBundle/Restore/Bootability/*` at its root, the bundle's other files under
/// `Restore/`, and its trust cache again as `Bootability.trustcache`.
fn bootability(t: &mut Tree, at: &str, names: &[String]) {
    const PREFIX: &str = "BootabilityBundle/Restore/";
    t.add(at, Node::Dir, 0o755, BOOTABILITY_OWNER);
    let add = |t: &mut Tree, rel: &str, member: Option<&str>| {
        let mut p = at.to_owned();
        let parts: Vec<&str> = rel.split('/').filter(|s| !s.is_empty()).collect();
        for (n, part) in parts.iter().enumerate() {
            p.push('/');
            p.push_str(part);
            let last = n + 1 == parts.len();
            match member {
                Some(m) if last => t.add(
                    &p,
                    Node::File(Data::Member(m.to_owned())),
                    0o644,
                    BOOTABILITY_OWNER,
                ),
                _ => t.add(&p.clone(), Node::Dir, 0o755, BOOTABILITY_OWNER),
            }
        }
    };
    let mut names: Vec<&String> = names.iter().filter(|n| n.starts_with(PREFIX)).collect();
    names.sort();
    for name in names {
        let rest = &name[PREFIX.len()..];
        if rest.is_empty() {
            continue;
        }
        let dir = name.ends_with('/');
        let rel = match rest.strip_prefix("Bootability/") {
            Some(r) => r.to_owned(),
            None => format!("Restore/{rest}"),
        };
        if rel.trim_end_matches('/').is_empty() {
            continue;
        }
        add(t, &rel, (!dir).then_some(name.as_str()));
        if rest == "Firmware/Bootability.dmg.trustcache" {
            add(t, "Bootability.trustcache", Some(name));
        }
    }
    add(t, "Restore/Bootability/", None);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs<'a>(
        bm: &'a BuildManifest,
        names: &'a [String],
        has: &'a dyn Fn(&str) -> bool,
    ) -> Inputs<'a> {
        let b = |s: &str| -> Arc<[u8]> { s.as_bytes().into() };
        Inputs {
            manifest: bm,
            has,
            names,
            install: bm.vma2_identity(crate::manifest::VARIANT_INSTALL).unwrap(),
            erase: bm.vma2_identity(crate::manifest::VARIANT_ERASE).unwrap(),
            ecid: 0x0123_4567_89ab_cdef,
            group: Uuid::from_u128(0xbf30_4653),
            boot_manifest: b("boot manifest"),
            sfr_manifest: b("sfr manifest"),
            cryptex1_ticket: b("cryptex1"),
            policy: ("P.img4".into(), b("policy")),
            policy_recovery: ("P.recovery.img4".into(), b("policy recovery")),
            recovery_os_policy: b("recovery os policy"),
            ronh: "RONH".into(),
            eacs_store: Uuid::from_u128(0x2471_2d74_0861_48b6_96ce_8cf3_1423_da73),
        }
    }

    /// A manifest with a virtual Mac's install and erase identities and every component the plan
    /// reads, written from scratch.
    fn manifest() -> Vec<u8> {
        let components = [
            ("OS", "os.dmg.aea"),
            (
                "Ap,SystemVolumeCanonicalMetadata",
                "Firmware/os.dmg.aea.mtree",
            ),
            ("BaseSystem", "base.dmg.aea"),
            ("KernelCache", "kernelcache.test"),
            ("DeviceTree", "Firmware/all_flash/dt.im4p"),
            ("SystemVolume", "Firmware/os.dmg.aea.root_hash"),
            ("BaseSystemVolume", "Firmware/base.dmg.aea.root_hash"),
            ("iBoot", "Firmware/all_flash/ib.im4p"),
            ("StaticTrustCache", "Firmware/os.dmg.aea.trustcache"),
            (
                "Ap,BaseSystemTrustCache",
                "Firmware/base.dmg.aea.trustcache",
            ),
            ("Cryptex1,SystemOS", "cx-os.dmg.aea"),
            ("Cryptex1,SystemVolume", "Firmware/cx-os.dmg.aea.root_hash"),
            (
                "Cryptex1,SystemTrustCache",
                "Firmware/cx-os.dmg.aea.trustcache",
            ),
            ("Cryptex1,AppOS", "cx-app.dmg"),
            ("Cryptex1,AppVolume", "Firmware/cx-app.dmg.root_hash"),
            ("Cryptex1,AppTrustCache", "Firmware/cx-app.dmg.trustcache"),
        ];
        let ident = |variant: &str| {
            let mut m = Dictionary::new();
            for (name, path) in components {
                let mut info = Dictionary::new();
                info.insert("Path".into(), Value::String(path.into()));
                let mut c = Dictionary::new();
                c.insert("Digest".into(), Value::Data(vec![1; 48]));
                c.insert("Info".into(), Value::Dictionary(info));
                m.insert(name.into(), Value::Dictionary(c));
            }
            let mut info = Dictionary::new();
            info.insert("DeviceClass".into(), Value::String("vma2macosap".into()));
            info.insert("Variant".into(), Value::String(variant.into()));
            let mut d = Dictionary::new();
            d.insert("ApChipID".into(), Value::String("0xFE00".into()));
            d.insert("ApBoardID".into(), Value::String("0x20".into()));
            d.insert("Info".into(), Value::Dictionary(info));
            d.insert("Manifest".into(), Value::Dictionary(m));
            Value::Dictionary(d)
        };
        let mut root = Dictionary::new();
        root.insert("ProductVersion".into(), Value::String("99.1".into()));
        root.insert("ProductBuildVersion".into(), Value::String("99A1".into()));
        root.insert("ManifestVersion".into(), Value::Integer(1.into()));
        root.insert(
            "BuildIdentities".into(),
            Value::Array(vec![
                ident(crate::manifest::VARIANT_ERASE),
                ident(crate::manifest::VARIANT_INSTALL),
            ]),
        );
        xml_plist::to_xml(&Value::Dictionary(root))
    }

    fn names() -> Vec<String> {
        [
            "BootabilityBundle/Restore/",
            "BootabilityBundle/Restore/Bootability/",
            "BootabilityBundle/Restore/Bootability/BootabilityBrain.framework/Resources",
            "BootabilityBundle/Restore/Bootability/System/Library/CoreServices/RestoreVersion.plist",
            "BootabilityBundle/Restore/BuildManifest.plist",
            "BootabilityBundle/Restore/Firmware/Bootability.dmg.trustcache",
            "BootabilityBundle/Restore/Restore.plist",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
    }

    fn by_path(v: &Volume) -> std::collections::BTreeMap<String, Entry> {
        match &v.content {
            Content::Entries(e) => e.iter().map(|e| (e.path.clone(), e.clone())).collect(),
            _ => Default::default(),
        }
    }

    /// No Data volume, and the cookie that has the first boot make it by EACS, naming the group
    /// and the main container's partition as the obliterator read them **[measured]**.
    #[test]
    fn the_data_volume_is_left_to_eacs_by_the_cookie() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let names = names();
        let has = |_: &str| true;
        let i = inputs(&bm, &names, &has);
        let cookie = format!("{}/{EACS_COOKIE}", i.group.to_string().to_uppercase());
        let p = plan(&i).unwrap();
        let roles: Vec<_> = p.main.iter().map(|v| v.role).collect();
        assert_eq!(
            roles,
            [Role::System, Role::Preboot, Role::Recovery, Role::Update]
        );
        assert_eq!(p.main[0].group, Some(i.group), "the system keeps its group");
        let pb = by_path(&p.main[1]);
        let e = &pb[&cookie];
        assert_eq!((e.mode, e.uid, e.gid), (0o644, 0, 0));
        let Node::File(Data::Bytes(b)) = &e.node else {
            panic!("{:?}", e.node)
        };
        assert_eq!(&b[..], &eacs_cookie(i.group, i.eacs_store)[..]);
        let v = Value::from_reader_xml(&b[..]).unwrap();
        let d = v.as_dictionary().unwrap();
        assert_eq!(d.get("MigrationMode"), Some(&Value::Integer(2.into())));
        let o = d
            .get("MobileObliterationOptions")
            .and_then(Value::as_dictionary)
            .unwrap();
        let s = |k: &str| o.get(k).and_then(Value::as_string).unwrap().to_owned();
        assert_eq!(s("ObliterationType"), "ObliterateDataPartition");
        assert_eq!(
            s("PreservedVolumeGroupUUID"),
            "00000000-0000-0000-0000-0000BF304653"
        );
        assert_eq!(
            s("boot-physical-store-UUID"),
            "24712D74-0861-48B6-96CE-8CF31423DA73"
        );
        assert_eq!(o.get("DisplayProgressBar"), Some(&Value::Boolean(true)));
        // Under the group's `var`, Preboot holds only the cookie and its directories.
        let var = format!("{}/var", i.group.to_string().to_uppercase());
        assert_eq!(pb.keys().filter(|k| k.starts_with(&var)).count(), 3);
    }

    #[test]
    fn the_plan_lays_out_vzs_volumes() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let names = names();
        let has = |_: &str| true;
        let p = plan(&inputs(&bm, &names, &has)).unwrap();
        let roles = |v: &[Volume]| v.iter().map(|v| (v.name, v.role)).collect::<Vec<_>>();
        assert_eq!(
            roles(&p.isc),
            [
                ("iSCPreboot", Role::Preboot),
                ("xART", Role::Xart),
                ("Hardware", Role::Hardware),
                ("Recovery", Role::Recovery)
            ]
        );
        assert_eq!(
            roles(&p.main),
            [
                ("Macintosh HD", Role::System),
                ("Preboot", Role::Preboot),
                ("Recovery", Role::Recovery),
                ("Update", Role::Update)
            ]
        );
        let g = "00000000-0000-0000-0000-0000BF304653";
        assert_eq!(p.main[0].group.unwrap().to_string().to_uppercase(), g);

        let pb = by_path(&p.main[1]);
        let h = hash_name(b"boot manifest");
        assert_eq!(
            pb[&format!("{g}/boot/active")].node,
            Node::File(Data::Bytes(h.clone().into_bytes().into()))
        );
        let k =
            &pb[&format!("{g}/boot/{h}/System/Library/Caches/com.apple.kernelcaches/kernelcache")];
        assert!(
            matches!(&k.node, Node::File(Data::Stitched { component, .. }) if component == "KernelCache")
        );
        assert_eq!(k.mode, 0o644);
        // Each boot directory's root hash is the seal's, stitched to its own manifest.
        let sfr = hash_name(b"sfr manifest");
        let r = RECOVERY_GROUP.to_string().to_uppercase();
        for (v, dir, manifest) in [
            (&p.main[1], format!("{g}/boot/{h}"), "boot manifest"),
            (&p.main[2], format!("{g}/boot/{h}"), "boot manifest"),
            (&p.recovery[0], format!("{r}/boot/{sfr}"), "sfr manifest"),
        ] {
            let e = &by_path(v)[&format!("{dir}/usr/standalone/firmware/root_hash.img4")];
            assert_eq!(
                e.node,
                Node::File(Data::SealedRootHash {
                    manifest: manifest.as_bytes().into()
                }),
                "{} {dir}",
                v.name
            );
        }
        assert_eq!(
            pb[&format!("{g}/cryptex1/current/os.clone.dmg")].node,
            Node::Clone(format!("{g}/cryptex1/current/os.dmg"))
        );
        assert!(pb.contains_key(&format!(
            "{g}/cryptex1/current/apticket.vma2macosap.0123456789ABCDEF.im4m"
        )));
        // Every entry's parent comes before it.
        for v in p.isc.iter().chain(&p.main).chain(&p.recovery) {
            if let Content::Entries(e) = &v.content {
                let mut seen = BTreeSet::new();
                for x in e {
                    if let Some((parent, _)) = x.path.rsplit_once('/') {
                        assert!(seen.contains(parent), "{} before its parent", x.path);
                    }
                    seen.insert(x.path.clone());
                }
            }
        }

        let isc = by_path(&p.isc[0]);
        assert_eq!(isc[&format!("{g}/LocalPolicy/P.img4")].mode, 0o666);
        let sfr = hash_name(b"sfr manifest");
        assert_eq!(isc["SystemRecovery/RONH"].node, Node::Symlink(sfr.clone()));
        assert!(
            isc.contains_key("3D3287DE-280D-4619-AAAB-D97469CA9C71/LocalPolicy/RONH.recovery.img4")
        );
        let recos = by_path(&p.recovery[0]);
        assert_eq!(
            recos[&format!(
                "3D3287DE-280D-4619-AAAB-D97469CA9C71/boot/{sfr}/System/Library/Caches/apticket.der"
            )]
                .mode,
            0o666
        );
    }

    #[test]
    fn the_images_copied_whole_are_those_the_plan_places() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let names = names();
        let has = |_: &str| true;
        let i = inputs(&bm, &names, &has);
        let p = plan(&i).unwrap();
        let mut placed: Vec<String> = [&p.isc, &p.main, &p.recovery]
            .into_iter()
            .flatten()
            .flat_map(|v| match &v.content {
                Content::Entries(e) => e.clone(),
                _ => Vec::new(),
            })
            .filter_map(|e| match e.node {
                Node::File(Data::Plain(m)) => Some(m),
                _ => None,
            })
            .collect();
        placed.sort();
        let mut copied = copied_images(&i.install).unwrap();
        copied.sort();
        assert_eq!(placed, copied);
        assert_eq!(copied.iter().filter(|m| *m == "base.dmg.aea").count(), 2);
    }

    #[test]
    fn the_bootability_bundle_unpacks_as_vzs() {
        let mut t = Tree::default();
        bootability(&mut t, "r/Bootability", &names());
        let got: Vec<(String, bool, (u32, u32))> = t
            .entries
            .iter()
            .map(|e| (e.path.clone(), e.node == Node::Dir, (e.uid, e.gid)))
            .collect();
        let want = [
            ("r/Bootability", true),
            ("r/Bootability/BootabilityBrain.framework", true),
            ("r/Bootability/BootabilityBrain.framework/Resources", false),
            ("r/Bootability/System", true),
            ("r/Bootability/System/Library", true),
            ("r/Bootability/System/Library/CoreServices", true),
            (
                "r/Bootability/System/Library/CoreServices/RestoreVersion.plist",
                false,
            ),
            ("r/Bootability/Restore", true),
            ("r/Bootability/Restore/BuildManifest.plist", false),
            ("r/Bootability/Restore/Firmware", true),
            (
                "r/Bootability/Restore/Firmware/Bootability.dmg.trustcache",
                false,
            ),
            ("r/Bootability/Bootability.trustcache", false),
            ("r/Bootability/Restore/Restore.plist", false),
            ("r/Bootability/Restore/Bootability", true),
        ];
        let want: Vec<(String, bool, (u32, u32))> = want
            .iter()
            .map(|(p, d)| ((*p).to_owned(), *d, (501, 20)))
            .collect();
        assert_eq!(got, want);
    }
}
