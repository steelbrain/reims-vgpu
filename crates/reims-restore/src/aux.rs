//! The VM's auxiliary storage: 32 MiB QEMU attaches as `aux.img.trimmed`, and the same behind
//! Virtualization.framework's 16 KiB header as `aux.img` (docs/design.md, "The auxiliary
//! storage"). A restore writes two regions of it, NVRAM and the ean storage; the firmware's own
//! regions stay empty.

use anyhow::{Result, ensure};
use uuid::Uuid;

use crate::gpt;

/// The auxiliary storage's size as QEMU attaches it.
pub const SIZE: usize = 32 << 20;
/// Virtualization.framework's header before it in `aux.img`: its magic, version 2 at 0x8, a 1
/// at 0x10, then zeroes to 16 KiB **[vz]**.
pub const HEADER_LEN: usize = 16 << 10;
const MAGIC: [u8; 8] = [0xa5, 0xd5, 0xb3, 0xfe, 0x97, 0xad, 0x1d, 0x27];

/// NVRAM's region (region 2 of the device tree's `/product/util` `regions`): 1 MiB at 0xA00000,
/// two banks of 512 KiB **[ipsw]**.
pub const NVRAM_AT: usize = 0xa0_0000;
const BANK: usize = 0x8_0000;
/// The banks' partitions: `common` from 0x20, `system` from 0x60000, to the bank's end
/// **[vz]**.
const COMMON_AT: usize = 0x20;
const SYSTEM_AT: usize = 0x6_0000;

/// The ean storage (region 6 of `regions`): 6 MiB at 0x1A00000 **[ipsw]**, where the kernel's
/// keystore keeps its xART entries (`/defaults` holds `ean-storage-present` and `has-xart`
/// **[ipsw]**). A fresh VZ restore leaves the SFR manifest in it as `aptk`, and the guest's own
/// `xrtP` and `xrtC` **[vz]**. Left empty, every write the guest makes to it fails ("failed to
/// eanWrite proposed (e00002bc)"), so the keystore never makes its D-key ("D-key is not set
/// yet") **[measured]**; with `aptk` alone, it makes the D-key and the M-key and saves them there
/// as two more entries **[measured]** (docs/design.md, "The ean storage").
pub const EAN_AT: usize = 0x1a0_0000;
const EAN_LEN: usize = 0x60_0000;
/// Its header: u32 1 at 0, 0xEAEAEAEA at 4, the number of entries at 8, the region's size at
/// 0xC, the end of the last entry's slot at 0x10 **[vz]**: the guest raises the last two as it
/// adds entries **[measured]**.
const EAN_MAGIC: u32 = 0xeaea_eaea;
/// Its entries, of 0x18 bytes from 0x1EC: the tag, its letters reversed (a little-endian
/// fourcc), the slot's offset and length, `field_0c`, the CRC-32 of the whole slot, and a zero
/// **[vz]**. `field_0c` is 1 for `aptk`, 5 for `xrtP` and 6 for `xrtC` in VZ's restores, 8 for
/// both of those a guest adds itself **[measured]**; what it means is not known.
const EAN_ENTRIES_AT: usize = 0x1ec;
/// The CRC-32 of the first 0xFFC bytes, at 0xFFC **[vz]**.
const EAN_HEADER_CRC_AT: usize = 0xffc;
/// Slots start at 0x1000 and take whole 4 KiB pages **[vz]**.
const EAN_PAGE: usize = 0x1000;
/// `aptk`'s `field_0c` **[vz]**.
const APTK_FIELD_0C: u32 = 1;

/// What a restore leaves in NVRAM: the values a fresh restore by Virtualization.framework
/// leaves, with this VM's own (docs/design.md, "NVRAM").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nvram {
    /// The policy nonces, `boop-storage-nonces`: seven of 16 bytes. The macOS LocalPolicy's
    /// `rpnh` and `lpnh` are the SHA-384 of the first and the second, the recoveryOS
    /// LocalPolicy's `ronh` of the fifth **[vz]**.
    pub policy_nonces: [[u8; 16]; 7],
    /// The main container's partition GUID.
    pub container: Uuid,
    /// The volume group's UUID, the Data volume's.
    pub group: Uuid,
}

/// `boop-storage-misc` as a fresh restore leaves it: 24 bytes, 1 at 0, 1 and 12 **[vz]**. What
/// they mean is not known.
const BOOP_STORAGE_MISC: [u8; 24] = {
    let mut m = [0u8; 24];
    m[0] = 1;
    m[1] = 1;
    m[12] = 1;
    m
};

impl Nvram {
    /// The policy nonce whose SHA-384 is the macOS LocalPolicy's `rpnh`.
    pub fn remote_policy_nonce(&self) -> &[u8; 16] {
        &self.policy_nonces[0]
    }

    /// The policy nonce whose SHA-384 is the macOS LocalPolicy's `lpnh`, and its file name.
    pub fn local_policy_nonce(&self) -> &[u8; 16] {
        &self.policy_nonces[1]
    }

    /// The policy nonce whose SHA-384 is the recoveryOS LocalPolicy's `ronh`.
    pub fn recovery_policy_nonce(&self) -> &[u8; 16] {
        &self.policy_nonces[4]
    }

    /// `boot-volume`: the main container's partition type and GUID, each in GPT's byte order
    /// read as one big-endian number, and the volume group, as a fresh restore writes it
    /// **[vz]**.
    pub fn boot_volume(&self) -> String {
        let as_stored = |u: Uuid| {
            Uuid::from_bytes(gpt::guid_bytes(u))
                .to_string()
                .to_uppercase()
        };
        format!(
            "{}:{}:{}",
            as_stored(gpt::APFS_TYPE),
            as_stored(self.container),
            self.group.to_string().to_uppercase()
        )
    }

    /// The `common` partition's entries, in order.
    fn common(&self) -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("restore-perform-shutdown", b"SHUTDOWN".to_vec()),
            ("restore-outcome", b"initial_engine_success".to_vec()),
        ]
    }

    /// The `system` partition's entries, in order.
    fn system(&self) -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("auto-boot", b"true".to_vec()),
            ("boop-storage-nonces", self.policy_nonces.concat()),
            ("boop-storage-misc", BOOP_STORAGE_MISC.to_vec()),
            ("boot-volume", self.boot_volume().into_bytes()),
            ("boot-args", Vec::new()),
            ("bootdelay", b"0".to_vec()),
        ]
    }

    /// One bank, of `generation`.
    fn bank(&self, generation: u32) -> Vec<u8> {
        let mut b = vec![0u8; BANK];
        b[..16].copy_from_slice(&chrp_header(0x5a, "nvram", 0x20));
        partition(&mut b[COMMON_AT..SYSTEM_AT], 0x71, "common", &self.common());
        partition(&mut b[SYSTEM_AT..], 0x70, "system", &self.system());
        b[0x14..0x18].copy_from_slice(&generation.to_le_bytes());
        let sum = adler32(&b[0x14..]);
        b[0x10..0x14].copy_from_slice(&sum.to_le_bytes());
        b
    }
}

/// The auxiliary storage as QEMU attaches it (`aux.img.trimmed`): NVRAM's two banks alike, the
/// first the newer, and the ean storage holding `sfr_manifest`.
pub fn trimmed(nvram: &Nvram, sfr_manifest: &[u8]) -> Result<Vec<u8>> {
    let mut a = vec![0u8; SIZE];
    a[NVRAM_AT..NVRAM_AT + BANK].copy_from_slice(&nvram.bank(2));
    a[NVRAM_AT + BANK..NVRAM_AT + 2 * BANK].copy_from_slice(&nvram.bank(1));
    let e = ean(sfr_manifest)?;
    a[EAN_AT..EAN_AT + e.len()].copy_from_slice(&e);
    Ok(a)
}

/// The ean storage's used bytes as a restore leaves them: its header, and one entry, `aptk`,
/// whose slot holds `sfr_manifest`. The rest of the region stays zero.
pub fn ean(sfr_manifest: &[u8]) -> Result<Vec<u8>> {
    let slot = sfr_manifest.len().div_ceil(EAN_PAGE) * EAN_PAGE;
    ensure!(
        slot > 0 && EAN_PAGE + slot <= EAN_LEN,
        "an SFR manifest of {} bytes doesn't fit the ean storage",
        sfr_manifest.len()
    );
    let end = EAN_PAGE + slot;
    let mut e = vec![0u8; end];
    e[EAN_PAGE..EAN_PAGE + sfr_manifest.len()].copy_from_slice(sfr_manifest);
    let words = |e: &mut [u8], at: usize, w: &[u32]| {
        for (i, x) in w.iter().enumerate() {
            e[at + 4 * i..at + 4 * i + 4].copy_from_slice(&x.to_le_bytes());
        }
    };
    let as_u32 = |n: usize| u32::try_from(n).expect("the ean storage is 6 MiB");
    words(&mut e, 0, &[1, EAN_MAGIC, 1, as_u32(EAN_LEN), as_u32(end)]);
    let crc = crc32fast::hash(&e[EAN_PAGE..end]);
    words(
        &mut e,
        EAN_ENTRIES_AT,
        &[
            u32::from_be_bytes(*b"aptk"),
            as_u32(EAN_PAGE),
            as_u32(slot),
            APTK_FIELD_0C,
            crc,
            0,
        ],
    );
    let header = crc32fast::hash(&e[..EAN_HEADER_CRC_AT]);
    words(&mut e, EAN_HEADER_CRC_AT, &[header]);
    Ok(e)
}

/// The number of entries in the ean storage `region` (from [`EAN_AT`]), where its header checks:
/// the restore writes one, and the guest's keystore adds its own as it makes its keys.
pub fn ean_entries(region: &[u8]) -> Option<u32> {
    let word = |at: usize| Some(u32::from_le_bytes(region.get(at..at + 4)?.try_into().ok()?));
    let crc = crc32fast::hash(region.get(..EAN_HEADER_CRC_AT)?);
    (word(0)? == 1 && word(4)? == EAN_MAGIC && word(EAN_HEADER_CRC_AT)? == crc).then(|| word(8))?
}

/// The NVRAM entries a guest's EACS boot deletes that the restore's LocalPolicies are bound to:
/// it leaves `system` without them **[measured]**.
pub const POLICY_ENTRIES: [&str; 2] = ["boop-storage-nonces", "boop-storage-misc"];

/// Puts [`POLICY_ENTRIES`] from the newest valid bank of `restored` (the auxiliary storage as
/// the restore wrote it) back into the `system` partition of each valid bank of `aux` (as the
/// guest left it), keeping the guest's other entries, and checksums each bank again. Refuses
/// where `restored` has no valid bank holding them, or `aux` none at all.
pub fn put_back_policy_entries(aux: &mut [u8], restored: &[u8]) -> Result<()> {
    ensure!(
        aux.len() == SIZE && restored.len() == SIZE,
        "an auxiliary storage is {} bytes",
        SIZE
    );
    let newest = (0..2)
        .filter_map(|i| {
            let b = &restored[NVRAM_AT + i * BANK..NVRAM_AT + (i + 1) * BANK];
            Some((
                u32::from_le_bytes(b[0x14..0x18].try_into().ok()?),
                read_bank(b)?,
            ))
        })
        .max_by_key(|(g, _)| *g)
        .map(|(_, p)| p);
    let wanted: Vec<(String, Vec<u8>)> = newest
        .iter()
        .flatten()
        .filter(|(n, _)| n == "system")
        .flat_map(|(_, e)| e.iter().cloned())
        .filter(|(k, _)| POLICY_ENTRIES.contains(&k.as_str()))
        .collect();
    ensure!(
        wanted.len() == POLICY_ENTRIES.len(),
        "the restore's NVRAM holds no {}",
        POLICY_ENTRIES.join(" or ")
    );
    let mut put = 0;
    for i in 0..2 {
        let bank = &mut aux[NVRAM_AT + i * BANK..NVRAM_AT + (i + 1) * BANK];
        let Some(parts) = read_bank(bank) else {
            continue;
        };
        let mut o = 0x20;
        for (name, entries) in parts {
            let len = usize::from(u16::from_le_bytes([bank[o + 2], bank[o + 3]])) * 16;
            if name == "system" {
                let sig = bank[o];
                let mut kept: Vec<(String, Vec<u8>)> = entries
                    .into_iter()
                    .filter(|(k, _)| !POLICY_ENTRIES.contains(&k.as_str()))
                    .collect();
                kept.extend(wanted.iter().cloned());
                let need: usize = 16
                    + kept
                        .iter()
                        .map(|(k, v)| k.len() + 2 + escape(v).len())
                        .sum::<usize>();
                ensure!(
                    need <= len,
                    "NVRAM's system partition has no room for its entries"
                );
                let at = &mut bank[o..o + len];
                at.fill(0);
                let refs: Vec<(&str, Vec<u8>)> =
                    kept.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
                partition(at, sig, "system", &refs);
                put += 1;
            }
            o += len;
        }
        let sum = adler32(&bank[0x14..]);
        bank[0x10..0x14].copy_from_slice(&sum.to_le_bytes());
    }
    ensure!(
        put > 0,
        "the VM's NVRAM has no valid bank with a system partition"
    );
    Ok(())
}

/// Virtualization.framework's header for `aux.img`.
pub fn vz_header() -> Vec<u8> {
    let mut h = vec![0u8; HEADER_LEN];
    h[..8].copy_from_slice(&MAGIC);
    h[8..12].copy_from_slice(&2u32.to_le_bytes());
    h[0x10..0x14].copy_from_slice(&1u32.to_le_bytes());
    h
}

/// A 16-byte CHRP partition header: signature, checksum, length in 16-byte units
/// (little-endian), name.
fn chrp_header(sig: u8, name: &str, len: usize) -> [u8; 16] {
    let mut h = [0u8; 16];
    h[0] = sig;
    h[2..4].copy_from_slice(&u16::try_from(len / 16).unwrap_or(u16::MAX).to_le_bytes());
    h[4..4 + name.len()].copy_from_slice(name.as_bytes());
    let mut s = u32::from(h[0]);
    for x in &h[2..] {
        s += u32::from(*x);
        s = (s & 0xff) + (s >> 8);
    }
    h[1] = s as u8;
    h
}

/// Fills `at` (the whole partition) with its header and its `name=value` entries, each
/// NUL-terminated, values escaped.
fn partition(at: &mut [u8], sig: u8, name: &str, entries: &[(&str, Vec<u8>)]) {
    let len = at.len();
    at[..16].copy_from_slice(&chrp_header(sig, name, len));
    let mut o = 16;
    for (k, v) in entries {
        let mut e = k.as_bytes().to_vec();
        e.push(b'=');
        e.extend(escape(v));
        e.push(0);
        at[o..o + e.len()].copy_from_slice(&e);
        o += e.len();
    }
}

/// A value as NVRAM keeps it: a run of up to 127 zero or 0xFF bytes becomes 0xFF and its count,
/// with the top bit set for 0xFF; every other byte stays.
pub fn escape(v: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len());
    let mut i = 0;
    while i < v.len() {
        let b = v[i];
        if b == 0 || b == 0xff {
            let mut n = 1;
            while i + n < v.len() && v[i + n] == b && n < 0x7f {
                n += 1;
            }
            out.push(0xff);
            out.push(n as u8 | if b == 0xff { 0x80 } else { 0 });
            i += n;
        } else {
            out.push(b);
            i += 1;
        }
    }
    out
}

/// The inverse of [`escape`].
pub fn unescape(v: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len());
    let mut i = 0;
    while i < v.len() {
        if v[i] == 0xff && i + 1 < v.len() {
            let fill = if v[i + 1] & 0x80 != 0 { 0xff } else { 0 };
            out.extend(std::iter::repeat_n(fill, usize::from(v[i + 1] & 0x7f)));
            i += 2;
        } else {
            out.push(v[i]);
            i += 1;
        }
    }
    out
}

/// Adler-32.
fn adler32(d: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in d.chunks(5552) {
        for x in chunk {
            a += u32::from(*x);
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

/// A partition read back: its name and its `name=value` entries, values unescaped.
pub type Partition = (String, Vec<(String, Vec<u8>)>);

/// A bank's entries, `partition -> [(name, value)]`, values unescaped: what the tests and
/// `inspect` read back.
pub fn read_bank(bank: &[u8]) -> Option<Vec<Partition>> {
    if bank.len() < 0x20
        || adler32(&bank[0x14..]) != u32::from_le_bytes(bank[0x10..0x14].try_into().ok()?)
    {
        return None;
    }
    let mut parts = Vec::new();
    let mut o = 0x20;
    while o + 16 <= bank.len() {
        let len = usize::from(u16::from_le_bytes([bank[o + 2], bank[o + 3]])) * 16;
        if bank[o] == 0 || len < 16 || o + len > bank.len() {
            break;
        }
        let name = String::from_utf8_lossy(&bank[o + 4..o + 16])
            .trim_end_matches('\0')
            .to_owned();
        let mut entries = Vec::new();
        for e in bank[o + 16..o + len]
            .split(|b| *b == 0)
            .filter(|e| !e.is_empty())
        {
            let at = e.iter().position(|b| *b == b'=').unwrap_or(e.len());
            let v = e.get(at + 1..).unwrap_or_default();
            entries.push((String::from_utf8_lossy(&e[..at]).into_owned(), unescape(v)));
        }
        parts.push((name, entries));
        o += len;
    }
    Some(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nvram() -> Nvram {
        let mut n = [[0u8; 16]; 7];
        for (i, x) in n.iter_mut().enumerate() {
            x.fill(i as u8 * 0x11);
        }
        n[3] = [0xff; 16];
        Nvram {
            policy_nonces: n,
            container: Uuid::from_u128(0x24712d74_0861_48b6_96ce_8cf31423da73),
            group: Uuid::from_u128(0xbf304653_4074_415b_8d52_3b779d769491),
        }
    }

    /// An SFR manifest of the tests' own: 5000 bytes, none of them Apple's.
    fn ticket() -> Vec<u8> {
        (0..5000u32).map(|i| (i * 7 + 3) as u8).collect()
    }

    fn word(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    /// The layout a fresh VZ restore leaves **[vz]**, with a manifest of the tests' own: the
    /// header, `aptk`'s entry and its slot, each CRC-32 over what the guest checks.
    #[test]
    fn the_ean_storage_holds_the_sfr_manifest_as_aptk() {
        assert_eq!(
            crc32fast::hash(b"123456789"),
            0xcbf4_3926,
            "CRC-32 (ISO-HDLC)"
        );
        let t = ticket();
        let e = ean(&t).unwrap();
        assert_eq!(e.len(), 0x3000, "the header page and two pages of slot");
        assert_eq!(
            (0..5).map(|i| word(&e, 4 * i)).collect::<Vec<_>>(),
            [1, 0xeaea_eaea, 1, 0x60_0000, 0x3000]
        );
        assert!(e[0x14..0x1ec].iter().all(|b| *b == 0));
        assert_eq!(&e[0x1ec..0x1f0], b"ktpa", "the tag's letters reversed");
        assert_eq!(
            (0..5).map(|i| word(&e, 0x1f0 + 4 * i)).collect::<Vec<_>>(),
            [0x1000, 0x2000, 1, crc32fast::hash(&e[0x1000..0x3000]), 0]
        );
        assert!(e[0x204..0xffc].iter().all(|b| *b == 0), "one entry");
        assert_eq!(word(&e, 0xffc), crc32fast::hash(&e[..0xffc]));
        assert_eq!(&e[0x1000..0x1000 + t.len()], &t[..]);
        assert!(e[0x1000 + t.len()..].iter().all(|b| *b == 0));
    }

    #[test]
    fn aptks_slot_takes_whole_pages() {
        for (len, slot) in [(1, 0x1000), (0x1000, 0x1000), (0x1001, 0x2000)] {
            let e = ean(&vec![0x5a; len]).unwrap();
            assert_eq!(e.len(), 0x1000 + slot, "{len}");
            assert_eq!(word(&e, 0x10), 0x1000 + slot as u32);
            assert_eq!(word(&e, 0x1f4), slot as u32);
        }
    }

    #[test]
    fn a_manifest_the_ean_storage_cant_hold_is_refused() {
        assert!(ean(&[]).is_err());
        assert!(ean(&vec![1; 0x60_0000 - 0x1000]).is_ok());
        assert!(ean(&vec![1; 0x60_0000 - 0x1000 + 1]).is_err());
    }

    /// The guest adds its entries after the restore's, raising the count; a header whose CRC-32
    /// fails doesn't count.
    #[test]
    fn the_ean_storage_counts_its_entries_where_its_header_checks() {
        let mut e = ean(&ticket()).unwrap();
        assert_eq!(ean_entries(&e), Some(1));
        for (k, tag) in [(1usize, *b"xrtC"), (2, *b"xrtP")] {
            let at = 0x1ec + 0x18 * k;
            e[at..at + 4].copy_from_slice(&u32::from_be_bytes(tag).to_le_bytes());
        }
        e[8..12].copy_from_slice(&3u32.to_le_bytes());
        let crc = crc32fast::hash(&e[..0xffc]);
        e[0xffc..0x1000].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(ean_entries(&e), Some(3));
        e[0x20] ^= 1;
        assert_eq!(ean_entries(&e), None);
        assert_eq!(ean_entries(&[0u8; 0x1000]), None, "an empty region");
        assert_eq!(ean_entries(&[]), None);
    }

    /// The guest's EACS boot leaves `system` without the policy entries and with entries of its
    /// own; putting them back keeps the guest's, in both banks, each checksummed again.
    #[test]
    fn the_policy_entries_go_back_beside_the_guests() {
        let n = nvram();
        let restored = trimmed(&n, &ticket()).unwrap();
        let mut after = restored.clone();
        for i in 0..2 {
            let bank = &mut after[NVRAM_AT + i * BANK..NVRAM_AT + (i + 1) * BANK];
            let at = &mut bank[SYSTEM_AT..];
            at.fill(0);
            partition(
                at,
                0x70,
                "system",
                &[
                    ("auto-boot", b"true".to_vec()),
                    ("guest-own", vec![0, 0, 7, 0xff]),
                ],
            );
            let sum = adler32(&bank[0x14..]);
            bank[0x10..0x14].copy_from_slice(&sum.to_le_bytes());
        }
        let common_before = after[NVRAM_AT + COMMON_AT..NVRAM_AT + SYSTEM_AT].to_vec();
        put_back_policy_entries(&mut after, &restored).unwrap();
        for i in 0..2 {
            let bank = &after[NVRAM_AT + i * BANK..NVRAM_AT + (i + 1) * BANK];
            let parts = read_bank(bank).expect("the bank checks again");
            let system = &parts.iter().find(|(p, _)| p == "system").unwrap().1;
            let names: Vec<&str> = system.iter().map(|(k, _)| k.as_str()).collect();
            assert_eq!(
                names,
                [
                    "auto-boot",
                    "guest-own",
                    "boop-storage-nonces",
                    "boop-storage-misc"
                ]
            );
            assert_eq!(system[1].1, [0, 0, 7, 0xff]);
            assert_eq!(system[2].1, n.policy_nonces.concat());
            assert_eq!(system[3].1, BOOP_STORAGE_MISC);
        }
        assert_eq!(
            &after[NVRAM_AT + COMMON_AT..NVRAM_AT + SYSTEM_AT],
            &common_before[..],
            "common stays"
        );
        assert_eq!(
            &after[EAN_AT..],
            &restored[EAN_AT..],
            "the ean storage stays"
        );
        // Again changes nothing.
        let once = after.clone();
        put_back_policy_entries(&mut after, &restored).unwrap();
        assert_eq!(after, once);
    }

    #[test]
    fn policy_entries_the_restore_never_wrote_are_refused() {
        let mut a = trimmed(&nvram(), &ticket()).unwrap();
        let blank = vec![0u8; SIZE];
        assert!(put_back_policy_entries(&mut a, &blank).is_err());
        let mut blank2 = vec![0u8; SIZE];
        let restored = trimmed(&nvram(), &ticket()).unwrap();
        assert!(put_back_policy_entries(&mut blank2, &restored).is_err());
    }

    /// The form a fresh restore writes **[vz]**, with UUIDs of the tests' own.
    #[test]
    fn boot_volume_names_the_container_as_gpt_stores_it() {
        assert_eq!(
            nvram().boot_volume(),
            "EF57347C-0000-AA11-AA11-00306543ECAC:742D7124-6108-B648-96CE-8CF31423DA73:\
             BF304653-4074-415B-8D52-3B779D769491"
        );
    }

    #[test]
    fn values_escape_their_zero_and_ff_runs() {
        let v = [
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(escape(&v), [1, 1, 0xff, 10, 1, 0xff, 11]);
        let long = [0xffu8; 200];
        assert_eq!(escape(&long), [0xff, 0xff, 0xff, 0x80 | 73]);
        for v in [&v[..], &long[..], b"plain", &[]] {
            assert_eq!(unescape(&escape(v)), v);
        }
        assert!(!escape(&nvram().policy_nonces.concat()).contains(&0));
    }

    #[test]
    fn both_banks_read_back_with_the_restores_values() {
        let a = trimmed(&nvram(), &ticket()).unwrap();
        assert_eq!(a.len(), SIZE);
        for (i, generation) in [(0, 2u32), (1, 1)] {
            let bank = &a[NVRAM_AT + i * BANK..NVRAM_AT + (i + 1) * BANK];
            assert_eq!(&bank[..16], b"\x5a\x82\x02\x00nvram\0\0\0\0\0\0\0");
            assert_eq!(
                u32::from_le_bytes(bank[0x14..0x18].try_into().unwrap()),
                generation
            );
            let parts = read_bank(bank).expect("the bank checks");
            let names: Vec<&str> = parts.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, ["common", "system"]);
            let system: Vec<&str> = parts[1].1.iter().map(|(k, _)| k.as_str()).collect();
            assert_eq!(
                system,
                [
                    "auto-boot",
                    "boop-storage-nonces",
                    "boop-storage-misc",
                    "boot-volume",
                    "boot-args",
                    "bootdelay"
                ]
            );
            assert_eq!(parts[1].1[1].1, nvram().policy_nonces.concat());
            assert_eq!(parts[1].1[2].1, BOOP_STORAGE_MISC);
            assert_eq!(
                parts[0].1[1],
                ("restore-outcome".into(), b"initial_engine_success".to_vec())
            );
        }
        // The partitions' headers carry the checksums a fresh restore's do **[vz]**.
        assert_eq!(
            &a[NVRAM_AT + COMMON_AT..NVRAM_AT + COMMON_AT + 4],
            &[0x71, 0x5b, 0xfe, 0x5f]
        );
        assert_eq!(
            &a[NVRAM_AT + SYSTEM_AT..NVRAM_AT + SYSTEM_AT + 4],
            &[0x70, 0x38, 0x00, 0x20]
        );
        // Nothing outside NVRAM and the ean storage's header and `aptk` is written.
        assert!(a[..NVRAM_AT].iter().all(|b| *b == 0));
        assert!(a[NVRAM_AT + 2 * BANK..EAN_AT].iter().all(|b| *b == 0));
        assert_eq!(&a[EAN_AT..EAN_AT + 0x3000], &ean(&ticket()).unwrap()[..]);
        assert!(a[EAN_AT + 0x3000..].iter().all(|b| *b == 0));
    }

    // Ambrosia boots from the valid bank of the highest generation, a bank valid where it opens
    // with 0x5a and `nvram` and its Adler-32 at 0x10 covers it from 0x14 (its `nvram.rs`); with
    // no valid bank it stops before the kernel. Both banks the restore writes pass that check,
    // the first current, and each carries the policy nonces the LocalPolicies are bound to.
    #[test]
    fn both_banks_pass_the_firmwares_check() {
        let n = nvram();
        let a = trimmed(&n, &ticket()).unwrap();
        let valid = |b: &[u8]| {
            b[0] == 0x5a
                && &b[4..9] == b"nvram"
                && adler32(&b[0x14..]) == u32::from_le_bytes(b[0x10..0x14].try_into().unwrap())
        };
        let banks: Vec<&[u8]> = (0..2)
            .map(|i| &a[NVRAM_AT + i * BANK..NVRAM_AT + (i + 1) * BANK])
            .collect();
        assert!(banks.iter().all(|b| valid(b)));
        let current = (0..2)
            .max_by_key(|&i| u32::from_le_bytes(banks[i][0x14..0x18].try_into().unwrap()))
            .unwrap();
        assert_eq!(current, 0);
        for b in banks {
            let parts = read_bank(b).unwrap();
            let nonces = &parts[1]
                .1
                .iter()
                .find(|(k, _)| k == "boop-storage-nonces")
                .unwrap()
                .1;
            assert_eq!(&nonces[..16], n.remote_policy_nonce());
            assert_eq!(&nonces[16..32], n.local_policy_nonce());
        }
    }

    #[test]
    fn a_damaged_bank_does_not_read() {
        let mut a = trimmed(&nvram(), &ticket()).unwrap();
        a[NVRAM_AT + 0x100] ^= 1;
        assert!(read_bank(&a[NVRAM_AT..NVRAM_AT + BANK]).is_none());
    }

    #[test]
    fn the_vz_header_is_magic_version_and_one() {
        let h = vz_header();
        assert_eq!(
            &h[..0x14],
            b"\xa5\xd5\xb3\xfe\x97\xad\x1d\x27\x02\0\0\0\0\0\0\0\x01\0\0\0"
        );
        assert!(h[0x14..].iter().all(|b| *b == 0));
    }
}
