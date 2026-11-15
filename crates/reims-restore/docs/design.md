# reims-restore: design

reims-restore installs macOS from an IPSW onto an arm64 VM's disk, as a restore by
Virtualization.framework (VZ) does, and writes the disk itself: it reads the IPSW where it lies,
asks Apple's signing server (TSS) for the VM's tickets, and lays out the GPT, the three APFS
containers and NVRAM as a fresh VZ restore leaves them. No VM runs, and none of Apple's code: no
AVPBooter, iBoot, restore OS or ASR. It is plain file I/O and HTTPS, so it runs on any host,
arm64 Linux and macOS among them. What it copies onto the disk is the IPSW's own files (the
system volume, the boot objects, the cryptexes, BaseSystem), as an install does.

The result is the raw disk image, the auxiliary storage (`aux.img`) and the ECID. A guest boots
from them under QEMU's `vmapple` on Ambrosia.

Facts tagged **[vz]** come from guests VZ installed: a fresh 27.2.1 (26B131) restore, never
booted, and the set-up 14.6.1, 15.6.1, 26.6.2 and 27.0.1 guests, all read through APFS clones or
read-only attachments. **[ipsw]** facts come from the IPSWs' manifests, of 14.6.1 (23G93), 15.6.1
(24G90), 26.6.2 (25G83), 27.0.1 (26A434) and 27.2.1 (26B131). **[measured]** facts come from runs
on this project's hosts, named where they are. Nothing from an IPSW or a guest is committed: the
tests use inputs written from scratch.

## Not goals

- **Booting under VZ.** The auxiliary storage's firmware region, where VZ's boot ROM finds the
  personalized LLB, stays empty, and the LocalPolicies are signed by our key, not Apple's
  hacktivation key. VZ can't boot our disks; Ambrosia can.
- **In-guest updates that patch the system volume.** Its seal is ours, not Apple's ("The system
  volume's seal"), so a delta update, which checks the running seal against Apple's, doesn't
  apply.
- **Software-update zips.** Their OS comes as file archives, not an APFS image, and their
  recoveryOS as another asset. IPSWs of every supported release stay on Apple's CDN.

## The steps

1. Read the IPSW's `BuildManifest.plist` and pick the virtual Mac's identities (`vma2macosap`):
   `macOS Customer` for the install, `Customer Erase Install (IPSW)` for the SFR manifest.
2. Check the VM's directory and the disk's size (at least 32 GiB); once the OS image is open,
   the space free.
3. Draw the VM's ECID (unless given), the AP nonce, the seven policy nonces and the disk's and
   volume group's UUIDs.
4. Ask TSS for the tickets ("Tickets").
5. Write the GPT, the three containers and their volumes ("The disk"), signing the LocalPolicies
   ("The LocalPolicy").
6. Write NVRAM and the ean storage into the auxiliary storage ("The auxiliary storage", "The
   ean storage").
7. Flush and move the files into place ("Failure handling").

## The disk

512-byte sectors, a protective MBR and a GPT of 128 entries of 128 bytes at LBA 2, with its
backup at the end. Three partitions, all attributes 0, every GUID random **[vz]**:

| # | type | name | where |
|---|---|---|---|
| 1 | `69646961-6700-11AA-AA11-00306543ECAC` | `iBootSystemContainer` | LBA 40, 500 MiB |
| 2 | `7C3457EF-0000-11AA-AA11-00306543ECAC` | `Container` | from the first's end to the third |
| 3 | `52637672-7900-11AA-AA11-00306543ECAC` | `RecoveryOSContainer` | 5 GiB less 40 KiB, ending 7 LBAs before the last usable one |

Every reference guest has this layout on its 64 GiB disk, LBA for LBA. On other sizes the first
and third partitions keep their places from each end (`gpt.rs`).

Each partition holds one APFS container (4 KiB blocks), its volumes unencrypted and
case-insensitive. A fresh restore has these, in this order **[vz]**:

- **iBootSystemContainer:** iSCPreboot (role Preboot), xART (role xART, empty), Hardware (role
  Hardware) and Recovery (role Recovery, empty). iSCPreboot holds:
  - `/<G>/LocalPolicy/<LPNH>.img4` and `<LPNH>.recovery.img4`, the macOS LocalPolicies ("The
    LocalPolicy");
  - `/3D3287DE-280D-4619-AAAB-D97469CA9C71/LocalPolicy/<RONH>.recovery.img4`, the recoveryOS
    container's LocalPolicy, which TSS signs;
  - `/SFR/current/` and `/SystemRecovery/<SFR hash>/`, each with the SFR manifest
    (`apticket.der`), `bootcaches.plist`, `RestoreVersion.plist` and `SystemVersion.plist`.

  Hardware holds the directories `FactoryData/System/Library/Caches/com.apple.factorydata` and
  `MobileActivation`; VZ's restore also leaves the logs of an activation request there, which
  this one doesn't make.
- **Container:** `Macintosh HD` (role System), Preboot, Recovery, Update and Data (role Data).
  System and Data form the volume group G, whose UUID is the Data volume's. There is no VM volume:
  the first boot makes it. For macOS 14 there is no Data volume either ("macOS 14's Data
  volume").
  - **The system volume** is the IPSW's OS image's volume, sealed by reims-apfs with its own
    node layout, so its root hash is ours, not the one the IPSW's signed `root_hash` carries
    ("The system volume's seal"). Its snapshot is `com.apple.os.update-<root hash>`.
  - **Preboot**, under `/<G>/`:
    - `boot/active`, the boot directory's name, and `boot/<H>/`, where H is the boot manifest's
      SHA-384 in upper-case hex: `System/Library/Caches/apticket.der` (the boot manifest), the
      kernel collection (`System/Library/Caches/com.apple.kernelcaches/kernelcache`), and
      `usr/standalone/firmware/{devicetree,root_hash,base_system_root_hash,iBoot}.img4` and
      `FUD/{StaticTrustCache,BaseSystemTrustCache}.img4`, each the IPSW's IM4P stitched to the
      boot manifest, except `root_hash.img4`, whose IM4P is ours ("The system volume's seal");
    - `cryptex1/current/`: the cryptexes' images (`os.dmg`, its clone `os.clone.dmg`, `app.dmg`
      and, where the build has it, `rosetta.dmg`) with their `.root_hash` and `.trustcache`, the
      Cryptex1 ticket (`apticket.vma2macosap.<ECID>.im4m`), the IPSW's unpersonalized one beside
      it, `BuildManifest.plist`, `RestoreVersion.plist` and `SystemVersion.plist`;
    - `restore/`: the restore bundle, `BuildManifest.plist`, both tickets, the restore kernel,
      device tree and iBoot IM4Ps, the images' `.root_hash` and `.trustcache`, the
      BootabilityBundle unpacked, `RestoreVersion.plist`, `SystemVersion.plist` and
      `usr/standalone/bootcaches.plist`;
    - `System/Library/CoreServices/boot.efi`, empty.

    The volume's root also holds empty `staging` and `Cryptexes` directories.
  - **Recovery**: `/<G>/boot/<H>/` with the same stitched objects, and
    `/<G>/usr/standalone/firmware/arm64eBaseSystem.dmg`, the IPSW's BaseSystem, which the boot
    directory links to.
  - **Update**: the directories `Firmware`, `Hardware`, `Controller` and `db`.
  - **Data**: the system volume's data template, `/System/Library/Templates/Data`, copied as
    it is (modes, owners, flags, extended attributes, its own links, and the times the OS image
    holds before it is sealed), except where the 9 patterns of
    `/System/Library/Templates/Data.symlink-patterns` match **[vz]**. Each pattern is a glob from
    the template's root, `*` within one component and `**` any depth:
    - one ending in `/` matches a directory, which becomes a symbolic link to
      `<one ../ per component of its parent>System/Library/Templates/Data/<path>/`, the target
      ending in `/`; nothing below it is copied;
    - any other matches files, each a link of the same form without the final `/`
      (`Library/User Pictures/**` links every file below and keeps its directories).

    A link keeps the permissions, owner and flags of the entry it replaces (`SF_RESTRICTED` on
    the asset directories, 0644 on the pictures), as VZ's do **[vz]**.

    The volume also gets the template's receipt,
    `Library/Apple/System/Library/Receipts/com.apple.files.data-template.plist` (XML, mode
    0666, `SF_RESTRICTED`), listing the patterns without their final `/` (`plan::receipt`). On
    27.2.1 this predicts the 12,796 entries of VZ's Data volume path for path, link for link. VZ's
    restore also rewrites the template's `com.apple.files.data-template.bom` (2,944,805 bytes to
    2,716,958); this restore copies the template's as it is.
- **RecoveryOSContainer:** one Recovery volume: `/3D3287DE-280D-4619-AAAB-D97469CA9C71/boot/<SFR
  hash>/` with `BuildManifest.plist`, the SFR manifest, the stitched objects and
  `arm64eBaseSystem.dmg`.

What Ambrosia reads of this, to boot: the GPT; the main container's system volume (for G); in
Preboot, `boot/active` and the boot directory's objects; in iSCPreboot, the one policy in
`/<G>/LocalPolicy/` that isn't the recovery one, and `/SFR/current/apticket.der`; and NVRAM.
Each of those files is written plain, with no compression: Ambrosia's reader has no decmpfs.

### Writing the containers

`reims-apfs`'s writer (`apfs/src/write/`) forms each container in one transaction, as a freshly
formatted container is before anything mounts it: one checkpoint, empty free queues, the space
manager's main device marking every block it places as used, and every other block left zero, so
a file destination stays sparse. Where the checkpoint areas, the space manager's internal pool and
its bitmap ring go follows from the block count by `newfs_apfs`'s own sizing
(`write/geometry.rs`), checked against containers `diskutil` formatted at 500 MiB, 5 GiB, 60 GiB
and 256 GiB **[measured]**. `ferrosys`, which formats containers to the same layout, is a test
oracle only: it cannot write a sealed volume, and a container holding one cannot be handed to it.

A container records the most volumes it may hold (`nx_max_file_systems`), and the size alone
gives one per 512 MiB: 1 for the 500 MiB iBootSystemContainer, which holds four. A reader takes
only that many of the superblock's volume slots (this crate's reader, given a 500 MiB container
of four volumes sized by the default, listed one **[measured]**), so the writer gives each
container at least as many slots as it holds volumes, and sizes the checkpoint data area, which
grows with that count, from the same figure.

## The system volume's seal

The kernel mounts the system volume only if its sealed root hash equals the one the boot hands it:
Ambrosia unpacks the boot directory's `root_hash.img4` and passes its payload alone, the IM4P's
contents without the manifest, as `/chosen/system-volume-auth-blob`, and names the root snapshot
`com.apple.os.update-<hex of payload bytes 16..48>` (Ambrosia's `tree.rs`). Nothing on that path
checks the payload against the boot manifest's `isys` digest. So this restore seals the volume with
reims-apfs's own node layout and writes its own `root_hash.img4`: an IM4P tagged `isys` whose
payload has the IPSW's layout (208 bytes on 27.2.1 **[ipsw]**: the `u64` 2, the `u32` 1, the
`u32` 0x20, then three 32-byte hashes each followed by 32 zero bytes), the volume's root hash
first. The boot manifest stays the one TSS signed; its `isys` digest names Apple's root hash
IM4P, not ours, which nothing reads under Ambrosia. Reproducing Apple's seal byte for byte would
need the kernel's own B-tree node builder (reims-apfs's notes); a seal of our own needs only a
valid one.

Open: what the payload's second and third hashes are, and whether the kernel reads them; and
whether the kernel's root hash validation has a check of its own beyond the payload (Ambrosia's
lane reads it).

The cost: an in-guest update that patches the system volume (a delta) checks the running seal
against Apple's root hash, so it doesn't apply to our seal; such updates fall back to a full
install at best. Ambrosia already can't re-sign the LocalPolicy in-guest (it runs no SEP).

## Tickets

TSS signs four, each bound to the ECID (`tss.rs`):
- the boot manifest, from the install identity, with the AP nonce (its `BNCH`) and VZ's SEP nonce,
  20 bytes of 0xAA (`snon`);
- the SFR manifest, from the erase identity, with the same nonces;
- the Cryptex1 ticket, with the AP nonce as `Cryptex1,Nonce` in nonce domain 0. Its `cnch` equals
  the boot manifest's `BNCH`, as on a fresh VZ restore **[vz]**;
- the recoveryOS container's LocalPolicy, with `Ap,NextStageIM4MHash` the SFR manifest's
  SHA-384, `Ap,RecoveryOSPolicyNonceHash` the SHA-384 of the fifth policy nonce, and
  `Ap,VolumeUUID` 3D3287DE-280D-4619-AAAB-D97469CA9C71.

The unpersonalized Cryptex1 and restore tickets are the IPSW's own files.

## The LocalPolicy

The macOS LocalPolicy (`policy.rs`) is the LocalPolicy IM4P (`lpol`, version `1.0`, payload one
zero byte) stitched to a manifest whose MANP holds, on a fresh VZ restore of 27.2.1 **[vz]**:

| property | value |
|---|---|
| `BORD`, `CHIP`, `SDOM`, `ECID` | the board, chip, security domain and ECID |
| `CEPO` | 1 |
| `CPRO`, `CSEC`, `lobo`, `hrlp` | true |
| `love` | the OS's version as the boot manifest's `love` carries it, e.g. `26.2.131.0.0,0` |
| `lpnh`, `rpnh` | the SHA-384 of the second and the first policy nonce |
| `nsih` | the boot manifest's SHA-384 |
| `spih` | the Cryptex1 ticket's SHA-384 |
| `stng` | 1 |
| `vuid` | the volume group's UUID, 16 bytes |

and an `lpol` object with the IM4P's SHA-384 (`DGST`) and `EKEY` true. The recoveryOS policy
beside it has `rolp` true in place of `hrlp`, `spih` and `stng`. Each is named for its `lpnh` in
upper-case hex.

VZ's restore OS signs these with a key built into Apple's restore OS, whose chain is `img4 test
secp384r1 Root Certificate Authority`, then `Hacktivation secp384r1 Intermediate Certificate`,
then `Hacktivation secp384r1 Leaf Certificate`; the manifest carries the intermediate and the leaf.
The signature is ECDSA P-384 with SHA-384 over the MANB `SET` **[vz]**. The 27.2.1 kernel checks
it at boot: a policy with one signature byte flipped panics (`boop.c:7620`) **[measured by
Ambrosia's lane]**.

That key is Apple's alone, so the restore signs with the project's own P-384 chain and has
Ambrosia swap the kernel's anchors for it (`anchors.rs`). The kernel (AppleImage4) anchors a
policy's chain on the img4 test root, which kexts of the boot kernel collection embed byte for
byte: `com.apple.security.AppleVPBootPolicy` and `com.apple.security.AppleImage4`, which runs the
check, on every release, and `com.apple.security.Image4` as well on 27.0.1 and 27.2.1, whose
collections are the first to have that kext (14.6.1, 15.6.1 and 26.6.2 don't) **[ipsw]**. The
root, intermediate and leaf are the same lengths on all five, and so is the key's offset. The
boot policy kext also
holds the Hacktivation intermediate and leaf, and a copy of the leaf's public key 84 bytes past
the leaf's end. The kexts keep these as runs of DER walked by length, so each of the project's
certificates is exactly as long as the one it stands in for (610, 833 and 825 bytes on 27.2.1
**[ipsw]**): a private, non-critical filler extension takes up the difference, and the serial is
drawn again until the signature's length lands it. The root is self-signed with `CA:TRUE`; the
intermediate and the leaf are `CA:FALSE` with digitalSignature, nonRepudiation, keyEncipherment
and dataEncipherment, and carry the guest's own img4 extension (1.2.840.113635.100.6.1.15)
verbatim. The policy's manifest keeps its MANB as it is and is signed with the project's leaf,
the intermediate and the leaf in its chain.

The restore reads all of the guest's side from the IPSW's own boot kernel collection
(`kernelcache.rs`: the install identity's `KernelCache`, decompressed, read as a Mach-O file set,
each kext by its `LC_FILESET_ENTRY`), the one Ambrosia loads from Preboot, and finds the anchors
by their shape: the one chain in the boot policy kext's `__TEXT,__const` whose two lower
certificates carry the img4 extension, and its root in each other entry that holds it. It
writes a `Kernel > Patch` entry for each anchor (six on 27, five before it) to Ambrosia's
configuration bundle beside the disks (`config.cpio`, a `newc` cpio of `config.plist`; Ambrosia
passes over other cpio forms): the boot policy kext's intermediate, leaf, leaf key and root,
then the root in each other kext, each expecting one replacement over the kext's segments but
`__LINKEDIT`, so a count that differs stops the boot.
The restore checks each count first, the key's once the leaf is replaced, and fails on any it
doesn't expect. Apple's certificates go only into that VM's bundle, as the patches' `Find`. The
kernel's two production roots, which anchor Cryptex1 tickets and firmware, are left alone: those
still need Apple's signatures. With the swap, the 27.2.1 kernel checks the policy's signature
and its bindings (`nsih`, the nonces, the Cryptex1 ticket) as it would Apple's, with SIP on
**[measured by Ambrosia's lane]**; patching out the check itself (`boop.c:7620`) proved
infeasible there.

The chain's keys are the project's (`--lp-keys`, `root.key`, `int.key`, `leaf.key`): the root and
the intermediate sign each install's certificates at its release's lengths. Whoever can write a
VM's disk can write its NVRAM and its bundle too, so a key per VM would protect nothing. Without
the keys the restore signs with the built-in chain (`keys/`) and writes no bundle.

## The auxiliary storage

`aux.img.trimmed` is the 32 MiB QEMU attaches; `aux.img` is the same behind VZ's 16 KiB header
(bytes `a5 d5 b3 fe 97 ad 1d 27`, then u32 2 at +0x8 and u32 1 at +0x10, little-endian)
**[vz]**. Its regions are listed in the device tree's `/product/util` `regions`: (id, offset,
size) triples, the same in every supported release **[ipsw]**:

| id | offset | size | what |
|---|---|---|---|
| 1 | 0 | 10 MiB | |
| 2 | 0xA00000 | 1 MiB | NVRAM |
| 3 | 0xB00000 | 5 MiB | |
| 7 | 0x1000000 | 4 KiB | |
| 4 | 0x1001000 | 5 MiB less 4 KiB | |
| 5 | 0x1500000 | 5 MiB | |
| 6 | 0x1A00000 | 6 MiB | the ean storage |

The restore writes regions 2 and 6 (`aux.rs`). In VZ's guests region 7 starts `UMPv`, u32 1,
then u32 3 (14.6.1 to 26.6.2) or 4 (27.0.1, 27.2.1) **[vz]**; the restore leaves it empty.

NVRAM is two banks of 512 KiB, each a 32-byte header (a CHRP partition header named `nvram`, then
the Adler-32 of the bank from +0x14 at +0x10, and the generation at +0x14) and two CHRP
partitions: `common` from 0x20 and `system` from 0x60000. Entries are `name=value`, NUL-ended; a
value's runs of up to 127 zero or 0xFF bytes are kept as 0xFF and the run's length, its top bit
set for 0xFF. A fresh VZ restore leaves **[vz]**:
- in `common`: `restore-perform-shutdown=SHUTDOWN`, `restore-outcome=initial_engine_success`;
- in `system`: `auto-boot=true`; `boop-storage-nonces`, seven 16-byte policy nonces (the SHA-384
  of the first is the macOS policy's `rpnh`, of the second its `lpnh`, of the fifth the recoveryOS
  container's policy's `ronh`); `boop-storage-misc`, 24 bytes with 1 at 0, 1 and 12;
  `boot-volume`, the main partition's type and GUID (each in GPT's byte order, read as one
  big-endian number) and G, joined by colons; an empty `boot-args`; `bootdelay=0`.

Both banks are written alike, the first with the higher generation.

## The ean storage

Region 6 is where the kernel's keystore keeps its xART entries: the device tree's `/defaults`
holds `ean-storage-present` and `has-xart` **[ipsw]**. A fresh VZ restore leaves in it
**[vz]**:
- a header: u32 1 at 0, 0xEAEAEAEA at 4, the number of entries at 8, the region's size
  (0x600000) at 0xC, and the end of the last entry's slot at 0x10;
- entries of 0x18 bytes from 0x1EC: the tag as a little-endian fourcc (its letters reversed on
  disk), the slot's offset and length, `field_0c`, the CRC-32 of the whole slot, and a zero;
- the CRC-32 of the first 0xFFC bytes at 0xFFC;
- slots from 0x1000, each whole 4 KiB pages: `aptk` (`field_0c` 1), the SFR manifest, then the
  guest's `xrtP` (5) and `xrtC` (6), a page each.

The restore writes the header and `aptk`. Left empty, every write the guest makes there fails
("AppleVirtualPlatformXART failed to eanWrite proposed (e00002bc)"), the keystore never makes its
D-key ("D-key is not set yet, cant derive M-key"), and the keys it would wrap with it can't be
made **[measured]**. With `aptk` alone, 14.6.1's keystore makes the D-key and the M-key and adds
`xrtC` and `xrtP` after `aptk`, a page each, raising the count to 3 and the end to 0x5000, with
`field_0c` 8 for both **[measured]**. What `field_0c` means is not known.

## macOS 14's Data volume

macOS 14's kernel refuses an unencrypted Data volume: its first boot panics, "unencrypted data
volume is not allowed" (`apfs_vfsops.c:2276`), in the BaseSystem environment before macOS
**[measured]**. VZ's 14.6.1 restore leaves Data encrypted (fs flags 0x100, keybags), and those of
15.6.1, 26.6.2 and 27.2.1 leave it unencrypted, as this restore writes it **[vz]**. Only the
guest's keystore can make an encrypted Data volume, so for macOS 14 (`pipeline::data_by_eacs`)
the restore leaves Data out and has the first boot make it, by EACS (erase all content and
settings):
- Preboot gets `/<G>/var/db/.com.apple.eacs`, an XML plist: `MigrationMode` 2 and
  `MobileObliterationOptions` {`ObliterationType` `ObliterateDataPartition`,
  `PreservedVolumeGroupUUID` G, `boot-physical-store-UUID` the main container's partition GUID,
  `DisplayProgressBar` true} (`plan::eacs_cookie`). At the boot, BaseSystem's
  `migrationModeSwitcher` commits boot mode `eacs`, and `mobile_obliterator` runs its phases,
  makes "Macintosh HD - Data" encrypted, makes the keystore's D-key and M-key in the ean storage,
  and resets the VM: 84 s from the boot mode to "EACS Obliteration completed" on the M3 Max
  under HVF **[measured]**.
- EACS deletes `/<G>/LocalPolicy/*` in iSCPreboot and `boop-storage-nonces` and
  `boop-storage-misc` from NVRAM **[measured]**, and only the restore can sign the macOS
  LocalPolicy, so Ambrosia then stops ("LocalPolicy holds no single policy").
- The restore keeps the iBoot system container's partition as it wrote it (`isc.pre-eacs`,
  sparse) and the auxiliary storage (`aux.pre-eacs`). `pipeline::finish_eacs`, after that boot,
  writes the partition back and puts the two NVRAM entries back into each bank, keeping what the
  guest wrote (the ean storage's keys among it), then removes the copies. It refuses while the
  ean storage holds only `aptk`. The next boot unwraps the D-key, unlocks Data (boot mode
  `_autounlock`) and goes to Setup Assistant **[measured]**.

So a macOS 14 VM takes one boot of the guest before it is ready (`pipeline::FirstBoot::Eacs`):
reims-ctl's `create` and the CLI's `restore` end by naming that boot and the step after it,
`reims-restore finish-eacs DIR`. On a
host, that boot needs what any boot does: QEMU's `vmapple` with Ambrosia, under HVF on a Mac or
KVM on arm64 Linux. Every later release restores without booting.

## The ECID and the result

The tool writes into the VM's directory `disk.img`, `aux.img`, `aux.img.trimmed`, and
`machine-identifier.bin`: VZ's form, a binary plist `{ECID: n}`. The ECID is random unless one is
given. Every ticket and policy is bound to it. For macOS 14 it also writes `isc.pre-eacs` and
`aux.pre-eacs` until `finish_eacs`.

## Reading the IPSW

The IPSW is never unpacked to disk:
- A stored zip member is read in place, at any offset.
- A deflated member is streamed. Every member's CRC-32 is checked as it streams.
- An Apple Encrypted Archive (`*.dmg.aea`, from 15.x on) is decrypted segment by segment
  (`aea.rs`). Its key comes from Apple's key service:
  1. The archive's auth data names `com.apple.wkms.fcs-key-url`. This is checked to be
     `https://wkms-public.apple.com/...`.
  2. That URL answers a P-256 private key in PEM.
  3. The key opens `com.apple.wkms.fcs-response`'s `wrapped-key` with HPKE base mode,
     DHKEM(P-256), HKDF-SHA256 and AES-256-GCM, with empty info and AAD. The result is the
     archive's 32-byte key (profile 1).
- A UDIF image is unpacked chunk by chunk (`udif.rs`): zero, raw, zlib, bzip2, LZFSE, LZMA and
  ADC chunks.
- A system image's APFS container is its UDIF's `Apple_APFS` partition (`image.rs`).

Each image is read where it lies, as the disk is written: nothing is unpacked to a scratch file.

**Peak scratch space:** none beyond the VM's disk and aux, and a few hundred MB of RAM for caches
(UDIF chunks, AEA segments).

## Failure handling

- The disk and aux are written under temporary names, flushed, and renamed into place only once
  every step has succeeded, the machine identifier last; a rename that fails takes back those
  made before it. An interrupted or failed restore leaves no file that looks finished.
- Before it writes, a restore needs the space it writes free where the VM goes: the system
  volume as the OS image allocates it, the images it copies whole as the IPSW stores them (the
  BaseSystem twice, the Cryptex1 images), 2 GiB for the rest (the plan's own files, 86 MB on
  14.6.1; the Data volume's template, 207 MB there **[measured]**; APFS's metadata), and the
  auxiliary storage twice. Every file is written uncompressed, so this is more than a restore by
  Virtualization.framework takes, which compresses some (a 27.2.1 restore took 25.1 GiB from a
  23.8 GiB IPSW **[measured]**). The disk is at least 32 GiB.
- SIGINT or SIGTERM cancels a restore from the CLI between steps, and the temporary files are
  removed. A second signal ends the process at once.
- A member that fails its CRC, an image that fails its manifest digest, or an archive segment
  that fails its HMAC or checksum stops the restore with what failed. A stored member read in
  place, such as the system image, has its CRC-32 checked as it is read front to back: the read
  that reaches its end fails if it is wrong.
- These also stop it with clear errors: TSS refusing (build not signed), the key service
  refusing, network errors after retries, a full disk, and an IPSW without a virtual Mac identity.

## Downloading IPSWs

`reims-restore download latest` (and `reims-ctl create --ipsw latest`) downloads the current
macOS's IPSW from Apple; `reims-restore list` lists it and those downloaded. Apple offers only
its current release: another is restored from an IPSW the user has.

Apple's sources, as **measured** on 2026-11-11 (macOS 27.0.1 host, 27.2.1 current):
- `https://mesu.apple.com/assets/macos/com_apple_macOSIPSW/com_apple_macOSIPSW.xml`, the
  catalog `VZMacOSRestoreImage.fetchLatestSupported` reads (its answer was this catalog's
  VirtualMac2,1 URL). It lists one release for each Mac and for the virtual Mac
  (VirtualMac2,1): the current one, 27.2.1 (26B131), with two builds for some new Macs alone
  (26B7131, 26B8132). Each release replaces the last. VirtualMac2,1's entry has the IPSW's
  version, build and URL but no digest; the physical Macs' entries for the same URL give its
  SHA-1 (`FirmwareSHA1`), which `catalog.rs` takes from them and `download.rs` checks.
- `https://gdmf.apple.com/v2/pmv` lists the versions and builds Apple offers each model,
  the virtual Mac's (`VMA2MACOSAP`) among them, with no URLs.
- Neither the iTunes version list (`s.mzstatic.com/version`) nor the software update catalogs
  list Mac IPSWs.

So no Apple catalog names the IPSWs of earlier releases: their URLs on Apple's CDN hold a UUID
and can't be derived, and the tool keeps no list of them. The owner's call (2026-11-11): the
current release is downloaded, and any other comes from an IPSW the user has.

A download (`download.rs`) goes into `<name>.part` in its directory and only ever appends to
it: it learns the file's length first (a request for its first byte), checks the space the
rest takes, continues a part by a ranged request, and starts over by unlinking the part and
making a new one, never by truncating or rewriting a file in place. A part as long as the file
is only hashed. The whole file's SHA-1 is checked against the catalog's before it is renamed to
Apple's name, so the name never holds a file that isn't whole and checked; one that doesn't
match is removed. Each download holds a lock (`.<name>.lock`), so one runs at a time, looks
for the finished file again once it holds it, and removes it after the rename. A cancel keeps
the part for the next.

## Network

The tool talks to Apple alone: TSS (`gs.apple.com`), the AEA key service
(`wkms-public.apple.com`), Apple's catalog of IPSWs (`mesu.apple.com`) and Apple's CDN for
IPSWs. Files come through one client (`http.rs`): HTTPS to those two hosts alone, no redirect
followed unless its target is one of them. Each request carries the User-Agent
`InetURL/1.0` (`USER_AGENT` in `lib.rs`), the one community restore tools send to these
endpoints, and nothing about the host, its user or this project.

## Dependencies

All are published crates under licenses the repository accepts (GPL-2.0-or-later OR
LGPL-3.0-or-later):
- `aea-tools` (MIT OR Apache-2.0), AEA decryption. Its segment index is replaced here: its
  stream counts clusters from the compressed size and panics past it;
- `hpke` (MIT/Apache-2.0) and `p256` (Apache-2.0 OR MIT), the archive key;
- `zip` (MIT), the IPSW's directory;
- `flate2`, `bzip2`, `lzfse_rust` (MIT OR Apache-2.0) and `lzma-rs` (MIT), decompression.
  `bzip2` decodes with `libbz2-rs-sys`, under bzip2's own license (`bzip2-1.0.6`, BSD-style), as
  the C libbzip2 is;
- `plist` (MIT), `serde_json` and `base64` (MIT OR Apache-2.0);
- `ureq` (MIT OR Apache-2.0) with rustls, HTTPS. Its default roots come from `webpki-roots`
  (CDLA-Permissive-2.0, Mozilla's root store as data); TSS is verified against Apple Root CA
  alone (`tss.rs`);
- `sha2`, `crc32fast`, `getrandom`, `libc`, `uuid`, `anyhow`, `clap` (MIT OR Apache-2.0);
- `p384` (Apache-2.0 OR MIT), the LocalPolicies' signatures;
- `reims-apfs` (`apfs/`, written here), the APFS containers.

Written here:
- Image4 (`img4.rs`): parsing, stitching and checking tickets against components;
- the UDIF reader. The published crates read whole partitions, not random offsets, which the
  system image needs to be read where it lies;
- the GPT, NVRAM and LocalPolicy writers.
