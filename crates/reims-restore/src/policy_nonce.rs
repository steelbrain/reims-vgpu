//! The guest's local policy nonce updates, refused (docs/design.md, "The LocalPolicy").
//!
//! At the first account, and at any later change to the macOS LocalPolicy, the guest begins a
//! local policy nonce update, has libbootpolicy decode the current policy to re-sign it for the
//! new nonce, and ends the update, which makes the new nonce the blessed one and removes the
//! policies named for the old. libbootpolicy anchors that decode on Apple's roots, compiled into
//! it in the OS cryptex, so it can't decode a policy of the project's chain: it skips the policy,
//! the update still ends, and the next boot finds no policy. So the restore has Ambrosia make the
//! boot policy kext refuse every local nonce update at its start, the way it already refuses one
//! while another, uninterruptible, runs: the update's first step fails, nothing is rolled or
//! removed, and the policy the restore wrote stays the blessed one.
//!
//! The kext's command for the update begins with
//!
//! ```text
//! ldrb wT, [xN]        ; the local update's state
//! cmp  wT, #2          ; one running?
//! b.ne new             ; no: make the new nonce
//! ldrb wT, [xN, #3]    ; is it interruptible?
//! cmp  wT, #1
//! b.ne refuse          ; no: refuse this one
//! ...                  ; log it, and mark it interrupted:
//! strb wzr, [xN, #3]
//! new: mov w0, #3      ; the proposed local nonce
//! bl   ...             ; made new
//! ```
//!
//! and the patch makes its first `b.ne` a `nop`, so the refusal is taken unless a running update
//! is interruptible, which only that update's own timer makes it. The kext holds the first six
//! instructions' shape in other commands too; the update's beginning is the one whose first
//! branch lands, eight instructions on, on `mov w0, #3` and a `bl` right after
//! `strb wzr, [xN, #3]`. The bytes are read from the IPSW's own boot kernel collection at
//! install time, as the anchors are, and the restore checks there is one such place, and that
//! its bytes occur once in the kext, before it writes the patch.

use anyhow::{Result, ensure};

use crate::anchors::BOOT_POLICY;
use crate::kernelcache::FileSet;

/// The patch's length: the six instructions of the update's state checks.
pub const LEN: usize = 24;
const NOP: u32 = 0xd503_201f;

/// The patch for the guest's boot policy kext: its bytes and what replaces them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub find: Vec<u8>,
    pub replace: Vec<u8>,
}

fn word(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at + 4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
}

/// Whether the six instructions at `at` in `b` begin the local nonce update (the module's doc).
fn begins_update(b: &[u8], at: usize) -> bool {
    let Some(w) = (0..6)
        .map(|i| word(b, at + 4 * i))
        .collect::<Option<Vec<u32>>>()
    else {
        return false;
    };
    // ldrb wT, [xN] (an unsigned offset of 0)
    if w[0] & 0xffff_fc00 != 0x3940_0000 {
        return false;
    }
    let (t, n) = (w[0] & 31, (w[0] >> 5) & 31);
    let b_ne = |x: u32| x & 0xff00_001f == 0x5400_0001;
    let first = w[1] == 0x7100_081f | (t << 5) // cmp wT, #2
        && b_ne(w[2])
        && w[3] == 0x3940_0000 | (3 << 10) | (n << 5) | t // ldrb wT, [xN, #3]
        && w[4] == 0x7100_041f | (t << 5) // cmp wT, #1
        && b_ne(w[5]);
    // The first branch lands eight instructions on: on `mov w0, #3` and a `bl`, right after
    // `strb wzr, [xN, #3]`.
    first
        && (w[2] >> 5) & 0x7ffff == 8
        && word(b, at + 36) == Some(0x3900_0c1f | (n << 5))
        && word(b, at + 40) == Some(0x5280_0060)
        && word(b, at + 44).is_some_and(|x| x >> 26 == 0x25)
}

/// Find the local nonce update's beginning in kernel collection `fs`'s boot policy kext.
///
/// # Errors
///
/// The collection has no boot policy kext, the kext holds no such place or more than one, or
/// the place's bytes occur more than once in the segments Ambrosia patches.
pub fn refusal(fs: &FileSet) -> Result<Refusal> {
    let segments: Vec<&[u8]> = fs
        .segments(BOOT_POLICY)?
        .into_iter()
        .filter(|(n, _)| n != "__LINKEDIT")
        .map(|(_, b)| b)
        .collect();
    let mut found = Vec::new();
    for s in &segments {
        for at in (0..s.len().saturating_sub(LEN - 1)).step_by(4) {
            if begins_update(s, at) {
                found.push(s[at..at + LEN].to_vec());
            }
        }
    }
    ensure!(
        found.len() == 1,
        "{BOOT_POLICY} begins a local policy nonce update in {} places, not one",
        found.len()
    );
    let find = found.remove(0);
    let n: usize = segments
        .iter()
        .map(|s| s.windows(LEN).filter(|w| *w == find.as_slice()).count())
        .sum();
    ensure!(
        n == 1,
        "{BOOT_POLICY} holds its local nonce update's checks {n} times, not once"
    );
    let mut replace = find.clone();
    replace[8..12].copy_from_slice(&NOP.to_le_bytes());
    Ok(Refusal { find, replace })
}

/// Add `r`'s `Kernel > Patch` entry, expecting one replacement, after `config`'s others.
pub fn add_to(config: &mut plist::Dictionary, r: &Refusal) {
    let mut d = plist::Dictionary::new();
    d.insert(
        "Identifier".into(),
        plist::Value::String(BOOT_POLICY.into()),
    );
    d.insert("Find".into(), plist::Value::Data(r.find.clone()));
    d.insert("Replace".into(), plist::Value::Data(r.replace.clone()));
    d.insert("Expect".into(), plist::Value::Integer(1.into()));
    if config.get("Kernel").is_none() {
        config.insert("Kernel".into(), plist::Dictionary::new().into());
    }
    let kernel = config
        .get_mut("Kernel")
        .and_then(plist::Value::as_dictionary_mut)
        .expect("the configuration's Kernel is a dictionary");
    if kernel.get("Patch").is_none() {
        kernel.insert("Patch".into(), plist::Value::Array(Vec::new()));
    }
    kernel
        .get_mut("Patch")
        .and_then(plist::Value::as_array_mut)
        .expect("the configuration's Kernel > Patch is an array")
        .push(plist::Value::Dictionary(d));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernelcache::tests::fileset;

    fn words(ws: &[u32]) -> Vec<u8> {
        ws.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The six checks on wT and xN, the first branch `first` instructions on, the second far.
    fn checks(t: u32, n: u32, first: u32) -> Vec<u32> {
        vec![
            0x3940_0000 | (n << 5) | t,
            0x7100_081f | (t << 5),
            0x5400_0001 | (first << 5),
            0x3940_0000 | (3 << 10) | (n << 5) | t,
            0x7100_041f | (t << 5),
            0x5400_0001 | (0x123 << 5),
        ]
    }

    /// The update's beginning on wT and xN, made here: the checks whose first branch lands eight
    /// instructions on, three instructions of logging, the interrupted mark, and the new nonce
    /// `nonce` made by a call. The branch lands on the `mov w0, #nonce`, the ninth instruction
    /// after it, so the logging is three instructions and the mark one.
    fn begin(t: u32, n: u32, nonce: u32) -> Vec<u32> {
        let mut w = checks(t, n, 8);
        w.extend([0x9000_0000, 0x9100_0000, 0x9400_0010]); // adrp x0; add x0, x0, #0; bl (the log)
        w.push(0x3900_0c1f | (n << 5)); // strb wzr, [xN, #3]
        w.push(0x5280_0000 | (nonce << 5)); // mov w0, #nonce  (the first branch's target)
        w.push(0x9400_0020); // bl
        w
    }

    /// A boot policy kext of `code`, after other bytes so the code isn't at the segment's start.
    fn kernel(code: &[u32]) -> Vec<u8> {
        let mut bp = words(&[0xdead_beef, 0xd65f_03c0]);
        bp.extend(words(code));
        bp.extend(words(&[0xd65f_03c0; 4]));
        fileset(&[("com.apple.kernel", b"the kernel"), (BOOT_POLICY, &bp)])
    }

    #[test]
    fn the_updates_beginning_is_found_among_the_other_checks() {
        // The other commands' checks: one whose first branch lands elsewhere, one that lands
        // eight on but makes another nonce, and the beginning on other registers.
        let mut code = checks(8, 19, 4);
        code.extend([0x3500_0228; 3]);
        code.extend(begin(8, 19, 2));
        code.extend(begin(9, 20, 3));
        code.extend(checks(8, 19, 343));
        let b = kernel(&code);
        let r = refusal(&FileSet::parse(&b).unwrap()).unwrap();
        assert_eq!(r.find, words(&begin(9, 20, 3))[..LEN]);
        // Only the first branch changes, to a nop.
        let mut want = checks(9, 20, 8);
        want[2] = NOP;
        assert_eq!(r.replace, words(&want));
    }

    #[test]
    fn a_kext_without_the_beginning_or_with_two_fails() {
        let mut one = checks(8, 19, 4);
        one.extend(begin(8, 19, 2));
        let b = kernel(&one);
        assert!(refusal(&FileSet::parse(&b).unwrap()).is_err());
        let mut two = begin(8, 19, 3);
        two.extend(begin(10, 21, 3));
        let b = kernel(&two);
        let e = refusal(&FileSet::parse(&b).unwrap()).unwrap_err();
        assert!(e.to_string().contains("in 2 places"), "{e}");
    }

    #[test]
    fn the_beginnings_bytes_must_occur_once() {
        // The same six instructions again elsewhere would take a second replacement.
        let mut code = begin(8, 19, 3);
        code.extend(checks(8, 19, 8));
        let b = kernel(&code);
        let e = refusal(&FileSet::parse(&b).unwrap()).unwrap_err();
        assert!(e.to_string().contains("2 times"), "{e}");
    }

    #[test]
    fn the_patch_goes_after_the_configurations_others() {
        let mut config = plist::Dictionary::new();
        let mut kernel = plist::Dictionary::new();
        kernel.insert(
            "Patch".into(),
            plist::Value::Array(vec![plist::Value::String("an anchor's patch".into())]),
        );
        config.insert("Kernel".into(), plist::Value::Dictionary(kernel));
        let r = Refusal {
            find: vec![1; LEN],
            replace: vec![2; LEN],
        };
        add_to(&mut config, &r);
        let patches = config["Kernel"].as_dictionary().unwrap()["Patch"]
            .as_array()
            .unwrap();
        assert_eq!(patches.len(), 2);
        let p = patches[1].as_dictionary().unwrap();
        assert_eq!(p["Identifier"].as_string(), Some(BOOT_POLICY));
        assert_eq!(p["Find"].as_data(), Some(&[1u8; LEN][..]));
        assert_eq!(p["Replace"].as_data(), Some(&[2u8; LEN][..]));
        assert_eq!(p["Expect"].as_unsigned_integer(), Some(1));
        // A configuration without patches gets the list.
        let mut empty = plist::Dictionary::new();
        add_to(&mut empty, &r);
        assert_eq!(
            empty["Kernel"].as_dictionary().unwrap()["Patch"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }
}
