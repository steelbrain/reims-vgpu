//! The library's link order on Linux: which of its functions the archive lays out first, and in
//! what order, so the code the device's lanes run for each Metal pass sits together and the
//! encoding thread fetches it from fewer cache lines and pages.
//!
//! [`ORDER_FILE`] lists the functions, one a line, by [`key`]: the name GNU `nm -C` prints,
//! without the crate disambiguators and hashes that change from build to build, so the list
//! survives a new metal2vulkan pin and most edits. Linux's `dist` renames the section of each
//! function the list names to `.text.sorted.NNNNNN`, NNNNNN its line's index
//! ([`renames`]); GNU ld's default linker script lays out `SORT(.text.sorted.*)` by name ahead
//! of the rest of `.text`, so QEMU linked by the compiler's default linker gets the order, and
//! a linker that does not sort those sections keeps the order it had. A function the list names
//! that the archive no longer has is skipped: a stale list places fewer functions, and fails
//! nothing.
//!
//! `cargo xtask link-order BINARY CALL_GRAPH` writes the list ([`cmd_link_order`]): the functions
//! of this tree's archive that `CALL_GRAPH` names (lld's `--call-graph-ordering-file`, a profile's
//! calls), in their address order in `BINARY`, a QEMU lld linked with that file, which clusters
//! callers with their callees (docs/development.md, "The link order").

// Linux's dist alone lays the list out; the other OSes build the module for its tests.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::strs_with;
use crate::util::Res;

/// The list, relative to the repository's root.
pub const ORDER_FILE: &str = "crates/reims-vgpu-c/link-order.txt";

/// The section name prefix GNU ld's default script sorts by name ahead of the rest of `.text`.
pub const SORTED: &str = ".text.sorted.";

/// The list's opening comment, which [`entries`] skips.
const HEADER: &str = "\
# The library's functions in the order cargo xtask dist lays them out on Linux, one a line, as
# GNU nm -C prints them without crate disambiguators. Written by cargo xtask link-order; refresh
# it as docs/development.md (\"The link order\") says. A function no longer in the library is
# skipped.
";

/// `cargo xtask link-order`'s usage line.
pub const USAGE: &str = "usage: cargo xtask link-order BINARY CALL_GRAPH";

/// A function's name as the list holds it: `demangled`, `nm -C`'s name, without the crate
/// disambiguators GNU's demangler prints in brackets (`metal2vulkan_runtime[2057445d05ea5865]`)
/// and without a legacy Rust name's hash (`::h` and 16 hex digits). Other names are kept as
/// they are.
pub fn key(demangled: &str) -> String {
    let hex = |s: &str| s.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'));
    let mut out = String::with_capacity(demangled.len());
    let mut rest = demangled;
    while let Some(open) = rest.find('[') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find(']') {
            Some(close) if (5..=16).contains(&close) && hex(&after[..close]) => {
                rest = &after[close + 1..];
            }
            _ => {
                out.push('[');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    let hashed = out.len().checked_sub(19).filter(|&at| {
        out.is_char_boundary(at) && out[at..].starts_with("::h") && hex(&out[at + 3..])
    });
    if let Some(at) = hashed {
        out.truncate(at);
    }
    out
}

/// The list's entries: its lines but the empty ones and the comments (`#`).
pub fn entries(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

/// A function `nm` lists: its address, its name and its [`key`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub address: u64,
    pub name: String,
    pub key: String,
}

/// `nm`'s arguments that list every symbol `file` defines in its symbol table's order, address,
/// type and name a line; demangled with `-C`.
pub fn nm_args(file: &Path, demangle: bool) -> Vec<String> {
    let flags: &[&str] = if demangle {
        &["-p", "--defined-only", "-C"]
    } else {
        &["-p", "--defined-only"]
    };
    strs_with(flags, file)
}

/// The functions (text symbols: `t`, `T`, `w`, `W`) of one file from `nm`'s two listings of it
/// ([`nm_args`]), `raw` and `demangled`, which list the same symbols in the same order. Fails
/// where they do not.
pub fn functions(raw: &str, demangled: &str) -> Res<Vec<Function>> {
    let (raw, demangled): (Vec<&str>, Vec<&str>) =
        (raw.lines().collect(), demangled.lines().collect());
    if raw.len() != demangled.len() {
        return Err(format!(
            "nm listed {} symbols, and {} demangled",
            raw.len(),
            demangled.len()
        ));
    }
    let mut out = Vec::new();
    for (r, d) in raw.iter().zip(&demangled) {
        let mut rf = r.splitn(3, ' ');
        let mut df = d.splitn(3, ' ');
        let (Some(address), Some(kind), Some(name)) = (rf.next(), rf.next(), rf.next()) else {
            continue;
        };
        let (Some(d_address), Some(d_kind), Some(d_name)) = (df.next(), df.next(), df.next())
        else {
            return Err(format!(
                "nm's demangled listing has no symbol for `{r}`: `{d}`"
            ));
        };
        if (address, kind) != (d_address, d_kind) {
            return Err(format!("nm's two listings differ: `{r}`, `{d}`"));
        }
        if !matches!(kind, "t" | "T" | "w" | "W") {
            continue;
        }
        let Ok(address) = u64::from_str_radix(address, 16) else {
            continue;
        };
        out.push(Function {
            address,
            name: name.to_owned(),
            key: key(d_name),
        });
    }
    Ok(out)
}

/// objcopy's arguments that rename the section of each function `order` names, `.text.<name>`,
/// to `.text.sorted.NNNNNN` ([`SORTED`]), NNNNNN the index of the entry that names it, and how
/// many of `order`'s entries name a function of `functions`. A name the archive defines more
/// than once (a function each codegen unit keeps a copy of) is renamed once.
pub fn renames(order: &[&str], functions: &[Function]) -> (Vec<String>, usize) {
    let mut by_key: HashMap<&str, Vec<&str>> = HashMap::new();
    for f in functions {
        by_key.entry(&f.key).or_default().push(&f.name);
    }
    let (mut args, mut placed, mut renamed) = (Vec::new(), 0, HashSet::new());
    for (index, entry) in order.iter().enumerate() {
        let Some(names) = by_key.get(entry) else {
            continue;
        };
        placed += 1;
        for name in names {
            if renamed.insert(*name) {
                args.push("--rename-section".to_owned());
                args.push(format!(".text.{name}={SORTED}{index:06}"));
            }
        }
    }
    (args, placed)
}

/// The list's text: the functions of `binary` that `graph` names (each line's first two
/// fields) and `archive` defines, in their address order in `binary`, each key once, after
/// [`HEADER`].
pub fn list(binary: &[Function], archive: &[Function], graph: &str) -> (String, usize) {
    let named: HashSet<&str> = graph
        .lines()
        .flat_map(|l| l.split_whitespace().take(2))
        .collect();
    let ours: HashSet<&str> = archive.iter().map(|f| f.key.as_str()).collect();
    let mut chosen: Vec<&Function> = binary
        .iter()
        .filter(|f| named.contains(f.name.as_str()) && ours.contains(f.key.as_str()))
        .collect();
    chosen.sort_by_key(|f| f.address);
    let mut text = HEADER.to_owned();
    let (mut seen, mut n) = (HashSet::new(), 0);
    for f in chosen {
        if seen.insert(f.key.as_str()) {
            text.push_str(&f.key);
            text.push('\n');
            n += 1;
        }
    }
    (text, n)
}

/// `cargo xtask link-order BINARY CALL_GRAPH`, on the Linux host that built `BINARY` and this
/// tree's dist: writes [`ORDER_FILE`] ([`list`]), the archive's functions read from the object
/// `dist` linked them into.
#[cfg(target_os = "linux")]
pub fn cmd_link_order(args: &[String]) -> Res<bool> {
    use crate::tools::{System, Tools, target_dir};
    use crate::util::{read, repo_root};
    let [binary, graph] = args else {
        return Err(USAGE.to_owned());
    };
    let mut tools = System;
    let mut listed = |file: &Path| -> Res<Vec<Function>> {
        let raw = tools.output("nm", &nm_args(file, false))?;
        let demangled = tools.output("nm", &nm_args(file, true))?;
        functions(&raw, &demangled)
    };
    let object = target_dir(&mut System)?
        .join(super::archive::LOCALIZED)
        .join(format!("{}.o", super::LIB));
    let archive = listed(&object)?;
    let binary = listed(Path::new(binary))?;
    let (text, n) = list(&binary, &archive, &read(Path::new(graph))?);
    let to = repo_root().join(ORDER_FILE);
    super::write(&to, &text)?;
    println!("link-order: {n} functions into {}", to.display());
    Ok(n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_drops_crate_disambiguators_and_legacy_hashes() {
        assert_eq!(
            key("<metal2vulkan_runtime[2057445d05ea5865]::render::RenderLowering>::end_pass"),
            "<metal2vulkan_runtime::render::RenderLowering>::end_pass"
        );
        assert_eq!(
            key("<alloc[1a2b3c4d5e]::vec::Vec<u8>>::push::<f32>"),
            "<alloc::vec::Vec<u8>>::push::<f32>"
        );
        assert_eq!(
            key("core::fmt::write::h0123456789abcdef"),
            "core::fmt::write"
        );
        // Slices, arrays, short or non-hex brackets and C names are kept.
        for kept in [
            "<[f32] as core::fmt::Debug>::fmt",
            "<[u8; 16] as Clone>::clone",
            "<x[ABCDEF0123]>::y",
            "_rjem_je_arena_ptr_array_flush",
            "a::b::h0123",
        ] {
            assert_eq!(key(kept), kept);
        }
    }

    #[test]
    fn entries_skip_comments_and_empty_lines() {
        assert_eq!(entries("# a\n\nf\n  \ng  \n# b\n"), ["f", "g"]);
    }

    const RAW: &str = "0000000000001000 t _RNvCs1_1a1f\n\
                       0000000000001100 T _RNvCs1_1a1g.llvm.42\n\
                       0000000000002000 d _RNvCs1_1a4DATA\n\
                       0000000000003000 T SDL_Init\n";
    const DEMANGLED: &str = "0000000000001000 t a[0123456789abcdef]::f\n\
                             0000000000001100 T a[0123456789abcdef]::g\n\
                             0000000000002000 d a[0123456789abcdef]::DATA\n\
                             0000000000003000 T SDL_Init\n";

    fn named(address: u64, name: &str, key: &str) -> Function {
        Function {
            address,
            name: name.to_owned(),
            key: key.to_owned(),
        }
    }

    #[test]
    fn functions_pair_the_two_listings_and_keep_text_alone() {
        assert_eq!(
            functions(RAW, DEMANGLED).unwrap(),
            [
                named(0x1000, "_RNvCs1_1a1f", "a::f"),
                named(0x1100, "_RNvCs1_1a1g.llvm.42", "a::g"),
                named(0x3000, "SDL_Init", "SDL_Init"),
            ]
        );
    }

    #[test]
    fn functions_refuse_listings_that_differ() {
        let short = functions(RAW, "0000000000001000 t a::f\n").unwrap_err();
        assert!(short.contains("4 symbols, and 1 demangled"), "{short}");
        let moved = DEMANGLED.replace("1100 T", "1104 T");
        let e = functions(RAW, &moved).unwrap_err();
        assert!(e.contains("differ"), "{e}");
    }

    #[test]
    fn renames_follow_the_order_and_skip_what_the_archive_lacks() {
        let fs = [
            named(0x1000, "_RNvCs1_1a1f", "a::f"),
            named(0x1100, "_RNvCs1_1a1g", "a::g"),
            named(0x1200, "_RNvCs1_1a1g", "a::g"),
            named(0x3000, "SDL_Init", "SDL_Init"),
        ];
        let (args, placed) = renames(&["a::g", "gone::h", "SDL_Init", "a::f"], &fs);
        assert_eq!(placed, 3);
        assert_eq!(
            args,
            [
                "--rename-section",
                ".text._RNvCs1_1a1g=.text.sorted.000000",
                "--rename-section",
                ".text.SDL_Init=.text.sorted.000002",
                "--rename-section",
                ".text._RNvCs1_1a1f=.text.sorted.000003",
            ]
        );
        assert_eq!(renames(&["gone::h"], &fs), (Vec::new(), 0));
    }

    #[test]
    fn list_takes_the_graphs_functions_of_the_archive_in_address_order() {
        let binary = [
            named(0x9000, "_RNvCs2_1a1f", "a::f"),
            named(0x1000, "_RNvCs2_1a1g", "a::g"),
            named(0x2000, "qemu_main", "qemu_main"),
            named(0x3000, "_RNvCs2_1a1h", "a::h"),
            named(0x4000, "_RNvCs2_1a1g", "a::g"),
        ];
        let archive = [
            named(0x10, "_RNvCs1_1a1f", "a::f"),
            named(0x20, "_RNvCs1_1a1g", "a::g"),
            named(0x30, "_RNvCs1_1a1h", "a::h"),
        ];
        let graph = "_RNvCs2_1a1f _RNvCs2_1a1g 40\nqemu_main _RNvCs2_1a1f 3\n";
        let (text, n) = list(&binary, &archive, graph);
        assert_eq!(n, 2);
        assert!(text.starts_with(HEADER), "{text}");
        assert_eq!(entries(&text), ["a::g", "a::f"]);
    }
}
