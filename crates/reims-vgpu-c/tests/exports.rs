//! The C API's boundary (`src/ffi.rs`), checked over this crate's sources, its committed header
//! and its dylib:
//! - every export is `#[unsafe(no_mangle)] pub [unsafe] extern "C" fn rvg_…`, and its whole body
//!   is one call of `ffi::export`, which installs the panic hook and marks the thread as the
//!   library's before anything else runs;
//! - no other function or function-pointer type of the product code has a C ABI, except the
//!   host's callback types, which stay `extern "C-unwind"` for `ffi::host_call`'s catch;
//! - the header declares exactly the exports, and the dylib exports every one of them.
//!
//! Test code (`tests.rs`, `testkit.rs`, the files of a `tests` directory and a file's trailing
//! `mod tests`) is not product code.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The crate's directory, which cargo and nextest give a test process: read at run time, so the
/// test is the same binary in every worktree (docs/development.md, "Building through kache").
fn crate_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .expect("CARGO_MANIFEST_DIR: run the tests with cargo or nextest")
}

/// The line with its `//` comment cut, outside string literals.
fn code(line: &str) -> &str {
    let mut in_str = false;
    let mut prev = ' ';
    for (i, c) in line.char_indices() {
        match c {
            '"' if prev != '\\' => in_str = !in_str,
            '/' if !in_str && prev == '/' => return line.get(..i - 1).unwrap_or(""),
            _ => {}
        }
        prev = c;
    }
    line
}

/// A product file's code lines, comments cut: the file up to its trailing `mod tests`.
fn product_lines(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let end = lines
        .windows(2)
        .position(|w| w[0].trim() == "#[cfg(test)]" && w[1].starts_with("mod tests {"))
        .unwrap_or(lines.len());
    lines[..end].iter().map(|l| code(l).to_owned()).collect()
}

/// Whether `word` stands in `line` as a word of its own.
fn has_word(line: &str, word: &str) -> bool {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    line.match_indices(word).any(|(k, _)| {
        !line[..k].chars().last().is_some_and(ident)
            && !line[k + word.len()..].chars().next().is_some_and(ident)
    })
}

/// What the check found in one file: its exports, and every line that breaks the rule.
#[derive(Debug, Default)]
struct Found {
    exports: BTreeSet<String>,
    errors: Vec<String>,
}

/// The name an export's signature line defines, if it has the export's shape.
fn export_name(line: &str) -> Option<&str> {
    let rest = line
        .strip_prefix("pub unsafe extern \"C\" fn ")
        .or_else(|| line.strip_prefix("pub extern \"C\" fn "))?;
    let name = rest.get(..rest.find('(')?)?;
    name.starts_with("rvg_").then_some(name)
}

/// The rule over one file's code lines (`path` names it in the errors).
fn check(path: &str, lines: &[String]) -> Found {
    let mut found = Found::default();
    let mut i = 0;
    while let Some(line) = lines.get(i) {
        let at = |k: usize| format!("{path}:{}", k + 1);
        if has_word(line, "export_name") {
            found.errors.push(format!("{}: export_name", at(i)));
        }
        if line.contains("no_mangle") {
            if line != "#[unsafe(no_mangle)]" {
                found
                    .errors
                    .push(format!("{}: an export is `#[unsafe(no_mangle)]`", at(i)));
            }
            match lines.get(i + 1).and_then(|l| export_name(l)) {
                None => found.errors.push(format!(
                    "{}: an export is `pub [unsafe] extern \"C\" fn rvg_…`",
                    at(i + 1)
                )),
                Some(name) => {
                    found.exports.insert(name.to_owned());
                    // The signature ends at the first unindented line that opens the body.
                    let Some(open) = (i + 1..lines.len())
                        .find(|&k| lines[k].ends_with('{') && !lines[k].starts_with(' '))
                    else {
                        found
                            .errors
                            .push(format!("{}: `{name}` has no body", at(i + 1)));
                        break;
                    };
                    let close = (open..lines.len()).find(|&k| lines[k] == "}");
                    let body = close.map_or(&[][..], |c| &lines[open + 1..c]);
                    let whole = body.first().is_some_and(|l| l == "    ffi::export(|| {")
                        && body.last().is_some_and(|l| l == "    })")
                        && body.len() >= 2
                        && body[1..body.len() - 1]
                            .iter()
                            .all(|l| l.is_empty() || l.starts_with("        "));
                    if !whole {
                        found.errors.push(format!(
                            "{}: `{name}`'s whole body is one `ffi::export(|| {{ … }})`",
                            at(open)
                        ));
                    }
                    i = close.unwrap_or(lines.len());
                    continue;
                }
            }
        }
        let callback_type = line.contains("extern \"C-unwind\" fn(");
        if has_word(line, "extern") && !line.contains("extern crate") && !callback_type {
            found.errors.push(format!(
                "{}: a C ABI outside the exports and the callback types",
                at(i)
            ));
        }
        i += 1;
    }
    found
}

/// Every product `.rs` file under `dir`, recursively: (path relative to the crate, text).
fn product_files(dir: &Path, out: &mut Vec<(String, String)>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if p.is_dir() {
            if name != "tests" {
                product_files(&p, out);
            }
        } else if name.ends_with(".rs") && name != "tests.rs" && name != "testkit.rs" {
            let rel = p.strip_prefix(crate_dir()).unwrap().display().to_string();
            out.push((rel, std::fs::read_to_string(&p).unwrap()));
        }
    }
}

/// The functions a C header declares (`rvg_name(`), its comments aside.
fn header_functions(text: &str) -> BTreeSet<String> {
    let mut code = String::new();
    let mut rest = text;
    while let Some(k) = rest.find("/*") {
        code.push_str(&rest[..k]);
        rest = rest[k..].find("*/").map_or("", |e| &rest[k + e + 2..]);
    }
    code.push_str(rest);
    let mut out = BTreeSet::new();
    for line in code.lines().map(|l| l.split("//").next().unwrap_or("")) {
        for (k, _) in line.match_indices("rvg_") {
            let before = line[..k].chars().last();
            if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                continue;
            }
            let name: String = line[k..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if line[k + name.len()..].trim_start().starts_with('(') {
                out.insert(name);
            }
        }
    }
    out
}

/// Every export has the boundary's shape, and nothing else of the product code has a C ABI but
/// the host's callback types
#[test]
fn every_export_runs_its_body_inside_ffi_export() {
    let mut files = Vec::new();
    product_files(&crate_dir().join("src"), &mut files);
    let mut exports = BTreeSet::new();
    let mut errors = Vec::new();
    for (path, text) in &files {
        let found = check(path, &product_lines(text));
        for name in found.exports {
            assert!(exports.insert(name.clone()), "{name} is exported twice");
        }
        errors.extend(found.errors);
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
    let header =
        std::fs::read_to_string(crate_dir().join("include/reims_vgpu.h")).expect("the header");
    assert_eq!(exports, header_functions(&header));
}

#[path = "support/cdylib.rs"]
mod cdylib;

/// The dylib exports every function the header declares, and no other `rvg_` function: its
/// defined external symbols, which carry a leading `_` on macOS and none on Linux
#[test]
fn the_dylib_exports_every_header_function() {
    let dylib = cdylib::dir().join(cdylib::file_name());
    assert!(dylib.exists(), "{} is missing", dylib.display());
    let out = Command::new("nm").arg("-gU").arg(&dylib).output().unwrap();
    assert!(out.status.success(), "nm: {out:?}");
    let exported: BTreeSet<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let symbol = l.split_whitespace().last()?;
            let c_name = symbol.strip_prefix('_').unwrap_or(symbol);
            c_name.strip_prefix("rvg_")
        })
        .map(|n| format!("rvg_{n}"))
        .collect();
    let header =
        std::fs::read_to_string(crate_dir().join("include/reims_vgpu.h")).expect("the header");
    assert_eq!(exported, header_functions(&header));
}

/// The rule refuses what it is for: an `extern "C-unwind"` export, a body outside
/// `ffi::export`, work after it, a C ABI elsewhere, `export_name`; it passes an export of the
/// boundary's shape and a callback type
#[test]
fn the_rule_refuses_what_it_is_for() {
    let lines = |t: &str| product_lines(t);
    let good = "#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn rvg_x(\n    a: u32,\n) -> u32 {\n    \
                ffi::export(|| {\n        // SAFETY: none.\n\n        a\n    })\n}\n\
                pub type F = Option<unsafe extern \"C-unwind\" fn(ctx: *mut c_void)>;\n\
                #[cfg(test)]\nmod tests {\n    extern \"C\" fn t() {}\n}\n";
    let f = check("good", &lines(good));
    assert!(f.errors.is_empty(), "{:?}", f.errors);
    assert_eq!(f.exports, BTreeSet::from(["rvg_x".to_owned()]));
    for bad in [
        "#[unsafe(no_mangle)]\npub unsafe extern \"C-unwind\" fn rvg_x() {\n    ffi::export(|| {\n    })\n}\n",
        "#[unsafe(no_mangle)]\npub extern \"C\" fn rvg_x() {\n    a();\n}\n",
        "#[unsafe(no_mangle)]\npub extern \"C\" fn rvg_x() {\n    ffi::export(|| {\n    });\n    a();\n}\n",
        "#[unsafe(no_mangle)]\npub extern \"C\" fn rvg_x() {\n    ffi::export(|| {\n    a();\n    })\n}\n",
        "#[no_mangle]\npub extern \"C\" fn rvg_x() {\n    ffi::export(|| {\n    })\n}\n",
        "extern \"C\" fn cb(ctx: *mut c_void) {}\n",
        "unsafe extern \"C-unwind\" fn cb(ctx: *mut c_void) {}\n",
        "pub type F = Option<extern \"C\" fn()>;\n",
        "#[unsafe(export_name = \"rvg_y\")]\n",
    ] {
        assert!(
            !check("bad", &lines(bad)).errors.is_empty(),
            "passed:\n{bad}"
        );
    }
    assert_eq!(
        header_functions("/* rvg_a(x) */ void rvg_b(int);\n// rvg_c(\nint rvg_d (void);\n"),
        BTreeSet::from(["rvg_b".to_owned(), "rvg_d".to_owned()])
    );
}
