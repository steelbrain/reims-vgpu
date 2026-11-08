//! Golden test: the crate sends only through the message layer.
//!
//! No product code outside `msg/` calls `objc_msgSend` itself, and none outside `spi/` (the typed
//! declarations objc2-metal lacks) sends with `msg_send!`, so a Metal call is objc2-metal's or
//! `spi`'s, run by the message layer. Every Objective-C call outside `msg/` and `spi/` runs inside
//! a call of the message layer's, which catches an exception at the call: no Objective-C
//! exception reaches a frame of the crate's callers.

// Sorting and searching lists of selector names: no order here is behaviour.
#![cfg(target_os = "macos")] // Metal is macOS's: the crate is empty elsewhere
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::disallowed_methods,
    missing_docs
)]

use std::path::{Path, PathBuf};

/// The production code of a source file: everything before its `#[cfg(test)]` module, without
/// its comment lines.
fn production(src: &str) -> String {
    let code = src.find("#[cfg(test)]").map_or(src, |i| &src[..i]);
    code.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The crate's `src/`, found from the crate's directory, which cargo and nextest give a test
/// process: read at run time, so the test is the same binary in every worktree
/// (docs/development.md, "Building through kache").
fn src_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .expect("CARGO_MANIFEST_DIR: run the tests with cargo or nextest")
        .join("src")
}

/// Every source file under `src/`, relative, with its production text.
fn sources() -> Vec<(PathBuf, String)> {
    let src = src_dir();
    let mut files = Vec::new();
    walk(&src, &mut files);
    files
        .into_iter()
        .map(|f| {
            let text = std::fs::read_to_string(&f).unwrap();
            let rel = f.strip_prefix(&src).unwrap().to_path_buf();
            (rel, production(&text))
        })
        .collect()
}

/// Every source file under `src/`, relative, with its text before its `#[cfg(test)]` module,
/// comments included (so line numbers are the file's).
fn product_sources() -> Vec<(PathBuf, String)> {
    let src = src_dir();
    let mut files = Vec::new();
    walk(&src, &mut files);
    files
        .into_iter()
        .map(|f| {
            let text = std::fs::read_to_string(&f).unwrap();
            let end = text.find("#[cfg(test)]").unwrap_or(text.len());
            let rel = f.strip_prefix(&src).unwrap().to_path_buf();
            (rel, text[..end].to_owned())
        })
        .collect()
}

/// No `objc_msgSend` outside msg/, and no `msg_send!` outside spi/, in product code
#[test]
fn metal_sends_go_through_the_message_layer() {
    let mut stray = Vec::new();
    for (rel, text) in sources() {
        let name = rel.display().to_string();
        let own = rel.starts_with("msg") || rel == Path::new("msg.rs");
        if !own && text.contains("objc_msgSend") {
            stray.push(format!("{name}: objc_msgSend"));
        }
        let foreign = rel.starts_with("spi");
        if !foreign && (text.contains("msg_send![") || text.contains("msg_send!(")) {
            stray.push(format!("{name}: msg_send!"));
        }
    }
    assert!(
        stray.is_empty(),
        "sends outside the message layer: {stray:?}"
    );
}

/// The message layer's calls: an Objective-C call inside the parentheses of a call to one of
/// these, or to one of [`HELPERS`], runs inside its catch.
const CATCHERS: &[&str] = &["send", "get", "catch", "fatal_send", "value"];

/// The crate's helpers that hand their closure (or function) to the message layer, each checked
/// to call it.
const HELPERS: &[&str] = &[
    "create_or_error",
    "create_or_die",
    "part",
    "arguments",
    "class_new",
];

/// Classes whose associated functions are Objective-C calls (`NSString::from_str`,
/// `NSNumber::new_u64`, `IOSurface::alloc`).
const CLASS_PREFIXES: &[&str] = &["NS", "MTL", "IOSurface"];

/// Metal's C functions, which raise as its methods do.
const METAL_FUNCTIONS: &[&str] = &["MTLCopy", "MTLCreate"];

/// `text` with every comment and string literal blanked out, byte for byte (newlines kept), so
/// positions and line numbers stay as they are.
fn blank(text: &str) -> Vec<u8> {
    let mut out = text.as_bytes().to_vec();
    let mut i = 0;
    while i < out.len() {
        if out[i..].starts_with(b"//") {
            while i < out.len() && out[i] != b'\n' {
                out[i] = b' ';
                i += 1;
            }
        } else if out[i] == b'"' {
            i += 1;
            while i < out.len() && out[i] != b'"' {
                let skip = if out[i] == b'\\' { 2 } else { 1 };
                for _ in 0..skip {
                    if i < out.len() && out[i] != b'\n' {
                        out[i] = b' ';
                    }
                    i += 1;
                }
            }
            i += 1;
        } else {
            i += 1;
        }
    }
    out
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Every identifier directly followed by `(`: its start, its end and the path before it (the
/// identifiers and `::` right before it, or `.` for a method).
fn calls(code: &[u8]) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < code.len() {
        if !is_ident(code[i]) || (i > 0 && is_ident(code[i - 1])) {
            i += 1;
            continue;
        }
        let start = i;
        while i < code.len() && is_ident(code[i]) {
            i += 1;
        }
        if code.get(i) != Some(&b'(') || code[start].is_ascii_digit() {
            continue;
        }
        let mut p = start;
        while p > 0 && (is_ident(code[p - 1]) || code[p - 1] == b':' || code[p - 1] == b'.') {
            p -= 1;
        }
        let path = String::from_utf8_lossy(&code[p..start]).into_owned();
        out.push((start, i, path));
    }
    out
}

/// The byte range of the parenthesised arguments that start at `open`.
fn arguments_at(code: &[u8], open: usize) -> std::ops::Range<usize> {
    let mut depth = 0usize;
    for (j, &b) in code.iter().enumerate().skip(open) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return open..j;
                }
            }
            _ => {}
        }
    }
    open..code.len()
}

/// Whether the word before `at` is `fn` (a definition, not a call).
fn defines(code: &[u8], at: usize) -> bool {
    let before = String::from_utf8_lossy(&code[..at]);
    before.trim_end().ends_with("fn")
}

/// Whether a call of `name` with `path` before it is an Objective-C call: a camelCase method or
/// associated function (the crate's own functions are snake_case), an associated function of an
/// Objective-C class, `+alloc`, or a Metal C function. The runtime's own functions (`objc_…`)
/// are not: a pool's pop runs deallocs as a release does, and no catch wraps a release.
fn is_objc_call(name: &str, path: &str) -> bool {
    let camel = !name.starts_with("objc_")
        && name.starts_with(|c: char| c.is_ascii_lowercase())
        && name.contains(|c: char| c.is_ascii_uppercase());
    let member = path.ends_with('.') || path.ends_with("::");
    let class = path.ends_with("::")
        && path
            .trim_end_matches("::")
            .rsplit("::")
            .next()
            .is_some_and(|c| CLASS_PREFIXES.iter().any(|p| c.starts_with(p)));
    let function = path.is_empty() && METAL_FUNCTIONS.iter().any(|p| name.starts_with(p));
    (camel && member) || class || (name == "alloc" && path.ends_with("::")) || function
}

/// Every Objective-C call in product code outside msg/ and spi/ runs inside a call of the
/// message layer's, which catches its exception at the call. The snake_case methods
/// objc2-foundation adds on instances (`insert`, `iter`, `to_string`) are not told apart from
/// Rust's here; review keeps those inside too
#[test]
fn objective_c_calls_run_inside_the_message_layer() {
    let mut stray = Vec::new();
    for (rel, text) in product_sources() {
        if rel.starts_with("msg") || rel == Path::new("msg.rs") || rel.starts_with("spi") {
            continue;
        }
        let code = blank(&text);
        let found = calls(&code);
        let caught: Vec<std::ops::Range<usize>> = found
            .iter()
            .filter(|(start, end, path)| {
                let name = std::str::from_utf8(&code[*start..*end]).unwrap();
                let bare = path.is_empty() || path.ends_with("msg::");
                bare && (CATCHERS.contains(&name) || HELPERS.contains(&name))
                    && !defines(&code, *start)
            })
            .map(|(_, end, _)| arguments_at(&code, *end))
            .collect();
        for (start, end, path) in &found {
            let name = std::str::from_utf8(&code[*start..*end]).unwrap();
            if is_objc_call(name, path) && !caught.iter().any(|r| r.contains(start)) {
                let line = code[..*start].iter().filter(|&&b| b == b'\n').count() + 1;
                stray.push(format!("{}:{line}: {path}{name}", rel.display()));
            }
        }
    }
    assert!(
        stray.is_empty(),
        "Objective-C calls outside the message layer: {stray:#?}"
    );
}

/// Each helper of [`HELPERS`] hands its closure to the message layer
#[test]
fn the_helpers_call_the_message_layer() {
    let all = product_sources();
    for helper in HELPERS {
        let body = all.iter().find_map(|(_, text)| {
            let code = blank(text);
            let text = String::from_utf8(code).unwrap();
            let at = text.find(&format!("fn {helper}"))?;
            let open = at + text[at..].find('{')?;
            let mut depth = 0usize;
            for (j, c) in text[open..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(text[open..=open + j].to_owned());
                        }
                    }
                    _ => {}
                }
            }
            None
        });
        let body = body.unwrap_or_else(|| panic!("no fn {helper} in src/"));
        assert!(
            ["msg::send(", "msg::get(", "msg::catch("]
                .iter()
                .any(|c| body.contains(c)),
            "fn {helper} does not call the message layer:\n{body}"
        );
    }
}
