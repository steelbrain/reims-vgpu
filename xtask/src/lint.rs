//! The tree's lints that read files, not code:
//! - the testkit manifest check (D50): the testkit stays out of the product. The Vulkan crate's
//!   `core` module declares its `testkit` module under `#[cfg(any(test, feature = "testkit"))]`,
//!   and only dev-dependencies turn the crate's feature on, so a product build cannot name it;
//! - the paths' check ([`windows_paths`]): every path git tracks checks out on Windows.

/// A check's result: errors fail the command.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Report {
    pub errors: Vec<String>,
}

/// The gate core's `core.rs` puts on its testkit module.
const GATE: &str = "#[cfg(any(test, feature = \"testkit\"))]";

/// Checks core's `core.rs` (`lib`, its text) and the crates' manifests (path, text).
pub fn testkit_manifests(lib: &str, manifests: &[(String, String)]) -> Report {
    let mut rep = Report::default();
    let mut prev = "";
    for (i, line) in lib.lines().enumerate() {
        let t = line.trim();
        if (t.starts_with("pub mod testkit") || t.starts_with("mod testkit")) && prev != GATE {
            rep.errors.push(format!(
                "crates/reims-vgpu-vulkan/src/core.rs:{}: `mod testkit` must be under {GATE} (D50)",
                i + 1
            ));
        }
        if !t.is_empty() {
            prev = t;
        }
    }
    for (path, text) in manifests {
        let mut section = String::new();
        for (i, line) in text.lines().enumerate() {
            let t = line.trim();
            if t.starts_with('[') {
                section = t.to_owned();
                continue;
            }
            let dev = section.contains("dev-dependencies");
            let deps = section.contains("dependencies");
            if deps && !dev && t.starts_with("reims-vgpu-vulkan") && t.contains("\"testkit\"") {
                rep.errors.push(format!(
                    "{path}:{}: the `testkit` feature outside [dev-dependencies] (D50)",
                    i + 1
                ));
            }
            if section == "[features]" && t.starts_with("default") && t.contains("\"testkit\"") {
                rep.errors.push(format!(
                    "{path}:{}: `testkit` in the default features (D50)",
                    i + 1
                ));
            }
        }
    }
    rep
}

/// The names Windows keeps for devices, beside `COM1` to `COM9` and `LPT1` to `LPT9`: no file or
/// directory there takes one, in any case, with or without an extension (`aux.rs` as much as
/// `aux`), as git for Windows reads them (`is_valid_win32_path`, under `core.protectNTFS`, on by
/// default).
const WINDOWS_DEVICES: [&str; 6] = ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"];

/// The characters no Windows file name holds, beside the control characters.
const WINDOWS_REFUSED: &str = "<>:\"|?*\\";

/// Whether `name`, one component of a path, is a device name Windows keeps ([`WINDOWS_DEVICES`],
/// `COM1` to `LPT9`): its part before the first dot, less the spaces that end it, in any case.
fn windows_device(name: &str) -> bool {
    let base = name
        .split('.')
        .next()
        .unwrap_or_default()
        .trim_end_matches(' ');
    let upper = base.to_ascii_uppercase();
    WINDOWS_DEVICES.contains(&upper.as_str())
        || (upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && matches!(upper.as_bytes()[3], b'1'..=b'9'))
}

/// Checks the paths git tracks (`git ls-files`, `/`-separated) for what a Windows checkout
/// refuses: a component that is a device name ([`windows_device`]), holds a character Windows
/// refuses or a control character, or ends in a dot or a space; and for two paths, or two
/// directories, whose names differ only by case, which Windows' file systems hold as one. Git
/// for Windows checks out no tree with the first ("invalid path"), so a Windows clone of it
/// fails, and potluck seeds no workspace of it there; the second leaves one file of the two.
pub fn windows_paths(paths: &[String]) -> Report {
    let mut rep = Report::default();
    // Each path and directory, by its lowercase form, to its first spelling.
    let mut seen: std::collections::BTreeMap<String, String> = Default::default();
    for path in paths {
        let mut prefix = String::new();
        for name in path.split('/') {
            if !prefix.is_empty() {
                prefix.push('/');
            }
            prefix.push_str(name);
            if windows_device(name) {
                rep.errors.push(format!(
                    "{path}: `{name}` is a device name Windows keeps (CON, PRN, AUX, NUL, COM1 to \
                     COM9, LPT1 to LPT9), which no file there takes, with or without an extension"
                ));
            } else if let Some(c) = name
                .chars()
                .find(|&c| WINDOWS_REFUSED.contains(c) || c.is_ascii_control())
            {
                rep.errors.push(format!(
                    "{path}: `{name}` holds {c:?}, which no Windows file name holds"
                ));
            } else if name.ends_with('.') || name.ends_with(' ') {
                rep.errors.push(format!(
                    "{path}: `{name}` ends in a dot or a space, which Windows drops from a name"
                ));
            }
            let first = seen
                .entry(prefix.to_lowercase())
                .or_insert_with(|| prefix.clone());
            if *first != prefix {
                rep.errors.push(format!(
                    "{prefix} and {first} differ only by case, and Windows holds them as one"
                ));
                // Once for the two, not for each path below them.
                *first = prefix.clone();
            }
        }
    }
    rep.errors.dedup();
    rep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path holding a device name Windows keeps, with an extension or none, in any case, fails,
    /// as `aux.rs` failed every Windows checkout; names that only start like one pass.
    #[test]
    fn windows_paths_refuses_the_device_names_windows_keeps() {
        let paths = |p: &[&str]| p.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for bad in [
            "crates/reims-restore/src/aux.rs",
            "aux",
            "docs/NUL",
            "a/Con.txt/b.rs",
            "prn.tar.gz",
            "COM1.log",
            "x/lpt9",
            "aux .rs",
            "conout$.md",
        ] {
            let r = windows_paths(&paths(&[bad]));
            assert_eq!(r.errors.len(), 1, "{bad}: {:#?}", r.errors);
            assert!(
                r.errors[0].contains("device name"),
                "{bad}: {}",
                r.errors[0]
            );
        }
        let fine = [
            "crates/reims-restore/src/aux_storage.rs",
            "auxiliary.rs",
            "console.rs",
            "nullable/x",
            "com10.rs",
            "lpt0",
            "com.rs",
            "src/con-test.rs",
        ];
        assert_eq!(windows_paths(&paths(&fine)).errors, Vec::<String>::new());
    }

    /// A name with a character Windows refuses, or ending in a dot or a space, fails; so do two
    /// paths or directories that differ only by case, once for the pair.
    #[test]
    fn windows_paths_refuses_characters_endings_and_case_twins() {
        let paths = |p: &[&str]| p.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for bad in [
            "a:b.rs",
            "q?",
            "x/<y>",
            "pipe|",
            "star*",
            "quote\"",
            "back\\slash",
            "tab\t",
            "dot.",
            "space ",
        ] {
            assert_eq!(windows_paths(&paths(&[bad])).errors.len(), 1, "{bad:?}");
        }
        let r = windows_paths(&paths(&["src/Foo.rs", "src/foo.rs"]));
        assert_eq!(r.errors.len(), 1, "{:#?}", r.errors);
        assert!(r.errors[0].contains("differ only by case"));
        let r = windows_paths(&paths(&["Docs/a.md", "docs/b.md", "docs/c.md"]));
        assert_eq!(r.errors.len(), 1, "{:#?}", r.errors);
        assert!(r.errors[0].starts_with("docs and Docs"), "{}", r.errors[0]);
        assert!(
            windows_paths(&paths(&["docs/a.md", "docs/b.md"]))
                .errors
                .is_empty()
        );
    }

    #[test]
    fn testkit_manifests_checks_the_gate_and_the_manifests() {
        let lib = "#[cfg(any(test, feature = \"testkit\"))]\npub mod testkit;\npub mod real;\n";
        let manifests = vec![(
            "crates/a/Cargo.toml".to_owned(),
            "[dependencies]\nreims-vgpu-vulkan = { path = \"x\", features = [\"testkit\"] }\n[dev-dependencies]\nreims-vgpu-vulkan = { path = \"x\", features = [\"testkit\"] }\n[features]\ndefault = [\"testkit\"]\n".to_owned(),
        )];
        let r = testkit_manifests(lib, &manifests);
        assert_eq!(r.errors.len(), 2, "{:#?}", r.errors);
        assert!(r.errors[0].contains("Cargo.toml:2"));
        assert!(r.errors[1].contains("Cargo.toml:6"));
        assert!(testkit_manifests(lib, &[]).errors.is_empty());
        let ungated = testkit_manifests("pub mod testkit;\n", &[]);
        assert_eq!(ungated.errors.len(), 1);
        assert!(ungated.errors[0].contains("core.rs:1"));
    }
}
