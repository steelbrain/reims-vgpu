//! The testkit manifest check (D50): the testkit stays out of the product. The Vulkan crate's
//! `core` module declares its `testkit` module under `#[cfg(any(test, feature = "testkit"))]`, and
//! only dev-dependencies turn the crate's feature on, so a product build cannot name it.

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

#[cfg(test)]
mod tests {
    use super::*;

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
