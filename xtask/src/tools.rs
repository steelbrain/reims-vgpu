//! What `dist` and `check` share: the external programs they run ([`Tools`]), Cargo's profiles
//! and target directory.

use std::path::PathBuf;

use crate::util::{Res, repo_root};

/// A Cargo profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile(pub String);

impl Profile {
    /// The directory under the target dir the profile writes to (`dev` writes to `debug`).
    pub fn dir(&self) -> &str {
        match self.0.as_str() {
            "dev" | "test" => "debug",
            "bench" => "release",
            other => other,
        }
    }
}

/// The external programs xtask runs: Cargo, `make`, `xcrun` tools and `codesign`. Tests record
/// the calls instead.
pub trait Tools {
    /// Runs `program` with `args`, failing on a non-zero exit.
    fn run(&mut self, program: &str, args: &[String]) -> Res<()>;
    /// Runs `program` with `args` and returns its trimmed stdout.
    fn output(&mut self, program: &str, args: &[String]) -> Res<String>;
}

/// The real tools, run in the repository's root.
pub struct System;

impl Tools for System {
    fn run(&mut self, program: &str, args: &[String]) -> Res<()> {
        println!("$ {program} {}", args.join(" "));
        let status = std::process::Command::new(program)
            .args(args)
            .current_dir(repo_root())
            .status()
            .map_err(|e| format!("{program}: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("{program} {} failed ({status})", args.join(" ")))
        }
    }

    fn output(&mut self, program: &str, args: &[String]) -> Res<String> {
        let mut cmd = std::process::Command::new(program);
        cmd.args(args).current_dir(repo_root());
        crate::util::run_capture(&mut cmd).map(|s| s.trim().to_owned())
    }
}

/// Cargo's target directory, as Cargo itself resolves it (`CARGO_TARGET_DIR`, relative to the
/// directory Cargo runs in, or `build.target-dir` from a config file): `cargo metadata`'s
/// `target_directory`, run where [`System`] runs Cargo.
pub fn target_dir(tools: &mut dyn Tools) -> Res<PathBuf> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let args = ["metadata", "--format-version", "1", "--no-deps"].map(str::to_owned);
    let json = tools.output(&cargo, &args)?;
    parse_target_directory(&json)
        .ok_or_else(|| "cargo metadata printed no target_directory".to_owned())
}

/// The `target_directory` string of `cargo metadata`'s JSON, unescaped.
pub fn parse_target_directory(json: &str) -> Option<PathBuf> {
    const KEY: &str = "\"target_directory\":";
    let rest = json[json.find(KEY)? + KEY.len()..].trim_start();
    let mut chars = rest.strip_prefix('"')?.chars();
    let mut out = String::new();
    loop {
        match chars.next()? {
            '"' => return Some(PathBuf::from(out)),
            '\\' => match chars.next()? {
                'u' => {
                    let hex: String = chars.by_ref().take(4).collect();
                    out.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
                }
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                c => out.push(c),
            },
            c => out.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records every call; `output` answers with `answer`.
    #[derive(Default)]
    struct Log {
        calls: Vec<(String, Vec<String>)>,
        answer: String,
    }

    impl Tools for Log {
        fn run(&mut self, program: &str, args: &[String]) -> Res<()> {
            self.calls.push((program.to_owned(), args.to_vec()));
            Ok(())
        }
        fn output(&mut self, program: &str, args: &[String]) -> Res<String> {
            self.calls.push((program.to_owned(), args.to_vec()));
            Ok(self.answer.clone())
        }
    }

    #[test]
    fn profile_dir_maps_dev_to_debug() {
        assert_eq!(Profile("dev".into()).dir(), "debug");
        assert_eq!(Profile("test".into()).dir(), "debug");
        assert_eq!(Profile("bench".into()).dir(), "release");
        assert_eq!(Profile("release".into()).dir(), "release");
        assert_eq!(Profile("checked".into()).dir(), "checked");
    }

    #[test]
    fn parse_target_directory_reads_cargo_metadata() {
        let json = r#"{"packages":[],"target_directory":"/a/b \"c\"é","version":1}"#;
        assert_eq!(
            parse_target_directory(json),
            Some(PathBuf::from("/a/b \"c\"\u{e9}"))
        );
        assert_eq!(parse_target_directory("{}"), None);
    }

    #[test]
    fn target_dir_asks_cargo_metadata() {
        let mut log = Log {
            answer: r#"{"target_directory":"/w/t"}"#.to_owned(),
            ..Log::default()
        };
        assert_eq!(target_dir(&mut log).unwrap(), PathBuf::from("/w/t"));
        assert_eq!(
            log.calls[0].1,
            ["metadata", "--format-version", "1", "--no-deps"].map(str::to_owned)
        );
    }

    #[test]
    fn target_dir_matches_cargo_for_a_relative_env_dir() {
        // The real System: cargo resolves a relative CARGO_TARGET_DIR against the directory it
        // runs in, which System makes repo_root() whatever the xtask's cwd.
        let got = target_dir(&mut System).unwrap();
        let want = match std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from) {
            Some(p) if p.is_absolute() => p,
            Some(p) => repo_root().join(p),
            None => got.clone(),
        };
        assert_eq!(got, want);
    }
}
