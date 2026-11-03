//! The runtime's work counts (`Device::work_counts`) in the host's log, for the lanes' benches:
//! every [`PERIOD`] (a second, so a bench cuts each of its windows to within one), while the
//! target [`TARGET`] is enabled at debug, one line of the counts that changed since the line
//! before, as deltas. A diagnostic: the counts are the runtime's own, which
//! no Metal call answers, and which lines the host writes is free under the contract.
//!
//! - **Off, it costs nothing.** [`start`] makes no thread unless the target is enabled at debug
//!   as the device is adopted, and the thread reads no count while the target is off.
//! - **Every field.** The counts are a struct of the runtime's, hundreds of fields in nested
//!   structs and arrays, which change with its commits. A line takes their `Debug` form and
//!   flattens it into paths ([`flatten`]: `shadow_copies`, `shadow_keeps.render_cut`,
//!   `copy_sites[3].calls`), so a field the runtime adds shows with no change here. Text and other
//!   values that are not whole numbers are left out.
//! - **A line** is `work counts over S s: PATH +N, PATH -N, ...`, its fields in the struct's
//!   order, S the time since the line before. The first is over the time since the device was
//!   adopted, from 0, so it holds every count that is not 0. A period in which nothing changed
//!   writes no line.
//! - **Reading the counts** has the runtime lower the render encoders' calls it logged and has
//!   not lowered yet (its `work_counts`), as it does before they run anyway.

use std::fmt::Write as _;
use std::time::Duration;

use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::os::clock::{Clock as _, Deadline, MonoTime};

/// The log target of the lines, at debug.
pub(crate) const TARGET: &str = "reims_vgpu::work";

/// How often the thread looks at the counts.
pub(crate) const PERIOD: Duration = Duration::from_secs(1);

/// The counts as paths and values, in their `Debug` form's order.
type Flat = Vec<(String, u64)>;

/// Starts the thread that logs `device`'s counts (the module docs), where the target is enabled
/// at debug.
pub(crate) fn start(device: &rt::Device) {
    if !log::log_enabled!(target: TARGET, log::Level::Debug) {
        return;
    }
    let device = device.clone();
    let _ = std::thread::Builder::new()
        .name("reims-vgpu-work".into())
        .spawn(move || {
            let clock = crate::os::OsClock::new();
            let mut lines = Lines::default();
            let mut tick = clock.now_mono();
            loop {
                tick = tick.saturating_add(PERIOD);
                clock.sleep_until(Deadline::Mono(tick));
                if !log::log_enabled!(target: TARGET, log::Level::Debug) {
                    continue;
                }
                let counts = format!("{:?}", device.work_counts());
                if let Some(line) = lines.next(&counts, clock.now_mono()) {
                    log::debug!(target: TARGET, "{line}");
                }
            }
        });
}

/// The line before's counts and when they were read.
#[derive(Default)]
pub(crate) struct Lines {
    last: Option<(Flat, MonoTime)>,
    started: Option<MonoTime>,
}

impl Lines {
    /// The line for the counts whose `Debug` form is `counts`, read `at`: the fields that changed
    /// since the line before (the first time, those that are not 0), over the time since it, or
    /// None where none did.
    pub(crate) fn next(&mut self, counts: &str, at: MonoTime) -> Option<String> {
        let now = flatten(counts);
        let started = *self.started.get_or_insert(at);
        let (before, since) = match self.last.take() {
            Some((flat, then)) => (flat, then),
            None => (Vec::new(), started),
        };
        let line = changes(&before, &now).map(|changed| {
            let over = at.saturating_duration_since(since);
            let mut line = format!("work counts over {:.1} s:", over.as_secs_f64());
            for (i, (path, delta)) in changed.iter().enumerate() {
                let _ = write!(line, "{} {path} {delta:+}", if i == 0 { "" } else { "," });
            }
            line
        });
        // A line's span starts at the line before's: a period with no change writes none.
        self.last = Some((now, if line.is_some() { at } else { since }));
        line
    }
}

/// The fields of `now` whose values differ from `before`'s (a field `before` lacks counts from
/// 0), with the difference, in `now`'s order; None where none differ.
fn changes(before: &[(String, u64)], now: &[(String, u64)]) -> Option<Vec<(String, i128)>> {
    let was: std::collections::HashMap<&str, u64> =
        before.iter().map(|(p, v)| (p.as_str(), *v)).collect();
    let changed: Vec<(String, i128)> = now
        .iter()
        .filter_map(|(path, v)| {
            let delta = i128::from(*v) - i128::from(was.get(path.as_str()).copied().unwrap_or(0));
            (delta != 0).then(|| (path.clone(), delta))
        })
        .collect();
    (!changed.is_empty()).then_some(changed)
}

/// The whole numbers of a value's `Debug` form, each at its path: a struct's field by its name
/// (`a.b`), a list's or a tuple's item by its index (`a[2]`), a one-item tuple (`Some(3)`) as its
/// item. Text, names and other values are left out.
pub(crate) fn flatten(debug: &str) -> Flat {
    let tokens = tokens(debug);
    let mut out = Vec::new();
    let mut at = 0;
    value(&tokens, &mut at, "", &mut out);
    out
}

#[derive(Debug, PartialEq)]
enum Token<'a> {
    Word(&'a str),
    Open(char),
    Close,
    Colon,
    Comma,
}

fn tokens(s: &str) -> Vec<Token<'_>> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'{' | b'[' | b'(' => {
                out.push(Token::Open(b[i] as char));
                i += 1;
            }
            b'}' | b']' | b')' => {
                out.push(Token::Close);
                i += 1;
            }
            b':' => {
                out.push(Token::Colon);
                i += 1;
            }
            b',' => {
                out.push(Token::Comma);
                i += 1;
            }
            b'"' | b'\'' => {
                // Text, its escapes skipped whole: a word of its own, never a number.
                let quote = b[i];
                let start = i;
                i += 1;
                while i < b.len() && b[i] != quote {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i = (i + 1).min(b.len());
                out.push(Token::Word(&s[start..i]));
            }
            c if c.is_ascii_whitespace() => i += 1,
            _ => {
                let start = i;
                while i < b.len() && !b"{}[](),:\"' \t\n".contains(&b[i]) {
                    i += 1;
                }
                out.push(Token::Word(&s[start..i]));
            }
        }
    }
    out
}

/// Reads the value at `tokens[*at]` into `out` under `path`, leaving `*at` past it.
fn value(tokens: &[Token<'_>], at: &mut usize, path: &str, out: &mut Flat) {
    // A struct or tuple struct's name, or a value of one word.
    if let Some(Token::Word(w)) = tokens.get(*at) {
        *at += 1;
        if !matches!(tokens.get(*at), Some(Token::Open(_))) {
            if let Ok(n) = w.parse::<u64>() {
                out.push((path.to_string(), n));
            }
            return;
        }
    }
    let Some(Token::Open(open)) = tokens.get(*at) else {
        return;
    };
    let open = *open;
    *at += 1;
    let mut index = 0;
    let mut items: Flat = Vec::new();
    loop {
        match tokens.get(*at) {
            None => break,
            Some(Token::Close) => {
                *at += 1;
                break;
            }
            Some(Token::Comma) => {
                *at += 1;
                continue;
            }
            _ => {}
        }
        if open == '{' {
            // `name: value`, or the `..` of a struct that shows only some fields.
            let name = match tokens.get(*at) {
                Some(Token::Word(w)) if matches!(tokens.get(*at + 1), Some(Token::Colon)) => *w,
                _ => {
                    skip(tokens, at);
                    continue;
                }
            };
            *at += 2;
            let child = if path.is_empty() {
                name.to_string()
            } else {
                format!("{path}.{name}")
            };
            value(tokens, at, &child, out);
        } else {
            value(tokens, at, &format!("{path}[{index}]"), &mut items);
            index += 1;
        }
    }
    // A one-item tuple (`Some(3)`, a newtype) is its item.
    if open == '(' && index == 1 {
        let prefix = format!("{path}[0]");
        for (p, v) in items {
            out.push((format!("{path}{}", &p[prefix.len()..]), v));
        }
    } else {
        out.extend(items);
    }
}

/// Moves `*at` past one token, or past a whole bracketed group where it opens one.
fn skip(tokens: &[Token<'_>], at: &mut usize) {
    let mut depth = 0usize;
    while let Some(t) = tokens.get(*at) {
        *at += 1;
        match t {
            Token::Open(_) => depth += 1,
            Token::Close if depth <= 1 => return,
            Token::Close => depth -= 1,
            _ if depth == 0 => return,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field the runtime's counts change shows in the next line, as its delta, and only it.
    #[test]
    fn a_change_of_a_field_shows_in_the_next_line() {
        let mut counts = rt::WorkCounts::default();
        let mut lines = Lines::default();
        let t0 = MonoTime::from_nanos(0);
        assert_eq!(
            lines.next(&format!("{counts:?}"), t0),
            None,
            "all 0: no line"
        );
        counts.shadow_copies += 3;
        counts.shadow_keeps.render_cut += 2;
        let s = Duration::from_secs(1);
        let line = (lines.next(&format!("{counts:?}"), t0.saturating_add(s))).expect("a line");
        assert!(line.starts_with("work counts over 1.0 s:"), "{line}");
        assert!(line.contains(" shadow_copies +3"), "{line}");
        assert!(line.contains(" shadow_keeps.render_cut +2"), "{line}");
        assert_eq!(
            line.matches(" +").count(),
            2,
            "only the changed fields: {line}"
        );
        assert_eq!(
            lines.next(&format!("{counts:?}"), t0.saturating_add(2 * s)),
            None,
            "no change, no line"
        );
        counts.shadow_copies += 1;
        let line = lines
            .next(&format!("{counts:?}"), t0.saturating_add(3 * s))
            .expect("a line");
        assert_eq!(line, "work counts over 2.0 s: shadow_copies +1");
    }

    /// The `Debug` form's numbers at their paths: nested structs, lists of structs, one-item
    /// tuples, and text and names left out.
    #[test]
    fn flatten_finds_each_number_at_its_path() {
        let flat = flatten(
            r#"Top { a: 1, inner: Inner { b: 2, name: "x, y: [z]" }, list: [Site { c: 3 }, Site { c: 4 }], some: Some(5), none: None, pair: (6, 7), f: 1.5 }"#,
        );
        let want: Vec<(String, u64)> = [
            ("a", 1),
            ("inner.b", 2),
            ("list[0].c", 3),
            ("list[1].c", 4),
            ("some", 5),
            ("pair[0]", 6),
            ("pair[1]", 7),
        ]
        .into_iter()
        .map(|(p, v)| (p.to_string(), v))
        .collect();
        assert_eq!(flat, want);
    }

    /// The runtime's own counts flatten to every one of their fields: as many paths as numbers in
    /// their `Debug` form, the copy sites' text left out.
    #[test]
    fn the_runtimes_counts_flatten_whole() {
        let counts = rt::WorkCounts {
            shadow_merges_fetched: 9,
            ..Default::default()
        };
        let flat = flatten(&format!("{counts:?}"));
        assert!(flat.len() > 500, "{} paths", flat.len());
        assert!(flat.contains(&("shadow_merges_fetched".to_string(), 9)));
        assert!(
            flat.iter().any(|(p, _)| p.starts_with("copy_sites[31].")),
            "the last copy site"
        );
    }
}
