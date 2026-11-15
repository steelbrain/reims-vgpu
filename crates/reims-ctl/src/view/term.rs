//! What the terminal can show and report, from its answers to one batch of queries, and the
//! image protocol chosen from them.
//!
//! The queries are written once raw mode is on and before anything else reads the terminal; the
//! answers are read until the primary device attributes' (DA1), which every terminal answers and
//! answers last, since terminals answer in order. A query the terminal does not know goes
//! unanswered. Over SSH the queries reach the terminal itself; inside tmux they reach tmux, which
//! answers for itself, so its sixel or half blocks are chosen there.

use std::io::Write;
use std::time::{Duration, Instant};

/// How the terminal shows a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Protocol {
    /// kitty's graphics protocol (kitty, Ghostty, Konsole)
    Kitty,
    /// iTerm2's inline images (iTerm2, WezTerm)
    Iterm,
    /// Sixel (foot, xterm, Windows Terminal, VS Code with images on)
    Sixel,
    /// Half-block characters in 24-bit color, two pixels a cell: any terminal
    Blocks,
}

impl Protocol {
    pub fn name(self) -> &'static str {
        match self {
            Protocol::Kitty => "kitty",
            Protocol::Iterm => "iTerm2",
            Protocol::Sixel => "sixel",
            Protocol::Blocks => "half blocks",
        }
    }
}

/// The terminal's answers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Caps {
    /// kitty's graphics protocol answered its query.
    pub kitty: bool,
    /// Its name and version (XTVERSION), such as `iTerm2 3.6.1`.
    pub name: Option<String>,
    /// A cell's width and height in pixels (`CSI 16 t`).
    pub cell: Option<(u16, u16)>,
    /// The text area's width and height in pixels (`CSI 14 t`).
    pub text_area: Option<(u32, u32)>,
    /// It reports the mouse in pixels (SGR-Pixels, mode 1016).
    pub sgr_pixels: bool,
    /// It speaks the kitty keyboard protocol.
    pub kitty_keyboard: bool,
    /// Its device attributes include sixel graphics (4).
    pub sixel: bool,
    /// It answered DA1, so every answer it gives is in.
    pub answered: bool,
}

/// The queries, DA1 last.
const QUERIES: &[u8] = b"\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\\
    \x1b[>0q\
    \x1b[16t\
    \x1b[14t\
    \x1b[?1016$p\
    \x1b[?u\
    \x1b[c";

/// Asks the terminal, on stdout, and reads its answers from stdin, which must be the terminal
/// in raw mode, for at most `timeout`.
pub fn query(timeout: Duration) -> Caps {
    let mut out = std::io::stdout();
    if out.write_all(QUERIES).and_then(|_| out.flush()).is_err() {
        return Caps::default();
    }
    let deadline = Instant::now() + timeout;
    let mut buf = Vec::new();
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        let mut chunk = [0u8; 1024];
        let r = match crate::os::read_terminal(&mut chunk, left) {
            Ok(0) => break,
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        buf.extend_from_slice(&chunk[..r]);
        if parse(&buf).answered {
            break;
        }
    }
    parse(&buf)
}

/// Reads the answers out of what the terminal sent, skipping anything else.
pub fn parse(b: &[u8]) -> Caps {
    let mut c = Caps::default();
    let mut i = 0;
    while i < b.len() {
        if b[i] != 0x1b || i + 1 >= b.len() {
            i += 1;
            continue;
        }
        match b[i + 1] {
            // APC and DCS run to the string terminator.
            kind @ (b'_' | b'P') => {
                let body_start = i + 2;
                let Some(end) = find(&b[body_start..], b"\x1b\\") else {
                    break;
                };
                let body = &b[body_start..body_start + end];
                if kind == b'_' {
                    if body.starts_with(b"Gi=31") && find(body, b";OK").is_some() {
                        c.kitty = true;
                    }
                } else if let Some(name) = body.strip_prefix(b">|") {
                    c.name = Some(String::from_utf8_lossy(name).trim().to_string());
                }
                i = body_start + end + 2;
            }
            b'[' => {
                let start = i + 2;
                let Some(len) = b[start..].iter().position(|&x| (0x40..=0x7e).contains(&x)) else {
                    break;
                };
                let params = String::from_utf8_lossy(&b[start..start + len]);
                csi(&mut c, &params, b[start + len]);
                i = start + len + 1;
            }
            _ => i += 1,
        }
    }
    c
}

fn csi(c: &mut Caps, params: &str, fin: u8) {
    let nums = |s: &str| -> Vec<u32> { s.split(';').filter_map(|p| p.parse().ok()).collect() };
    match fin {
        b't' => match nums(params).as_slice() {
            [6, h, w] if *w > 0 && *h > 0 => c.cell = Some((*w as u16, *h as u16)),
            [4, h, w] if *w > 0 && *h > 0 => c.text_area = Some((*w, *h)),
            _ => {}
        },
        b'y' => {
            if let Some(rest) = params.strip_prefix("?1016;")
                && let Some(v) = rest.strip_suffix('$')
            {
                c.sgr_pixels = matches!(v, "1" | "2" | "3" | "4");
            }
        }
        b'u' if params.starts_with('?') => c.kitty_keyboard = true,
        b'c' if params.starts_with('?') => {
            c.answered = true;
            c.sixel = params[1..].split(';').skip(1).any(|a| a == "4");
        }
        _ => {}
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The protocol for a terminal that answered `c`: kitty's where it answered kitty's query,
/// iTerm2's where it is iTerm2 or WezTerm, sixel where its attributes say so, and half blocks
/// otherwise. `env` reads the environment, which names the terminal when its name did not come
/// back; inside tmux or screen the environment names the terminal outside, which the escape
/// sequences do not reach.
pub fn choose(c: &Caps, env: impl Fn(&str) -> Option<String>) -> Protocol {
    if c.kitty {
        return Protocol::Kitty;
    }
    let name = c.name.as_deref().unwrap_or("");
    let multiplexed = env("TMUX").is_some()
        || env("STY").is_some()
        || name.starts_with("tmux")
        || name.starts_with("screen");
    let iterm_named = name.starts_with("iTerm2") || name.starts_with("WezTerm");
    let iterm_env = matches!(
        env("TERM_PROGRAM").as_deref(),
        Some("iTerm.app" | "WezTerm")
    ) || env("LC_TERMINAL").as_deref() == Some("iTerm2");
    if iterm_named || (iterm_env && !multiplexed && c.name.is_none()) {
        return Protocol::Iterm;
    }
    if c.sixel {
        return Protocol::Sixel;
    }
    Protocol::Blocks
}

/// A cell's size in pixels, for a terminal of `cols` by `rows` cells: from the window size the
/// terminal reports with its rows and columns (which SSH carries), else from its answers.
pub fn cell_size(c: &Caps, cols: u16, rows: u16) -> Option<(u16, u16)> {
    if let Ok(ws) = ratatui::crossterm::terminal::window_size()
        && ws.width > 0
        && ws.height > 0
        && cols > 0
        && rows > 0
    {
        return Some((ws.width / cols, ws.height / rows));
    }
    if c.cell.is_some() {
        return c.cell;
    }
    match c.text_area {
        Some((w, h)) if cols > 0 && rows > 0 => {
            Some(((w / cols as u32) as u16, (h / rows as u32) as u16))
        }
        _ => None,
    }
    .filter(|(w, h)| *w > 0 && *h > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kittys_answers_are_read() {
        let a = b"\x1b_Gi=31;OK\x1b\\\x1bP>|kitty(0.39.1)\x1b\\\x1b[6;20;10t\x1b[4;1000;1600t\
                  \x1b[?1016;2$y\x1b[?0u\x1b[?62;c";
        let c = parse(a);
        assert_eq!(
            c,
            Caps {
                kitty: true,
                name: Some("kitty(0.39.1)".into()),
                cell: Some((10, 20)),
                text_area: Some((1600, 1000)),
                sgr_pixels: true,
                kitty_keyboard: true,
                sixel: false,
                answered: true,
            }
        );
        assert_eq!(choose(&c, |_| None), Protocol::Kitty);
    }

    #[test]
    fn a_terminal_that_knows_little_answers_da1_alone() {
        let c = parse(b"\x1b[?1016;0$y\x1b[?1;2c");
        assert!(c.answered && !c.sgr_pixels && !c.kitty && !c.sixel);
        assert_eq!(choose(&c, |_| None), Protocol::Blocks);
        // Nothing back at all.
        assert!(!parse(b"").answered);
    }

    #[test]
    fn iterm2_by_name_and_sixel_by_attributes() {
        let c = parse(b"\x1bP>|iTerm2 3.6.1\x1b\\\x1b[?62;4;22c");
        assert_eq!(choose(&c, |_| None), Protocol::Iterm);
        let foot = parse(b"\x1bP>|foot(1.20.0)\x1b\\\x1b[?62;4;22c");
        assert_eq!(choose(&foot, |_| None), Protocol::Sixel);
        // A terminal that did not say its name, but whose environment does.
        let env = |k: &str| (k == "LC_TERMINAL").then(|| "iTerm2".to_string());
        assert_eq!(choose(&parse(b"\x1b[?62;22c"), env), Protocol::Iterm);
    }

    #[test]
    fn inside_tmux_the_outer_terminals_environment_is_not_trusted() {
        let c = parse(b"\x1bP>|tmux 3.4\x1b\\\x1b[?62;4c");
        let env = |k: &str| match k {
            "TMUX" => Some("/tmp/tmux-1/default,1,0".into()),
            "LC_TERMINAL" => Some("iTerm2".into()),
            _ => None,
        };
        assert_eq!(choose(&c, env), Protocol::Sixel);
    }

    #[test]
    fn unknown_sequences_and_text_are_skipped() {
        let c = parse(b"junk\x1b[0n\x1bOA\x1b]11;rgb:0/0/0\x07\x1b[6;16;8t\x1b[?64;4c");
        assert_eq!(c.cell, Some((8, 16)));
        assert!(c.sixel && c.answered);
    }
}
