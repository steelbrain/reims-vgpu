//! A form for the interface: text, path, choice and toggle fields, some shown only for some
//! answers of another field (`show_if`).
//!
//! Keys: Up and Down (or Tab and Shift-Tab) move between fields; Left and Right, or Space,
//! change a choice or toggle; typing edits text; Tab in a path field completes it as a shell
//! would ([`complete`]; on Windows whatever the case typed, as its file systems match names),
//! lists the names that match below the fields where several do and it can add nothing, and
//! moves on when there is nothing to complete; Enter moves on, and on the last field submits;
//! Ctrl-S submits from anywhere; Esc cancels. While the work a submit started runs (`busy`), the
//! form shows its step and takes only Esc, which closes it.

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Wrap};
use std::path::{Path, PathBuf};

pub enum Value {
    Text {
        text: String,
        cursor: usize,
        placeholder: String,
        path: bool,
    },
    Choice {
        options: Vec<(String, String)>,
        index: usize,
    },
    Toggle(bool),
}

pub struct Field {
    pub key: &'static str,
    pub label: String,
    pub help: String,
    pub value: Value,
    pub show_if: Option<(&'static str, Vec<&'static str>)>,
}

impl Field {
    pub fn text(key: &'static str, label: &str, value: &str, help: &str) -> Field {
        Field {
            key,
            label: label.into(),
            help: help.into(),
            value: Value::Text {
                text: value.into(),
                cursor: value.chars().count(),
                placeholder: String::new(),
                path: false,
            },
            show_if: None,
        }
    }

    pub fn path(key: &'static str, label: &str, value: &str, help: &str) -> Field {
        let mut f = Field::text(key, label, value, help);
        if let Value::Text { path, .. } = &mut f.value {
            *path = true;
        }
        f
    }

    pub fn choice(
        key: &'static str,
        label: &str,
        options: Vec<(String, String)>,
        selected: &str,
        help: &str,
    ) -> Field {
        let index = options
            .iter()
            .position(|(id, _)| id == selected)
            .unwrap_or(0);
        Field {
            key,
            label: label.into(),
            help: help.into(),
            value: Value::Choice { options, index },
            show_if: None,
        }
    }

    pub fn toggle(key: &'static str, label: &str, on: bool, help: &str) -> Field {
        Field {
            key,
            label: label.into(),
            help: help.into(),
            value: Value::Toggle(on),
            show_if: None,
        }
    }

    pub fn placeholder(mut self, p: &str) -> Field {
        if let Value::Text { placeholder, .. } = &mut self.value {
            *placeholder = p.into();
        }
        self
    }

    /// Shows the field while the field `key` shows with one of `values`, and always in a form
    /// without a field `key`.
    pub fn show_if(mut self, key: &'static str, values: &[&'static str]) -> Field {
        self.show_if = Some((key, values.to_vec()));
        self
    }
}

pub fn opts(list: &[(&str, &str)]) -> Vec<(String, String)> {
    list.iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

pub struct Form {
    pub title: String,
    pub fields: Vec<Field>,
    pub focus: usize,
    pub error: Option<String>,
    /// The names a path field's Tab found where several match ([`Completion::Several`]), until
    /// the next key.
    hint: Option<String>,
    /// The work its submit started, while that runs: the caller's id for it, and its step.
    pub busy: Option<(u64, String)>,
    pub submit: String,
}

pub enum Event {
    None,
    Submit,
    Cancel,
}

impl Form {
    pub fn new(title: &str, submit: &str, fields: Vec<Field>) -> Form {
        Form {
            title: title.into(),
            fields,
            focus: 0,
            error: None,
            hint: None,
            busy: None,
            submit: submit.into(),
        }
    }

    fn shown(&self, i: usize) -> bool {
        match &self.fields[i].show_if {
            None => true,
            Some((key, values)) => {
                let Some(j) = self.fields.iter().position(|f| f.key == *key) else {
                    return true;
                };
                self.shown(j) && values.iter().any(|v| self.choice_at(j) == *v)
            }
        }
    }

    fn visible(&self) -> Vec<usize> {
        (0..self.fields.len()).filter(|&i| self.shown(i)).collect()
    }

    pub fn is_shown(&self, key: &str) -> bool {
        self.fields
            .iter()
            .position(|f| f.key == key)
            .is_some_and(|i| self.shown(i))
    }

    fn choice_at(&self, i: usize) -> String {
        match &self.fields[i].value {
            Value::Choice { options, index } => {
                options.get(*index).map(|o| o.0.clone()).unwrap_or_default()
            }
            Value::Toggle(b) => if *b { "yes" } else { "no" }.into(),
            Value::Text { text, .. } => text.clone(),
        }
    }

    fn field(&self, key: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.key == key)
    }

    /// A text field's value, trimmed; empty when hidden.
    pub fn text(&self, key: &str) -> String {
        if !self.is_shown(key) {
            return String::new();
        }
        match self.field(key).map(|f| &f.value) {
            Some(Value::Text { text, .. }) => text.trim().to_string(),
            _ => String::new(),
        }
    }

    pub fn choice(&self, key: &str) -> String {
        self.fields
            .iter()
            .position(|f| f.key == key)
            .map(|i| self.choice_at(i))
            .unwrap_or_default()
    }

    pub fn toggle(&self, key: &str) -> bool {
        matches!(self.field(key).map(|f| &f.value), Some(Value::Toggle(true)))
    }

    /// Replaces a choice's options, keeping its selection: one the new options lack stays, after
    /// them, as a VM's GPU the host's list lacks does in its edit form.
    pub fn set_options(&mut self, key: &str, mut new: Vec<(String, String)>) {
        if let Some(f) = self.fields.iter_mut().find(|f| f.key == key)
            && let Value::Choice { options, index } = &mut f.value
        {
            *index = match options.get(*index) {
                Some(cur) => new.iter().position(|o| o.0 == cur.0).unwrap_or_else(|| {
                    new.push(cur.clone());
                    new.len() - 1
                }),
                None => 0,
            };
            *options = new;
        }
    }

    #[cfg(test)]
    pub fn focused_key(&self) -> &'static str {
        self.fields[self.focus].key
    }

    fn move_focus(&mut self, forward: bool) {
        let v = self.visible();
        if v.is_empty() {
            return;
        }
        let pos = v.iter().position(|&i| i == self.focus).unwrap_or(0);
        let next = if forward {
            (pos + 1) % v.len()
        } else {
            (pos + v.len() - 1) % v.len()
        };
        self.focus = v[next];
    }

    fn at_last(&self) -> bool {
        self.visible().last() == Some(&self.focus)
    }

    pub fn handle_key(&mut self, k: KeyEvent) -> Event {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let close = k.code == KeyCode::Esc || (ctrl && k.code == KeyCode::Char('c'));
        if self.busy.is_some() && !close {
            return Event::None;
        }
        self.hint = None;
        if !self.visible().contains(&self.focus) {
            self.focus = self.visible().first().copied().unwrap_or(0);
        }
        match k.code {
            KeyCode::Esc => return Event::Cancel,
            KeyCode::Char('s') if ctrl => return Event::Submit,
            KeyCode::Char('c') if ctrl => return Event::Cancel,
            KeyCode::Up | KeyCode::BackTab => {
                self.move_focus(false);
                return Event::None;
            }
            KeyCode::Down => {
                self.move_focus(true);
                return Event::None;
            }
            KeyCode::Enter => {
                if self.at_last() {
                    return Event::Submit;
                }
                self.move_focus(true);
                return Event::None;
            }
            _ => {}
        }
        self.error = None;
        let field = &mut self.fields[self.focus];
        match &mut field.value {
            Value::Choice { options, index } => match k.code {
                KeyCode::Left => {
                    *index = (*index + options.len().max(1) - 1) % options.len().max(1)
                }
                KeyCode::Right | KeyCode::Char(' ') => *index = (*index + 1) % options.len().max(1),
                KeyCode::Tab => self.move_focus(true),
                _ => {}
            },
            Value::Toggle(b) => match k.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Char(' ') => *b = !*b,
                KeyCode::Char('y') => *b = true,
                KeyCode::Char('n') => *b = false,
                KeyCode::Tab => self.move_focus(true),
                _ => {}
            },
            Value::Text {
                text, cursor, path, ..
            } => {
                let byte =
                    |t: &str, c: usize| t.char_indices().nth(c).map(|(i, _)| i).unwrap_or(t.len());
                match k.code {
                    KeyCode::Char('a') if ctrl => *cursor = 0,
                    KeyCode::Char('e') if ctrl => *cursor = text.chars().count(),
                    KeyCode::Char('u') if ctrl => {
                        text.clear();
                        *cursor = 0;
                    }
                    KeyCode::Char('w') if ctrl => {
                        let b = byte(text, *cursor);
                        let head = text[..b].trim_end_matches('/');
                        let cut = head.rfind(['/', ' ']).map(|i| i + 1).unwrap_or(0);
                        text.replace_range(cut..b, "");
                        *cursor = text[..cut].chars().count();
                    }
                    KeyCode::Char(c) if !ctrl => {
                        let b = byte(text, *cursor);
                        text.insert(b, c);
                        *cursor += 1;
                    }
                    KeyCode::Backspace if *cursor > 0 => {
                        let b = byte(text, *cursor - 1);
                        text.remove(b);
                        *cursor -= 1;
                    }
                    KeyCode::Delete if *cursor < text.chars().count() => {
                        let b = byte(text, *cursor);
                        text.remove(b);
                    }
                    KeyCode::Left => *cursor = cursor.saturating_sub(1),
                    KeyCode::Right => *cursor = (*cursor + 1).min(text.chars().count()),
                    KeyCode::Home => *cursor = 0,
                    KeyCode::End => *cursor = text.chars().count(),
                    KeyCode::Tab => {
                        let found = if *path {
                            complete(text)
                        } else {
                            Completion::Nothing
                        };
                        match found {
                            Completion::Done(done) => {
                                *text = done;
                                *cursor = text.chars().count();
                            }
                            Completion::Several(names) => self.hint = Some(several(&names)),
                            Completion::Nothing => self.move_focus(true),
                        }
                    }
                    _ => {}
                }
            }
        }
        Event::None
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let visible = self.visible();
        let width = area.width.saturating_sub(4).clamp(area.width.min(40), 110);
        let help = &self.fields[self.focus].help;
        let help_lines = wrap_count(help, width.saturating_sub(4)) as u16;
        // The running work's step, the error, or what Tab found, above the help, and a blank
        // line.
        let note = match (&self.busy, &self.error, &self.hint) {
            (Some((_, step)), ..) => Some((step, Style::new().fg(Color::Yellow))),
            (None, Some(e), _) => Some((e, Style::new().fg(Color::Red).bold())),
            (None, None, Some(h)) => Some((h, Style::new())),
            (None, None, None) => None,
        };
        let note_lines = note.map_or(0, |(n, _)| {
            wrap_count(n, width.saturating_sub(4)) as u16 + 1
        });
        let rows = visible.len() as u16;
        let height = (rows + help_lines + 6 + note_lines)
            .min(area.height.saturating_sub(2))
            .max(8);
        let popup = area.centered(Constraint::Length(width), Constraint::Length(height));
        f.render_widget(Clear, popup);
        let block = Block::bordered()
            .title(Line::from(format!(" {} ", self.title)).bold())
            .title_bottom(
                Line::from(if self.busy.is_some() {
                    " Esc close: the work goes on ".to_string()
                } else {
                    format!(" Enter next · Ctrl-S {} · Esc cancel ", self.submit)
                })
                .dim(),
            )
            .border_style(Style::new().fg(Color::Cyan));
        let inner = block.inner(popup);
        f.render_widget(block, popup);
        let [list_area, help_area] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(help_lines + 1 + note_lines),
        ])
        .areas(inner);

        let label_w = 26u16.min(list_area.width / 2);
        let n = list_area.height as usize;
        let pos = visible.iter().position(|&i| i == self.focus).unwrap_or(0);
        let start = if pos >= n { pos + 1 - n } else { 0 };
        for (row, &i) in visible.iter().enumerate().skip(start).take(n) {
            let fld = &self.fields[i];
            let y = list_area.y + (row - start) as u16;
            let focused = i == self.focus;
            let label_style = if focused {
                Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::new()
            };
            let marker = if focused { "› " } else { "  " };
            let label = Paragraph::new(Line::from(vec![
                Span::styled(marker, label_style),
                Span::styled(fld.label.clone(), label_style),
            ]));
            f.render_widget(
                label,
                Rect {
                    x: list_area.x,
                    y,
                    width: label_w,
                    height: 1,
                },
            );
            let vx = list_area.x + label_w;
            let vw = list_area.width.saturating_sub(label_w);
            let value_area = Rect {
                x: vx,
                y,
                width: vw,
                height: 1,
            };
            match &fld.value {
                Value::Text {
                    text,
                    cursor,
                    placeholder,
                    ..
                } => {
                    let chars: Vec<char> = text.chars().collect();
                    let w = vw.saturating_sub(1) as usize;
                    let off = (*cursor).saturating_sub(w);
                    let shown: String = chars.iter().skip(off).take(w).collect();
                    let line = if text.is_empty() {
                        Line::from(Span::styled(
                            placeholder.clone(),
                            Style::new().dim().italic(),
                        ))
                    } else {
                        Line::from(shown)
                    };
                    let style = if focused {
                        Style::new().bg(Color::Indexed(236))
                    } else {
                        Style::new()
                    };
                    f.render_widget(Paragraph::new(line).style(style), value_area);
                    if focused {
                        f.set_cursor_position((vx + (*cursor - off) as u16, y));
                    }
                }
                Value::Choice { options, index } => {
                    let label = options.get(*index).map(|o| o.1.clone()).unwrap_or_default();
                    let text = if focused && options.len() > 1 {
                        format!("◀ {label} ▶")
                    } else {
                        label
                    };
                    let style = if focused {
                        Style::new().fg(Color::Yellow)
                    } else {
                        Style::new()
                    };
                    f.render_widget(Paragraph::new(text).style(style), value_area);
                }
                Value::Toggle(b) => {
                    let text = if *b { "[x] yes" } else { "[ ] no" };
                    let style = if focused {
                        Style::new().fg(Color::Yellow)
                    } else {
                        Style::new()
                    };
                    f.render_widget(Paragraph::new(text).style(style), value_area);
                }
            }
        }
        let mut lines = Vec::new();
        if let Some((n, style)) = note {
            lines.extend(n.lines().map(|l| Line::styled(l, style)));
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(help.clone(), Style::new().dim())));
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), help_area);
    }
}

fn wrap_count(s: &str, width: u16) -> usize {
    let w = width.max(10) as usize;
    s.lines()
        .map(|l| l.chars().count().div_ceil(w).max(1))
        .sum::<usize>()
        .max(1)
}

/// What Tab finds for a path field's text ([`complete`]).
#[derive(Debug, PartialEq, Eq)]
pub enum Completion {
    /// The text completed: the one name that matches, whole (a directory's with its separator,
    /// `/`, on Windows `\`), or the longest start the names that match share; or a drive or
    /// other directory typed without its separator, with it (`C:` is `C:\`).
    Done(String),
    /// The names that match, sorted (a directory's with its separator), where several do and
    /// they share no more than the text has.
    Several(Vec<String>),
    /// Nothing to complete: the text is empty, no name matches, or it names a file whole.
    Nothing,
}

/// `text` completed as a path, as a shell's Tab would ([`Completion`]), matching names whatever
/// their case on Windows, whose file systems ignore it: `c:\users\me\doc` completes to
/// `c:\users\me\Documents\`.
pub fn complete(text: &str) -> Completion {
    complete_in(text, cfg!(windows))
}

/// [`complete`], matching names whatever their case where `fold_case`.
fn complete_in(text: &str, fold_case: bool) -> Completion {
    let typed = text.trim();
    if typed.is_empty() {
        return Completion::Nothing;
    }
    let sep = std::path::MAIN_SEPARATOR_STR;
    let expanded = crate::paths::expand(typed);
    let name = expanded
        .file_name()
        .map(|n| n.to_string_lossy().into_owned());
    let (dir, prefix) = if typed.ends_with('/') || typed.ends_with(std::path::MAIN_SEPARATOR) {
        (expanded, String::new())
    } else if let Some(name) = name.filter(|n| typed.ends_with(n.as_str())) {
        let dir = expanded
            .parent()
            .map(Path::to_path_buf)
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or_else(|| PathBuf::from("."));
        (dir, name)
    } else {
        // No name typed last to complete, as in a drive (`C:`), `~`, `..` or `.`: a directory
        // takes its separator.
        return if expanded.is_dir() {
            Completion::Done(format!("{typed}{sep}"))
        } else {
            Completion::Nothing
        };
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Completion::Nothing;
    };
    let typed_len = prefix.chars().count();
    let mut matches: Vec<(String, bool)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            (shared_start(&name, &prefix, fold_case) == typed_len
                && (!name.starts_with('.') || prefix.starts_with('.')))
            .then(|| (name, e.path().is_dir()))
        })
        .collect();
    matches.sort();
    let shown = |(name, is_dir): &(String, bool)| {
        if *is_dir {
            format!("{name}{sep}")
        } else {
            name.clone()
        }
    };
    let completion = match &matches[..] {
        [] => return Completion::Nothing,
        [one] => shown(one),
        [(first, _), rest @ ..] => {
            let n = rest
                .iter()
                .map(|(m, _)| shared_start(first, m, fold_case))
                .min()
                .unwrap_or(0);
            if n <= typed_len {
                return Completion::Several(matches.iter().map(shown).collect());
            }
            first.chars().take(n).collect()
        }
    };
    if completion == prefix {
        return Completion::Nothing;
    }
    // The name typed, which the text ends with, gives way to the name found, whose case may
    // differ.
    let head = &typed[..typed.len() - prefix.len()];
    Completion::Done(format!("{head}{completion}"))
}

/// How many characters `a` and `b` share from their start, whatever their case where
/// `fold_case`.
fn shared_start(a: &str, b: &str, fold_case: bool) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(x, y)| x == y || (fold_case && x.to_lowercase().eq(y.to_lowercase())))
        .count()
}

/// What a path field's Tab shows where several names match: the first few, and how many more.
fn several(names: &[String]) -> String {
    const SHOWN: usize = 8;
    let mut list = names[..names.len().min(SHOWN)].join(", ");
    if names.len() > SHOWN {
        list.push_str(&format!(" and {} more", names.len() - SHOWN));
    }
    format!(
        "{} names match: {list}. Type more of one, then Tab again.",
        names.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completes_paths() {
        let d = std::env::temp_dir().join(format!("reims-ctl-complete-{}", std::process::id()));
        std::fs::create_dir_all(d.join("opencore-dir")).unwrap();
        std::fs::write(d.join("opencore.qcow2"), b"").unwrap();
        std::fs::write(d.join("other.img"), b"").unwrap();
        std::fs::write(d.join(".hidden"), b"").unwrap();
        let sep = std::path::MAIN_SEPARATOR;
        let base = format!("{}{sep}", d.display());
        let done = Completion::Done;
        assert_eq!(
            complete_in(&format!("{base}ot"), false),
            done(format!("{base}other.img"))
        );
        assert_eq!(
            complete_in(&format!("{base}op"), false),
            done(format!("{base}opencore"))
        );
        assert_eq!(
            complete_in(&format!("{base}opencore-"), false),
            done(format!("{base}opencore-dir{sep}"))
        );
        assert_eq!(
            complete_in(&format!("{base}zz"), false),
            Completion::Nothing
        );
        assert_eq!(
            complete_in(&format!("{base}other.img"), false),
            Completion::Nothing,
            "a file named whole has nothing to complete, so Tab moves on"
        );
        assert_eq!(complete_in("", false), Completion::Nothing);
        assert_eq!(complete_in("  ", false), Completion::Nothing);
        // Where several names match and share no more than was typed, Tab lists them, hiding
        // names that start with `.` unless one was typed.
        assert_eq!(
            complete_in(&format!("{base}opencore"), false),
            Completion::Several(vec![
                format!("opencore-dir{sep}"),
                "opencore.qcow2".to_owned()
            ])
        );
        assert_eq!(complete_in(&base, false), done(format!("{base}o")));
        assert_eq!(
            complete_in(&format!("{base}o"), false),
            Completion::Several(vec![
                format!("opencore-dir{sep}"),
                "opencore.qcow2".to_owned(),
                "other.img".to_owned()
            ])
        );
        assert_eq!(
            complete_in(&format!("{base}."), false),
            done(format!("{base}.{sep}")),
            "`.` names a directory"
        );
        assert_eq!(
            complete_in(&format!("{base}.h"), false),
            done(format!("{base}.hidden"))
        );
        // A directory typed with its separator lists what it holds.
        assert_eq!(
            complete_in(&format!("{base}opencore-dir{sep}"), false),
            Completion::Nothing,
            "an empty directory completes nothing"
        );
        // A path whose last part is no name takes the directory's separator.
        assert_eq!(
            complete_in(&format!("{base}opencore-dir{sep}.."), false),
            done(format!("{base}opencore-dir{sep}..{sep}"))
        );
        assert_eq!(complete_in("~", false), done(format!("~{sep}")));
        // On Windows a path typed with `/` completes there too, adding `\` after a directory,
        // and a drive takes its `\`.
        #[cfg(windows)]
        {
            assert_eq!(
                complete(&format!("{}/opencore-", d.display())),
                done(format!("{}/opencore-dir\\", d.display()))
            );
            assert_eq!(complete("C:"), done("C:\\".to_owned()));
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Windows' file systems match names whatever their case, so its Tab does: `doc` finds
    /// `Documents`.
    #[test]
    fn completes_paths_whatever_their_case_where_asked() {
        let d = std::env::temp_dir().join(format!("reims-ctl-case-{}", std::process::id()));
        std::fs::create_dir_all(d.join("Documents")).unwrap();
        std::fs::create_dir_all(d.join("Downloads")).unwrap();
        std::fs::write(d.join("other.img"), b"").unwrap();
        let sep = std::path::MAIN_SEPARATOR;
        let base = format!("{}{sep}", d.display());
        let done = Completion::Done;
        assert_eq!(
            complete_in(&format!("{base}doc"), true),
            done(format!("{base}Documents{sep}"))
        );
        assert_eq!(
            complete_in(&format!("{base}OTHER.IMG"), true),
            done(format!("{base}other.img")),
            "a name typed whole in another case takes the name's own"
        );
        assert_eq!(
            complete_in(&format!("{base}d"), true),
            done(format!("{base}Do")),
            "names that match whatever their case complete as far as they share"
        );
        assert_eq!(
            complete_in(&format!("{base}do"), true),
            Completion::Several(vec![format!("Documents{sep}"), format!("Downloads{sep}")])
        );
        assert_eq!(
            complete_in(&format!("{base}doc"), false),
            Completion::Nothing
        );
        assert_eq!(
            complete(&format!("{base}doc")),
            complete_in(&format!("{base}doc"), cfg!(windows))
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Tab where several names match lists them below the fields and keeps the field, until the
    /// next key; where nothing matches it moves on.
    #[test]
    fn tab_lists_the_names_that_match_and_keeps_the_field() {
        let d = std::env::temp_dir().join(format!("reims-ctl-tab-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for n in ["disk-a.img", "disk-b.img"] {
            std::fs::write(d.join(n), b"").unwrap();
        }
        let sep = std::path::MAIN_SEPARATOR;
        let mut f = Form::new(
            "t",
            "save",
            vec![
                Field::path("path", "Path", &format!("{}{sep}disk-", d.display()), ""),
                Field::text("size", "Size", "", ""),
            ],
        );
        let tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        f.handle_key(tab);
        assert_eq!(f.focused_key(), "path");
        assert_eq!(
            f.hint.as_deref(),
            Some("2 names match: disk-a.img, disk-b.img. Type more of one, then Tab again.")
        );
        f.handle_key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE));
        assert_eq!(f.hint, None);
        f.handle_key(tab);
        assert_eq!(f.focused_key(), "size");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn several_shows_the_first_names_and_how_many_more() {
        let names: Vec<String> = (0..10).map(|i| format!("n{i}")).collect();
        assert_eq!(
            several(&names),
            "10 names match: n0, n1, n2, n3, n4, n5, n6, n7 and 2 more. Type more of one, then \
             Tab again."
        );
    }

    #[test]
    fn hidden_fields_read_empty_and_are_skipped() {
        let mut f = Form::new(
            "t",
            "save",
            vec![
                Field::choice(
                    "kind",
                    "Disk",
                    opts(&[("new", "New"), ("existing", "Existing")]),
                    "new",
                    "",
                ),
                Field::text("size", "Size", "128G", "").show_if("kind", &["new"]),
                Field::path("path", "Path", "/x", "").show_if("kind", &["existing"]),
            ],
        );
        assert_eq!(f.text("size"), "128G");
        assert_eq!(f.text("path"), "");
        f.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert_eq!(f.choice("kind"), "existing");
        assert_eq!(f.text("size"), "");
        f.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(f.focused_key(), "path");
    }

    /// The edit form's GPU, when the host's GPUs are listed after it opened: a VM's GPU the list
    /// lacks (its driver moved, say) stays the VM's, rather than turning into Automatic, which
    /// the form would then save.
    #[test]
    fn new_options_keep_a_selection_they_lack() {
        let mut f = Form::new(
            "t",
            "save",
            vec![Field::choice(
                "gpu",
                "GPU",
                opts(&[("auto", "Automatic"), ("/icd/a.json", "A")]),
                "/icd/a.json",
                "",
            )],
        );
        f.set_options("gpu", opts(&[("auto", "Automatic"), ("/icd/b.json", "B")]));
        assert_eq!(f.choice("gpu"), "/icd/a.json");
        f.set_options(
            "gpu",
            opts(&[("auto", "Automatic"), ("/icd/a.json", "A, listed")]),
        );
        assert_eq!(f.choice("gpu"), "/icd/a.json");
        match &f.fields[0].value {
            Value::Choice { options, .. } => assert_eq!(options.len(), 2),
            _ => unreachable!(),
        }
    }

    #[test]
    fn a_field_shows_in_a_form_without_the_field_it_follows() {
        let field = Field::text("resolution", "Resolution", "1", "").show_if("graphics", &["x"]);
        assert_eq!(Form::new("t", "save", vec![field]).text("resolution"), "1");
    }
}
