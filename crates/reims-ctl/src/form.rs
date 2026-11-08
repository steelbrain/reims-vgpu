//! A form for the interface: text, path, choice and toggle fields, some shown only for some
//! answers of another field (`show_if`).
//!
//! Keys: Up and Down (or Tab and Shift-Tab) move between fields; Left and Right, or Space,
//! change a choice or toggle; typing edits text; Tab in a path field completes it as a shell
//! would, and moves on when there is nothing to complete; Enter moves on, and on the last field
//! submits; Ctrl-S submits from anywhere; Esc cancels. While the work a submit started runs
//! (`busy`), the form shows its step and takes only Esc, which closes it.

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
                        if *path && let Some(done) = complete(text) {
                            *text = done;
                            *cursor = text.chars().count();
                        } else {
                            self.move_focus(true);
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
        // The running work's step, or the error, above the help, and a blank line.
        let note = match (&self.busy, &self.error) {
            (Some((_, step)), _) => Some((step, Style::new().fg(Color::Yellow))),
            (None, Some(e)) => Some((e, Style::new().fg(Color::Red).bold())),
            (None, None) => None,
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

/// `text` completed as a path, as a shell's Tab would: the one match whole (a directory with
/// its `/`), or the matches' longest common prefix. None when nothing would change.
pub fn complete(text: &str) -> Option<String> {
    let typed = text.to_string();
    let expanded = crate::paths::expand(&typed);
    let exp = expanded.to_string_lossy().into_owned();
    let (dir, prefix) = if exp.ends_with('/') {
        (PathBuf::from(&exp), String::new())
    } else {
        let p = Path::new(&exp);
        let dir = p
            .parent()
            .map(Path::to_path_buf)
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or_else(|| PathBuf::from("."));
        (
            dir,
            p.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
        )
    };
    let mut matches: Vec<(String, bool)> = std::fs::read_dir(&dir)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            (name.starts_with(&prefix) && (!name.starts_with('.') || prefix.starts_with('.')))
                .then(|| (name, e.path().is_dir()))
        })
        .collect();
    if matches.is_empty() {
        return None;
    }
    matches.sort();
    let completion = if matches.len() == 1 {
        let (name, is_dir) = &matches[0];
        format!("{name}{}", if *is_dir { "/" } else { "" })
    } else {
        let first = &matches[0].0;
        let mut n = first.len();
        for (m, _) in &matches[1..] {
            n = n.min(
                first
                    .bytes()
                    .zip(m.bytes())
                    .take_while(|(a, b)| a == b)
                    .count(),
            );
        }
        first[..n].to_string()
    };
    let addition = completion.strip_prefix(&prefix)?;
    if addition.is_empty() {
        return None;
    }
    Some(format!("{typed}{addition}"))
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
        let base = format!("{}/", d.display());
        assert_eq!(
            complete(&format!("{base}ot")),
            Some(format!("{base}other.img"))
        );
        assert_eq!(
            complete(&format!("{base}op")),
            Some(format!("{base}opencore"))
        );
        assert_eq!(
            complete(&format!("{base}opencore-")),
            Some(format!("{base}opencore-dir/"))
        );
        assert_eq!(complete(&format!("{base}zz")), None);
        std::fs::remove_dir_all(&d).unwrap();
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
