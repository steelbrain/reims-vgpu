//! The library's `log` logger: silent by default, stderr with `REIMS_VGPU_LOG`, or the host's
//! hook.
//!
//! The library writes its lines through the `log` crate's macros, each under its module's path as
//! its target (`reims_vgpu::graphics::decode`). Where they go:
//!
//! - a host that installed its own `log` logger gets them there, filtered as it filters;
//! - otherwise nowhere, unless `REIMS_VGPU_LOG` is set or the host gave a hook ([`set_hook`]).
//!   Then [`init`] or [`set_hook`] installs this module's logger, once per process, and each line
//!   goes to the hook if there is one, and to stderr otherwise, as
//!   `reims-vgpu <LEVEL> <target>: <message>`.
//!
//! `REIMS_VGPU_LOG` is a level (`off`, `error`, `warn`, `info`, `debug` or `trace`), optionally
//! followed by `,target=level` pairs: `warn,reims_vgpu::graphics=debug` ([`Filter::parse`]).
//! When it is set, its filter decides which lines are written, to the hook too; when it isn't,
//! the hook's own level does. A diagnostic whose own work costs the work it watches runs only
//! where the variable names its target itself ([`names`]), not under a lone level.
//!
//! Which lines are written, and whether a line's arguments are evaluated, are diagnostics: a line
//! evaluates its arguments only when a logger takes it, so no argument has an effect. A value a
//! line shows that has one (a read of guest memory, a question to Metal) is read before the line,
//! whether or not it is written.
//!
//! **A warn or error line is written once per kind.** Such a line says a fallback, a refusal or a
//! failure, whose cause may come again at every event that meets it (a guest's message, a frame,
//! an object's release). This logger writes the first line of each kind at its level, and each
//! later one at debug, where the filter takes debug for its target, counting it ([`Kinds`]). A
//! kind is the line's call site (its file and line) and, where the line names one, the class of
//! its [`KIND`] key-value, which tells apart the causes one site reports: the cause itself,
//! `log::warn!(target: "…", kind:% = error; "… was refused: {error}")`, whose text up to its
//! first digit, quote or `;` is its class ([`class`]), so a size, an address or a name in it
//! makes no kind of its own. An info line that names a [`KIND`] is written once per kind too: a
//! fallback each session or object may take, said once per process. [`write_repeats`] writes how
//! many lines of each kind came after its first since its last line, at a device's reset and at
//! the process's exit (`crate::answer::write_counts`). Other info lines, and debug and trace
//! lines, are written as they come: a line at info is never on a path every event takes
//! (docs/conventions.md).

use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use log::{Level, LevelFilter, Log, Metadata, Record};

use crate::core::sync::Mutex;

/// The environment variable that turns the stderr logger on, and holds its filter.
pub const ENV_VAR: &str = "REIMS_VGPU_LOG";

/// A host's receiver of lines: a line's level, target and message, on the thread that writes
/// the line.
pub type Hook = Box<dyn Fn(Level, &str, &str) + Send + Sync>;

/// A hook as the logger keeps it: its level, and the hook, shared with the lines being written.
type HookEntry = (LevelFilter, Arc<dyn Fn(Level, &str, &str) + Send + Sync>);

/// The level each target is written up to: a level per target prefix, and one for the rest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    rest: LevelFilter,
    targets: Vec<(String, LevelFilter)>,
}

impl Filter {
    /// Every target up to `level`.
    pub fn all(level: LevelFilter) -> Filter {
        Filter {
            rest: level,
            targets: Vec::new(),
        }
    }

    /// `spec`, read as `REIMS_VGPU_LOG` is: comma-separated entries, each a level (case aside)
    /// or `target=level`.
    ///
    /// - A lone level covers every target no pair names; without one, those are `off`.
    /// - A pair covers its target and the targets below it at a `::`: `reims_vgpu` covers
    ///   `reims_vgpu::graphics`. The longest covering target decides, and among equal ones the
    ///   last.
    /// - An entry that is neither is ignored.
    pub fn parse(spec: &str) -> Filter {
        let mut filter = Filter::all(LevelFilter::Off);
        for entry in spec.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            match entry.split_once('=') {
                Some((target, level)) => {
                    if let Ok(level) = level.trim().parse() {
                        filter.targets.push((target.trim().to_owned(), level));
                    }
                }
                None => {
                    if let Ok(level) = entry.parse() {
                        filter.rest = level;
                    }
                }
            }
        }
        filter
    }

    /// The level lines of `target` are written up to.
    pub fn level(&self, target: &str) -> LevelFilter {
        self.targets
            .iter()
            .filter(|(prefix, _)| covers(prefix, target))
            .max_by_key(|(prefix, _)| prefix.len())
            .map_or(self.rest, |(_, level)| *level)
    }

    /// The most verbose level any target is written up to.
    pub fn max(&self) -> LevelFilter {
        self.targets
            .iter()
            .map(|(_, level)| *level)
            .fold(self.rest, Ord::max)
    }

    /// Whether a pair names `target` itself, up to `level` or finer: a lone level does not, nor
    /// does a pair of a target above it (`reims_vgpu` for `reims_vgpu::work`) or below it. The
    /// last pair of `target` decides, as in [`Self::level`].
    pub fn names(&self, target: &str, level: Level) -> bool {
        (self.targets.iter().rev())
            .find(|(named, _)| named == target)
            .is_some_and(|(_, up_to)| level <= *up_to)
    }
}

/// Whether `REIMS_VGPU_LOG` names `target` itself, up to `level` or finer ([`Filter::names`]).
/// False where the variable is not set: a hook's level is a lone level. For a diagnostic whose
/// own work costs the work it watches, which runs only where its target is asked for by name,
/// not under a lone level such as a bug report's `debug`. Whether its lines are then written is
/// still the logger's to say (`log::log_enabled!`).
pub fn names(target: &str, level: Level) -> bool {
    LOGGER
        .env()
        .is_some_and(|filter| filter.names(target, level))
}

/// Whether the filter's `prefix` covers `target`: it is `target`, or `target`'s path above a `::`.
fn covers(prefix: &str, target: &str) -> bool {
    target
        .strip_prefix(prefix)
        .is_some_and(|below| below.is_empty() || below.starts_with("::"))
}

/// A line as stderr shows it, with its newline.
pub fn stderr_line(level: Level, target: &str, message: &str) -> String {
    format!("reims-vgpu {level} {target}: {message}\n")
}

/// The key-value a warn or error line names its cause by, where its call site reports more than
/// one (the module docs): the [`class`] of its value's `Display` text, with the call site, is the
/// line's kind.
pub const KIND: &str = "kind";

/// How many kinds of warn and error lines the logger tells apart; a line of a kind past them is
/// written at debug, and counted with the others past them ([`Kinds::past`]).
const KINDS_KEPT: usize = 256;

/// The class of a cause's text, which a line's [`KIND`] names: its text up to its first ASCII
/// digit, `"`, `` ` `` or `;`, less the spaces and opening punctuation it then ends with. What
/// varies between causes of one class (a size, an address, an index, a quoted name, a clause of
/// detail) comes after it: `too small: 16 x 16` is of the class `too small`,
/// `NoFunction("main0")` of `NoFunction`.
pub fn class(text: &str) -> &str {
    let end = text
        .find(|c: char| c.is_ascii_digit() || matches!(c, '"' | '`' | ';'))
        .unwrap_or(text.len());
    text[..end].trim_end_matches(|c: char| c.is_whitespace() || "([{<:,=#".contains(c))
}

/// The target of [`write_repeats`]'s line.
const REPEATS_TARGET: &str = "reims_vgpu_vulkan::core::logging";

/// How many characters of a kind's first line [`write_repeats`]'s line shows.
const FIRST_SHOWN: usize = 120;

/// A line's kind: its call site, and the [`class`] of the [`KIND`] it names ("" for none).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Site {
    file: &'static str,
    line: u32,
    kind: String,
}

impl Site {
    /// `record`'s, where it is written once per kind (the module docs): a warn or error line, or
    /// an info line that names a [`KIND`], whose call site is known, as it is for every line of
    /// `log`'s macros.
    fn of(record: &Record<'_>) -> Option<Site> {
        let kind = (record.key_values().get(log::kv::Key::from_str(KIND)))
            .map(|kind| class(&kind.to_string()).to_owned());
        if record.level() > Level::Warn && (record.level() > Level::Info || kind.is_none()) {
            return None;
        }
        Some(Site {
            file: record.file_static()?,
            line: record.line()?,
            kind: kind.unwrap_or_default(),
        })
    }
}

/// The lines of one kind the logger wrote or counted.
struct Kind {
    site: Site,
    /// The first line's target and message, cut to [`FIRST_SHOWN`] characters.
    first: String,
    /// How many lines of the kind came after the first.
    after: u64,
    /// Of them, how many [`write_repeats`] has counted.
    counted: u64,
}

/// The kinds of warn and error lines the logger took, in the order of their first lines (the
/// module docs). A process meets a few dozen at most, so a list serves.
struct Kinds {
    kept: Vec<Kind>,
    /// Lines of kinds past the first [`KINDS_KEPT`], each written at debug.
    past: u64,
    /// Of them, how many [`write_repeats`] has counted.
    past_counted: u64,
}

impl Kinds {
    /// An empty list, for a `static`'s initializer.
    const fn new() -> Kinds {
        Kinds {
            kept: Vec::new(),
            past: 0,
            past_counted: 0,
        }
    }

    /// Notes a line of `site`: whether it is its kind's first, which is written at its level.
    fn note(&mut self, site: &Site) -> bool {
        if let Some(kind) = self.kept.iter_mut().find(|kind| kind.site == *site) {
            kind.after += 1;
            return false;
        }
        if self.kept.len() == KINDS_KEPT {
            self.past += 1;
            return false;
        }
        self.kept.push(Kind {
            site: site.clone(),
            first: String::new(),
            after: 0,
            counted: 0,
        });
        true
    }

    /// Keeps the first line of `site`'s kind, of `target` and `message`, for [`Self::repeats`].
    fn describe(&mut self, site: &Site, target: &str, message: &str) {
        if let Some(kind) = self.kept.iter_mut().find(|kind| kind.site == *site) {
            let mut first = format!("{target}: {message}");
            if let Some((cut, _)) = first.char_indices().nth(FIRST_SHOWN) {
                first.truncate(cut);
                first.push('…');
            }
            kind.first = first;
        }
    }

    /// The line counting each kind that had lines after its first since the last count, in the
    /// order of their first lines, then the lines of kinds past the first [`KINDS_KEPT`]; `None`
    /// where there were none.
    fn repeats(&mut self) -> Option<String> {
        let mut line = String::new();
        for kind in &mut self.kept {
            let more = kind.after - kind.counted;
            if more == 0 {
                continue;
            }
            kind.counted = kind.after;
            if !line.is_empty() {
                line.push_str("; ");
            }
            let Site {
                file,
                line: at,
                kind: named,
            } = &kind.site;
            let _ = write!(line, "{more} more of {file}:{at}");
            if !named.is_empty() {
                let _ = write!(line, " [{named}]");
            }
            let _ = write!(line, " ({})", kind.first);
        }
        let past = self.past - self.past_counted;
        if past > 0 {
            self.past_counted = self.past;
            if !line.is_empty() {
                line.push_str("; ");
            }
            let _ = write!(line, "{past} of kinds past the first {KINDS_KEPT}");
        }
        (!line.is_empty()).then(|| format!("lines repeated since the last count: {line}"))
    }
}

/// The logger: the environment's filter, the host's hook, and stderr.
struct Logger {
    /// `REIMS_VGPU_LOG`'s filter, read on first use; `None` if it is not set.
    env: OnceLock<Option<Filter>>,
    hook: Mutex<Option<HookEntry>>,
    /// The hook's level ([`hook_code`]), written under `hook`'s lock as the hook is set or
    /// cleared ([`Self::keep`]): what [`Log::enabled`] and `log`'s filter read, so neither takes
    /// the lock for a line that is not written. Every encoder call of the Vulkan platform asks
    /// whether its debug line is on, from as many walk threads as run at once.
    hook_level: AtomicUsize,
    /// The kinds of warn and error lines taken (the module docs).
    kinds: Mutex<Kinds>,
}

/// A hook's level as [`Logger::hook_level`] keeps it: 0 for no hook, else the level's place in
/// [`LevelFilter::iter`] plus one.
fn hook_code(level: Option<LevelFilter>) -> usize {
    level
        .and_then(|level| LevelFilter::iter().position(|l| l == level))
        .map_or(0, |at| at + 1)
}

/// The hook's level [`hook_code`] gave `code`.
fn hook_level_of(code: usize) -> Option<LevelFilter> {
    LevelFilter::iter().nth(code.checked_sub(1)?)
}

impl Logger {
    const fn new() -> Logger {
        Logger {
            env: OnceLock::new(),
            hook: Mutex::new(None),
            hook_level: AtomicUsize::new(0),
            kinds: Mutex::new(Kinds::new()),
        }
    }

    /// Puts `entry` in the hook's `slot`, which the caller holds locked, and its level where
    /// [`Self::hook_level`] reads it.
    fn keep(&self, slot: &mut Option<HookEntry>, entry: Option<HookEntry>) {
        let code = hook_code(entry.as_ref().map(|(level, _)| *level));
        self.hook_level.store(code, Ordering::Release);
        *slot = entry;
    }

    /// The hook's level, `None` without a hook, read without its lock.
    fn hook_level(&self) -> Option<LevelFilter> {
        hook_level_of(self.hook_level.load(Ordering::Acquire))
    }

    fn env(&self) -> Option<&Filter> {
        self.env
            .get_or_init(|| std::env::var(ENV_VAR).ok().map(|spec| Filter::parse(&spec)))
            .as_ref()
    }

    /// The level lines of `target` are written up to, given the hook's level if there is a
    /// hook.
    fn level(&self, target: &str, hook: Option<LevelFilter>) -> LevelFilter {
        match (self.env(), hook) {
            (Some(env), _) => env.level(target),
            (None, Some(level)) => level,
            (None, None) => LevelFilter::Off,
        }
    }

    /// The most verbose level any line is written up to, given the hook's level.
    fn max_level(&self, hook: Option<LevelFilter>) -> LevelFilter {
        match (self.env(), hook) {
            (Some(env), _) => env.max(),
            (None, Some(level)) => level,
            (None, None) => LevelFilter::Off,
        }
    }
}

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= self.level(metadata.target(), self.hook_level())
    }

    fn log(&self, record: &Record<'_>) {
        // A line not written takes no lock; one written is filtered again by the hook it goes
        // to, as a hook set or cleared meanwhile may change its level.
        if !self.enabled(record.metadata()) {
            return;
        }
        // The hook is called without the lock: it may write lines of its own, or be replaced.
        let hook = self.hook.lock().clone();
        let up_to = self.level(record.target(), hook.as_ref().map(|(level, _)| *level));
        if record.level() > up_to {
            return;
        }
        // A warn or error line after its kind's first is a debug line, counted (the module
        // docs).
        let site = Site::of(record);
        let first = (site.as_ref()).is_none_or(|site| self.kinds.lock().note(site));
        let level = if first { record.level() } else { Level::Debug };
        if level > up_to {
            return;
        }
        let mut message = String::new();
        // A `Display` impl that fails leaves what it wrote.
        let _ = write!(message, "{}", record.args());
        if let (true, Some(site)) = (first, &site) {
            self.kinds.lock().describe(site, record.target(), &message);
        }
        write_to(hook, level, record.target(), &message);
    }

    fn flush(&self) {}
}

/// Writes `message`, a line of `level` and `target` the filter passed, to `hook`, or to stderr
/// without one.
fn write_to(hook: Option<HookEntry>, level: Level, target: &str, message: &str) {
    match hook {
        Some((_, hook)) => hook(level, target, message),
        None => {
            let line = stderr_line(level, target, message);
            // One write per line, so lines from different threads do not interleave.
            let _ = std::io::stderr().write_all(line.as_bytes());
        }
    }
}

impl Logger {
    /// Writes the line counting the lines that came after their kind's first since the last
    /// such line, where any did ([`Kinds::repeats`]): a summary, at info, under
    /// [`REPEATS_TARGET`], where the filter takes it.
    fn write_repeats(&self) {
        let hook = self.hook.lock().clone();
        if Level::Info > self.level(REPEATS_TARGET, hook.as_ref().map(|(level, _)| *level)) {
            return;
        }
        let Some(line) = self.kinds.lock().repeats() else {
            return;
        };
        write_to(hook, Level::Info, REPEATS_TARGET, &line);
    }
}

/// Writes how many lines of each kind came after its first since the last count, one line at
/// info (the module docs), where any did and the process's logger is this module's:
/// `reims-vgpu INFO reims_vgpu_vulkan::core::logging: lines repeated since the last count: <n>
/// more of <file>:<line>[ [<kind>]] (<the first line's target>: <its message>); …[; <n> of kinds
/// past the first 256]`, the first line cut at 120 characters. A device's reset and the process's
/// exit write it (`crate::answer::write_counts`).
pub fn write_repeats() {
    if OURS.get() == Some(&true) {
        LOGGER.write_repeats();
    }
}

static LOGGER: Logger = Logger::new();

/// Whether this module's logger is the process's: `Some(true)` once installed, `Some(false)` if
/// the host's was there first, `None` before anything asked.
static OURS: OnceLock<bool> = OnceLock::new();

/// Installs this module's logger unless the process has one: once per process.
fn install() -> bool {
    *OURS.get_or_init(|| log::set_logger(&LOGGER).is_ok())
}

/// Sets `log`'s global level to what the logger writes now. `hook` is the hook's level, read
/// under the hook's lock so a concurrent [`set_hook`] cannot leave a stale level.
fn refresh_max_level(hook: Option<LevelFilter>) {
    if OURS.get() == Some(&true) {
        log::set_max_level(LOGGER.max_level(hook));
    }
}

/// Installs the stderr logger if `REIMS_VGPU_LOG` is set, unless the process already has a
/// `log` logger (the host's). The product calls it when it makes its platform; a later call
/// changes nothing.
pub fn init() {
    if LOGGER.env().is_some() && install() {
        let hook = LOGGER.hook.lock();
        refresh_max_level(hook.as_ref().map(|(level, _)| *level));
    }
}

/// Sends every line up to `level` to `hook`, in place of stderr, from now on: the hook is called
/// synchronously on the thread that writes the line. When `REIMS_VGPU_LOG` is set, its filter
/// decides instead of `level`.
///
/// Installs this module's logger if it is not yet the process's. Answers false, and keeps no
/// hook, if the host installed its own `log` logger first: the lines go there.
pub fn set_hook(level: LevelFilter, hook: Hook) -> bool {
    if !install() {
        return false;
    }
    let mut slot = LOGGER.hook.lock();
    LOGGER.keep(&mut slot, Some((level, Arc::from(hook))));
    refresh_max_level(Some(level));
    true
}

/// Removes the hook: lines go to stderr again if `REIMS_VGPU_LOG` is set, and nowhere otherwise.
/// A line being written on another thread may still reach the old hook.
pub fn clear_hook() {
    let mut slot = LOGGER.hook.lock();
    LOGGER.keep(&mut slot, None);
    refresh_max_level(None);
}

#[cfg(test)]
mod tests {
    use super::*;

    type Seen = Arc<Mutex<Vec<(Level, String, String)>>>;

    /// A hook that keeps what it is given.
    fn recording_hook() -> (Seen, Hook) {
        let seen: Seen = Arc::default();
        let kept = Arc::clone(&seen);
        let hook: Hook = Box::new(move |level, target, message| {
            kept.lock()
                .push((level, target.to_owned(), message.to_owned()));
        });
        (seen, hook)
    }

    /// A logger that read `env` from the environment, with no hook yet.
    fn logger(env: Option<&str>) -> Logger {
        Logger {
            env: OnceLock::from(env.map(Filter::parse)),
            hook: Mutex::new(None),
            hook_level: AtomicUsize::new(0),
            kinds: Mutex::new(Kinds::new()),
        }
    }

    /// `l` with a hook that keeps every line its filter passes.
    fn hooked(l: &Logger) -> Seen {
        let (seen, hook) = recording_hook();
        l.keep(
            &mut l.hook.lock(),
            Some((LevelFilter::Trace, Arc::from(hook))),
        );
        seen
    }

    /// Writes `text` at `level` from the call site `x.rs:line`, naming `kind` where given, as
    /// `log`'s macros do.
    fn write_from(l: &Logger, level: Level, line: u32, kind: Option<&str>, text: &str) {
        let kinds = kind.map(|kind| [(KIND, kind)]);
        let none: [(&str, &str); 0] = [];
        l.log(
            &Record::builder()
                .args(format_args!("{text}"))
                .level(level)
                .target("reims_vgpu::graphics")
                .file_static(Some("x.rs"))
                .line(Some(line))
                .key_values(match &kinds {
                    Some(kinds) => kinds as &dyn log::kv::Source,
                    None => &none,
                })
                .build(),
        );
    }

    /// The levels and messages `seen` kept, taken.
    fn taken(seen: &Seen) -> Vec<(Level, String)> {
        (seen.lock().drain(..))
            .map(|(level, _, message)| (level, message))
            .collect()
    }

    /// A warn or error line of a kind already written is a debug line, and counted: the count
    /// names its call site and its first line, once
    #[test]
    fn a_kinds_later_lines_are_debug_lines_and_counted() {
        let l = logger(Some("debug"));
        let seen = hooked(&l);
        write_from(&l, Level::Warn, 7, None, "refused 1");
        write_from(&l, Level::Warn, 7, None, "refused 2");
        write_from(&l, Level::Error, 9, None, "broken");
        write_from(&l, Level::Warn, 7, None, "refused 3");
        assert_eq!(
            taken(&seen),
            [
                (Level::Warn, "refused 1".to_owned()),
                (Level::Debug, "refused 2".to_owned()),
                (Level::Error, "broken".to_owned()),
                (Level::Debug, "refused 3".to_owned()),
            ]
        );
        l.write_repeats();
        assert_eq!(
            taken(&seen),
            [(
                Level::Info,
                "lines repeated since the last count: 2 more of x.rs:7 (reims_vgpu::graphics: \
                 refused 1)"
                    .to_owned()
            )]
        );
        l.write_repeats();
        assert_eq!(taken(&seen), [], "nothing repeated since");
        write_from(&l, Level::Error, 9, None, "broken again");
        l.write_repeats();
        assert_eq!(
            taken(&seen),
            [
                (Level::Debug, "broken again".to_owned()),
                (
                    Level::Info,
                    "lines repeated since the last count: 1 more of x.rs:9 \
                     (reims_vgpu::graphics: broken)"
                        .to_owned()
                ),
            ]
        );
    }

    /// A line naming a new kind at a site written before is written at its level; a line of a
    /// kind written before, at debug
    #[test]
    fn a_new_kind_at_a_site_is_written_at_its_level() {
        let l = logger(Some("debug"));
        let seen = hooked(&l);
        write_from(&l, Level::Warn, 7, Some("Abi"), "refused: abi");
        write_from(&l, Level::Warn, 7, Some("Driver"), "refused: driver");
        write_from(&l, Level::Warn, 7, Some("Abi"), "refused: abi again");
        write_from(&l, Level::Warn, 7, None, "refused");
        assert_eq!(
            taken(&seen),
            [
                (Level::Warn, "refused: abi".to_owned()),
                (Level::Warn, "refused: driver".to_owned()),
                (Level::Debug, "refused: abi again".to_owned()),
                (Level::Warn, "refused".to_owned()),
            ]
        );
        l.write_repeats();
        assert_eq!(
            taken(&seen),
            [(
                Level::Info,
                "lines repeated since the last count: 1 more of x.rs:7 [Abi] \
                 (reims_vgpu::graphics: refused: abi)"
                    .to_owned()
            )]
        );
    }

    /// Under a filter that takes no debug line, a kind's later lines are not written, but still
    /// counted; info lines, and lines with no call site, are written at every occurrence
    #[test]
    fn a_kinds_later_lines_are_counted_where_debug_is_off() {
        let l = logger(Some("info"));
        let seen = hooked(&l);
        for n in 0..3 {
            write_from(&l, Level::Warn, 7, None, &format!("refused {n}"));
            write_from(&l, Level::Info, 8, None, &format!("shown {n}"));
        }
        write(&l, Level::Warn, "reims_vgpu::graphics", "no site");
        write(&l, Level::Warn, "reims_vgpu::graphics", "no site");
        assert_eq!(
            taken(&seen),
            [
                (Level::Warn, "refused 0".to_owned()),
                (Level::Info, "shown 0".to_owned()),
                (Level::Info, "shown 1".to_owned()),
                (Level::Info, "shown 2".to_owned()),
                (Level::Warn, "no site".to_owned()),
                (Level::Warn, "no site".to_owned()),
            ]
        );
        l.write_repeats();
        assert_eq!(
            taken(&seen),
            [(
                Level::Info,
                "lines repeated since the last count: 2 more of x.rs:7 (reims_vgpu::graphics: \
                 refused 0)"
                    .to_owned()
            )]
        );
    }

    /// An info line that names a kind is written once per kind, its later lines at debug and
    /// counted, as a warn line's
    #[test]
    fn an_info_line_naming_a_kind_is_written_once_per_kind() {
        let l = logger(Some("debug"));
        let seen = hooked(&l);
        write_from(&l, Level::Info, 8, Some("in place"), "landed in place");
        write_from(&l, Level::Info, 8, Some("copied"), "landed copied");
        write_from(&l, Level::Info, 8, Some("in place"), "landed in place");
        assert_eq!(
            taken(&seen),
            [
                (Level::Info, "landed in place".to_owned()),
                (Level::Info, "landed copied".to_owned()),
                (Level::Debug, "landed in place".to_owned()),
            ]
        );
        l.write_repeats();
        assert_eq!(
            taken(&seen),
            [(
                Level::Info,
                "lines repeated since the last count: 1 more of x.rs:8 [in place] \
                 (reims_vgpu::graphics: landed in place)"
                    .to_owned()
            )]
        );
    }

    /// A kind's class is its text up to its first digit or quote: the causes one site reports
    /// differ by it, and not by a size, an address or a name past it
    #[test]
    fn a_kinds_class_is_its_text_before_any_digit_or_quote() {
        assert_eq!(class("Abi"), "Abi");
        assert_eq!(class("too small: 16 x 16"), "too small");
        assert_eq!(class("NoFunction(\"main0\")"), "NoFunction");
        assert_eq!(class("ERROR_DEVICE_LOST"), "ERROR_DEVICE_LOST");
        assert_eq!(
            class("the library's function `f` is missing"),
            "the library's function"
        );
        assert_eq!(class("0x1000"), "");
        assert_eq!(class("nil; its functions: vertex"), "nil");

        let l = logger(Some("debug"));
        let seen = hooked(&l);
        write_from(&l, Level::Warn, 7, Some("too small: 16 x 16"), "a");
        write_from(&l, Level::Warn, 7, Some("too small: 8 x 8"), "b");
        write_from(&l, Level::Warn, 7, Some("too large: 8 x 8"), "c");
        assert_eq!(
            taken(&seen),
            [
                (Level::Warn, "a".to_owned()),
                (Level::Debug, "b".to_owned()),
                (Level::Warn, "c".to_owned()),
            ]
        );
    }

    /// Past the first 256 kinds, a line of a new kind is written at debug and counted with the
    /// others past them
    #[test]
    fn kinds_past_the_first_256_are_counted_together() {
        let l = logger(Some("info"));
        let seen = hooked(&l);
        for line in 0..KINDS_KEPT as u32 + 2 {
            write_from(&l, Level::Warn, line, None, "refused");
        }
        assert_eq!(taken(&seen).len(), KINDS_KEPT);
        l.write_repeats();
        assert_eq!(
            taken(&seen),
            [(
                Level::Info,
                "lines repeated since the last count: 2 of kinds past the first 256".to_owned()
            )]
        );
    }

    /// The count of a kind cuts its first line at 120 characters; under a filter that drops
    /// info lines for its target, no count is written
    #[test]
    fn the_count_cuts_long_first_lines_and_follows_the_filter() {
        let l = logger(Some("debug"));
        let seen = hooked(&l);
        let long = "x".repeat(200);
        write_from(&l, Level::Warn, 7, None, &long);
        write_from(&l, Level::Warn, 7, None, &long);
        let _ = taken(&seen);
        l.write_repeats();
        let shown = format!("reims_vgpu::graphics: {long}");
        let cut: String = shown.chars().take(FIRST_SHOWN).collect();
        assert_eq!(
            taken(&seen),
            [(
                Level::Info,
                format!("lines repeated since the last count: 1 more of x.rs:7 ({cut}…)")
            )]
        );

        let l = logger(Some("debug,reims_vgpu_vulkan::core::logging=error"));
        let seen = hooked(&l);
        write_from(&l, Level::Warn, 7, None, "refused");
        write_from(&l, Level::Warn, 7, None, "refused");
        l.write_repeats();
        assert_eq!(taken(&seen).len(), 2, "the two lines, and no count");
    }

    fn write(l: &Logger, level: Level, target: &str, text: &str) {
        l.log(
            &Record::builder()
                .args(format_args!("{text}"))
                .level(level)
                .target(target)
                .build(),
        );
    }

    fn enabled(l: &Logger, level: Level, target: &str) -> bool {
        l.enabled(&Metadata::builder().level(level).target(target).build())
    }

    /// A lone level covers every target; pairs cover their target and those below it
    #[test]
    fn filters_read_a_level_and_target_pairs() {
        let f = Filter::parse("warn,reims_vgpu::graphics=debug");
        assert_eq!(f.level("reims_vgpu::graphics"), LevelFilter::Debug);
        assert_eq!(f.level("reims_vgpu::graphics::decode"), LevelFilter::Debug);
        assert_eq!(f.level("reims_vgpu::graphics_host"), LevelFilter::Warn);
        assert_eq!(f.level("reims_vgpu::device"), LevelFilter::Warn);
        assert_eq!(f.max(), LevelFilter::Debug);

        let f = Filter::parse(" INFO , reims_vgpu = error , reims_vgpu::tasks=trace ");
        assert_eq!(f.level("other"), LevelFilter::Info);
        assert_eq!(f.level("reims_vgpu::graphics"), LevelFilter::Error);
        assert_eq!(f.level("reims_vgpu::tasks"), LevelFilter::Trace);
        assert_eq!(f.max(), LevelFilter::Trace);
    }

    /// Without a lone level the rest is off; unreadable entries are ignored; the last of equal
    /// pairs decides
    #[test]
    fn filters_default_to_off_and_skip_what_they_cannot_read() {
        let f = Filter::parse("reims_vgpu::graphics=info");
        assert_eq!(f.level("reims_vgpu::graphics"), LevelFilter::Info);
        assert_eq!(f.level("reims_vgpu::device"), LevelFilter::Off);
        assert_eq!(Filter::parse(""), Filter::all(LevelFilter::Off));
        assert_eq!(
            Filter::parse("loud,debug,x=loud,,=,y"),
            Filter::all(LevelFilter::Debug)
        );
        let f = Filter::parse("a=debug,a=error");
        assert_eq!(f.level("a"), LevelFilter::Error);
        assert_eq!(f.max(), LevelFilter::Debug);
    }

    /// A filter names a target only by a pair of the target itself, the last one deciding: not
    /// by a lone level, nor by a pair of a target above or below it
    #[test]
    fn a_filter_names_a_target_only_by_its_own_pair() {
        const WORK: &str = "reims_vgpu::work";
        let names = |spec: &str| Filter::parse(spec).names(WORK, Level::Debug);
        assert_eq!(Filter::parse("debug").level(WORK), LevelFilter::Debug);
        assert!(
            !names("debug"),
            "a lone level, which writes the target's lines"
        );
        assert!(!names("trace"));
        assert!(names("info,reims_vgpu::work=debug"));
        assert!(names("reims_vgpu::work=trace"));
        assert!(
            !names("reims_vgpu::work=info"),
            "named, but not up to debug"
        );
        assert!(!names("reims_vgpu=debug"), "a target above it");
        assert!(!names("reims_vgpu::work::lines=debug"), "a target below it");
        assert!(!names("reims_vgpu::works=debug"));
        assert!(!names("reims_vgpu::work=debug,reims_vgpu::work=warn"));
        assert!(names("reims_vgpu::work=warn,reims_vgpu::work=debug"));
    }

    /// The process names a target only where the variable's own pair does: not without the
    /// variable, nor under a lone level or a pair above it, a hook at any level aside
    #[test]
    fn the_process_names_a_target_only_by_the_variables_pair() {
        if std::env::var_os(CHILD).is_some() {
            let named = names("reims_vgpu::work", Level::Debug);
            let (_, hook) = recording_hook();
            assert!(set_hook(LevelFilter::Trace, hook));
            assert_eq!(names("reims_vgpu::work", Level::Debug), named);
            println!("named {named}");
            return;
        }
        for (env, named) in [
            (None, false),
            (Some("debug"), false),
            (Some("trace"), false),
            (Some("reims_vgpu=debug"), false),
            (Some("info,reims_vgpu::work=debug"), true),
        ] {
            let out = child(
                "core::logging::tests::the_process_names_a_target_only_by_the_variables_pair",
                env,
            );
            assert!(out.status.success(), "{env:?}: {out:?}");
            let said = String::from_utf8_lossy(&out.stdout);
            assert!(said.contains(&format!("named {named}")), "{env:?}: {said}");
        }
    }

    /// A stderr line names the library, the level and the target
    #[test]
    fn stderr_lines_name_level_and_target() {
        assert_eq!(
            stderr_line(Level::Warn, "reims_vgpu::graphics", "stalled 3"),
            "reims-vgpu WARN reims_vgpu::graphics: stalled 3\n"
        );
    }

    /// With neither the variable nor a hook, nothing is enabled
    #[test]
    fn silent_without_the_variable_or_a_hook() {
        let l = logger(None);
        assert!(!enabled(&l, Level::Error, "reims_vgpu::device"));
        assert_eq!(l.max_level(None), LevelFilter::Off);
    }

    /// A hook takes every line up to its level; the variable's filter, when set, decides for it
    #[test]
    fn a_hook_takes_the_lines_its_filter_passes() {
        let l = logger(None);
        let (seen, hook) = recording_hook();
        l.keep(
            &mut l.hook.lock(),
            Some((LevelFilter::Info, Arc::from(hook))),
        );
        assert!(enabled(&l, Level::Info, "reims_vgpu::graphics"));
        assert!(!enabled(&l, Level::Debug, "reims_vgpu::graphics"));
        write(&l, Level::Warn, "reims_vgpu::graphics", "kept");
        write(&l, Level::Debug, "reims_vgpu::graphics", "dropped");
        assert_eq!(
            *seen.lock(),
            [(
                Level::Warn,
                "reims_vgpu::graphics".to_owned(),
                "kept".to_owned()
            )]
        );

        let l = logger(Some("error,reims_vgpu::tasks=debug"));
        let (seen, hook) = recording_hook();
        l.keep(
            &mut l.hook.lock(),
            Some((LevelFilter::Trace, Arc::from(hook))),
        );
        assert_eq!(l.max_level(Some(LevelFilter::Trace)), LevelFilter::Debug);
        write(&l, Level::Warn, "reims_vgpu::graphics", "dropped");
        write(&l, Level::Debug, "reims_vgpu::tasks", "kept");
        assert_eq!(
            *seen.lock(),
            [(
                Level::Debug,
                "reims_vgpu::tasks".to_owned(),
                "kept".to_owned()
            )]
        );
    }

    /// A line not written takes no lock: `enabled`, and `log` of a line the filter drops, return
    /// while another thread holds the hook's lock, under the hook's level and under the
    /// variable's filter alike. Every encoder call of the Vulkan platform asks `enabled` for its
    /// debug line, from every walk thread at once.
    #[test]
    fn a_line_not_written_takes_no_lock() {
        for env in [None, Some("info,reims_vgpu_graphics=trace")] {
            let l = logger(env);
            let (seen, hook) = recording_hook();
            l.keep(
                &mut l.hook.lock(),
                Some((LevelFilter::Info, Arc::from(hook))),
            );
            let held = l.hook.lock();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::scope(|s| {
                s.spawn(|| {
                    let asked = enabled(&l, Level::Debug, "reims_vgpu_vulkan::gpu::timing");
                    write(
                        &l,
                        Level::Debug,
                        "reims_vgpu_vulkan::gpu::timing",
                        "dropped",
                    );
                    let _ = tx.send(asked);
                });
                let answer = rx.recv_timeout(std::time::Duration::from_secs(10));
                // Let go before the asserts, so a thread that waited for the lock can end.
                drop(held);
                assert_eq!(
                    answer,
                    Ok(false),
                    "{env:?}: the filtered line waited for the lock"
                );
            });
            assert!(seen.lock().is_empty(), "{env:?}");
            // A line the filter passes still reaches the hook.
            write(&l, Level::Info, "reims_vgpu_graphics", "kept");
            assert_eq!(seen.lock().len(), 1, "{env:?}");
        }
    }

    /// The marker that makes a test the child of `the_process_logger_*`.
    const CHILD: &str = "REIMS_VGPU_LOGGING_CHILD";

    /// Re-runs this binary on the test `name`, as a child, with `REIMS_VGPU_LOG` set to `env`.
    fn child(name: &str, env: Option<&str>) -> std::process::Output {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", name, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .env_remove(ENV_VAR);
        if let Some(env) = env {
            cmd.env(ENV_VAR, env);
        }
        cmd.output().unwrap()
    }

    /// With the variable set, init installs the stderr logger; a hook then takes the lines in
    /// its place, and stderr has them back once it is cleared
    #[test]
    fn the_process_logger_writes_stderr_until_a_hook_takes_over() {
        if std::env::var_os(CHILD).is_some() {
            init();
            init();
            assert_eq!(log::max_level(), LevelFilter::Debug);
            log::debug!(target: "reims_vgpu::graphics", "to stderr {}", 1);
            log::debug!(target: "reims_vgpu::device", "filtered");
            log::info!(target: "reims_vgpu::device", "to stderr {}", 2);
            let (seen, hook) = recording_hook();
            assert!(set_hook(LevelFilter::Error, hook));
            log::info!(target: "reims_vgpu::device", "to the hook");
            clear_hook();
            log::error!(target: "reims_vgpu::tasks", "to stderr {}", 3);
            assert_eq!(
                *seen.lock(),
                [(
                    Level::Info,
                    "reims_vgpu::device".to_owned(),
                    "to the hook".to_owned()
                )]
            );
            return;
        }
        let out = child(
            "core::logging::tests::the_process_logger_writes_stderr_until_a_hook_takes_over",
            Some("info,reims_vgpu::graphics=debug"),
        );
        assert!(out.status.success(), "{out:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        let lines: Vec<&str> = err
            .lines()
            .filter(|l| l.starts_with("reims-vgpu "))
            .collect();
        assert_eq!(
            lines,
            [
                "reims-vgpu DEBUG reims_vgpu::graphics: to stderr 1",
                "reims-vgpu INFO reims_vgpu::device: to stderr 2",
                "reims-vgpu ERROR reims_vgpu::tasks: to stderr 3",
            ],
            "{err}"
        );
    }

    /// Without the variable, init installs nothing and the process stays silent; a hook still
    /// gets the lines up to its own level
    #[test]
    fn the_process_logger_is_silent_without_the_variable() {
        if std::env::var_os(CHILD).is_some() {
            init();
            assert_eq!(log::max_level(), LevelFilter::Off);
            log::error!(target: "reims_vgpu::device", "nowhere");
            let (seen, hook) = recording_hook();
            assert!(set_hook(LevelFilter::Warn, hook));
            assert_eq!(log::max_level(), LevelFilter::Warn);
            log::warn!(target: "reims_vgpu::device", "hooked");
            log::info!(target: "reims_vgpu::device", "too fine");
            clear_hook();
            assert_eq!(log::max_level(), LevelFilter::Off);
            log::error!(target: "reims_vgpu::device", "nowhere again");
            assert_eq!(
                *seen.lock(),
                [(
                    Level::Warn,
                    "reims_vgpu::device".to_owned(),
                    "hooked".to_owned()
                )]
            );
            return;
        }
        let out = child(
            "core::logging::tests::the_process_logger_is_silent_without_the_variable",
            None,
        );
        assert!(out.status.success(), "{out:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!err.contains("reims-vgpu "), "{err}");
    }

    /// A logger the host installed first keeps the lines: init leaves it, and set_hook refuses
    #[test]
    fn the_process_logger_yields_to_the_hosts() {
        struct Host;
        impl Log for Host {
            fn enabled(&self, _: &Metadata<'_>) -> bool {
                true
            }
            fn log(&self, record: &Record<'_>) {
                eprintln!("host {}", record.args());
            }
            fn flush(&self) {}
        }
        static HOST: Host = Host;

        if std::env::var_os(CHILD).is_some() {
            log::set_logger(&HOST).unwrap();
            log::set_max_level(LevelFilter::Info);
            init();
            let (_, hook) = recording_hook();
            assert!(!set_hook(LevelFilter::Trace, hook));
            clear_hook();
            assert_eq!(log::max_level(), LevelFilter::Info);
            log::warn!(target: "reims_vgpu::device", "to the host");
            return;
        }
        let out = child(
            "core::logging::tests::the_process_logger_yields_to_the_hosts",
            Some("trace"),
        );
        assert!(out.status.success(), "{out:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("host to the host"), "{err}");
        assert!(!err.contains("reims-vgpu "), "{err}");
    }
}
