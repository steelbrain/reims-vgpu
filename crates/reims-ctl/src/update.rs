//! Self-update: the checkout reims-ctl was installed from pulls, and scripts/install-ctl.sh
//! builds and installs the new reims-ctl and QEMU. The new QEMU goes beside the old one, so
//! running VMs keep theirs; the new reims-ctl replaces this binary, so restart it afterwards.
//!
//! The checkout's branch is the channel an update follows: master is stable, and development
//! gets changes first. Choosing a channel checks its branch out before the script runs, so the
//! script that builds is that branch's own, and later updates stay on it.

use crate::install::Install;
use crate::paths;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The branches an update can follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Channel {
    /// The master branch
    Stable,
    /// The development branch, which gets changes before stable
    Development,
}

impl Channel {
    pub const ALL: [Channel; 2] = [Channel::Stable, Channel::Development];

    pub fn name(self) -> &'static str {
        match self {
            Channel::Stable => "stable",
            Channel::Development => "development",
        }
    }

    pub fn branch(self) -> &'static str {
        match self {
            Channel::Stable => "master",
            Channel::Development => "development",
        }
    }

    /// The channel that follows `branch`, if one does.
    pub fn of(branch: &str) -> Option<Channel> {
        Channel::ALL.into_iter().find(|c| c.branch() == branch)
    }
}

/// The branch checked out in `repo`; None on a detached HEAD.
pub fn current_branch(repo: &Path) -> Option<String> {
    git_output(repo, &["symbolic-ref", "--short", "-q", "HEAD"])
}

/// What an update of `repo` follows, for people: "the stable channel (master)", "the branch
/// NAME", or "a detached HEAD".
pub fn following(repo: &Path) -> String {
    match current_branch(repo) {
        Some(b) => match Channel::of(&b) {
            Some(c) => format!("the {} channel ({b})", c.name()),
            None => format!("the branch {b}"),
        },
        None => "a detached HEAD".into(),
    }
}

/// Updates from the checkout, first switching it to `channel`'s branch where one is given.
pub fn update(channel: Option<Channel>) -> Result<()> {
    let i = Install::load().context(
        "no install record: run scripts/install-ctl.sh in your reims-vgpu checkout once, and \
         reims-ctl update can use it from then on",
    )?;
    if let Some(c) = channel {
        switch(&i.repo, c)?;
    }
    let script = i.repo.join("scripts/install-ctl.sh");
    if !script.is_file() {
        bail!("{} is missing: was the checkout moved?", script.display());
    }
    let mut bash = Command::new("bash");
    for v in GIT_REPOSITORY_ENV {
        bash.env_remove(v);
    }
    let status = bash
        .arg(&script)
        .arg("--update")
        .current_dir(&i.repo)
        .status()
        .context("run bash")?;
    if !status.success() {
        bail!("the update failed (exit {})", status.code().unwrap_or(-1));
    }
    Ok(())
}

/// Checks out `c`'s branch in `repo`, fetched from the remote the current branch tracks (origin
/// where it tracks none). A branch the checkout lacks is made tracking the remote's, and one
/// that tracks nothing is set to track it, so the script's pull moves it on. git tracks only a
/// branch the remote's fetch refspecs cover, which a single-branch clone's do not: the branch is
/// added to them there. The committed Cargo.lock goes back first: the build writes it again, and
/// git refuses to switch over a changed file the branches differ in.
fn switch(repo: &Path, c: Channel) -> Result<()> {
    let branch = c.branch();
    if !repo.join(".git").exists() {
        bail!("{} is not a git checkout: was it moved?", repo.display());
    }
    let current = current_branch(repo);
    if current.as_deref() == Some(branch) {
        return Ok(());
    }
    let remote = current
        .and_then(|b| git_output(repo, &["config", &format!("branch.{b}.remote")]))
        .filter(|r| r != ".")
        .unwrap_or_else(|| "origin".into());
    let upstream = format!("{remote}/{branch}");
    println!(
        "Switching {} to the {} channel ({branch})",
        paths::tilde(repo),
        c.name()
    );
    let src = format!("refs/heads/{branch}");
    let covered = git_output(
        repo,
        &["config", "--get-all", &format!("remote.{remote}.fetch")],
    )
    .unwrap_or_default()
    .lines()
    .filter_map(|r| r.trim_start_matches('+').split_once(':'))
    .any(|(from, _)| from == "refs/heads/*" || from == src);
    if !covered {
        git(repo, &["remote", "set-branches", "--add", &remote, branch])?;
    }
    git(
        repo,
        &["fetch", &remote, &format!("+{src}:refs/remotes/{upstream}")],
    )
    .with_context(|| format!("fetch {branch} from {remote}"))?;
    git(repo, &["checkout", "-q", "HEAD", "--", "Cargo.lock"])?;
    let have = git_output(repo, &["rev-parse", "--verify", "-q", &src]);
    if have.is_none() {
        git(repo, &["switch", "--track", "-c", branch, &upstream])
    } else {
        git(repo, &["switch", branch])
    }
    .with_context(|| {
        format!(
            "switch {} to {branch}: commit or set aside its local changes, then update again",
            paths::tilde(repo)
        )
    })?;
    if git_output(repo, &["config", &format!("branch.{branch}.remote")]).is_none() {
        git(repo, &["branch", "--set-upstream-to", &upstream, branch])?;
    }
    Ok(())
}

/// The variables through which git takes its repository from the environment, which a git hook
/// and `git rebase -x` export to what they run. `git -C` does not override them, so each git and
/// script reims-ctl runs on a checkout drops them, as it names the checkout itself.
pub(crate) const GIT_REPOSITORY_ENV: [&str; 6] = [
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
];

/// git on `repo` alone, whatever repository the environment names ([`GIT_REPOSITORY_ENV`]).
pub(crate) fn git_command(repo: &Path) -> Command {
    let mut c = Command::new("git");
    c.arg("-C").arg(repo);
    for v in GIT_REPOSITORY_ENV {
        c.env_remove(v);
    }
    c
}

fn git(repo: &Path, args: &[&str]) -> Result<()> {
    let status = git_command(repo).args(args).status().context("run git")?;
    if !status.success() {
        bail!("git {} failed", args.join(" "));
    }
    Ok(())
}

/// `git args`' output, trimmed; None where it fails or prints nothing.
fn git_output(repo: &Path, args: &[&str]) -> Option<String> {
    let o = git_command(repo).args(args).output().ok()?;
    let out = String::from_utf8_lossy(&o.stdout).trim().to_string();
    (o.status.success() && !out.is_empty()).then_some(out)
}

/// The binary to run after an update, from the path this process was started from
/// (`std::env::current_exe`). The update installs the new reims-ctl over that path (cargo install
/// --force writes it anew), so it names the new binary. On Linux the path is read from
/// /proc/self/exe, which names a file replaced since as "PATH (deleted)": PATH is the new one.
pub fn restart_path(exe: PathBuf) -> PathBuf {
    if exe.exists() {
        return exe;
    }
    match exe.to_str().and_then(|p| p.strip_suffix(" (deleted)")) {
        Some(p) => PathBuf::from(p),
        None => exe,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A binary replaced since this process started is run from its path again, which holds the
    /// new one; a path that is there, or names no replaced file, is run as it is.
    #[test]
    fn a_restart_runs_the_binary_the_update_installed_at_the_same_path() {
        let d = std::env::temp_dir().join(format!("reims-ctl-restart-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let bin = d.join("reims-ctl");
        std::fs::write(&bin, "new").unwrap();
        assert_eq!(restart_path(d.join("reims-ctl (deleted)")), bin);
        assert_eq!(restart_path(bin.clone()), bin);
        // A file of that name is the one to run.
        let named = d.join("odd (deleted)");
        std::fs::write(&named, "").unwrap();
        assert_eq!(restart_path(named.clone()), named);
        assert_eq!(restart_path(d.join("gone")), d.join("gone"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// What /proc/self/exe names for a running binary replaced as an update replaces it: the
    /// path with " (deleted)", which [`restart_path`] maps back to the path, now the new binary.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_names_a_replaced_running_binary_deleted() {
        let d = std::env::temp_dir().join(format!("reims-ctl-replaced-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let bin = d.join("sleeper");
        std::fs::copy("/bin/sleep", &bin).unwrap();
        let mut child = Command::new(&bin).arg("30").spawn().unwrap();
        // spawn returns once the child's exec lets go of this process's memory, which the kernel
        // does a moment before it names the new binary as the child's exe: wait for that name.
        let exe = format!("/proc/{}/exe", child.id());
        let start = std::time::Instant::now();
        while std::fs::read_link(&exe).ok().as_deref() != Some(bin.as_path()) {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(10),
                "the sleeper never ran"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // cargo install --force: the new binary written beside, then renamed over the old.
        let new = d.join("sleeper.new");
        std::fs::copy("/bin/sleep", &new).unwrap();
        std::fs::rename(&new, &bin).unwrap();
        let named = std::fs::read_link(format!("/proc/{}/exe", child.id())).unwrap();
        let _ = child.kill();
        let _ = child.wait();
        assert_eq!(named, d.join("sleeper (deleted)"));
        assert_eq!(restart_path(named), bin);
        let _ = std::fs::remove_dir_all(&d);
    }

    fn run(dir: &Path, args: &[&str]) {
        let ok = git_command(dir)
            .args(args)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .unwrap()
            .status
            .success();
        assert!(ok, "git {}", args.join(" "));
    }

    #[test]
    fn git_drops_every_variable_that_names_another_repository() {
        let c = git_command(Path::new("/checkout"));
        let dropped: Vec<_> = c
            .get_envs()
            .filter(|(_, v)| v.is_none())
            .map(|(k, _)| k)
            .collect();
        for v in GIT_REPOSITORY_ENV {
            assert!(dropped.contains(&std::ffi::OsStr::new(v)), "{v}");
        }
    }

    #[test]
    fn channels_follow_their_branches() {
        assert_eq!(Channel::of("master"), Some(Channel::Stable));
        assert_eq!(Channel::of("development"), Some(Channel::Development));
        assert_eq!(Channel::of("steelbrain/feature"), None);
    }

    #[test]
    fn switching_fetches_and_tracks_the_channels_branch() {
        let d = std::env::temp_dir().join(format!("reims-ctl-channel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let (origin, clone) = (d.join("origin"), d.join("clone"));
        std::fs::create_dir_all(&origin).unwrap();
        run(&origin, &["init", "-q", "-b", "master"]);
        run(&origin, &["config", "user.name", "t"]);
        run(&origin, &["config", "user.email", "t@t"]);
        // The branches' locks differ, as a pull of a lock change finds them.
        let lock = |dir: &Path, s: &str| std::fs::write(dir.join("Cargo.lock"), s).unwrap();
        lock(&origin, "stable");
        run(&origin, &["add", "Cargo.lock"]);
        run(&origin, &["commit", "-q", "-m", "stable"]);
        run(&origin, &["switch", "-q", "-c", "development"]);
        lock(&origin, "development");
        run(&origin, &["commit", "-q", "-a", "-m", "development"]);
        run(&origin, &["switch", "-q", "master"]);
        // A single-branch clone, whose fetches keep only master.
        run(
            &d,
            &[
                "clone",
                "-q",
                "--single-branch",
                "-b",
                "master",
                origin.to_str().unwrap(),
                "clone",
            ],
        );
        // The build rewrote the clone's lock.
        lock(&clone, "built");

        switch(&clone, Channel::Development).unwrap();
        assert_eq!(current_branch(&clone).as_deref(), Some("development"));
        assert_eq!(
            std::fs::read_to_string(clone.join("Cargo.lock")).unwrap(),
            "development"
        );
        assert_eq!(
            git_output(&clone, &["rev-parse", "--abbrev-ref", "@{u}"]).as_deref(),
            Some("origin/development")
        );
        assert_eq!(following(&clone), "the development channel (development)");

        switch(&clone, Channel::Stable).unwrap();
        assert_eq!(following(&clone), "the stable channel (master)");
        let _ = std::fs::remove_dir_all(&d);
    }
}
