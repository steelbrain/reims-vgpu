//! Bug reports: a zip of a VM's logs, its settings, the exact QEMU command, what the host is
//! and vulkaninfo's whole report of it, in the downloads folder, for someone to attach to a
//! report.

use crate::host::{Host, Os};
use crate::runtime;
use crate::vm::Vm;
use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

/// The most of one log a report takes: its end. The most of each of a tool's streams it takes:
/// their start.
const LOG_CAP: u64 = 64 << 20;

/// How long one run of vulkaninfo may take: a GPU driver that hangs can hold it forever, and
/// the report goes on without the rest of its output.
pub const VULKANINFO_LIMIT: Duration = Duration::from_secs(30);

pub fn bundle(vm: &Vm, host: Host) -> Result<PathBuf> {
    let stamp = runtime::now_stamp("%Y%m%d-%H%M%S");
    let out = crate::paths::downloads_dir().join(format!("reims-ctl-{}-{stamp}.zip", vm.name()));
    let mut zip = Zip::create(&out)?;
    let prefix = format!("reims-ctl-{}-{stamp}", vm.name());
    let vulkaninfo = run_limited(Command::new("vulkaninfo"), VULKANINFO_LIMIT);
    let system = system_info(host, &runtime::warnings(vm, host), &vulkaninfo.end);
    zip.add(&format!("{prefix}/system.txt"), system.as_bytes())?;
    zip.add(
        &format!("{prefix}/vulkaninfo.txt"),
        vulkaninfo.report("vulkaninfo").as_bytes(),
    )?;
    if let Ok(t) = std::fs::read(vm.dir.join(crate::vm::CONFIG_FILE)) {
        zip.add(&format!("{prefix}/vm.toml"), &t)?;
    }
    if let Ok(inv) = runtime::invocation_for_print(vm, host) {
        zip.add(
            &format!("{prefix}/launch.sh"),
            inv.shell_script("the QEMU command reims-ctl runs")
                .as_bytes(),
        )?;
    }
    let state = match runtime::pid(vm) {
        Some(p) => format!("running, pid {p}\n"),
        None => "stopped\n".to_string(),
    };
    zip.add(&format!("{prefix}/state.txt"), state.as_bytes())?;
    if let Ok(rd) = std::fs::read_dir(vm.logs_dir()) {
        let mut files: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        for f in files {
            let data = read_tail(&f, LOG_CAP)?;
            zip.add(
                &format!("{prefix}/logs/{}", f.file_name().unwrap().to_string_lossy()),
                &data,
            )?;
        }
    }
    zip.finish()?;
    Ok(out)
}

fn read_tail(p: &Path, cap: u64) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(p)?;
    let len = f.metadata()?.len();
    if len > cap {
        f.seek(SeekFrom::Start(len - cap))?;
    }
    let mut b = Vec::new();
    f.read_to_end(&mut b)?;
    Ok(b)
}

fn run(cmd: &str, args: &[&str]) -> String {
    match Command::new(cmd).args(args).output() {
        Ok(o) => {
            let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
            if !o.status.success() {
                s.push_str(&String::from_utf8_lossy(&o.stderr));
            }
            s.trim_end().to_string()
        }
        Err(e) => format!("({cmd}: {e})"),
    }
}

/// A tool's run under a time limit ([`run_limited`]): what it wrote to each stream, and how it
/// ended.
struct Limited {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    end: End,
}

/// How a run under a time limit ended.
#[derive(Debug)]
enum End {
    /// It exited, with this status.
    Exited(std::process::ExitStatus),
    /// It still ran at its limit, this long, and was killed.
    TimedOut(Duration),
    /// It did not start, or could not be waited for: `NotFound` from its start where the tool
    /// is not installed.
    Failed(std::io::Error),
}

impl End {
    /// One line that says how `tool`'s run ended.
    fn says(&self, tool: &str) -> String {
        match self {
            End::Exited(status) => format!("{tool} ended with {status}"),
            End::TimedOut(limit) => {
                format!(
                    "{tool} timed out after {} s and was stopped",
                    limit.as_secs()
                )
            }
            End::Failed(e) if e.kind() == std::io::ErrorKind::NotFound => {
                format!("{tool} is not installed on this host")
            }
            End::Failed(e) => format!("{tool} did not run: {e}"),
        }
    }
}

impl Limited {
    /// The run's file in a report, as `vulkaninfo.txt` holds vulkaninfo's: the line that says
    /// how it ended, then what it wrote to stdout and to stderr, each under its title, unless
    /// it failed with nothing written.
    fn report(&self, tool: &str) -> String {
        let mut s = format!("{}\n", self.end.says(tool));
        if matches!(self.end, End::Failed(_)) && self.stdout.is_empty() && self.stderr.is_empty() {
            return s;
        }
        for (title, out) in [("stdout", &self.stdout), ("stderr", &self.stderr)] {
            s.push_str(&format!("\n== {title}\n{}", String::from_utf8_lossy(out)));
        }
        s
    }
}

/// Runs `cmd`, with no input, for at most `limit`, keeping what it writes to stdout and to
/// stderr (the first [`LOG_CAP`] bytes of each). A run still going at its limit is killed, and
/// what it wrote until then is kept: a GPU driver that hangs can hold a tool forever, and the
/// report goes on without it. The killed process is waited for on a thread of its own, which
/// nothing joins, since a process stuck in a driver can outlive its kill; its streams' readers
/// get a second more.
fn run_limited(mut cmd: Command, limit: Duration) -> Limited {
    use std::process::Stdio;
    let child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            return Limited {
                stdout: Vec::new(),
                stderr: Vec::new(),
                end: End::Failed(e),
            };
        }
    };
    let (done, drained) = mpsc::channel();
    let stdout = drain(child.stdout.take(), done.clone());
    let stderr = drain(child.stderr.take(), done);
    let deadline = Instant::now() + limit;
    let end = loop {
        match child.try_wait() {
            Ok(Some(status)) => break End::Exited(status),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            waited => {
                let _ = child.kill();
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                break match waited {
                    Err(e) => End::Failed(e),
                    _ => End::TimedOut(limit),
                };
            }
        }
    };
    // Its streams end with it, unless a process it started holds them open: their readers get
    // a second more.
    let by = Instant::now() + Duration::from_secs(1);
    for _ in 0..2 {
        if drained
            .recv_timeout(by.saturating_duration_since(Instant::now()))
            .is_err()
        {
            break;
        }
    }
    let take =
        |b: &Arc<Mutex<Vec<u8>>>| std::mem::take(&mut *b.lock().unwrap_or_else(|e| e.into_inner()));
    Limited {
        stdout: take(&stdout),
        stderr: take(&stderr),
        end,
    }
}

/// Reads `pipe` to its end on a thread of its own, into the buffer it returns, which keeps its
/// first [`LOG_CAP`] bytes, and then tells `done`.
fn drain(
    pipe: Option<impl std::io::Read + Send + 'static>,
    done: mpsc::Sender<()>,
) -> Arc<Mutex<Vec<u8>>> {
    let buf = Arc::new(Mutex::new(Vec::new()));
    let Some(mut pipe) = pipe else {
        let _ = done.send(());
        return buf;
    };
    let kept = Arc::clone(&buf);
    std::thread::spawn(move || {
        let mut chunk = vec![0u8; 64 << 10];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    let mut b = kept.lock().unwrap_or_else(|e| e.into_inner());
                    let room = (LOG_CAP as usize).saturating_sub(b.len());
                    b.extend_from_slice(&chunk[..n.min(room)]);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        let _ = done.send(());
    });
    buf
}

/// system.txt's `vulkaninfo --summary`: the first 80 lines it writes, stdout then stderr, and
/// how it ended where it did not exit with success. It runs only where `whole`, the run of the
/// whole report, exited: where vulkaninfo is not installed, or that run timed out or failed,
/// the section says so instead, and holds the report up no longer.
fn vulkan_summary(whole: &End) -> String {
    match whole {
        End::Exited(_) => {}
        End::Failed(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return whole.says("vulkaninfo");
        }
        _ => return format!("not run: {} (vulkaninfo.txt)", whole.says("vulkaninfo")),
    }
    let mut cmd = Command::new("vulkaninfo");
    cmd.arg("--summary");
    let run = run_limited(cmd, VULKANINFO_LIMIT);
    let out = [run.stdout, run.stderr].concat();
    let mut s = String::from_utf8_lossy(&out)
        .lines()
        .take(80)
        .collect::<Vec<_>>()
        .join("\n");
    if !matches!(run.end, End::Exited(status) if status.success()) {
        s.push('\n');
        s.push_str(&run.end.says("vulkaninfo --summary"));
    }
    s
}

/// What the host is: the parts a report about a guest's graphics needs.
/// What the host is, and `warnings`, the ones reims-ctl gives the VM as the report is made: what
/// its user was shown. `vulkaninfo` is how the run of vulkaninfo's whole report ended
/// ([`vulkan_summary`]).
fn system_info(host: Host, warnings: &[String], vulkaninfo: &End) -> String {
    let mut s = String::new();
    let mut section = |title: &str, body: String| {
        s.push_str(&format!("== {title}\n{body}\n\n"));
    };
    section(
        "reims-ctl",
        format!("version {}\nhost {host}", env!("CARGO_PKG_VERSION")),
    );
    section("warnings", warnings_list(warnings));
    if let Some(i) = crate::install::Install::load() {
        section(
            "install",
            format!(
                "repo {}\ncommit {}\nqemu {}\ninstalled {}",
                i.repo.display(),
                i.commit,
                i.qemu.display(),
                i.installed_at
            ),
        );
    }
    if let Some(profile) = host.default_profile()
        && let Ok(q) = crate::install::qemu_binary(profile)
    {
        section(
            "qemu",
            format!(
                "{}\n{}\naudio backends: {}",
                q.display(),
                crate::install::qemu_version(&q).unwrap_or_default(),
                crate::install::audio_backends(&q).join(", ")
            ),
        );
    }
    section("uname", run("uname", &["-a"]));
    match host.os {
        Os::Mac => {
            section("sw_vers", run("sw_vers", &[]));
            section(
                "cpu",
                run(
                    "sysctl",
                    &["machdep.cpu.brand_string", "hw.ncpu", "hw.memsize"],
                ),
            );
            section("displays", run("system_profiler", &["SPDisplaysDataType"]));
        }
        Os::Linux => {
            section(
                "os-release",
                std::fs::read_to_string("/etc/os-release").unwrap_or_default(),
            );
            section(
                "cpu",
                run(
                    "sh",
                    &[
                        "-c",
                        "grep -m1 'model name' /proc/cpuinfo; echo \"cpus: $(nproc)\"; grep MemTotal /proc/meminfo",
                    ],
                ),
            );
            section(
                "pci display devices",
                run(
                    "sh",
                    &[
                        "-c",
                        "lspci -nnk 2>/dev/null | grep -A3 -iE 'vga|3d|display'",
                    ],
                ),
            );
            let gpus = crate::gpu::list()
                .iter()
                .map(|g| {
                    format!(
                        "{}  [{:04x}:{:04x}] {}",
                        g.label(),
                        g.vendor_id,
                        g.device_id,
                        g.driver_file.display()
                    )
                })
                .collect::<Vec<_>>();
            section("vulkan gpus", gpus.join("\n"));
            section("vulkaninfo --summary", vulkan_summary(vulkaninfo));
            section(
                "nvidia",
                run(
                    "sh",
                    &[
                        "-c",
                        "command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv",
                    ],
                ),
            );
            section(
                "kernel",
                run(
                    "sh",
                    &[
                        "-c",
                        "cat /proc/sys/vm/max_map_count; ls -l /dev/kvm /dev/udmabuf; \
                         grep -H . /sys/module/udmabuf/parameters/*; id; cat /proc/cmdline",
                    ],
                ),
            );
            section(
                "session",
                format!(
                    "XDG_SESSION_TYPE={}\nWAYLAND_DISPLAY={}\nDISPLAY={}",
                    std::env::var("XDG_SESSION_TYPE").unwrap_or_default(),
                    std::env::var("WAYLAND_DISPLAY").unwrap_or_default(),
                    std::env::var("DISPLAY").unwrap_or_default()
                ),
            );
        }
        Os::Windows | Os::Other => {}
    }
    s
}

/// One line a warning, or `none`.
fn warnings_list(warnings: &[String]) -> String {
    if warnings.is_empty() {
        return "none".into();
    }
    warnings
        .iter()
        .map(|w| format!("- {w}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A zip archive, written as it goes: deflated entries, no ZIP64 (each entry is capped well
/// below 4 GiB).
struct Zip {
    file: std::io::BufWriter<std::fs::File>,
    offset: u32,
    central: Vec<u8>,
    count: u16,
    dos_time: u16,
    dos_date: u16,
    path: PathBuf,
}

impl Zip {
    fn create(path: &Path) -> Result<Zip> {
        let file =
            std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
        let (dos_time, dos_date) = dos_now();
        Ok(Zip {
            file: std::io::BufWriter::new(file),
            offset: 0,
            central: Vec::new(),
            count: 0,
            dos_time,
            dos_date,
            path: path.to_path_buf(),
        })
    }

    fn add(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let crc = crc32fast::hash(data);
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(data)?;
        let comp = enc.finish()?;
        let name_b = name.as_bytes();
        let mut h = Vec::new();
        h.extend_from_slice(&0x04034b50u32.to_le_bytes());
        h.extend_from_slice(&20u16.to_le_bytes()); // version needed
        h.extend_from_slice(&0x0800u16.to_le_bytes()); // UTF-8 names
        h.extend_from_slice(&8u16.to_le_bytes()); // deflate
        h.extend_from_slice(&self.dos_time.to_le_bytes());
        h.extend_from_slice(&self.dos_date.to_le_bytes());
        h.extend_from_slice(&crc.to_le_bytes());
        h.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        h.extend_from_slice(&(data.len() as u32).to_le_bytes());
        h.extend_from_slice(&(name_b.len() as u16).to_le_bytes());
        h.extend_from_slice(&0u16.to_le_bytes());
        h.extend_from_slice(name_b);
        self.file.write_all(&h)?;
        self.file.write_all(&comp)?;

        let c = &mut self.central;
        c.extend_from_slice(&0x02014b50u32.to_le_bytes());
        c.extend_from_slice(&0x031eu16.to_le_bytes()); // made by: Unix, 3.0
        c.extend_from_slice(&20u16.to_le_bytes());
        c.extend_from_slice(&0x0800u16.to_le_bytes());
        c.extend_from_slice(&8u16.to_le_bytes());
        c.extend_from_slice(&self.dos_time.to_le_bytes());
        c.extend_from_slice(&self.dos_date.to_le_bytes());
        c.extend_from_slice(&crc.to_le_bytes());
        c.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        c.extend_from_slice(&(data.len() as u32).to_le_bytes());
        c.extend_from_slice(&(name_b.len() as u16).to_le_bytes());
        c.extend_from_slice(&[0; 8]); // extra, comment, disk, internal attributes
        c.extend_from_slice(&((0o100644u32) << 16).to_le_bytes()); // external: a regular file
        c.extend_from_slice(&self.offset.to_le_bytes());
        c.extend_from_slice(name_b);

        self.offset += (h.len() + comp.len()) as u32;
        self.count += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        let cd_offset = self.offset;
        self.file.write_all(&self.central)?;
        let mut e = Vec::new();
        e.extend_from_slice(&0x06054b50u32.to_le_bytes());
        e.extend_from_slice(&[0; 4]);
        e.extend_from_slice(&self.count.to_le_bytes());
        e.extend_from_slice(&self.count.to_le_bytes());
        e.extend_from_slice(&(self.central.len() as u32).to_le_bytes());
        e.extend_from_slice(&cd_offset.to_le_bytes());
        e.extend_from_slice(&0u16.to_le_bytes());
        self.file.write_all(&e)?;
        self.file
            .flush()
            .with_context(|| format!("write {}", self.path.display()))?;
        Ok(())
    }
}

fn dos_now() -> (u16, u16) {
    let t = crate::os::local_now();
    let time = ((t.hour as u16) << 11) | ((t.minute as u16) << 5) | ((t.second as u16) / 2);
    let date = (((t.year - 1980).max(0) as u16) << 9) | ((t.month as u16) << 5) | t.day as u16;
    (time, date)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warnings_list_one_a_line_or_none() {
        assert_eq!(warnings_list(&[]), "none");
        assert_eq!(warnings_list(&["one".into(), "two".into()]), "- one\n- two");
    }

    /// What the run tests run as a tool: this test binary, which, where `REIMS_CTL_TEST_TOOL` is
    /// `write`, writes more than a pipe holds to stdout and a line to stderr and exits with 3,
    /// and where it is `hang`, writes a line and sleeps past the limit the test gives it.
    #[test]
    fn a_tool() {
        let mut out = std::io::stdout();
        match std::env::var("REIMS_CTL_TEST_TOOL").as_deref() {
            Ok("write") => {
                for i in 0..20_000 {
                    writeln!(out, "line {i} of the tool's report").unwrap();
                }
                out.flush().unwrap();
                std::io::stderr().write_all(b"a warning\n").unwrap();
                std::process::exit(3);
            }
            Ok("hang") => {
                out.write_all(b"before the hang\n").unwrap();
                out.flush().unwrap();
                std::thread::sleep(Duration::from_secs(60));
            }
            _ => {}
        }
    }

    /// This test binary as a tool that does what `mode` names ([`a_tool`]).
    fn tool(mode: &str) -> Command {
        let mut c = Command::new(std::env::current_exe().unwrap());
        c.args(["--exact", "logs::tests::a_tool", "--nocapture"])
            .env("REIMS_CTL_TEST_TOOL", mode);
        c
    }

    /// A tool that exits is kept whole, past what a pipe holds, each of its streams under its
    /// title, below a first line that says how it exited.
    #[test]
    fn a_tool_that_exits_is_kept_whole_below_its_exit_status() {
        let run = run_limited(tool("write"), Duration::from_secs(60));
        assert!(
            matches!(run.end, End::Exited(s) if s.code() == Some(3)),
            "{:?}",
            run.end
        );
        let report = run.report("vulkaninfo");
        let (first, rest) = report.split_once('\n').unwrap();
        assert!(
            first.starts_with("vulkaninfo ended with ") && first.ends_with('3'),
            "{first}"
        );
        let (stdout, stderr) = rest.split_once("\n== stderr\n").unwrap();
        assert!(stdout.starts_with("\n== stdout\n"), "{stdout:.300}");
        assert!(stdout.contains("line 0 of the tool's report\n"));
        assert!(stdout.ends_with("\nline 19999 of the tool's report\n"));
        assert!(stderr.contains("a warning\n"), "{stderr}");
    }

    /// A tool still running at its limit is stopped there, and its file keeps what it wrote
    /// until then, below a first line that says it timed out; system.txt's summary does not run
    /// it again.
    #[test]
    fn a_tool_past_its_limit_is_stopped_and_what_it_wrote_kept() {
        let started = Instant::now();
        let run = run_limited(tool("hang"), Duration::from_secs(3));
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
        let report = run.report("vulkaninfo");
        assert!(
            report.starts_with("vulkaninfo timed out after 3 s and was stopped\n"),
            "{report}"
        );
        assert!(report.contains("before the hang\n"), "{report}");
        assert_eq!(
            vulkan_summary(&run.end),
            "not run: vulkaninfo timed out after 3 s and was stopped (vulkaninfo.txt)"
        );
    }

    /// A tool that is not installed is said to be, in one line, in its file and in system.txt's
    /// summary.
    #[test]
    fn a_tool_not_installed_says_so_in_one_line() {
        let run = run_limited(
            Command::new("reims-ctl-test-no-such-tool"),
            Duration::from_secs(10),
        );
        assert_eq!(
            run.report("vulkaninfo"),
            "vulkaninfo is not installed on this host\n"
        );
        assert_eq!(
            vulkan_summary(&run.end),
            "vulkaninfo is not installed on this host"
        );
    }

    #[test]
    fn zip_is_readable_by_unzip() {
        let dir = std::env::temp_dir().join(format!("reims-ctl-zip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.zip");
        let mut z = Zip::create(&p).unwrap();
        z.add("a/one.txt", b"hello hello hello").unwrap();
        z.add("a/two.txt", &vec![b'x'; 100_000]).unwrap();
        z.finish().unwrap();
        if let Ok(o) = Command::new("unzip").arg("-t").arg(&p).output() {
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stdout));
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
