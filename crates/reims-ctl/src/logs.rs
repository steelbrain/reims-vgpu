//! Bug reports: a zip of a VM's logs, its settings, the exact QEMU command and what the host
//! is, in the downloads folder, for someone to attach to a report.

use crate::host::{Host, Os};
use crate::runtime;
use crate::vm::Vm;
use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The most of one log a report takes: its end.
const LOG_CAP: u64 = 64 << 20;

pub fn bundle(vm: &Vm, host: Host) -> Result<PathBuf> {
    let stamp = runtime::now_stamp("%Y%m%d-%H%M%S");
    let out = crate::paths::downloads_dir().join(format!("reims-ctl-{}-{stamp}.zip", vm.name()));
    let mut zip = Zip::create(&out)?;
    let prefix = format!("reims-ctl-{}-{stamp}", vm.name());
    zip.add(
        &format!("{prefix}/system.txt"),
        system_info(host, &runtime::warnings(vm, host)).as_bytes(),
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

/// What the host is: the parts a report about a guest's graphics needs.
/// What the host is, and `warnings`, the ones reims-ctl gives the VM as the report is made: what
/// its user was shown.
pub fn system_info(host: Host, warnings: &[String]) -> String {
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
            section(
                "vulkaninfo --summary",
                run(
                    "sh",
                    &[
                        "-c",
                        "command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>&1 | head -80",
                    ],
                ),
            );
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
        Os::Other => {}
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
