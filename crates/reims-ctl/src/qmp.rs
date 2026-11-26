//! A QMP client: one command at a time over the VM's Unix socket, events skipped. On Windows the
//! socket is Winsock's AF_UNIX ([`crate::os::UnixStream`]).
//!
//! QEMU answers one client at a time on a `-qmp` socket: while one is connected, its socket
//! chardev accepts no other (`tcp_chr_new_client` in QEMU's `chardev/char-socket.c` takes the
//! listener's handler away until that client leaves), so a second client's connection waits in
//! the socket's queue and hears nothing, not even QEMU's greeting, and its read ends at its
//! timeout. Read from QEMU's code.

use crate::os::UnixStream;
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub struct Qmp {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    socket: PathBuf,
    timeout: Duration,
}

impl Qmp {
    pub fn connect(socket: &Path, timeout: Duration) -> Result<Qmp> {
        let s = UnixStream::connect(socket)
            .with_context(|| format!("connect to {}", socket.display()))?;
        s.set_read_timeout(Some(timeout))?;
        s.set_write_timeout(Some(timeout))?;
        let mut q = Qmp {
            reader: BufReader::new(s.try_clone()?),
            writer: s,
            socket: socket.to_path_buf(),
            timeout,
        };
        let greeting = q.read()?;
        if greeting.get("QMP").is_none() {
            bail!("{} is not a QMP socket", socket.display());
        }
        q.execute("qmp_capabilities", None)?;
        Ok(q)
    }

    fn read(&mut self) -> Result<Value> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => bail!("QMP closed the connection"),
            Ok(_) => Ok(serde_json::from_str(&line)?),
            Err(e) => Err(self.failed(e)),
        }
    }

    /// What `e`, an error reading or writing the socket, says: where the socket's timeout ended
    /// the read or write (EAGAIN on Linux and macOS, WSAETIMEDOUT on Windows), that QEMU did not
    /// answer, with why that can be (the module's doc); otherwise `e`, on the socket.
    fn failed(&self, e: std::io::Error) -> anyhow::Error {
        let socket = self.socket.display();
        match e.kind() {
            ErrorKind::WouldBlock | ErrorKind::TimedOut => anyhow!(
                "QEMU did not answer on {socket} in {} s: QEMU answers one QMP client at a time, \
                 and no other while a program stays connected to that socket",
                self.timeout.as_secs_f64()
            ),
            _ => anyhow::Error::new(e).context(format!("QMP on {socket}")),
        }
    }

    pub fn execute(&mut self, command: &str, arguments: Option<Value>) -> Result<Value> {
        let mut msg = json!({ "execute": command });
        if let Some(a) = arguments {
            msg["arguments"] = a;
        }
        let mut bytes = serde_json::to_vec(&msg)?;
        bytes.push(b'\n');
        if let Err(e) = self.writer.write_all(&bytes) {
            return Err(self.failed(e));
        }
        loop {
            let v = self.read()?;
            if let Some(r) = v.get("return") {
                return Ok(r.clone());
            }
            if let Some(e) = v.get("error") {
                let desc = e.get("desc").and_then(Value::as_str).unwrap_or("error");
                bail!("QMP {command}: {desc}");
            }
        }
    }
}

/// Runs one command on the VM's socket.
pub fn run(socket: &Path, command: &str, arguments: Option<Value>) -> Result<Value> {
    Qmp::connect(socket, Duration::from_secs(10))?.execute(command, arguments)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A QMP socket at `sock` served as QEMU serves one: a client at a time, greeted and each of
    /// its commands answered, and the next accepted only once that one has left (the module's
    /// doc).
    fn one_client_qmp(sock: &Path) {
        let listener = crate::os::UnixListener::bind(sock).unwrap();
        std::thread::spawn(move || {
            for mut conn in listener.incoming().map_while(Result::ok) {
                let _ = conn.write_all(b"{\"QMP\": {}}\n");
                let reader = BufReader::new(conn.try_clone().unwrap());
                for _ in reader.lines().map_while(Result::ok) {
                    let _ = conn.write_all(b"{\"return\": {}}\n");
                }
            }
        });
    }

    /// While a client is connected to a QEMU's QMP socket, another hears nothing and says so at
    /// its timeout, naming the socket and why, where the OS says only EAGAIN ("Resource
    /// temporarily unavailable (os error 11)" on Linux); the connected client is answered
    /// throughout, and once it leaves, a new one is.
    #[test]
    fn a_client_qemu_does_not_answer_says_why_at_its_timeout() {
        let env = crate::testenv::Env::new("qmp-held");
        let sock = env.root.join("qmp.sock");
        one_client_qmp(&sock);
        let mut first = Qmp::connect(&sock, Duration::from_secs(5)).unwrap();
        let second = Qmp::connect(&sock, Duration::from_millis(200))
            .err()
            .map(|e| format!("{e:#}"));
        let answered = first.execute("query-status", None).map(|_| ());
        drop(first);
        let after = Qmp::connect(&sock, Duration::from_secs(5)).map(|_| ());
        assert_eq!(
            second,
            Some(format!(
                "QEMU did not answer on {} in 0.2 s: QEMU answers one QMP client at a time, and \
                 no other while a program stays connected to that socket",
                sock.display()
            ))
        );
        answered.unwrap();
        after.unwrap();
    }

    /// The QEMU that `REIMS_CTL_TEST_QEMU` names answers this client on its `-qmp unix:` socket,
    /// as a start asks it for one (qemu.rs): a check of a host's QEMU, which only that host has.
    /// On Windows its directory holds the DLLs it was linked with, or PATH does.
    #[test]
    #[ignore = "needs a QEMU: REIMS_CTL_TEST_QEMU names its qemu-system-x86_64"]
    fn a_qemu_answers_on_its_unix_socket() {
        let qemu = std::env::var_os("REIMS_CTL_TEST_QEMU").expect("REIMS_CTL_TEST_QEMU");
        let env = crate::testenv::Env::new("qmp-qemu");
        let sock = env.root.join("qmp.sock");
        let mut child = std::process::Command::new(qemu)
            .args([
                "-machine",
                "none",
                "-nodefaults",
                "-display",
                "none",
                "-qmp",
            ])
            .arg(format!("unix:{},server=on,wait=off", sock.display()))
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let answer = loop {
            match Qmp::connect(&sock, Duration::from_secs(2)) {
                Ok(mut q) => break q.execute("query-version", None),
                Err(_) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => break Err(e),
            }
        };
        let _ = child.kill();
        let _ = child.wait();
        let v = answer.unwrap();
        assert!(v["qemu"]["major"].as_u64().is_some(), "{v}");
    }
}
