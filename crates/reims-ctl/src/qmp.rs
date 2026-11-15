//! A QMP client: one command at a time over the VM's Unix socket, events skipped.

#[cfg(windows)]
use crate::os::UnixStream;
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

pub struct Qmp {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
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
        if self.reader.read_line(&mut line)? == 0 {
            bail!("QMP closed the connection");
        }
        Ok(serde_json::from_str(&line)?)
    }

    pub fn execute(&mut self, command: &str, arguments: Option<Value>) -> Result<Value> {
        let mut msg = json!({ "execute": command });
        if let Some(a) = arguments {
            msg["arguments"] = a;
        }
        let mut bytes = serde_json::to_vec(&msg)?;
        bytes.push(b'\n');
        self.writer.write_all(&bytes)?;
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
