//! Unix sockets on Windows (AF_UNIX, in Winsock since Windows 10 1803), which QEMU's `-qmp unix:`
//! listens on there too: the connection QMP takes, as std's `UnixStream` is on macOS and Linux,
//! and, for tests, a listener.

use std::io;
use std::os::windows::io::{AsRawSocket, FromRawSocket, OwnedSocket};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use windows_sys::Win32::Networking::WinSock as ws;

/// Starts Winsock for this process, once; it stays started until the process ends.
fn start() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        // SAFETY: an all-zero WSADATA is valid, and WSAStartup fills it.
        unsafe {
            let mut data: ws::WSADATA = std::mem::zeroed();
            ws::WSAStartup(0x0202, &mut data);
        }
    });
}

fn last_error() -> io::Error {
    // SAFETY: WSAGetLastError has no preconditions.
    io::Error::from_raw_os_error(unsafe { ws::WSAGetLastError() })
}

/// `path` as a Unix socket's address: its UTF-8, NUL-terminated, in `sun_path`'s 108 bytes.
fn address(path: &Path) -> io::Result<ws::SOCKADDR_UN> {
    let bytes = path
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "a socket's path is UTF-8"))?
        .as_bytes();
    let mut a = ws::SOCKADDR_UN {
        sun_family: ws::AF_UNIX,
        sun_path: [0; 108],
    };
    if bytes.len() >= a.sun_path.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a socket's path is shorter than 108 bytes",
        ));
    }
    for (to, from) in a.sun_path.iter_mut().zip(bytes) {
        *to = *from as i8;
    }
    Ok(a)
}

/// A new Unix stream socket.
fn socket() -> io::Result<OwnedSocket> {
    start();
    // SAFETY: socket has no preconditions; INVALID_SOCKET is its failure.
    let s = unsafe { ws::socket(i32::from(ws::AF_UNIX), ws::SOCK_STREAM, 0) };
    if s == ws::INVALID_SOCKET {
        return Err(last_error());
    }
    // SAFETY: s is a new socket that nothing else owns.
    Ok(unsafe { OwnedSocket::from_raw_socket(s as _) })
}

fn raw(s: &OwnedSocket) -> ws::SOCKET {
    s.as_raw_socket() as ws::SOCKET
}

/// A connection to a Unix socket. Its clones share the socket, which closes when the last drops.
pub struct UnixStream(Arc<OwnedSocket>);

impl UnixStream {
    pub fn connect(path: &Path) -> io::Result<UnixStream> {
        let a = address(path)?;
        let s = socket()?;
        // SAFETY: a is a SOCKADDR_UN of the size given, alive for the call.
        let r = unsafe {
            ws::connect(
                raw(&s),
                (&a as *const ws::SOCKADDR_UN).cast(),
                std::mem::size_of::<ws::SOCKADDR_UN>() as i32,
            )
        };
        if r != 0 {
            return Err(last_error());
        }
        Ok(UnixStream(Arc::new(s)))
    }

    /// Sets the socket's option `opt`, a timeout, to `t`: none where None, and, as std refuses
    /// it on Unix, not 0.
    fn timeout(&self, opt: i32, t: Option<Duration>) -> io::Result<()> {
        let ms: u32 = match t {
            None => 0,
            Some(d) if d.is_zero() => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "cannot set a 0 duration timeout",
                ));
            }
            Some(d) => d.as_millis().clamp(1, u128::from(u32::MAX)) as u32,
        };
        // SAFETY: the option's value is a u32 of milliseconds, alive for the call.
        let r = unsafe {
            ws::setsockopt(
                raw(&self.0),
                ws::SOL_SOCKET,
                opt,
                (&ms as *const u32).cast(),
                4,
            )
        };
        if r != 0 { Err(last_error()) } else { Ok(()) }
    }

    pub fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.timeout(ws::SO_RCVTIMEO, t)
    }

    pub fn set_write_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.timeout(ws::SO_SNDTIMEO, t)
    }

    pub fn try_clone(&self) -> io::Result<UnixStream> {
        Ok(UnixStream(self.0.clone()))
    }
}

impl io::Read for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let len = buf.len().min(i32::MAX as usize) as i32;
        // SAFETY: buf holds `len` bytes or more.
        let n = unsafe { ws::recv(raw(&self.0), buf.as_mut_ptr(), len, 0) };
        if n < 0 {
            Err(last_error())
        } else {
            Ok(n as usize)
        }
    }
}

impl io::Write for UnixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len().min(i32::MAX as usize) as i32;
        // SAFETY: buf holds `len` bytes or more.
        let n = unsafe { ws::send(raw(&self.0), buf.as_ptr(), len, 0) };
        if n < 0 {
            Err(last_error())
        } else {
            Ok(n as usize)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A Unix socket listening at a path, as std's `UnixListener` on macOS and Linux, for tests.
#[cfg(test)]
pub struct UnixListener(OwnedSocket);

#[cfg(test)]
impl UnixListener {
    pub fn bind(path: impl AsRef<Path>) -> io::Result<UnixListener> {
        let a = address(path.as_ref())?;
        let s = socket()?;
        // SAFETY: a is a SOCKADDR_UN of the size given, alive for the call.
        let bound = unsafe {
            ws::bind(
                raw(&s),
                (&a as *const ws::SOCKADDR_UN).cast(),
                std::mem::size_of::<ws::SOCKADDR_UN>() as i32,
            )
        };
        // SAFETY: listen on the socket this function holds.
        if bound != 0 || unsafe { ws::listen(raw(&s), ws::SOMAXCONN as i32) } != 0 {
            return Err(last_error());
        }
        Ok(UnixListener(s))
    }

    pub fn accept(&self) -> io::Result<UnixStream> {
        // SAFETY: no address is asked for.
        let s = unsafe { ws::accept(raw(&self.0), std::ptr::null_mut(), std::ptr::null_mut()) };
        if s == ws::INVALID_SOCKET {
            return Err(last_error());
        }
        // SAFETY: s is a new socket that nothing else owns.
        Ok(UnixStream(Arc::new(unsafe {
            OwnedSocket::from_raw_socket(s as _)
        })))
    }

    pub fn incoming(&self) -> impl Iterator<Item = io::Result<UnixStream>> + '_ {
        std::iter::repeat_with(move || self.accept())
    }
}
