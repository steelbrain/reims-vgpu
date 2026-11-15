//! A local HTTP/1.1 server for the tests of what the tool fetches (http.rs, catalog.rs,
//! download.rs): each request is answered by a handler, one connection per request, and the
//! requests are recorded.

use std::io::{Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

/// An answer: its status, headers and body, and where the body is cut off by closing the
/// connection (a download that drops part way).
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub cut: Option<usize>,
}

impl From<(u16, Vec<(String, String)>, Vec<u8>)> for Reply {
    fn from((status, headers, body): (u16, Vec<(String, String)>, Vec<u8>)) -> Reply {
        Reply {
            status,
            headers,
            body,
            cut: None,
        }
    }
}

/// A request as the server saw it: its path, and its headers with lowercased names.
#[derive(Clone, Debug)]
pub struct Seen {
    pub path: String,
    pub headers: Vec<(String, String)>,
}

impl Seen {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

type Handler = dyn Fn(&str, &Seen) -> Reply + Send + Sync;

pub struct Server {
    port: u16,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Server {
    /// Starts a server answering each request with `handler(path, request)`; it runs until the
    /// test process ends.
    pub fn start<R: Into<Reply>>(
        handler: impl Fn(&str, &Seen) -> R + Send + Sync + 'static,
    ) -> Server {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let handler: Arc<Handler> = Arc::new(move |p: &str, s: &Seen| handler(p, s).into());
        let log = seen.clone();
        std::thread::spawn(move || {
            for s in l.incoming() {
                let Ok(s) = s else { continue };
                let (handler, log) = (handler.clone(), log.clone());
                std::thread::spawn(move || serve(s, &*handler, &log));
            }
        });
        Server { port, seen }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    /// The paths requested, in order.
    pub fn paths(&self) -> Vec<String> {
        self.requests().into_iter().map(|s| s.path).collect()
    }

    pub fn requests(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

fn serve(mut s: TcpStream, handler: &Handler, log: &Mutex<Vec<Seen>>) {
    let mut req = Vec::new();
    let mut b = [0u8; 4096];
    let end = loop {
        if let Some(i) = req.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        match s.read(&mut b) {
            Ok(0) | Err(_) => return,
            Ok(n) => req.extend_from_slice(&b[..n]),
        }
    };
    let head = String::from_utf8_lossy(&req[..end]).to_string();
    let mut lines = head.split("\r\n");
    let path = lines
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .unwrap_or_default()
        .to_owned();
    let headers = lines
        .take_while(|l| !l.is_empty())
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
        .collect();
    let seen = Seen { path, headers };
    log.lock().unwrap().push(seen.clone());
    let r = handler(&seen.path, &seen);
    let mut out = format!("HTTP/1.1 {} X\r\n", r.status);
    for (k, v) in &r.headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str(&format!(
        "Content-Length: {}\r\nConnection: close\r\n\r\n",
        r.body.len()
    ));
    let _ = s.write_all(out.as_bytes());
    let body = &r.body[..r.cut.unwrap_or(r.body.len()).min(r.body.len())];
    let _ = s.write_all(body);
    let _ = s.flush();
}
