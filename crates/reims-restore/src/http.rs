//! HTTPS to Apple's hosts for files: the client every such request goes through ([`Client`]),
//! and a file on Apple's CDN read at offsets by ranged requests ([`HttpRange`]): the parts of an
//! IPSW the tool needs without downloading it, such as AVPBooter from 14.6.1's system image
//! (avpbooter.rs). Apple's catalog of IPSWs (catalog.rs) and whole IPSWs (download.rs) come
//! through the same client.
//!
//! A request goes only to one of Apple's hosts ([`check_url`]), over HTTPS, with the fixed
//! User-Agent and no header naming the host, its user or this project. The client follows no
//! redirect by itself: [`Client::get`] follows each only to a URL [`check_url`] allows, so a
//! redirect can't take a request off Apple's hosts.

use std::collections::HashMap;
use std::io;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};

use crate::source::ReadAt;

/// The hosts files are read from: Apple's CDN for IPSWs, and mesu for Apple's catalog of them.
const HOSTS: &[&str] = &["updates.cdn-apple.com", "mesu.apple.com"];

/// The most redirects a request follows.
const REDIRECTS: usize = 5;

/// Reads go in aligned blocks of this size, and this many are kept.
const BLOCK: u64 = 1 << 20;
const CACHED: usize = 64;

/// A file on Apple's CDN, read by ranged GETs and kept in a cache of blocks.
pub struct HttpRange {
    client: Client,
    url: String,
    len: u64,
    cache: Mutex<Cache>,
}

#[derive(Default)]
struct Cache {
    blocks: HashMap<u64, (u64, Vec<u8>)>,
    tick: u64,
}

/// Checks that `url` is HTTPS on one of Apple's hosts for IPSWs and their catalog.
pub fn check_url(url: &str) -> Result<()> {
    let rest = url
        .strip_prefix("https://")
        .with_context(|| format!("{url} is not HTTPS"))?;
    let host = rest.split(['/', ':', '?', '#']).next().unwrap_or_default();
    ensure!(
        HOSTS.contains(&host),
        "{url} is not on Apple's hosts for IPSWs ({})",
        HOSTS.join(", ")
    );
    Ok(())
}

/// The client requests for files go through: on Apple's hosts, or in tests a local server.
pub struct Client {
    agent: ureq::Agent,
    allow: fn(&str) -> Result<()>,
}

impl Client {
    /// For Apple's hosts over HTTPS ([`check_url`]). A request ends when its connection, its
    /// answer's head or any read of its body stalls for a minute, however long the body is.
    pub fn apple() -> Client {
        Client::with(check_url, true)
    }

    /// For a server on this host, over HTTP: the tests'.
    #[cfg(test)]
    pub fn local() -> Client {
        fn allow(url: &str) -> Result<()> {
            ensure!(url.starts_with("http://127.0.0.1:"), "{url} is not local");
            Ok(())
        }
        Client::with(allow, false)
    }

    fn with(allow: fn(&str) -> Result<()>, https_only: bool) -> Client {
        let minute = Some(Duration::from_secs(60));
        let agent = ureq::Agent::config_builder()
            .user_agent(crate::USER_AGENT)
            .https_only(https_only)
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_connect(minute)
            .timeout_recv_response(minute)
            .timeout_per_read(minute)
            .build()
            .into();
        Client { agent, allow }
    }

    /// GETs `url`, with a `Range` header where `range` is given (`bytes=0-99`), following up to
    /// [`REDIRECTS`] redirects, each only to a URL this client allows. Answers with any status
    /// but a redirect; the URL that answered, and the answer.
    pub fn get(
        &self,
        url: &str,
        range: Option<&str>,
    ) -> Result<(String, ureq::http::Response<ureq::Body>)> {
        let mut url = url.to_owned();
        for _ in 0..=REDIRECTS {
            (self.allow)(&url)?;
            let mut req = self.agent.get(&url);
            if let Some(r) = range {
                req = req.header("Range", r);
            }
            let resp = req.call().with_context(|| format!("GET {url}"))?;
            if !resp.status().is_redirection() {
                return Ok((url, resp));
            }
            let to = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .with_context(|| format!("{url} redirected with no Location"))?;
            url = redirect_target(&url, to).with_context(|| format!("{url} redirected to {to}"))?;
        }
        bail!("{url}: more than {REDIRECTS} redirects")
    }
}

/// Where a redirect from `from` to `location` goes: `location` if it is a URL, else its path on
/// `from`'s scheme and host.
fn redirect_target(from: &str, location: &str) -> Result<String> {
    if location.contains("://") {
        return Ok(location.to_owned());
    }
    ensure!(
        location.starts_with('/') && !location.starts_with("//"),
        "a Location that is neither a URL nor a path"
    );
    let (scheme, rest) = from.split_once("://").context("a URL with no scheme")?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or_default();
    Ok(format!("{scheme}://{host}{location}"))
}

/// Parses a `Content-Range: bytes a-b/total` header into (a, b, total).
pub(crate) fn content_range(v: &str) -> Option<(u64, u64, u64)> {
    let v = v.trim().strip_prefix("bytes ")?;
    let (range, total) = v.split_once('/')?;
    let (a, b) = range.split_once('-')?;
    Some((a.parse().ok()?, b.parse().ok()?, total.parse().ok()?))
}

impl HttpRange {
    /// Opens `url`, learning its length from a one-byte ranged request.
    pub fn open(url: &str) -> Result<Self> {
        Self::open_with(url, Client::apple())
    }

    fn open_with(url: &str, client: Client) -> Result<Self> {
        (client.allow)(url)?;
        let mut s = Self {
            client,
            url: url.to_owned(),
            len: 0,
            cache: Mutex::default(),
        };
        let (_, total) = s.get(0, 0).with_context(|| format!("reach {url}"))?;
        s.len = total;
        Ok(s)
    }

    /// Bytes `a..=b` of the file, and the file's length.
    fn get(&self, a: u64, b: u64) -> Result<(Vec<u8>, u64)> {
        let mut last = None;
        for attempt in 0..4 {
            if attempt > 0 {
                std::thread::sleep(Duration::from_secs(1 << attempt));
            }
            let mut r = match self.client.get(&self.url, Some(&format!("bytes={a}-{b}"))) {
                Ok((_, r)) if (400..500).contains(&r.status().as_u16()) => {
                    bail!("Apple's CDN refused {} ({})", self.url, r.status())
                }
                Ok((_, r)) if r.status().is_server_error() => {
                    last = Some(anyhow::anyhow!("Apple's CDN answered {}", r.status()));
                    continue;
                }
                Ok((_, r)) => r,
                Err(e) => {
                    last = Some(e);
                    continue;
                }
            };
            ensure!(
                r.status().as_u16() == 206,
                "Apple's CDN answered {} to a ranged request",
                r.status()
            );
            let cr = r
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(content_range)
                .context("Apple's CDN answered a ranged request without its range")?;
            // b is within the file (block() clamps it), so the answer holds the whole range: a
            // shorter one would leave a block short of what its readers slice from it.
            ensure!(
                cr.0 == a && cr.1 == b,
                "Apple's CDN answered bytes {}-{} for {a}-{b}",
                cr.0,
                cr.1
            );
            // ureq's limit refuses a read once it is reached, so a body of exactly the limit
            // fails (measured, ureq 3.5.1): one byte more, and the length is checked below.
            match r.body_mut().with_config().limit(b - a + 2).read_to_vec() {
                Ok(body) if body.len() as u64 == cr.1 - cr.0 + 1 => return Ok((body, cr.2)),
                Ok(body) => {
                    last = Some(anyhow::anyhow!(
                        "a ranged answer of {} bytes for {}",
                        body.len(),
                        cr.1 - cr.0 + 1
                    ))
                }
                Err(e) => last = Some(e.into()),
            }
        }
        Err(last.unwrap()).with_context(|| format!("read bytes {a}-{b} of {}", self.url))
    }

    fn block(&self, index: u64) -> io::Result<Vec<u8>> {
        {
            let mut c = self.cache.lock().unwrap();
            c.tick += 1;
            let tick = c.tick;
            if let Some(e) = c.blocks.get_mut(&index) {
                e.0 = tick;
                return Ok(e.1.clone());
            }
        }
        let a = index * BLOCK;
        let b = (a + BLOCK).min(self.len) - 1;
        let (data, _) = self
            .get(a, b)
            .map_err(|e| io::Error::other(format!("{e:#}")))?;
        let mut c = self.cache.lock().unwrap();
        if c.blocks.len() >= CACHED
            && let Some(old) = c
                .blocks
                .iter()
                .min_by_key(|(_, (t, _))| *t)
                .map(|(k, _)| *k)
        {
            c.blocks.remove(&old);
        }
        let tick = c.tick;
        c.blocks.insert(index, (tick, data.clone()));
        Ok(data)
    }
}

impl ReadAt for HttpRange {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if offset >= self.len {
            return Ok(0);
        }
        let n = (buf.len() as u64).min(self.len - offset) as usize;
        let mut done = 0;
        while done < n {
            let at = offset + done as u64;
            let block = self.block(at / BLOCK)?;
            let from = (at % BLOCK) as usize;
            let take = (n - done).min(block.len() - from);
            buf[done..done + take].copy_from_slice(&block[from..from + take]);
            done += take;
        }
        Ok(n)
    }

    fn len(&self) -> u64 {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_apples_hosts_over_https() {
        assert!(check_url("https://updates.cdn-apple.com/a/b.ipsw").is_ok());
        assert!(check_url("https://mesu.apple.com/assets/macos/c.xml").is_ok());
        assert!(check_url("http://updates.cdn-apple.com/a/b.ipsw").is_err());
        assert!(check_url("https://updates.cdn-apple.com.example.net/b.ipsw").is_err());
        assert!(check_url("https://example.com/updates.cdn-apple.com/b.ipsw").is_err());
        assert!(check_url("https://example.com#updates.cdn-apple.com").is_err());
    }

    #[test]
    fn a_ranged_request_sends_only_the_pinned_headers() {
        let h = crate::sent_headers::of(|url| {
            Client::local().get(url, Some("bytes=0-0")).unwrap();
        });
        crate::sent_headers::assert_pinned(&h, &["range"]);
    }

    #[test]
    fn redirects_are_followed_only_where_allowed() {
        use crate::test_server::Server;
        let s = Server::start(|path, _| match path {
            "/a" => (302, vec![("Location".into(), "/b".into())], vec![]),
            "/b" => (200, vec![], b"here".to_vec()),
            "/away" => (
                301,
                vec![("Location".into(), "https://example.com/x.ipsw".into())],
                vec![],
            ),
            "/loop" => (302, vec![("Location".into(), "/loop".into())], vec![]),
            _ => (404, vec![], vec![]),
        });
        let c = Client::local();
        let (url, mut r) = c.get(&s.url("/a"), None).unwrap();
        assert_eq!(url, s.url("/b"));
        assert_eq!(r.body_mut().read_to_vec().unwrap(), b"here");
        let e = format!("{:#}", c.get(&s.url("/away"), None).unwrap_err());
        assert!(e.contains("https://example.com/x.ipsw is not local"), "{e}");
        let e = format!("{:#}", c.get(&s.url("/loop"), None).unwrap_err());
        assert!(e.contains("more than 5 redirects"), "{e}");
        // the redirect to example.com was refused before any request went there
        assert_eq!(
            s.paths(),
            [
                "/a", "/b", "/away", "/loop", "/loop", "/loop", "/loop", "/loop", "/loop"
            ]
        );
    }

    /// A ranged answer must hold the range asked for: a shorter one, or another range, is an
    /// error, not a short block.
    #[test]
    fn a_short_ranged_answer_is_refused() {
        use crate::test_server::Server;
        let s = Server::start(|path, req| {
            let total = 4u64 << 20;
            let (a, b) = req
                .header("range")
                .and_then(|r| r.strip_prefix("bytes="))
                .and_then(|r| r.split_once('-'))
                .map(|(a, b)| (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap()))
                .unwrap();
            // "/short" answers one byte short of what was asked past the first byte
            let b2 = if path == "/short" && b > 0 { b - 1 } else { b };
            (
                206,
                vec![("Content-Range".into(), format!("bytes {a}-{b2}/{total}"))],
                vec![7u8; (b2 - a + 1) as usize],
            )
        });
        let mut buf = [0u8; 16];
        let ok = HttpRange::open_with(&s.url("/whole"), Client::local()).unwrap();
        assert_eq!(ok.read_at(5, &mut buf).unwrap(), 16);
        let short = HttpRange::open_with(&s.url("/short"), Client::local()).unwrap();
        let e = short.read_at(5, &mut buf).unwrap_err().to_string();
        assert!(e.contains("answered bytes 0-1048574 for 0-1048575"), "{e}");
    }

    #[test]
    fn a_redirect_keeps_the_scheme_and_host() {
        assert_eq!(
            redirect_target("https://mesu.apple.com/a/b.xml", "/c.xml").unwrap(),
            "https://mesu.apple.com/c.xml"
        );
        assert_eq!(
            redirect_target("https://a.example/x", "https://b.example/y").unwrap(),
            "https://b.example/y"
        );
        assert!(redirect_target("https://a.example/x", "//b.example/y").is_err());
        assert!(redirect_target("https://a.example/x", "y").is_err());
    }

    #[test]
    fn content_ranges_parse() {
        assert_eq!(content_range("bytes 0-0/1234"), Some((0, 0, 1234)));
        assert_eq!(content_range("bytes 5-9/10"), Some((5, 9, 10)));
        assert_eq!(content_range("bytes */10"), None);
    }
}
