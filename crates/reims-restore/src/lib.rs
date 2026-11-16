//! reims-restore: restores macOS from an IPSW onto an arm64 VM's disk, as Apple's restore would,
//! without Apple's tools or code (docs/design.md). It reads the IPSW where it lies, decrypts and
//! decompresses its images as it copies them, asks Apple's TSS for the VM's tickets, and writes
//! the disk, its auxiliary storage and the VM's ECID itself, on any host.

pub mod aea;
pub mod anchors;
// The file is not `aux.rs`: AUX is a device name Windows keeps, which no file there takes, so git
// for Windows checks out no tree holding it.
#[path = "aux_storage.rs"]
pub mod aux;
pub mod catalog;
pub mod containers;
pub mod download;
pub mod gpt;
pub mod http;
pub mod image;
pub mod img4;
pub mod ipsw;
pub mod kernelcache;
pub mod manifest;
pub mod mtree;
pub mod pipeline;
pub mod plan;
pub mod policy;
pub mod policy_nonce;
pub mod source;
pub mod sys;
pub mod tss;
pub mod udif;
pub mod xml_plist;

/// The User-Agent of every request the tool makes (Apple's TSS, key service and CDN): the one
/// community restore tools send to these endpoints, fixed, naming nothing of the host, its user
/// or this project.
pub const USER_AGENT: &str = "InetURL/1.0";

/// The headers each of the tool's HTTP clients sends, read off a local listener.
#[cfg(test)]
mod test_server;

#[cfg(test)]
pub(crate) mod sent_headers {
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;

    /// The header lines of the one request `send` makes to the URL it is given: lowercased
    /// names, in the order sent.
    pub fn of(send: impl FnOnce(&str)) -> Vec<(String, String)> {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/k", l.local_addr().unwrap());
        let t = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut req = Vec::new();
            let mut b = [0u8; 4096];
            let end = loop {
                if let Some(i) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                    break i + 4;
                }
                let n = s.read(&mut b).unwrap();
                assert!(n > 0, "the request ended before its headers did");
                req.extend_from_slice(&b[..n]);
            };
            let head = String::from_utf8(req[..end].to_vec()).unwrap();
            let body = head
                .lines()
                .filter_map(|l| l.split_once(':'))
                .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .map_or(0, |(_, v)| v.trim().parse::<usize>().unwrap());
            // read the body before answering, so closing resets nothing the client still sends
            let mut have = req.len() - end;
            while have < body {
                let n = s.read(&mut b).unwrap();
                assert!(n > 0, "the request ended before its body did");
                have += n;
            }
            s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
            head
        });
        send(&url);
        let head = t.join().unwrap();
        head.split("\r\n")
            .skip(1)
            .take_while(|l| !l.is_empty())
            .map(|l| {
                let (k, v) = l.split_once(':').unwrap();
                (k.trim().to_ascii_lowercase(), v.trim().to_owned())
            })
            .collect()
    }

    /// Checks that `h` holds the fixed User-Agent and no header beyond the host, the User-Agent,
    /// ureq's accept headers and `extra`.
    pub fn assert_pinned(h: &[(String, String)], extra: &[&str]) {
        let ua = h
            .iter()
            .find(|(k, _)| k == "user-agent")
            .map(|(_, v)| v.as_str());
        assert_eq!(ua, Some(crate::USER_AGENT));
        for (k, v) in h {
            assert!(
                ["host", "user-agent", "accept", "accept-encoding"].contains(&k.as_str())
                    || extra.contains(&k.as_str()),
                "unexpected header {k}: {v}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_user_agent_is_the_fixed_one() {
        assert_eq!(crate::USER_AGENT, "InetURL/1.0");
    }
}
