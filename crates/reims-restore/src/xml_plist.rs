//! XML property lists in the form Apple's own writer gives them: a dictionary's keys sorted, one
//! tab per level, and data in base64 lines that end by column 76 with a tab counted as 8. The
//! restore writes the plists a fresh restore leaves in this form, so that each matches its
//! counterpart byte for byte (docs/design.md, "The disk").

use std::fmt::Write as _;

use base64::Engine as _;
use plist::Value;

const HEAD: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \
                    \"-//Apple//DTD PLIST 1.0//EN\" \
                    \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist \
                    version=\"1.0\">\n";

/// `v` as an XML property list.
pub fn to_xml(v: &Value) -> Vec<u8> {
    let mut s = String::from(HEAD);
    value(&mut s, v, 0);
    s.push_str("</plist>\n");
    s.into_bytes()
}

fn indent(s: &mut String, level: usize) {
    for _ in 0..level {
        s.push('\t');
    }
}

fn line(s: &mut String, level: usize, text: &str) {
    indent(s, level);
    s.push_str(text);
    s.push('\n');
}

fn escape(t: &str) -> String {
    t.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn value(s: &mut String, v: &Value, level: usize) {
    match v {
        Value::Dictionary(d) if d.is_empty() => line(s, level, "<dict/>"),
        Value::Dictionary(d) => {
            line(s, level, "<dict>");
            let mut keys: Vec<&String> = d.keys().collect();
            keys.sort();
            for k in keys {
                line(s, level + 1, &format!("<key>{}</key>", escape(k)));
                value(s, &d[k], level + 1);
            }
            line(s, level, "</dict>");
        }
        Value::Array(a) if a.is_empty() => line(s, level, "<array/>"),
        Value::Array(a) => {
            line(s, level, "<array>");
            for x in a {
                value(s, x, level + 1);
            }
            line(s, level, "</array>");
        }
        Value::String(t) => line(s, level, &format!("<string>{}</string>", escape(t))),
        Value::Integer(i) => line(s, level, &format!("<integer>{i}</integer>")),
        Value::Real(r) => line(s, level, &format!("<real>{r:?}</real>")),
        Value::Boolean(b) => line(s, level, if *b { "<true/>" } else { "<false/>" }),
        Value::Date(d) => line(s, level, &format!("<date>{}</date>", d.to_xml_format())),
        Value::Data(d) => {
            line(s, level, "<data>");
            // Each line holds as many whole 4-character groups as fit before column 76.
            let width = 76usize.saturating_sub(8 * level).max(16) / 4 * 3;
            let b64 = base64::engine::general_purpose::STANDARD;
            for chunk in d.chunks(width) {
                indent(s, level);
                let _ = writeln!(s, "{}", b64.encode(chunk));
            }
            line(s, level, "</data>");
        }
        Value::Uid(u) => line(s, level, &format!("<integer>{}</integer>", u.get())),
        _ => line(s, level, "<string></string>"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plist::Dictionary;

    #[test]
    fn a_plist_is_written_in_apples_form() {
        let mut inner = Dictionary::new();
        inner.insert("Digest".into(), Value::Data((0u8..40).collect()));
        inner.insert("Trusted".into(), Value::Boolean(true));
        let mut d = Dictionary::new();
        d.insert("b".into(), Value::Integer(7.into()));
        d.insert("a & <z>".into(), Value::String("x<y".into()));
        d.insert(
            "c".into(),
            Value::Array(vec![Value::Dictionary(inner), Value::Array(vec![])]),
        );
        d.insert("e".into(), Value::Dictionary(Dictionary::new()));
        let got = String::from_utf8(to_xml(&Value::Dictionary(d))).unwrap();
        let want = format!(
            "{HEAD}<dict>\n\t<key>a &amp; &lt;z&gt;</key>\n\t<string>x&lt;y</string>\n\
             \t<key>b</key>\n\t<integer>7</integer>\n\t<key>c</key>\n\t<array>\n\t\t<dict>\n\
             \t\t\t<key>Digest</key>\n\t\t\t<data>\n\
             \t\t\tAAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUm\n\t\t\tJw==\n\t\t\t</data>\n\t\t\t<key>Trusted</key>\n\t\t\t<true/>\n\
             \t\t</dict>\n\t\t<array/>\n\t</array>\n\t<key>e</key>\n\t<dict/>\n</dict>\n</plist>\n"
        );
        assert_eq!(got, want);
    }

    #[test]
    fn what_is_written_reads_back() {
        let mut d = Dictionary::new();
        d.insert("k".into(), Value::Data(vec![0xab; 300]));
        d.insert("n".into(), Value::Integer(u64::MAX.into()));
        let v = Value::Dictionary(d);
        let back = Value::from_reader_xml(std::io::Cursor::new(to_xml(&v))).unwrap();
        assert_eq!(back, v);
    }
}
