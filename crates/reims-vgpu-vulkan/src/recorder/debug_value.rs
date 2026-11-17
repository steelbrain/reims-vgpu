//! An argument's value, read from its `Debug` text: the platform's argument types all print
//! themselves (derived `Debug`, a few by hand), so one reader turns every argument into the
//! stream's values without a writer per type.
//!
//! - A struct is a map of its fields, each named in camel case (`pixel_format` is
//!   `pixelFormat`, Metal's property names where the field is named after one); its type's name
//!   is dropped. A tuple variant is a map of one pair, its name and its value (several values as
//!   a list); a unit variant is its name, a string.
//! - `Some(x)` is `x`; `None` is nil; a list, tuple, set or range is a list; a map is a list of
//!   `[key, value]` lists. An integer is unsigned, or signed where negative; any other number is
//!   a double.
//! - Objects: `ObjRef(n)` is the object of stream id `n` (an `Obj` prints so while it is
//!   formatted for the stream), `ObjId(p)` the object whose pointer is `p`, and
//!   `Unrecorded(..)` (an `Obj` the stream never made) nil, counted.
//! - Host memory (`HostRange { .. }`, `HostPtr(..)`) is nil: its bytes reach the stream as
//!   writes, never its address. A lock (`Mutex { data: x, .. }`) is `x`.
//!
//! Text the reader does not take is kept whole as a string, counted.

use super::format::Value;

/// What a read found besides its value.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Notes {
    /// Objects the stream never made, read as nil.
    pub unrecorded: u64,
    /// Whether the text did not read, and was kept whole as a string.
    pub unread: bool,
}

/// The value `text`, an argument's `Debug` text, stands for; `ptr_ids` gives the stream id of
/// the object at a pointer, where the stream made one there.
pub fn parse(text: &str, ptr_ids: &dyn Fn(u64) -> Option<u64>) -> (Value, Notes) {
    let mut p = Parser {
        s: text.as_bytes(),
        at: 0,
        ptr_ids,
        notes: Notes::default(),
    };
    let v = p.value();
    p.ws();
    match v {
        Some(v) if p.at == p.s.len() => (v, p.notes),
        _ => (
            Value::Str(text.to_owned()),
            Notes {
                unread: true,
                ..p.notes
            },
        ),
    }
}

/// `snake_case` as `snakeCase`.
pub fn camel(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut up = false;
    for c in name.chars() {
        if c == '_' {
            up = !out.is_empty();
        } else if up {
            out.extend(c.to_uppercase());
            up = false;
        } else {
            out.push(c);
        }
    }
    out
}

struct Parser<'a> {
    s: &'a [u8],
    at: usize,
    ptr_ids: &'a dyn Fn(u64) -> Option<u64>,
    notes: Notes,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.at).copied()
    }

    fn ws(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_whitespace()) {
            self.at += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.peek() == Some(c) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn starts(&self, t: &[u8]) -> bool {
        self.s.get(self.at..).is_some_and(|r| r.starts_with(t))
    }

    fn value(&mut self) -> Option<Value> {
        let v = self.single()?;
        self.ws();
        if self.starts(b"..") && !self.starts(b"...") {
            self.at += 2;
            if self.peek() == Some(b'=') {
                self.at += 1;
            }
            self.ws();
            // An open range (`a..`) ends here.
            if matches!(self.peek(), None | Some(b',' | b')' | b']' | b'}')) {
                return Some(Value::List(vec![v]));
            }
            let end = self.single()?;
            return Some(Value::List(vec![v, end]));
        }
        Some(v)
    }

    fn single(&mut self) -> Option<Value> {
        self.ws();
        match self.peek()? {
            b'"' => self.string().map(Value::Str),
            b'\'' => self.char_lit().map(Value::Str),
            b'[' => {
                self.at += 1;
                self.seq(b']').map(Value::List)
            }
            b'(' => {
                self.at += 1;
                self.seq(b')').map(Value::List)
            }
            b'{' => {
                self.at += 1;
                self.braces()
            }
            b'-' | b'0'..=b'9' => self.number(),
            c if c.is_ascii_alphabetic() || c == b'_' => self.named(),
            _ => None,
        }
    }

    fn seq(&mut self, close: u8) -> Option<Vec<Value>> {
        let mut items = Vec::new();
        loop {
            if self.eat(close) {
                return Some(items);
            }
            items.push(self.value()?);
            if !self.eat(b',') {
                return self.eat(close).then_some(items);
            }
        }
    }

    /// A map (`{k: v, ..}`) as `[k, v]` lists, or a set (`{a, b}`) as a list.
    fn braces(&mut self) -> Option<Value> {
        let mut items = Vec::new();
        loop {
            if self.eat(b'}') {
                return Some(Value::List(items));
            }
            let k = self.value()?;
            if self.eat(b':') {
                let v = self.value()?;
                items.push(Value::List(vec![k, v]));
            } else {
                items.push(k);
            }
            if !self.eat(b',') {
                return self.eat(b'}').then_some(Value::List(items));
            }
        }
    }

    /// A name, its path's last part, with any generic arguments after it skipped
    /// (`PhantomData<&[u8]>`).
    fn ident(&mut self) -> String {
        let start = self.at;
        loop {
            while self
                .peek()
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_')
            {
                self.at += 1;
            }
            if self.starts(b"::") {
                self.at += 2;
                continue;
            }
            break;
        }
        let text =
            std::str::from_utf8(self.s.get(start..self.at).unwrap_or_default()).unwrap_or_default();
        let name = text.rsplit("::").next().unwrap_or(text).to_owned();
        if self.peek() == Some(b'<') {
            let mut depth = 0usize;
            while let Some(c) = self.peek() {
                self.at += 1;
                match c {
                    b'<' => depth += 1,
                    b'>' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
        name
    }

    fn named(&mut self) -> Option<Value> {
        let name = self.ident();
        match name.as_str() {
            "true" => return Some(Value::Bool(true)),
            "false" => return Some(Value::Bool(false)),
            "None" => return Some(Value::Nil),
            "NaN" => return Some(Value::F64(f64::NAN)),
            "inf" => return Some(Value::F64(f64::INFINITY)),
            _ => {}
        }
        self.ws();
        match self.peek() {
            Some(b'(') => {
                self.at += 1;
                let mut args = self.seq(b')')?;
                Some(match name.as_str() {
                    "Some" if args.len() == 1 => args.remove(0),
                    "ObjRef" => match args.first() {
                        Some(Value::Uint(id)) => Value::Object(*id),
                        _ => Value::Nil,
                    },
                    "ObjId" => match args.first() {
                        Some(Value::Uint(p)) => match (self.ptr_ids)(*p) {
                            Some(id) => Value::Object(id),
                            None => {
                                self.notes.unrecorded += 1;
                                Value::Nil
                            }
                        },
                        _ => Value::Nil,
                    },
                    "Unrecorded" => {
                        self.notes.unrecorded += 1;
                        Value::Nil
                    }
                    "HostPtr" | "NonNull" => Value::Nil,
                    _ if args.len() == 1 => Value::Map(vec![(name, args.remove(0))]),
                    _ => Value::Map(vec![(name, Value::List(args))]),
                })
            }
            Some(b'{') => {
                self.at += 1;
                let fields = self.fields()?;
                Some(match name.as_str() {
                    "HostRange" => Value::Nil,
                    "Mutex" | "RwLock" | "RefCell" | "Cell" => fields
                        .into_iter()
                        .find(|(k, _)| k == "data" || k == "value")
                        .map_or(Value::Nil, |(_, v)| v),
                    _ => Value::Map(fields),
                })
            }
            _ => Some(Value::Str(name)),
        }
    }

    /// A struct's fields up to its `}`; `..` (a non-exhaustive struct) is skipped.
    fn fields(&mut self) -> Option<Vec<(String, Value)>> {
        let mut fields = Vec::new();
        loop {
            if self.eat(b'}') {
                return Some(fields);
            }
            self.ws();
            if self.starts(b"..") {
                self.at += 2;
            } else {
                let name = camel(&self.ident());
                if name.is_empty() || !self.eat(b':') {
                    return None;
                }
                let v = self.value()?;
                fields.push((name, v));
            }
            if !self.eat(b',') {
                return self.eat(b'}').then_some(fields);
            }
        }
    }

    fn number(&mut self) -> Option<Value> {
        let start = self.at;
        let neg = self.peek() == Some(b'-');
        if neg {
            self.at += 1;
        }
        if self.starts(b"inf") {
            self.at += 3;
            return Some(Value::F64(f64::NEG_INFINITY));
        }
        if self.starts(b"0x") {
            self.at += 2;
            let h = self.at;
            while self.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
                self.at += 1;
            }
            let text = std::str::from_utf8(self.s.get(h..self.at)?).ok()?;
            let v = u64::from_str_radix(text, 16).ok()?;
            return Some(if neg {
                Value::Sint(0i64.checked_sub_unsigned(v)?)
            } else {
                Value::Uint(v)
            });
        }
        let mut float = false;
        while let Some(c) = self.peek() {
            let next = self.s.get(self.at + 1).copied();
            match c {
                b'0'..=b'9' => {}
                b'.' if next.is_some_and(|n| n.is_ascii_digit()) => float = true,
                b'e' | b'E' => {
                    float = true;
                    if matches!(next, Some(b'-' | b'+')) {
                        self.at += 1;
                    }
                }
                _ => break,
            }
            self.at += 1;
        }
        let text = std::str::from_utf8(self.s.get(start..self.at)?).ok()?;
        if float {
            return text.parse().ok().map(Value::F64);
        }
        if neg {
            return text.parse().ok().map(Value::Sint);
        }
        text.parse().ok().map(Value::Uint)
    }

    fn string(&mut self) -> Option<String> {
        self.at += 1;
        let mut out = String::new();
        loop {
            let c = self.peek()?;
            self.at += 1;
            match c {
                b'"' => return Some(out),
                b'\\' => out.push(self.escape()?),
                _ => {
                    // A UTF-8 sequence, copied whole.
                    let start = self.at - 1;
                    while self.peek().is_some_and(|b| b & 0xc0 == 0x80) {
                        self.at += 1;
                    }
                    out.push_str(std::str::from_utf8(self.s.get(start..self.at)?).ok()?);
                }
            }
        }
    }

    fn char_lit(&mut self) -> Option<String> {
        self.at += 1;
        let mut out = String::new();
        loop {
            let c = self.peek()?;
            self.at += 1;
            match c {
                b'\'' => return Some(out),
                b'\\' => out.push(self.escape()?),
                _ => out.push(char::from(c)),
            }
        }
    }

    fn escape(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.at += 1;
        Some(match c {
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'0' => '\0',
            b'u' => {
                if !self.eat(b'{') {
                    return None;
                }
                let start = self.at;
                while self.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
                    self.at += 1;
                }
                let text = std::str::from_utf8(self.s.get(start..self.at)?).ok()?;
                let code = u32::from_str_radix(text, 16).ok()?;
                if !self.eat(b'}') {
                    return None;
                }
                char::from_u32(code)?
            }
            other => char::from(other),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Value {
        let (v, notes) = parse(text, &|p| (p == 0xabc).then_some(9));
        assert!(!notes.unread, "{text} did not read");
        v
    }

    fn map(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
        )
    }

    /// A struct is a map of camel-cased fields, an option its value, a unit variant its name
    #[test]
    fn structs_options_and_variants() {
        let v = read(
            r#"TextureDesc { pixel_format: 80, width: Some(64), usage: None, storage: Private, label: "a \"b\"", depth: -1 }"#,
        );
        assert_eq!(
            v,
            map(&[
                ("pixelFormat", Value::Uint(80)),
                ("width", Value::Uint(64)),
                ("usage", Value::Nil),
                ("storage", Value::Str("Private".to_owned())),
                ("label", Value::Str("a \"b\"".to_owned())),
                ("depth", Value::Sint(-1)),
            ])
        );
    }

    /// Objects read as their stream ids; an object the stream never made is nil, counted
    #[test]
    fn objects_and_unrecorded_objects() {
        let (v, notes) = parse(
            "[ObjRef(3), ObjId(2748), ObjId(5), Unrecorded(\"TextureObject\")]",
            &|p| (p == 0xabc).then_some(9),
        );
        assert_eq!(
            v,
            Value::List(vec![
                Value::Object(3),
                Value::Object(9),
                Value::Nil,
                Value::Nil
            ])
        );
        assert_eq!(notes.unrecorded, 2);
    }

    /// Lists, tuples, ranges, maps, floats and tuple variants
    #[test]
    fn lists_ranges_maps_floats_and_tuple_variants() {
        let got = read(
            "([1, 2], 0..10, 3..=4, {1: 2}, 1.5, -2.5e-3, NaN, ResourceOptions(16), Two(1, 2))",
        );
        let want = Value::List(vec![
            Value::List(vec![Value::Uint(1), Value::Uint(2)]),
            Value::List(vec![Value::Uint(0), Value::Uint(10)]),
            Value::List(vec![Value::Uint(3), Value::Uint(4)]),
            Value::List(vec![Value::List(vec![Value::Uint(1), Value::Uint(2)])]),
            Value::F64(1.5),
            Value::F64(-2.5e-3),
            Value::F64(f64::NAN),
            map(&[("ResourceOptions", Value::Uint(16))]),
            map(&[("Two", Value::List(vec![Value::Uint(1), Value::Uint(2)]))]),
        ]);
        // NaN equals nothing, itself included: compare the texts.
        assert_eq!(format!("{got:?}"), format!("{want:?}"));
    }

    /// Host memory is nil, a lock is its data, and a non-exhaustive struct reads
    #[test]
    fn host_memory_locks_and_non_exhaustive_structs() {
        assert_eq!(
            read(
                "BufferDesc { contents: Some(HostRange { ptr: HostPtr(4096), len: 64, _mapped: PhantomData<&[u8]> }), length: 64 }"
            ),
            map(&[("contents", Value::Nil), ("length", Value::Uint(64))])
        );
        assert_eq!(
            read("Mutex { data: Fields { width: 1 }, poisoned: false, .. }"),
            map(&[("width", Value::Uint(1))])
        );
    }

    /// Text the reader does not take is kept whole, counted
    #[test]
    fn unread_text_is_kept_whole() {
        let (v, notes) = parse("<0x10 Thing>", &|_| None);
        assert!(notes.unread);
        assert_eq!(v, Value::Str("<0x10 Thing>".to_owned()));
    }

    /// Snake case becomes camel case
    #[test]
    fn camel_case() {
        assert_eq!(camel("pixel_format"), "pixelFormat");
        assert_eq!(camel("_mapped"), "mapped");
        assert_eq!(camel("width"), "width");
    }
}
