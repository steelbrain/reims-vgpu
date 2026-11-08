//! Four-character codes: the message types, parameter keys, value types,
//! atom types, codec types and pixel formats the wire spells as one
//! big-endian u32.

use std::fmt;

/// A four-character code, as the u32 whose big-endian bytes spell it
/// (`FourCc::new(*b"vtpv")` is 0x76747076).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct FourCc(pub u32);

impl FourCc {
    /// The code its four bytes spell, first byte most significant.
    pub const fn new(bytes: [u8; 4]) -> Self {
        Self(u32::from_be_bytes(bytes))
    }

    /// The four bytes, as the wire carries them.
    pub const fn to_be_bytes(self) -> [u8; 4] {
        self.0.to_be_bytes()
    }
}

impl From<u32> for FourCc {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl fmt::Debug for FourCc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'{self}'")
    }
}

impl fmt::Display for FourCc {
    /// The printable characters as they are and the others as `.`, as
    /// VideoToolbox's own logs print a FourCC.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.to_be_bytes() {
            let shown = if byte.is_ascii_graphic() || byte == b' ' {
                char::from(byte)
            } else {
                '.'
            };
            write!(f, "{shown}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_code_is_its_bytes_big_endian() {
        assert_eq!(FourCc::new(*b"vtpv").0, 0x7674_7076);
        assert_eq!(FourCc(0x2c65_7272).to_be_bytes(), *b",err");
    }

    #[test]
    fn unprintable_bytes_show_as_dots() {
        assert_eq!(FourCc::new(*b"=u32").to_string(), "=u32");
        assert_eq!(FourCc(0x0000_0041).to_string(), "...A");
    }
}
