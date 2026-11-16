//! Format descriptions as `=fds` parameters carry them: a big-endian QuickTime ImageDescription,
//! its 86-byte header and then extension atoms such as `avcC`, which the guest's VideoToolbox
//! writes with `CMVideoFormatDescriptionCopyAsBigEndianImageDescriptionBlockBuffer`.

use crate::core::fourcc::FourCc;
use crate::paravirt::WireError;
use crate::paravirt::atom::{read_u16, read_u32};

/// The ImageDescription header's size.
pub const HEADER_SIZE: usize = 86;

/// The `avcC` extension: an AVCDecoderConfigurationRecord.
pub const AVC_CONFIGURATION: FourCc = FourCc::new(*b"avcC");
/// The `hvcC` extension: an HEVCDecoderConfigurationRecord.
pub const HEVC_CONFIGURATION: FourCc = FourCc::new(*b"hvcC");

/// A video format description, as far as the host reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageDescription {
    /// The codec type, such as `avc1`.
    pub codec: FourCc,
    /// The width.
    pub width: u16,
    /// The height.
    pub height: u16,
    /// The extension atoms in order, each as its type and payload.
    pub extensions: Vec<(FourCc, Vec<u8>)>,
}

impl ImageDescription {
    /// Reads an ImageDescription. Its own size field bounds it; the extensions end at that size,
    /// at a zero size field, or at a tail too short for an atom.
    pub fn parse(bytes: &[u8]) -> Result<Self, WireError> {
        let size = read_u32(bytes, 0)
            .and_then(|size| usize::try_from(size).ok())
            .ok_or(WireError::Malformed("image description size"))?;
        if size < HEADER_SIZE || size > bytes.len() {
            return Err(WireError::Malformed("image description size"));
        }
        let description = bytes
            .get(..size)
            .ok_or(WireError::Malformed("image description size"))?;
        let field16 =
            |offset| read_u16(description, offset).ok_or(WireError::Malformed("image description"));
        let mut extensions = Vec::new();
        let mut offset = HEADER_SIZE;
        while offset + 8 <= description.len() {
            let atom_size = read_u32(description, offset)
                .and_then(|size| usize::try_from(size).ok())
                .ok_or(WireError::Malformed("extension size"))?;
            if atom_size == 0 {
                break;
            }
            if atom_size < 8 || atom_size > description.len() - offset {
                return Err(WireError::Malformed("extension size"));
            }
            let kind =
                read_u32(description, offset + 4).ok_or(WireError::Malformed("extension type"))?;
            let payload = description
                .get(offset + 8..offset + atom_size)
                .ok_or(WireError::Malformed("extension size"))?;
            extensions.push((FourCc(kind), payload.to_vec()));
            offset += atom_size;
        }
        Ok(Self {
            codec: FourCc(read_u32(description, 4).ok_or(WireError::Malformed("codec type"))?),
            width: field16(32)?,
            height: field16(34)?,
            extensions,
        })
    }

    /// The first extension of type `kind`.
    pub fn extension(&self, kind: FourCc) -> Option<&[u8]> {
        self.extensions
            .iter()
            .find(|(found, _)| *found == kind)
            .map(|(_, payload)| payload.as_slice())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A description of `codec` at `width` by `height` with `record` as its configuration
    /// record, written byte by byte from the QuickTime layout.
    pub(crate) fn hand_written(codec: &[u8; 4], width: u16, height: u16, record: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0, 0, 0, 0];
        bytes.extend_from_slice(codec);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0xff, 0xff]);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
        bytes.extend_from_slice(&[0, 0, 2, 0, 0, 0, 2, 0]);
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[0, 0x48, 0, 0, 0, 0x48, 0, 0]);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 1]);
        let mut name = [0; 32];
        name[..6].copy_from_slice(b"\x05H.264");
        bytes.extend_from_slice(&name);
        bytes.extend_from_slice(&[0, 24, 0xff, 0xff]);
        bytes.extend_from_slice(&(record.len() as u32 + 8).to_be_bytes());
        bytes.extend_from_slice(if codec == b"hvc1" { b"hvcC" } else { b"avcC" });
        bytes.extend_from_slice(record);
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        let size = bytes.len() as u32;
        bytes[..4].copy_from_slice(&size.to_be_bytes());
        bytes
    }

    const RECORD: [u8; 7] = [1, 0x64, 0, 0x0b, 0xff, 0xe0, 0];

    #[test]
    fn reads_a_hand_written_description() -> Result<(), WireError> {
        let bytes = hand_written(b"avc1", 64, 48, &RECORD);
        assert_eq!(bytes.len(), 105);
        let description = ImageDescription::parse(&bytes)?;
        assert_eq!(
            description,
            ImageDescription {
                codec: FourCc::new(*b"avc1"),
                width: 64,
                height: 48,
                extensions: vec![(AVC_CONFIGURATION, RECORD.to_vec())],
            }
        );
        assert_eq!(description.extension(AVC_CONFIGURATION), Some(&RECORD[..]));
        assert_eq!(description.extension(HEVC_CONFIGURATION), None);
        Ok(())
    }

    #[test]
    fn a_description_shorter_than_its_header_is_malformed() {
        let mut bytes = hand_written(b"avc1", 64, 48, &RECORD);
        bytes.truncate(80);
        assert_eq!(
            ImageDescription::parse(&bytes),
            Err(WireError::Malformed("image description size"))
        );
    }

    #[test]
    fn an_extension_past_the_end_is_malformed() {
        let mut bytes = hand_written(b"avc1", 64, 48, &RECORD);
        bytes[HEADER_SIZE + 3] = 40;
        assert_eq!(
            ImageDescription::parse(&bytes),
            Err(WireError::Malformed("extension size"))
        );
    }
}
