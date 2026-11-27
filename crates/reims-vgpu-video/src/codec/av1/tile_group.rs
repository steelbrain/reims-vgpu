//! Tile groups (AV1 5.11.1 and 6.10.1): the tiles a tile group OBU, or an `OBU_FRAME` after
//! its frame header, holds, each tile's bytes found by the sizes the group codes. Nothing
//! here reads a tile: the hardware decodes it from the group's tile data.

use super::Error;
use super::bits::BitReader;
use super::frame::TileInfo;
use crate::codec::num::ToUsize;

/// One tile of a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    /// Where its bytes start in [`TileGroup::data`].
    pub offset: u32,
    /// Its bytes, one at least.
    pub size: u32,
    /// Its tile row.
    pub row: u32,
    /// Its tile column.
    pub column: u32,
}

/// A tile group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileGroup<'a> {
    /// The group's tile data: its OBU's bytes after `byte_alignment()`, each tile's size
    /// field and bytes.
    pub data: &'a [u8],
    /// `tg_start`.
    pub start: u32,
    /// `tg_end`.
    pub end: u32,
    /// Its tiles, `tg_start` to `tg_end`.
    pub tiles: Vec<Tile>,
}

impl<'a> TileGroup<'a> {
    /// Reads the tile group `obu` holds, for a frame of `info`'s tiles.
    pub fn parse(obu: &'a [u8], info: &TileInfo) -> Result<Self, Error> {
        let num_tiles = info.num_tiles();
        let mut r = BitReader::new(obu);
        let (start, end) = if num_tiles > 1 && r.bit()? {
            let bits = info.cols_log2.saturating_add(info.rows_log2);
            (r.f(bits)?, r.f(bits)?)
        } else {
            (0, num_tiles.saturating_sub(1))
        };
        r.byte_alignment()?;
        if start > end || end >= num_tiles {
            return Err(Error::OutOfRange("tg_end"));
        }
        let data = obu.get(r.byte_position()..).ok_or(Error::Truncated)?;
        let columns =
            std::num::NonZeroU32::new(info.cols()).ok_or(Error::OutOfRange("TileCols"))?;
        let size_bytes = info.tile_size_bytes.to_usize();
        let mut offset = 0usize;
        let mut tiles = Vec::new();
        for number in start..=end {
            let size = if number == end {
                data.len().saturating_sub(offset)
            } else {
                let field = data
                    .get(offset..offset.saturating_add(size_bytes))
                    .filter(|field| field.len() == size_bytes)
                    .ok_or(Error::Truncated)?;
                offset = offset.saturating_add(size_bytes);
                let mut bytes = [0u8; 4];
                for (byte, &coded) in bytes.iter_mut().zip(field) {
                    *byte = coded;
                }
                // tile_size_minus_1 + 1, at most 2^32.
                let size = u32::from_le_bytes(bytes).to_usize().saturating_add(1);
                // The tiles after this one have a byte each at least.
                if size >= data.len().saturating_sub(offset) {
                    return Err(Error::Truncated);
                }
                size
            };
            if size == 0 {
                return Err(Error::EmptyTile);
            }
            tiles.push(Tile {
                offset: u32::try_from(offset).map_err(|_| Error::OutOfRange("tile offset"))?,
                size: u32::try_from(size).map_err(|_| Error::OutOfRange("tile size"))?,
                row: number / columns,
                column: number % columns,
            });
            offset = offset.saturating_add(size);
        }
        Ok(Self {
            data,
            start,
            end,
            tiles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(cols: u32, rows: u32, cols_log2: u32, rows_log2: u32, size_bytes: u32) -> TileInfo {
        TileInfo {
            uniform: true,
            cols_log2,
            rows_log2,
            col_widths_sb: vec![1; cols as usize],
            row_heights_sb: vec![1; rows as usize],
            context_update_tile_id: 0,
            tile_size_bytes: size_bytes,
        }
    }

    #[test]
    fn a_groups_tiles_are_found_by_their_sizes_and_the_last_takes_the_rest() {
        // 2x2 tiles; the group holds tiles 1 to 3: flag 1, tg_start 01, tg_end 11, aligned.
        let mut obu = vec![0b1011_1000];
        obu.extend([0x01, 0xa0, 0xa1]); // tile 1: 2 bytes
        obu.extend([0x00, 0xb0]); // tile 2: 1 byte
        obu.extend([0xc0, 0xc1, 0xc2]); // tile 3: the rest
        let group = TileGroup::parse(&obu, &info(2, 2, 1, 1, 1)).expect("a group");
        assert_eq!((group.start, group.end), (1, 3));
        assert_eq!(
            group.tiles,
            vec![
                Tile {
                    offset: 1,
                    size: 2,
                    row: 0,
                    column: 1
                },
                Tile {
                    offset: 4,
                    size: 1,
                    row: 1,
                    column: 0
                },
                Tile {
                    offset: 5,
                    size: 3,
                    row: 1,
                    column: 1
                },
            ]
        );
        assert_eq!(group.data, &obu[1..]);
    }

    #[test]
    fn a_size_past_the_group_an_empty_last_tile_or_a_bad_range_is_refused() {
        let info = info(2, 1, 1, 0, 2);
        // No flag: tiles 0 and 1; tile 0's size of 3 leaves nothing for tile 1.
        assert_eq!(
            TileGroup::parse(&[0x00, 0x02, 0x00, 1, 2, 3], &info),
            Err(Error::Truncated)
        );
        assert_eq!(
            TileGroup::parse(&[0x00, 0x02, 0x00, 1, 2, 3, 4], &info),
            Ok(TileGroup {
                data: &[0x02, 0x00, 1, 2, 3, 4],
                start: 0,
                end: 1,
                tiles: vec![
                    Tile {
                        offset: 2,
                        size: 3,
                        row: 0,
                        column: 0
                    },
                    Tile {
                        offset: 5,
                        size: 1,
                        row: 0,
                        column: 1
                    },
                ],
            })
        );
        // Flag 1, tg_start 1, tg_end 0.
        assert_eq!(
            TileGroup::parse(&[0b1100_0000, 1], &info),
            Err(Error::OutOfRange("tg_end"))
        );
        // One tile and no bytes.
        let one = super::super::frame::TileInfo {
            col_widths_sb: vec![1],
            row_heights_sb: vec![1],
            ..info.clone()
        };
        assert_eq!(TileGroup::parse(&[], &one), Err(Error::EmptyTile));
    }
}
