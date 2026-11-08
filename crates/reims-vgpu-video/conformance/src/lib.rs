//! The test streams' readers and writers, with no dependency: what the
//! backends' examples and hardware tests read their streams with and hash
//! their frames by.
//!
//! - [`mp4`] reads a track's configuration record and samples out of an
//!   MP4, and [`mp4_write`] writes one of what an encoder gave.
//! - [`md5`] hashes frames, as ffmpeg's `framemd5` does.

pub mod md5;
pub mod mp4;
pub mod mp4_write;
