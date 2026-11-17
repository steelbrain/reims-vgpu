//! The DXGI adapter an example's decodes run on, on Windows: its first
//! argument, a LUID in hex (`0x` optional), or `default` for the adapter the
//! driver picks.

/// The LUID `arg` names, as `Display::open_adapter` takes it: its 8 bytes,
/// little-endian; `None` for `default`.
pub fn luid(arg: &str) -> Result<Option<[u8; 8]>, String> {
    if arg == "default" {
        return Ok(None);
    }
    let digits = arg.strip_prefix("0x").unwrap_or(arg);
    u64::from_str_radix(digits, 16)
        .map(|luid| Some(luid.to_le_bytes()))
        .map_err(|_| format!("{arg}: neither `default` nor a DXGI adapter's LUID in hex"))
}
