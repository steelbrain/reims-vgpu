//! MTLB sole-function load with content-hash cache.

use crate::backend::blob::BlobKey;
use crate::backend::metal::cache::{fn_cache_insert, fn_cache_lookup};
use crate::backend::metal::util::{set_err, ErrOut, Status};
use metal::{Device, Function};

pub fn load_only_function(
    device: &Device,
    mtlb: &[u8],
    label: &str,
    err: ErrOut<'_>,
) -> Result<Function, Status> {
    validate_mtlb(mtlb, label, err)?;
    let key = BlobKey::new(mtlb);
    if let Some(hit) = fn_cache_lookup(&key) {
        return Ok(hit);
    }
    let function = load_only_function_uncached(device, mtlb, label, err)?;
    Ok(fn_cache_insert(&key, function))
}

fn validate_mtlb(mtlb: &[u8], label: &str, err: ErrOut<'_>) -> Result<(), Status> {
    if mtlb.is_empty() {
        set_err(err, format!("{label} MTLB is empty"));
        Err(Status::args("metal_function_mtlb_empty"))
    } else {
        Ok(())
    }
}

fn load_only_function_uncached(
    device: &Device,
    mtlb: &[u8],
    label: &str,
    err: ErrOut<'_>,
) -> Result<Function, Status> {
    let library = match device.new_library_with_data(mtlb) {
        Ok(lib) => lib,
        Err(e) => {
            set_err(err, format!("{label} newLibraryWithData failed: {e}"));
            return Err(Status::execute("metal_function_library_create_failed")
                .field("mtlb_len", mtlb.len()));
        }
    };
    let names = library.function_names();
    if names.len() != 1 {
        set_err(
            err,
            format!("{label} exposed {} functions, expected one", names.len()),
        );
        return Err(Status::args("metal_function_count_not_one")
            .field("count", names.len())
            .field("expected", 1usize));
    }
    match library.get_function(&names[0], None) {
        Ok(f) => Ok(f),
        Err(_) => {
            set_err(err, format!("{label} function lookup failed"));
            Err(Status::execute("metal_function_lookup_failed"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Refusal;
    use std::ffi::CStr;
    use std::os::raw::c_char;

    #[test]
    fn empty_mtlb_is_rejected_before_device_access_with_a_reason() {
        let mut error = [0 as c_char; 64];
        let status = validate_mtlb(&[], "vertex", (error.as_mut_ptr(), error.len()))
            .expect_err("empty input must fail");
        assert_eq!(status.refusal(), Some("metal_function_mtlb_empty"));
        assert!(status.is_args());
        let message = unsafe { CStr::from_ptr(error.as_ptr()) };
        assert_eq!(message.to_str().unwrap(), "vertex MTLB is empty");
    }

    #[test]
    fn nonempty_mtlb_passes_the_pre_device_validation() {
        assert_eq!(
            validate_mtlb(&[1], "fragment", (std::ptr::null_mut(), 0)),
            Ok(())
        );
    }
}
