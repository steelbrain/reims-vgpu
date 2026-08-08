//! Typed Metal-backend status and C error-buffer helpers.

use crate::backend::metal::abi::{REIMS_VGPU_ERR_ARGS, REIMS_VGPU_ERR_EXECUTE, REIMS_VGPU_OK};
use crate::observe::Refusal;
use std::os::raw::c_char;

/// Maximum structured fields carried from a Metal check to its runtime
/// emission boundary.
///
/// This is deliberately fixed-size so [`Status`] stays `Copy`: the render and
/// compute rails pass it through several leaf helpers before the runtime turns
/// it into `EncodeStatus` / `ComputeStatus`. Six covers the widest current
/// check (geometry + expected/actual length) without allocating on success.
const MAX_STATUS_FIELDS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FieldValue {
    Unsigned(u64),
    Signed(i64),
    Text(&'static str),
}

impl std::fmt::Display for FieldValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsigned(value) => value.fmt(f),
            Self::Signed(value) => value.fmt(f),
            Self::Text(value) => value.fmt(f),
        }
    }
}

macro_rules! unsigned_field_value {
    ($($ty:ty),+ $(,)?) => {
        $(impl From<$ty> for FieldValue {
            fn from(value: $ty) -> Self {
                Self::Unsigned(value as u64)
            }
        })+
    };
}

macro_rules! signed_field_value {
    ($($ty:ty),+ $(,)?) => {
        $(impl From<$ty> for FieldValue {
            fn from(value: $ty) -> Self {
                Self::Signed(value as i64)
            }
        })+
    };
}

unsigned_field_value!(u8, u16, u32, u64, usize);
signed_field_value!(i8, i16, i32, i64, isize);

impl From<bool> for FieldValue {
    fn from(value: bool) -> Self {
        Self::Unsigned(value as u64)
    }
}

impl From<&'static str> for FieldValue {
    fn from(value: &'static str) -> Self {
        Self::Text(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StatusField {
    key: &'static str,
    value: FieldValue,
}

/// Result of a direct-Metal backend operation.
///
/// The C ABI still observes the historical integer code, but Rust never
/// constructs a payload-free `ARGS` / `EXECUTE` value. Every refusal carries
/// the registered slug of the exact check plus its numeric protocol facts.
/// `Ok` remains in the same status type so [`Refusal::refusal`] makes the
/// success-vs-decline judgement exhaustive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StatusClass {
    Ok,
    Args,
    Execute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status {
    class: StatusClass,
    slug: Option<&'static str>,
    fields: [Option<StatusField>; MAX_STATUS_FIELDS],
}

impl Status {
    pub const OK: Self = Self {
        class: StatusClass::Ok,
        slug: None,
        fields: [None; MAX_STATUS_FIELDS],
    };

    pub fn args(slug: &'static str) -> Self {
        Self {
            class: StatusClass::Args,
            slug: Some(slug),
            fields: [None; MAX_STATUS_FIELDS],
        }
    }

    pub fn execute(slug: &'static str) -> Self {
        Self {
            class: StatusClass::Execute,
            slug: Some(slug),
            fields: [None; MAX_STATUS_FIELDS],
        }
    }

    /// Add one load-bearing numeric or static-token fact to the refusal.
    ///
    /// Overflow is an authoring defect, not a runtime fallback: every current
    /// constructor is kept within [`MAX_STATUS_FIELDS`] and the unit test pins
    /// that a seventh field cannot silently replace an earlier one.
    pub(crate) fn field(mut self, key: &'static str, value: impl Into<FieldValue>) -> Self {
        if self.class == StatusClass::Ok {
            return self;
        }
        let Some(slot) = self.fields.iter_mut().find(|slot| slot.is_none()) else {
            panic!("Metal Status field capacity exceeded for {key}");
        };
        *slot = Some(StatusField {
            key,
            value: value.into(),
        });
        self
    }

    pub fn code(self) -> i32 {
        match self.class {
            StatusClass::Ok => REIMS_VGPU_OK,
            StatusClass::Args => REIMS_VGPU_ERR_ARGS,
            StatusClass::Execute => REIMS_VGPU_ERR_EXECUTE,
        }
    }

    pub fn is_ok(self) -> bool {
        self.class == StatusClass::Ok
    }

    pub fn is_args(self) -> bool {
        self.class == StatusClass::Args
    }
}

impl Refusal for Status {
    fn refusal(&self) -> Option<&'static str> {
        self.slug
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let class = match self.class {
            StatusClass::Ok => return Vec::new(),
            StatusClass::Args => "args",
            StatusClass::Execute => "execute",
        };
        let mut out = Vec::with_capacity(1 + MAX_STATUS_FIELDS);
        out.push(("class", class.to_string()));
        out.extend(
            self.fields
                .iter()
                .flatten()
                .map(|field| (field.key, field.value.to_string())),
        );
        out
    }
}

/// Copy `msg` into the shim's error buffer, NUL-terminated and truncated to fit.
///
/// # Safety
///
/// `err` must be null, or valid for writes of `err_cap` bytes. Null and a zero
/// capacity are both checked here, so the caller's obligation is only that a
/// non-null pointer really has the capacity it claims — which is the contract
/// `reims_vgpu_qemu_abi.h` states for every `(char *err, size_t err_cap)` pair
/// crossing the boundary.
pub unsafe fn write_err(err: *mut c_char, err_cap: usize, msg: &str) {
    // SAFETY: forwarded unchanged — the caller's promise about `err` and
    // `err_cap` is exactly what `write_c_str` asks for.
    unsafe { crate::qemu::cstr::write_c_str(err, err_cap, msg) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Emit;

    #[test]
    fn status_preserves_abi_class_reason_and_structured_fields() {
        let status = Status::args("metal_test_bad_extent")
            .field("width", 19u32)
            .field("offset", -4i32);
        assert_eq!(status.code(), REIMS_VGPU_ERR_ARGS);
        assert!(status.is_args());
        assert_eq!(status.refusal(), Some("metal_test_bad_extent"));
        assert_eq!(
            Emit::refusal("metal_backend", &status)
                .expect("a refusal must render")
                .render(),
            "metal_backend reason=metal_test_bad_extent class=args width=19 offset=-4"
        );
    }

    #[test]
    fn ok_cannot_be_emitted_as_a_refusal() {
        assert_eq!(Status::OK.code(), REIMS_VGPU_OK);
        assert!(Status::OK.is_ok());
        assert!(Emit::refusal("metal_backend", &Status::OK).is_none());
    }

    #[test]
    #[should_panic(expected = "Metal Status field capacity exceeded")]
    fn structured_field_overflow_is_not_silent() {
        let _ = Status::execute("metal_test_field_overflow")
            .field("a", 1u8)
            .field("b", 2u8)
            .field("c", 3u8)
            .field("d", 4u8)
            .field("e", 5u8)
            .field("f", 6u8)
            .field("g", 7u8);
    }
}
