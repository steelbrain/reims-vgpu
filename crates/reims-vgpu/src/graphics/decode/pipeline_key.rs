//! Pipeline keys: what a made render or compute pipeline's reply carries after its reflection
//! where its create asked for it ([`RGFX_PIPELINE_REFLECT_KEYED`]), so the guest's kext answers
//! an equal create of any process from the reflection it kept.
//!
//! The key names the create's whole message with its object ids taken out, and each library it
//! names by the content key the device holds it under: the host vouches for a library's key only
//! where it checked the bytes against it ([`super::library`]), so a key the host gives always
//! names the libraries' content. A create naming a library the device holds under no key, a
//! mesh create and a compute create that links dynamic libraries are not keyed: their answers
//! hang on objects the key cannot name.

use std::mem::offset_of;

use reims_vgpu_graphics::wire::{
    RGFX_CMD_COMPUTE_PIPELINE_CREATE, RGFX_CMD_RENDER_PIPELINE_CREATE,
    RGFX_COMPUTE_PIPELINE_ANSWER_KEY, RGFX_RENDER_PIPELINE_ANSWER_KEY, RgfxComputePipelineCreate,
    RgfxLinkedFunction, RgfxMsgHeader, RgfxRenderPipelineCreate,
};
use sha2::{Digest, Sha256};

use super::library::LibraryKey;
use super::read;

/// A pipeline create's key ([`RGFX_PIPELINE_REFLECT_KEYED`]).
///
/// [`RGFX_PIPELINE_REFLECT_KEYED`]: reims_vgpu_graphics::wire::RGFX_PIPELINE_REFLECT_KEYED
pub(crate) type PipelineKey = [u8; 32];

/// The bytes a pipeline key's digest starts with.
const KEY_PREFIX: &[u8; 16] = b"rgfx-pipeline-v1";

/// Where the header's `seqno` and the create's `obj_id` lie, both zeroed in the key.
const SEQNO_AT: usize = offset_of!(RgfxMsgHeader, seqno);
const OBJ_ID_AT: usize = size_of::<RgfxMsgHeader>();

/// The key of the render or compute create `message`, its header's `size` bytes, where its flags
/// ask for one and `library_key` gives every library it names a content key (a library id of 0
/// names none, and adds zeros): the SHA-256 of [`KEY_PREFIX`], the message with its `seqno`, its
/// `obj_id` and its library ids zeroed, then the libraries' keys in the order their ids lie
/// ([`library_ids`]). `None` for any other message, one that does not read, and a compute create
/// that links dynamic libraries.
pub(crate) fn pipeline_key(
    message: &[u8],
    library_key: impl Fn(u64) -> Option<LibraryKey>,
) -> Option<PipelineKey> {
    let header = read::<RgfxMsgHeader>(message)?;
    let message = message.get(..usize::try_from(header.size).ok()?)?;
    let ids = library_ids(message)?;
    let mut zeroed = message.to_vec();
    zeroed[SEQNO_AT..SEQNO_AT + 8].fill(0);
    zeroed[OBJ_ID_AT..OBJ_ID_AT + 8].fill(0);
    let mut keys = Vec::with_capacity(ids.len());
    for at in ids {
        let id = u64::from_le_bytes(message.get(at..at + 8)?.try_into().ok()?);
        keys.push(if id == 0 { [0; 32] } else { library_key(id)? });
        zeroed[at..at + 8].fill(0);
    }
    let mut hasher = Sha256::new();
    hasher.update(KEY_PREFIX);
    hasher.update(&zeroed);
    for key in &keys {
        hasher.update(key);
    }
    Some(hasher.finalize().into())
}

/// The byte offsets of the library ids a keyed create names, in the key's order: a compute
/// create's function library, then each linked function's; a render create's vertex and
/// fragment libraries, then each vertex linked function's and each fragment linked function's.
/// `None` where the create does not ask for a key, links dynamic libraries, or a run of linked
/// functions does not lie inside it.
fn library_ids(message: &[u8]) -> Option<Vec<usize>> {
    let header = read::<RgfxMsgHeader>(message)?;
    let mut ids = Vec::new();
    match header.r#type {
        RGFX_CMD_COMPUTE_PIPELINE_CREATE => {
            let m = read::<RgfxComputePipelineCreate>(message)?;
            if m.flags & RGFX_COMPUTE_PIPELINE_ANSWER_KEY == 0 || m.library_count != 0 {
                return None;
            }
            ids.push(offset_of!(
                RgfxComputePipelineCreate,
                function_library_obj_id
            ));
            linked_ids(message, m.linked_offset, m.linked_count, &mut ids)?;
        }
        RGFX_CMD_RENDER_PIPELINE_CREATE => {
            let m = read::<RgfxRenderPipelineCreate>(message)?;
            if m.flags & RGFX_RENDER_PIPELINE_ANSWER_KEY == 0 {
                return None;
            }
            ids.push(offset_of!(RgfxRenderPipelineCreate, vertex_library_obj_id));
            ids.push(offset_of!(
                RgfxRenderPipelineCreate,
                fragment_library_obj_id
            ));
            linked_ids(
                message,
                m.vertex_linked_offset,
                m.vertex_linked_count,
                &mut ids,
            )?;
            linked_ids(
                message,
                m.fragment_linked_offset,
                m.fragment_linked_count,
                &mut ids,
            )?;
        }
        _ => return None,
    }
    Some(ids)
}

/// Appends the library ids' offsets of the run of `count` linked functions at byte `offset` of
/// `message`: `None` where the run does not lie inside it.
fn linked_ids(message: &[u8], offset: u32, count: u32, ids: &mut Vec<usize>) -> Option<()> {
    let record = size_of::<RgfxLinkedFunction>();
    let start = usize::try_from(offset).ok()?;
    let end = record
        .checked_mul(usize::try_from(count).ok()?)?
        .checked_add(start)?;
    if count != 0 && end > message.len() {
        return None;
    }
    let at = offset_of!(RgfxLinkedFunction, library_obj_id);
    ids.extend((0..count as usize).map(|i| start + i * record + at));
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_graphics::wire::{RGFX_COMPUTE_PIPELINE_SUPPORT_ICB, RGFX_LINKED_FUNCTIONS};
    use zerocopy::IntoBytes;

    /// The shared vector's compute create (`guest/bundle/tests/known_pipeline_tests.mm` builds
    /// the same bytes): kernel `k` of library `lib`, asking for a key, as object `obj` at `seqno`.
    fn compute_create(obj: u64, lib: u64, seqno: u64, flags: u32) -> Vec<u8> {
        let head = size_of::<RgfxComputePipelineCreate>();
        let mut m = RgfxComputePipelineCreate::default();
        m.header.r#type = RGFX_CMD_COMPUTE_PIPELINE_CREATE;
        m.header.size = (head + 8) as u32;
        m.header.seqno = seqno;
        m.obj_id = obj;
        m.function_library_obj_id = lib;
        m.name_offset = head as u32;
        m.name_len = 1;
        m.constants_offset = (head + 1) as u32;
        m.linked_offset = (head + 1) as u32;
        m.flags = flags;
        let mut bytes = m.as_bytes().to_vec();
        bytes.extend_from_slice(b"k\0\0\0\0\0\0\0");
        bytes
    }

    /// Library key `i`: its bytes `i, i + 1, ...`.
    fn library(i: u8) -> LibraryKey {
        std::array::from_fn(|b| i.wrapping_add(b as u8))
    }

    fn hex(key: &PipelineKey) -> String {
        key.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The key of the shared vector, which the bundle computes too (`known_pipeline_tests.mm`,
    /// `test_the_shared_key_vector`): the two sides key a create alike.
    #[test]
    fn pipeline_key_of_the_shared_vector() {
        let message = compute_create(
            (1 << 40) | 7,
            (1 << 40) | 2,
            41,
            RGFX_COMPUTE_PIPELINE_ANSWER_KEY,
        );
        let key = pipeline_key(&message, |id| (id == (1 << 40) | 2).then(|| library(0x10)));
        assert_eq!(
            key.map(|k| hex(&k)).as_deref(),
            Some("2be3bb91e78f55b505082cb4c42a647817694dcb5cf47b9974d643f958e991f7")
        );
    }

    /// Equal creates of two processes (other ids, seqnos and contexts) whose libraries hold equal
    /// bytes have one key; a create that differs in any byte else, or names a library of other
    /// bytes, has another.
    #[test]
    fn pipeline_key_names_the_create_and_its_libraries_alone() {
        let flags = RGFX_COMPUTE_PIPELINE_ANSWER_KEY;
        let first = compute_create((1 << 40) | 7, (1 << 40) | 2, 41, flags);
        let second = compute_create((3 << 40) | 9, (3 << 40) | 5, 900, flags);
        let same = |_| Some(library(0x10));
        let a = pipeline_key(&first, same).unwrap();
        assert_eq!(pipeline_key(&second, same), Some(a));
        assert_ne!(pipeline_key(&second, |_| Some(library(0x11))), Some(a));
        let other = compute_create(
            (1 << 40) | 7,
            (1 << 40) | 2,
            41,
            flags | RGFX_COMPUTE_PIPELINE_SUPPORT_ICB,
        );
        assert_ne!(pipeline_key(&other, same), Some(a));
    }

    /// No key where the create does not ask for one, a library it names has no key, it links
    /// dynamic libraries, or its linked functions lie outside it.
    #[test]
    fn pipeline_key_none_where_the_create_cannot_be_keyed() {
        let some = |_| Some(library(1));
        let unasked = compute_create(7, 2, 1, 0);
        assert_eq!(pipeline_key(&unasked, some), None);
        let asked = compute_create(7, 2, 1, RGFX_COMPUTE_PIPELINE_ANSWER_KEY);
        assert_eq!(pipeline_key(&asked, |_| None), None);
        let mut dynamic = asked.clone();
        let at = offset_of!(RgfxComputePipelineCreate, library_count);
        dynamic[at..at + 4].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(pipeline_key(&dynamic, some), None);
        let mut outside = asked.clone();
        let at = offset_of!(RgfxComputePipelineCreate, linked_count);
        outside[at..at + 4].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(pipeline_key(&outside, some), None);
    }

    /// A linked function's library is keyed after the kernel's, by its own key.
    #[test]
    fn pipeline_key_names_linked_functions_libraries() {
        let head = size_of::<RgfxComputePipelineCreate>();
        let record = size_of::<RgfxLinkedFunction>();
        let message = |linked_lib: u64| {
            let mut m = RgfxComputePipelineCreate::default();
            m.header.r#type = RGFX_CMD_COMPUTE_PIPELINE_CREATE;
            m.header.size = (head + 8 + record) as u32;
            m.obj_id = 7;
            m.function_library_obj_id = 2;
            m.name_offset = head as u32;
            m.name_len = 1;
            m.linked_offset = (head + 8) as u32;
            m.linked_count = 1;
            m.flags = RGFX_COMPUTE_PIPELINE_ANSWER_KEY;
            let linked = RgfxLinkedFunction {
                library_obj_id: linked_lib,
                name_offset: head as u32,
                name_len: 1,
                list: RGFX_LINKED_FUNCTIONS,
                ..RgfxLinkedFunction::default()
            };
            let mut bytes = m.as_bytes().to_vec();
            bytes.extend_from_slice(b"k\0\0\0\0\0\0\0");
            bytes.extend_from_slice(linked.as_bytes());
            bytes
        };
        let keys = |id: u64| Some(library(id as u8));
        // Libraries 3 and 4 hold other bytes; the linked library's id itself is not keyed.
        let three = pipeline_key(&message(3), keys).unwrap();
        assert_ne!(pipeline_key(&message(4), keys), Some(three));
        assert_eq!(
            pipeline_key(&message(5), |id| Some(library(if id == 5 {
                3
            } else {
                id as u8
            }))),
            Some(three)
        );
        assert_eq!(
            pipeline_key(&message(3), |id| (id == 2).then(|| library(2))),
            None
        );
    }
}
