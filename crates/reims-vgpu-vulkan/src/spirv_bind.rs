//! Reflection projection and final-module safety checks.
//!
//! metal2vulkan owns descriptor layout, storage-format specialization, sampler
//! specialization, and source ABI layout. This module projects its reflection
//! into Reims semantic types. The remaining SPIR-V inspection answers facts not
//! present in reflection: static descriptor use, storage-image read/write use,
//! required Vulkan capabilities, and validator acceptance. It never mutates a
//! translated module.

/// SPIR-V `OpDecorate` opcode.
const OP_DECORATE: u16 = 71;
/// The declarations that name a variable id without referencing it, which
/// [`descriptor_static_use`] must skip. `OpEntryPoint` is the one that matters:
/// from SPIR-V 1.4 its interface list carries every global variable in the
/// module, so counting it would make every declared descriptor read as used.
const OP_NAME: u16 = 5;
const OP_MEMBER_NAME: u16 = 6;
const OP_ENTRY_POINT: u16 = 15;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_DECORATE_ID: u16 = 332;
const OP_DECORATE_STRING: u16 = 5632;
const OP_MEMBER_DECORATE_STRING: u16 = 5633;
const OP_TYPE_IMAGE: u16 = 25;
#[cfg(any(test, feature = "test-fixtures"))]
const OP_TYPE_SAMPLER: u16 = 26;
#[cfg(any(test, feature = "test-fixtures"))]
const OP_TYPE_POINTER: u16 = 32;
const OP_VARIABLE: u16 = 59;
const OP_FUNCTION_CALL: u16 = 57;
const OP_IMAGE_TEXEL_POINTER: u16 = 60;
const OP_LOAD: u16 = 61;
const OP_COPY_OBJECT: u16 = 83;
const OP_IMAGE_READ: u16 = 98;
const OP_IMAGE_WRITE: u16 = 99;
const OP_IMAGE_QUERY_FORMAT: u16 = 101;
const OP_IMAGE_QUERY_ORDER: u16 = 102;
const OP_IMAGE_QUERY_SIZE_LOD: u16 = 103;
const OP_IMAGE_QUERY_SIZE: u16 = 104;
const OP_IMAGE_QUERY_LOD: u16 = 105;
const OP_IMAGE_QUERY_LEVELS: u16 = 106;
const OP_IMAGE_QUERY_SAMPLES: u16 = 107;
const OP_SELECT: u16 = 169;
const OP_PHI: u16 = 245;
const OP_RETURN_VALUE: u16 = 254;
// The three storage-image capability numbers, from SPIR-V's `Capability` enum.
// These are read-only classifications of the translator's final module. Reims
// never injects them: metal2vulkan owns capability declaration, while the host
// gate below verifies that the enabled Vulkan features cover what it emitted.
/// SPIR-V `Capability StorageImageWriteWithoutFormat` (writes to an `Unknown`
/// format storage image). Paired with the Vulkan feature of the same name.
const CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT: u32 = 56;
/// SPIR-V `Capability StorageImageReadWithoutFormat` (reads from an `Unknown`
/// format storage image). Paired with the Vulkan feature of the same name.
const CAPABILITY_STORAGE_IMAGE_READ_WITHOUT_FORMAT: u32 = 55;
/// SPIR-V `Capability StorageImageExtendedFormats`, required by any storage
/// image whose declared format is outside the core set. Paired with the Vulkan
/// feature `shaderStorageImageExtendedFormats`.
const CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS: u32 = 49;

// The three are distinct and none is `Shader` (1), which is the one every module
// already declares. A collision here would make a splice a silent no-op.
const _: () = assert!(
    CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT != CAPABILITY_STORAGE_IMAGE_READ_WITHOUT_FORMAT
        && CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT
            != CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS
        && CAPABILITY_STORAGE_IMAGE_READ_WITHOUT_FORMAT
            != CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS
);
/// `OpTypeImage`'s `Sampled` operand value for "used as a storage image".
///
/// Operand 7 of `OpTypeImage`; 1 means sampled, 2 means storage, 0 means either.
/// Only a storage image's format operand carries a capability requirement, so
/// this is what keeps a sampled image with an exotic format from being read as
/// one.
const IMAGE_SAMPLED_STORAGE: u32 = 2;
/// SPIR-V `Decoration Binding`.
const DECORATION_BINDING: u32 = 33;
const HEADER_WORDS: usize = 5;
/// First binding of the translator's sampled-resource band, and therefore the
/// exclusive end of its buffer band — the two are the same number because the
/// bands abut.
///
/// A second constant used to spell this `BUFFER_BINDING_LIMIT: u32 = 32` on the
/// line above, which read as Apple's buffer bind limit and was not one: that is
/// `reims_vgpu_wire::ops::bind_limit::BUFFER`, and it is **31**. Nothing here
/// bounds how many buffers a guest may bind; this only says where one band stops
/// and the next starts, which is a fact about our own translated layout.
const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;
/// SPIR-V storage classes used by buffer descriptors emitted by the translator.
const STORAGE_CLASS_UNIFORM: u32 = 2;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

/// SPIR-V `StorageClass PushConstant`.
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;

/// Whether this module declares a push-constant variable.
///
/// The one thing a consumer must know without consulting reflection: a module
/// that reads push constants cannot execute correctly under a pipeline layout
/// that exposes none. Reflection remains the authority on *where* the kernel
/// grid sits — this answers only whether something is there to satisfy, so a
/// prepared module that lost its range can be refused by name instead of
/// reading whatever the driver leaves in that storage.
///
/// A guard reading undefined push constants is not a subtle failure: zeros cull
/// every invocation, the dispatch writes nothing, and no counter moves.
#[must_use]
pub fn declares_push_constants(words: &[u32]) -> bool {
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        // OpVariable: result type, result id, storage class.
        if opcode == OP_VARIABLE && word_count >= 4 && words[i + 3] == STORAGE_CLASS_PUSH_CONSTANT {
            return true;
        }
        i += word_count;
    }
    false
}

// Descriptor numbering is selected through metal2vulkan transform options before
// SPIR-V is emitted. These defaults remain public for tests and fixed engine
// facilities; executable resource locations come from ShaderReflection and
// each reflected DescriptorLocation.
pub use metal2vulkan::reflect::{
    COLOR_INPUT_BINDING_BASE, SAMPLER_BINDING_BASE, TEXTURE_BINDING_BASE,
};

/// Widest ordinary descriptor class in the translator's default set-zero map.
pub const MAX_DESCRIPTOR_CLASS_BINDINGS: u32 = SAMPLER_BINDING_BASE - TEXTURE_BINDING_BASE;

const _: () = assert!(
    TEXTURE_BINDING_BASE + reims_vgpu_wire::ops::bind_limit::TEXTURE == SAMPLER_BINDING_BASE
);
const _: () = assert!(
    SAMPLER_BINDING_BASE + reims_vgpu_wire::ops::bind_limit::SAMPLER <= COLOR_INPUT_BINDING_BASE
);
pub use reims_vgpu_core::{
    DescriptorUse, ImageAccess, ReflectedBufferAccess, ReflectedComputeTexture,
    ReflectedSampledKind, ReflectedSamplerDescriptor, ReflectedTextureAccess,
    ReflectedTextureDescriptor, SampledImageKind, StorageImageAccess,
};

#[derive(Clone, Copy)]
struct Instruction {
    opcode: u16,
    word_count: usize,
    at: usize,
}

/// Decode every instruction header in a module body, or `None` if the stream is
/// not one.
///
/// # Why the scans below take the output of this rather than `&[u32]`
///
/// A `&[Instruction]` only exists if the whole stream walked cleanly, so a scan
/// holding one may index `words[at + n]` for any `n < word_count` without a
/// bounds check, and cannot spin. That is not a new rule — it is the rule the
/// provenance scans have always run under, taken from `descriptor_root` having
/// been called first and having returned `Some`. Nothing recorded the
/// dependency: `propagate_derived` and both escape scans re-walked the raw
/// `&[u32]` with no guard of their own, and were correct only because that one
/// guard, in another function, had already rejected every stream that would
/// break them.
///
/// The cost of leaving it implicit is not a style point. `words[at + 1 ..
/// at + word_count]` on a truncated final instruction panics, and a zero word
/// count makes `i += word_count` spin forever — a device abort and a wedged
/// guest, from a reflector whose contract is to fail closed. Making the
/// entitlement a type means a scan cannot be reached without it.
///
/// Decoding once also removes the re-decode from `propagate_derived`, which
/// re-walks to a fixpoint and used to re-split every header on every pass.
///
/// # What a driven boot says about it, and what it cannot
///
/// Driven x86/PCI boot on the consolidated walk (web-content probe, 10 captures):
/// `linux_m2v_async` 160, so 160 real guest shaders were translated and reflected
/// through it; `m2v_reflect_malformed` and `spirv_reloc_unclassified_binding` both
/// absent; 10 of 10 regions measured their declared colour. Against the boot
/// before it (158 translations) the fail-channel reason ranking gained no new
/// entry. That is the regression evidence that matters here — the reflectors'
/// answers for well-formed modules are what must not move, and 160 shaders is a
/// wider sample than the unit tests reach.
///
/// It says **nothing about the malformed cases**, and no boot can. This SPIR-V is
/// the translator's own output rather than anything a guest sends, so a guest
/// cannot drive a stream that fails this parse; only a translator bug produces
/// one, which is exactly why the failure had to stop being a panic. The coverage
/// for those is `tests::a_module_that_does_not_walk_reflects_nothing`.
fn instructions(words: &[u32]) -> Option<Vec<Instruction>> {
    if words.len() < HEADER_WORDS {
        return None;
    }
    let mut out = Vec::new();
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        if word_count == 0 || i + word_count > words.len() {
            return None;
        }
        out.push(Instruction {
            opcode: (words[i] & 0xffff) as u16,
            word_count,
            at: i,
        });
        i += word_count;
    }
    Some(out)
}

/// The one descriptor variable declaring `wanted_binding` in `storage_class`,
/// with the module's id bound.
///
/// `OpDecorate Binding` and `OpVariable` are the only two declarations either
/// provenance question in this module needs, and both ask for exactly one
/// match: two variables sharing a binding means neither can be reflected, which
/// is `Root::Ambiguous`.
///
/// `None` is a module this reflector cannot parse at all — a zero id bound, or
/// no variable on that binding. A stream whose instructions do not walk cleanly
/// never reaches here at all: [`instructions`] rejects it, and its `None` is
/// this one. Every case must fail closed rather than reflect a guess.
enum Root {
    One { id: usize, bound: usize },
    Ambiguous,
}

fn descriptor_root(
    words: &[u32],
    instrs: &[Instruction],
    wanted_binding: u32,
    storage_class: u32,
) -> Option<Root> {
    let bound = *words.get(3)? as usize;
    if bound == 0 {
        return None;
    }
    let mut bindings = vec![None; bound];
    let mut storage = vec![None; bound];
    for &Instruction {
        opcode,
        word_count,
        at: i,
    } in instrs
    {
        match opcode {
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_BINDING => {
                let id = words[i + 1] as usize;
                if id < bound {
                    bindings[id] = Some(words[i + 3]);
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                let id = words[i + 2] as usize;
                if id < bound {
                    storage[id] = Some(words[i + 3]);
                }
            }
            _ => {}
        }
    }
    let mut roots = bindings.iter().enumerate().filter_map(|(id, binding)| {
        (*binding == Some(wanted_binding) && storage[id] == Some(storage_class)).then_some(id)
    });
    let id = roots.next()?;
    Some(if roots.next().is_some() {
        Root::Ambiguous
    } else {
        Root::One { id, bound }
    })
}

/// Whether the module actually declares a descriptor variable at `binding`.
///
/// The reflection is derived from the AIR entry point's signature, not from the
/// translated module, so a Metal function that names `[[texture(n)]]` and never
/// samples it produces a reflection entry for a descriptor the SPIR-V may not
/// declare at all. The render path's declared-but-unprovided scan reported those
/// as gaps, which is a false alarm: nothing references the binding, so nothing
/// is unbound.
///
/// This is the question that separates the two, asked of the module rather than
/// of the signature. `false` means the reflection named a resource the shader
/// does not carry, and there is nothing to bind. `true` means the module has a
/// variable on that binding and a draw that leaves it out of the descriptor
/// layout is building a pipeline whose module references a binding its layout
/// does not contain.
///
/// Deliberately narrower than "is it sampled": a declared-and-unused variable
/// still participates in layout consistency, so declaration is the right bar for
/// the question the caller is asking.
pub fn declares_descriptor(words: &[u32], binding: u32) -> bool {
    let Some(instrs) = instructions(words) else {
        return false;
    };
    descriptor_root(words, &instrs, binding, STORAGE_CLASS_UNIFORM_CONSTANT).is_some()
}

/// Whether a declared descriptor is *statically used*, which is the bar Vulkan
/// actually sets.
///
/// [`declares_descriptor`] answers the weaker question, and the two are not the
/// same rule. Vulkan requires the pipeline layout to contain a descriptor for
/// every resource the shader **statically uses**; a variable that is declared
/// and never referenced is legal to omit. So a draw whose module declares a
/// binding its layout does not contain is a specification violation only if this
/// says [`DescriptorUse::Used`], and a fail line that does not separate the two
/// is reporting a population it cannot tell apart.
///
/// "Statically used" here is the spec's own wording — the variable is referenced
/// by an instruction — and the test is the direct one: does the root id appear as
/// an operand anywhere outside the declarations that necessarily name it?
///
/// The exclusion list is the whole subtlety. `OpVariable` declares the id,
/// `OpDecorate`/`OpMemberDecorate` and their `Id`/`String` forms carry its
/// binding, `OpName`/`OpMemberName` carry its debug name, and from SPIR-V 1.4
/// **`OpEntryPoint` lists every global variable in its interface** whether or not
/// the body touches it. Counting any of those as a reference would make every
/// declared descriptor read as used, which is the failure that looks like
/// thoroughness. Everything else counts, including an `OpAccessChain` or an
/// `OpImageTexelPointer` that never gets loaded, because those reference the
/// variable and the spec asks about references rather than about loads.
pub fn descriptor_static_use(words: &[u32], binding: u32) -> DescriptorUse {
    descriptor_static_use_in_storage_classes(
        words,
        binding,
        &[
            STORAGE_CLASS_UNIFORM_CONSTANT,
            STORAGE_CLASS_UNIFORM,
            STORAGE_CLASS_STORAGE_BUFFER,
        ],
    )
}

/// Static executable use of one translated buffer descriptor.
///
/// Buffer reflection describes the source function's argument table. This
/// answers the independent executable-side question used to validate an
/// `Unused` answer before guest bytes may be substituted. Both buffer storage
/// classes are admitted because the selected SPIR-V environment decides which
/// representation a storage buffer uses.
pub fn buffer_descriptor_static_use(words: &[u32], binding: u32) -> DescriptorUse {
    descriptor_static_use_in_storage_classes(
        words,
        binding,
        &[STORAGE_CLASS_UNIFORM, STORAGE_CLASS_STORAGE_BUFFER],
    )
}

fn descriptor_static_use_in_storage_classes(
    words: &[u32],
    binding: u32,
    storage_classes: &[u32],
) -> DescriptorUse {
    let Some(instrs) = instructions(words) else {
        return DescriptorUse::NotDeclared;
    };
    let mut root = None;
    for storage_class in storage_classes {
        match descriptor_root(words, &instrs, binding, *storage_class) {
            None => {}
            Some(Root::Ambiguous) => return DescriptorUse::Ambiguous,
            Some(Root::One { id, .. }) if root.replace(id as u32).is_some() => {
                return DescriptorUse::Ambiguous;
            }
            Some(Root::One { .. }) => {}
        }
    }
    let Some(root) = root else {
        return DescriptorUse::NotDeclared;
    };
    for &Instruction {
        opcode,
        word_count,
        at: i,
    } in &instrs
    {
        if matches!(
            opcode,
            OP_VARIABLE
                | OP_DECORATE
                | OP_MEMBER_DECORATE
                | OP_DECORATE_ID
                | OP_DECORATE_STRING
                | OP_MEMBER_DECORATE_STRING
                | OP_NAME
                | OP_MEMBER_NAME
                | OP_ENTRY_POINT
        ) {
            continue;
        }
        // Word 0 is the opcode/length header; every operand after it is a
        // candidate reference. A result id cannot collide with the root, because
        // the root's own result id belongs to the `OpVariable` skipped above.
        if words[i + 1..i + word_count].contains(&root) {
            return DescriptorUse::Used;
        }
    }
    DescriptorUse::DeclaredUnused
}

/// Mark every id whose value derives from an already-marked id, to a fixpoint.
///
/// Both provenance questions here have the same shape — seed a set of ids, then
/// re-walk the instruction stream marking any result built from a marked
/// operand, until a pass changes nothing — and differ only in which opcodes
/// propagate. `propagates` is that difference: given an instruction, it answers
/// whether that instruction's result derives from something already marked.
///
/// The opcodes handled here are the ones that merge or rename an SSA value
/// without regard to what kind of value it is: `OpCopyObject` renames,
/// `OpSelect` and `OpPhi` merge. Provenance flows through all three for a
/// pointer and for an image alike, so they are not the caller's business.
///
/// The `word_count >= 3` guard is what makes a result id exist to mark; an
/// instruction shorter than that has no result operand.
fn propagate_derived(
    words: &[u32],
    instrs: &[Instruction],
    bound: usize,
    seed: Option<usize>,
    propagates: impl Fn(u16, usize, usize, &[bool]) -> bool,
) -> Vec<bool> {
    let mut derived = vec![false; bound];
    if let Some(root) = seed {
        derived[root] = true;
    }
    loop {
        let mut changed = false;
        for &Instruction {
            opcode,
            word_count,
            at: i,
        } in instrs
        {
            let marked = |id: u32| derived.get(id as usize).copied() == Some(true);
            let result_from = match opcode {
                OP_COPY_OBJECT if word_count >= 4 => marked(words[i + 3]),
                OP_SELECT if word_count >= 6 => marked(words[i + 4]) || marked(words[i + 5]),
                OP_PHI if word_count >= 5 => (i + 3..i + word_count)
                    .step_by(2)
                    .any(|at| marked(words[at])),
                _ => propagates(opcode, word_count, i, &derived),
            };
            if result_from && word_count >= 3 {
                let result = words[i + 2] as usize;
                if result < bound && !derived[result] {
                    derived[result] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    derived
}

/// Reflect whether a storage image consumes its pre-dispatch contents.
///
/// This follows `OpLoad` from the descriptor variable through the SSA image
/// operations that preserve identity. `OpImageRead` and `OpImageWrite` then
/// provide the content-access contract. Queries do not consume texels;
/// pointer/image escapes fail closed as [`StorageImageAccess::Unknown`].
/// The tracked set is image *values*, so the root variable is deliberately not
/// seeded: `OpLoad` from it is what produces the first image value. Seeding the
/// variable id would also be unsound here, because the escape scan below ends in
/// a catch-all — `OpDecorate root Binding N` and `OpEntryPoint`'s interface list
/// both name the variable, and either would then read as an escape and force
/// every storage image to `Unknown`. `buffer_access` seeds its root only because
/// its scan enumerates instead.
pub fn storage_image_access(words: &[u32], wanted_binding: u32) -> Option<StorageImageAccess> {
    let instrs = instructions(words)?;
    let (root, bound) = match descriptor_root(
        words,
        &instrs,
        wanted_binding,
        STORAGE_CLASS_UNIFORM_CONSTANT,
    )? {
        Root::One { id, bound } => (id, bound),
        Root::Ambiguous => return Some(StorageImageAccess::AmbiguousBinding),
    };

    let derived = propagate_derived(
        words,
        &instrs,
        bound,
        None,
        |opcode, word_count, i, _derived| {
            opcode == OP_LOAD && word_count >= 4 && words[i + 3] as usize == root
        },
    );

    let is_derived = |id: u32| derived.get(id as usize).copied() == Some(true);
    let mut read = false;
    let mut write = false;
    let mut unknown = false;
    for &Instruction {
        opcode,
        word_count,
        at: i,
    } in &instrs
    {
        match opcode {
            OP_IMAGE_READ if word_count >= 5 && is_derived(words[i + 3]) => read = true,
            OP_IMAGE_WRITE if word_count >= 4 && is_derived(words[i + 1]) => write = true,
            OP_IMAGE_QUERY_FORMAT
            | OP_IMAGE_QUERY_ORDER
            | OP_IMAGE_QUERY_SIZE_LOD
            | OP_IMAGE_QUERY_SIZE
            | OP_IMAGE_QUERY_LOD
            | OP_IMAGE_QUERY_LEVELS
            | OP_IMAGE_QUERY_SAMPLES => {
                // Shape/format queries do not consume image contents.
            }
            OP_IMAGE_TEXEL_POINTER if word_count >= 5 && is_derived(words[i + 3]) => unknown = true,
            OP_FUNCTION_CALL
                if word_count >= 5
                    && words[i + 4..i + word_count].iter().copied().any(is_derived) =>
            {
                unknown = true
            }
            OP_RETURN_VALUE if word_count >= 2 && is_derived(words[i + 1]) => unknown = true,
            OP_LOAD | OP_COPY_OBJECT | OP_SELECT | OP_PHI => {}
            _ if words[i + 1..i + word_count].iter().copied().any(is_derived) => unknown = true,
            _ => {}
        }
    }
    Some(if unknown || (!read && !write) {
        StorageImageAccess::Unknown
    } else if read && write {
        StorageImageAccess::ReadWrite
    } else if read {
        StorageImageAccess::ReadOnly
    } else {
        StorageImageAccess::WriteOnly
    })
}

/// What a validator said about a module this device was about to hand a driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpirvValidation {
    /// The module is valid as far as the validator can tell — including the
    /// case where no validator is installed, which is not evidence of anything
    /// and must not become a refusal.
    Accepted,
    /// The validator rejected it. Carries its first line, which names the
    /// instruction.
    Rejected(String),
}

/// Ask a validator whether `words` is a module a driver can be given.
///
/// # Why this is not the driver's job
///
/// It is, and that is the problem. A Vulkan driver is entitled to undefined
/// behaviour on invalid SPIR-V, and NVIDIA's takes it: a module `metal2vulkan`
/// produced for a macOS 14 guest's compositor kernel segmentation-faults inside
/// its SPIR-V compiler and ends the QEMU process — the guest, the device and
/// every other rail with it. A guest authors its own kernels, so "the module
/// was invalid" has to be a declined dispatch and can never be a dead VM.
///
/// `metal2vulkan` validates every candidate before returning it. Reims validates
/// the unchanged final bytes again at the driver boundary as an independent
/// deployment guard: a bad module must become a typed refusal rather than reach
/// undefined driver behaviour.
///
/// # When there is no validator
///
/// The check is external (`spirv-val`, from SPIRV-Tools, which
/// `vm/boot-x86.sh` already requires and which `metal2vulkan` spawns during
/// translation). A host without it gets [`SpirvValidation::Accepted`] and one
/// fail-log line: an absent instrument is not a verdict, and refusing every
/// dispatch because a developer tool is missing would be the widening this
/// device is not allowed to do in either direction.
pub fn validate(words: &[u32]) -> SpirvValidation {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    // A private directory per call, because `spirv_val_bytes` writes a **fixed**
    // file name inside whatever directory it is given. Handed the shared
    // `/tmp`, two concurrent validations — this device's, or one of
    // `metal2vulkan`'s own async translations — write and delete the same path,
    // and the loser validates bytes it did not produce or finds no file at all.
    // Measured: three modules on a working macos-13 boot rejected that way,
    // which would have cost the guest three shaders to fix a crash on a
    // different rail.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("reims-vgpu-spirv-val-{}-{seq}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        if reims_vgpu_observe::first_sight("spirv_val_no_tmp", 0) {
            reims_vgpu_observe::fail(format!(
                "spirv_validate reason=validator_unavailable detail={e}"
            ));
        }
        return SpirvValidation::Accepted;
    }
    let verdict = metal2vulkan::tools::spirv_val_bytes(&bytes, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    match verdict {
        Ok(()) => SpirvValidation::Accepted,
        Err(why) => {
            // A missing or unrunnable tool reads as an error from the same call
            // as a rejected module, and the two must not be confused: one is a
            // module that would take the process down, the other is a laptop
            // without SPIRV-Tools installed.
            if why.contains("No such file") || why.contains("spawn") || why.contains("not found") {
                if reims_vgpu_observe::first_sight("spirv_val_absent", 0) {
                    reims_vgpu_observe::fail(format!(
                        "spirv_validate reason=validator_unavailable detail={}",
                        why.lines().next().unwrap_or("")
                    ));
                }
                return SpirvValidation::Accepted;
            }
            // The whole message on one line. Its first line is the wrapper's
            // own "spirv-val failed:" and the validator's diagnosis — the part
            // that names the instruction — is on the ones after it, so keeping
            // only the first is how a rejection reads as having no reason.
            let flattened: Vec<&str> = why
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect();
            SpirvValidation::Rejected(flattened.join(" | "))
        }
    }
}

/// Vulkan feature requirements observed in the translator's final module.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequiredImageCapabilities {
    /// A storage image declares a format outside SPIR-V's core set.
    pub extended_formats: bool,
    /// An `OpImageWrite` targets a storage image whose format is `Unknown`.
    pub write_without_format: bool,
    /// An `OpImageRead` targets a storage image whose format is `Unknown`.
    pub read_without_format: bool,
}

impl RequiredImageCapabilities {
    pub fn any(&self) -> bool {
        self.extended_formats || self.write_without_format || self.read_without_format
    }
}

/// Whether a SPIR-V storage-image format needs `StorageImageExtendedFormats`.
///
/// The core set — usable with `Shader` alone — is `Unknown`, the four `Rgba*32/16`
/// and `R32` float/int/uint widths, `Rgba8`/`Rgba8Snorm`, and the 8-bit `Rgba8i`
/// / `Rgba8ui` forms. Anything narrower than four channels below 32 bits, which
/// is where `Rg16f`, `R16f`, `Rg8` and `R8` all live, is extended.
///
/// Written as an explicit list of the core values rather than a range, because
/// the core set is not contiguous in the enum and a `<=` test would silently
/// admit the two-channel formats sitting between the four-channel ones.
fn storage_format_is_extended(raw: u32) -> bool {
    !matches!(
        raw,
        0     // Unknown
        | 1   // Rgba32f
        | 2   // Rgba16f
        | 3   // R32f
        | 4   // Rgba8
        | 5   // Rgba8Snorm
        | 21  // Rgba32i
        | 22  // Rgba16i
        | 23  // Rgba8i
        | 24  // R32i
        | 30  // Rgba32ui
        | 31  // Rgba16ui
        | 32  // Rgba8ui
        | 33 // R32ui
    )
}

/// Derive which storage-image capabilities a module needs, from the module.
///
/// # Why this is derived and not tracked
///
/// The device used to add `StorageImageWriteWithoutFormat` when *it* had
/// retargeted a binding to `Unknown`, on the reasoning that this was the only
/// way a module could come to need it. That is a claim about provenance, and it
/// was wrong in the direction that reads as careful: the translator emits
/// `Unknown`-format storage images of its own accord, and those modules reached
/// the validator without the capability and were **rejected**, losing the
/// dispatch. Both x86 rails measured here lose compute work to exactly that —
/// including the rail that otherwise renders correctly, which is why it went
/// unnoticed.
///
/// Asking the module what it contains cannot go stale that way. A new source of
/// extended-format or format-less storage images — another translator version, a
/// new specialization, a guest that simply emits one — is covered without anyone
/// remembering to add a case, which is the property the provenance test did not
/// have.
///
/// # Why it over-approximates on purpose
///
/// The format requirement is exact: a storage image's declared format is right
/// there in its `OpTypeImage`. The *format-less* requirement is not attributed
/// to a specific write. This walk asks "does the module declare an
/// `Unknown`-format storage image, and does it contain any `OpImageWrite`",
/// rather than proving that some particular write targets that image.
///
/// That is deliberate, and the first version of this function did it the other
/// way. Resolving the write's image operand back through
/// `OpTypePointer`/`OpVariable`/`OpLoad` attributed most writes correctly and
/// **missed three modules per boot** — the def-use chain reached them through a
/// shape the walk did not follow, and the only symptom was the same rejection
/// the function existed to prevent. Every shape it misses fails closed in the
/// direction that loses guest work, and there is no bound on how many shapes a
/// translator can emit.
///
/// Over-approximating cannot fail that way, and it costs nothing real:
/// declaring a capability a module does not use is valid SPIR-V, and the caller
/// has already established that the device enabled the matching feature. The
/// exchange is a capability that is occasionally redundant for a class of bug
/// that is silent and recurring.
///
/// The walk is structural throughout and never looks at debug names or guest
/// object ids.
pub fn required_image_capabilities(words: &[u32]) -> RequiredImageCapabilities {
    let mut need = RequiredImageCapabilities::default();
    if words.len() < HEADER_WORDS {
        return need;
    }
    let mut has_unknown_storage_image = false;
    let mut has_image_write = false;
    let mut has_image_read = false;

    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        match opcode {
            OP_TYPE_IMAGE if word_count >= 9 => {
                // [7] = Sampled, [8] = Image Format. Sampled 0 means "either",
                // so it is a possible storage use and counts.
                let sampled = words[i + 7];
                let format = words[i + 8];
                if sampled == IMAGE_SAMPLED_STORAGE || sampled == 0 {
                    if storage_format_is_extended(format) {
                        need.extended_formats = true;
                    }
                    if format == 0 {
                        has_unknown_storage_image = true;
                    }
                }
            }
            OP_IMAGE_WRITE => has_image_write = true,
            OP_IMAGE_READ => has_image_read = true,
            _ => {}
        }
        i += word_count;
    }
    need.write_without_format = has_unknown_storage_image && has_image_write;
    need.read_without_format = has_unknown_storage_image && has_image_read;
    need
}

/// Every `OpTypeImage` in a module, as `(sampled, format)` pairs.
///
/// Diagnostic only. When the validator refuses a module for a capability the
/// derivation above did not ask for, the disagreement is between what
/// `spirv-val` sees and what this walk sees, and the only way to settle it is to
/// print what the walk found. Ordered as encountered, deduplicated.
pub fn image_type_census(words: &[u32]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    if words.len() < HEADER_WORDS {
        return out;
    }
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        if opcode == OP_TYPE_IMAGE && word_count >= 9 {
            let pair = (words[i + 7], words[i + 8]);
            if !out.contains(&pair) {
                out.push(pair);
            }
        }
        i += word_count;
    }
    out
}

/// Add every capability [`required_image_capabilities`] found missing.
///
/// Returns what it added. Declaring a capability whose Vulkan feature the device
/// did not enable is invalid usage, so the caller must gate on the features and
/// decline by name rather than hand the driver a module it may not survive.
/// A module carrying two set-0 sampled images: one referenced by an `OpLoad`,
/// one declared and never touched.
///
/// Lives here, out of any one `mod tests`, because three modules need the same
/// fixture and because the opcode numbers it is built from are already constants
/// in this file — a copy elsewhere would be those numbers spelled a second time,
/// which is the duplication that goes stale silently.
///
/// Both halves are the point. Only the referenced variable is what Vulkan calls
/// *statically used*, so a caller that cannot tell the two apart either refuses
/// dispatches that are legal or admits the one that makes a driver divide by
/// zero.
#[cfg(any(test, feature = "test-fixtures"))]
pub fn test_module_with_two_sampled_images(used: u32, declared_unused: u32) -> Vec<u32> {
    const IMAGE_TY: u32 = 10;
    const POINTER_TY: u32 = 11;
    const USED_VAR: u32 = 12;
    const UNUSED_VAR: u32 = 13;

    let mut w = vec![
        0x0723_0203,       // magic
        0x0001_0600,       // version
        0,                 // generator
        32,                // bound
        0,                 // schema
        (2u32 << 16) | 17, // OpCapability
        1,                 // Shader
        (3u32 << 16) | 14, // OpMemoryModel
        0,                 // Logical
        1,                 // GLSL450
    ];
    for (var, binding) in [(USED_VAR, used), (UNUSED_VAR, declared_unused)] {
        w.extend_from_slice(&[
            (4u32 << 16) | OP_DECORATE as u32,
            var,
            DECORATION_BINDING,
            binding,
        ]);
    }
    // OpTypeImage %IMAGE_TY %2 2D 0 0 0 1 Unknown — `Sampled` 1 is the
    // separate-image form, which is what makes these SAMPLED_IMAGE rather than
    // the storage class that shares the opcode.
    w.extend_from_slice(&[
        (9u32 << 16) | OP_TYPE_IMAGE as u32,
        IMAGE_TY,
        2, // sampled type id
        1, // Dim2D
        0, // depth
        0, // arrayed
        0, // MS
        IMAGE_SAMPLED_WITH_SAMPLER,
        0, // Unknown format
    ]);
    w.extend_from_slice(&[
        (4u32 << 16) | OP_TYPE_POINTER as u32,
        POINTER_TY,
        STORAGE_CLASS_UNIFORM_CONSTANT,
        IMAGE_TY,
    ]);
    for var in [USED_VAR, UNUSED_VAR] {
        w.extend_from_slice(&[
            (4u32 << 16) | OP_VARIABLE as u32,
            POINTER_TY,
            var,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
    }
    // The reference that makes `USED_VAR` statically used, and the only
    // difference between the two variables.
    w.extend_from_slice(&[(4u32 << 16) | OP_LOAD as u32, IMAGE_TY, 20, USED_VAR]);
    w
}

/// A module carrying one set-0 `OpTypeSampler` variable at each of `bindings`.
///
/// Beside [`test_module_with_two_sampled_images`] and for the same reason: the
/// opcode numbers are constants in this file, so a copy in another module's test
/// would be those numbers spelled a second time.
///
/// A *sampler*, not a sampled image: the test oracle partitions set 0 by the
/// pointee type, so a fixture built out of images answers its question with an
/// empty vector and would make any test over it vacuous.
#[cfg(any(test, feature = "test-fixtures"))]
pub fn test_module_with_samplers(bindings: &[u32]) -> Vec<u32> {
    const SAMPLER_TY: u32 = 10;
    const POINTER_TY: u32 = 11;
    const FIRST_VAR: u32 = 12;

    let mut w = vec![
        0x0723_0203,       // magic
        0x0001_0600,       // version
        0,                 // generator
        64,                // bound
        0,                 // schema
        (2u32 << 16) | 17, // OpCapability
        1,                 // Shader
        (3u32 << 16) | 14, // OpMemoryModel
        0,                 // Logical
        1,                 // GLSL450
    ];
    for (i, binding) in bindings.iter().enumerate() {
        w.extend_from_slice(&[
            (4u32 << 16) | OP_DECORATE as u32,
            FIRST_VAR + i as u32,
            DECORATION_BINDING,
            *binding,
        ]);
    }
    w.extend_from_slice(&[(2u32 << 16) | OP_TYPE_SAMPLER as u32, SAMPLER_TY]);
    w.extend_from_slice(&[
        (4u32 << 16) | OP_TYPE_POINTER as u32,
        POINTER_TY,
        STORAGE_CLASS_UNIFORM_CONSTANT,
        SAMPLER_TY,
    ]);
    for i in 0..bindings.len() as u32 {
        w.extend_from_slice(&[
            (4u32 << 16) | OP_VARIABLE as u32,
            POINTER_TY,
            FIRST_VAR + i,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
    }
    w
}

/// Every distinct `Binding` decoration in the module, whatever it decorates.
///
/// Deliberately class-blind and deliberately cheap: this is the candidate list
/// for "does the pipeline layout describe everything the module uses", and the
/// question of *what* each binding is has already been answered by the walks
/// that build the layout. Pair it with [`descriptor_static_use`], which answers
/// `NotDeclared` for anything that is not a `UniformConstant` descriptor and so
/// narrows this to the population that walk can reason about exactly.
#[cfg(any(test, feature = "test-fixtures"))]
pub fn declared_binding_numbers(words: &[u32]) -> Vec<u32> {
    let mut bindings = Vec::new();
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        if opcode == OP_DECORATE && word_count >= 4 && words[i + 2] == DECORATION_BINDING {
            bindings.push(words[i + 3]);
        }
        i += word_count;
    }
    bindings.sort_unstable();
    bindings.dedup();
    bindings
}

/// SPIR-V `OpTypeImage` `Sampled` operand value 1 — "will be used with a
/// sampler". 2 is the storage-image form, which is a different descriptor type,
/// and 0 means "either", which cannot be given a descriptor type without a guess.
#[cfg(any(test, feature = "test-fixtures"))]
const IMAGE_SAMPLED_WITH_SAMPLER: u32 = 1;

// ---------------------------------------------------------------------------
// Reflection-derived reflectors (single source of truth) + divergence census
// ---------------------------------------------------------------------------
//
// `metal2vulkan::reflect::ShaderReflection` already carries the decoded texture
// shape / access per binding, parsed from the AIR by the same decoder the emit
// path uses to write the `OpTypeImage`. The functions below read those facts
// directly, so a consumer never re-walks the emitted SPIR-V. They are keyed on
// the exact descriptor binding reflection reports for the selected layout.
//
// The `census_reflection_wellformed` guard runs once per translate (miss path)
// and validates, on the live guest's own shaders, that the AIR-derived reflection
// is internally consistent and ABI-versioned. It is the always-on regression
// proxy for the hot path now that texture shape/access is read solely from
// reflection (no second SPIR-V walk to cross-check against).

use metal2vulkan::meta::{TextureDimension, TextureShape};
use metal2vulkan::reflect::{
    BufferExtent, ResourceAccess, ResourceKind, ShaderReflection, ShaderStage, REFLECTION_VERSION,
};

/// Every descriptor binding in the executable module, from the reflection
/// emitted alongside those exact bytes.
///
/// Imageblock descriptors live outside `bindings`, so the three reflection
/// populations are deliberately joined here. Keeping this as the sole
/// projection prevents callers from reconstructing the translator's selected
/// descriptor layout from binding-band arithmetic.
pub fn reflected_descriptor_bindings(reflection: &ShaderReflection) -> Vec<u32> {
    let mut bindings = reflection
        .bindings
        .iter()
        .filter_map(|resource| resource.descriptor.map(|descriptor| descriptor.binding))
        .chain(
            reflection
                .implicit_imageblock_attachments
                .iter()
                .map(|attachment| attachment.binding),
        )
        .chain(
            reflection
                .fragment_imageblock
                .iter()
                .flat_map(|imageblock| imageblock.members.iter())
                .filter_map(|member| member.binding),
        )
        .collect::<Vec<_>>();
    bindings.sort_unstable();
    bindings.dedup();
    bindings
}

/// Statically-used sampled-image bindings absent from `bound`.
///
/// Reflection owns descriptor kind and location; the remaining SPIR-V query is
/// intentionally only executable static use, which reflection does not claim
/// to report.
pub fn reflected_null_sampled_image_bindings(
    reflection: &ShaderReflection,
    words: &[u32],
    bound: &[u32],
) -> Vec<u32> {
    let candidates = reflection
        .bindings
        .iter()
        .filter(|resource| {
            resource.kind == ResourceKind::Texture
                || matches!(
                    resource.kind,
                    ResourceKind::TextureArray | ResourceKind::EmbeddedArgBufferTexture
                ) && resource.access == Some(ResourceAccess::Sampled)
        })
        .filter_map(|resource| resource.descriptor.map(|descriptor| descriptor.binding))
        .collect::<Vec<_>>();
    null_statically_used_bindings(words, &candidates, bound)
}

/// Candidate descriptors the executable statically uses but the caller did
/// not bind. Descriptor class comes from reflection at product call sites;
/// this function answers only the executable-use relation.
pub fn null_statically_used_bindings(words: &[u32], candidates: &[u32], bound: &[u32]) -> Vec<u32> {
    let mut bindings = candidates
        .iter()
        .copied()
        .filter(|binding| {
            !bound.contains(binding) && descriptor_static_use(words, *binding).is_violation()
        })
        .collect::<Vec<_>>();
    bindings.sort_unstable();
    bindings.dedup();
    bindings
}

/// Map a decoded [`TextureShape`] to a [`SampledImageKind`] via its `OpTypeImage`
/// Dim + Arrayed. `None` for shapes `SampledImageKind` cannot express (a texel
/// `Buffer`, or a 3D array) — those are legitimate reflection shapes the sampled
/// render path does not support and rejects fail-visibly at the call site.
fn sampled_image_kind_from_shape(shape: &TextureShape) -> Option<SampledImageKind> {
    match (shape.dimension, shape.arrayed, shape.multisampled) {
        (TextureDimension::D1, false, false) => Some(SampledImageKind::D1),
        (TextureDimension::D1, true, false) => Some(SampledImageKind::D1Array),
        (TextureDimension::D2, false, false) => Some(SampledImageKind::D2),
        (TextureDimension::D2, false, true) => Some(SampledImageKind::D2Multisample),
        (TextureDimension::D2, true, false) => Some(SampledImageKind::D2Array),
        (TextureDimension::D3, false, false) => Some(SampledImageKind::D3),
        (TextureDimension::Cube, false, false) => Some(SampledImageKind::Cube),
        (TextureDimension::Cube, true, false) => Some(SampledImageKind::CubeArray),
        _ => None,
    }
}

/// Find the texture shape reflection reports for descriptor `binding`.
/// `None` when no binding matches or it carries no shape.
fn texture_shape_for_binding(reflection: &ShaderReflection, binding: u32) -> Option<&TextureShape> {
    reflection.bindings.iter().find_map(|b| {
        (is_texture_kind(b.kind) && b.descriptor.map(|d| d.binding) == Some(binding))
            .then_some(b.texture_shape.as_ref())
            .flatten()
    })
}

/// Whether a reflected resource is one of the kinds a `[[texture(n)]]` index
/// names, as opposed to a sampler, a buffer, or a framebuffer-fetch input.
///
/// The kind is checked as well as the binding because resource class is part of
/// the reflected contract and must not be inferred from numeric position.
fn is_texture_kind(kind: ResourceKind) -> bool {
    matches!(
        kind,
        ResourceKind::Texture
            | ResourceKind::TextureArray
            | ResourceKind::StorageImage
            | ResourceKind::EmbeddedArgBufferTexture
    )
}

/// First reflected resource that needs a Vulkan runtime provisioning path this
/// device does not implement. These must not collapse into an ordinary bind's
/// `Absent` answer: each represents real shader work with no descriptor or
/// synthesized input behind it.
pub fn first_unsupported_vulkan_resource(
    reflection: &ShaderReflection,
) -> Option<&metal2vulkan::reflect::ResourceBinding> {
    reflection
        .bindings
        .iter()
        .find(|resource| unsupported_vulkan_resource_kind_name(resource.kind).is_some())
}

pub fn unsupported_vulkan_resource_kind_name(kind: ResourceKind) -> Option<&'static str> {
    match kind {
        ResourceKind::KernelStageInput => Some("kernel_stage_input"),
        ResourceKind::AccelerationStructureShadow => Some("acceleration_structure_shadow"),
        ResourceKind::PrimitiveAccelerationStructure => Some("primitive_acceleration_structure"),
        ResourceKind::EmbeddedArgBufferTexture => Some("embedded_texture"),
        ResourceKind::EmbeddedArgBufferBuffer => Some("embedded_buffer"),
        ResourceKind::BufferAddressTable => Some("buffer_address_table"),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedVulkanInterface {
    pub feature: &'static str,
    pub count: usize,
}

/// First stage-level reflected interface the current Vulkan runtime cannot
/// provision. Unlike a resource binding, these contracts live on the shader as
/// a whole and otherwise have no bind loop in which to become visible.
pub fn first_unsupported_vulkan_interface(
    reflection: &ShaderReflection,
    expected_stage: ShaderStage,
) -> Option<UnsupportedVulkanInterface> {
    if reflection.stage != expected_stage {
        let feature = match reflection.stage {
            ShaderStage::Vertex => "shader_stage_vertex",
            ShaderStage::TessellationEvaluation => "shader_stage_tessellation_evaluation",
            ShaderStage::Fragment => "shader_stage_fragment",
            ShaderStage::Kernel => "shader_stage_kernel",
        };
        return Some(UnsupportedVulkanInterface { feature, count: 1 });
    }
    if reflection.tessellation.is_some() {
        return Some(UnsupportedVulkanInterface {
            feature: "tessellation",
            count: 1,
        });
    }
    if !reflection.imageblock_layouts.is_empty() {
        return Some(UnsupportedVulkanInterface {
            feature: "kernel_imageblock",
            count: reflection.imageblock_layouts.len(),
        });
    }
    if !reflection.implicit_imageblock_attachments.is_empty() {
        return Some(UnsupportedVulkanInterface {
            feature: "implicit_imageblock_attachments",
            count: reflection.implicit_imageblock_attachments.len(),
        });
    }
    reflection
        .fragment_imageblock
        .as_ref()
        .map(|imageblock| UnsupportedVulkanInterface {
            feature: "fragment_imageblock",
            count: imageblock.members.len(),
        })
}

/// One Metal texture-table slot's place in the Vulkan descriptor interface.
/// Scalar textures have `array_element = 0, descriptor_count = 1`; a texture
/// handle array maps consecutive Metal slots onto elements of one descriptor
/// binding instead of pretending each handle is a separate binding.
/// Resolve one reflected sampler at the descriptor location baked into the
/// executable module. The translator's effective layout is authoritative.
pub fn reflected_sampler_binding(resource: &metal2vulkan::reflect::ResourceBinding) -> Option<u32> {
    if !matches!(
        resource.kind,
        ResourceKind::Sampler | ResourceKind::StaticSampler
    ) {
        return None;
    }
    Some(resource.descriptor?.binding)
}

fn project_static_sampler(
    state: metal2vulkan::reflect::StaticSamplerState,
) -> reims_vgpu_core::ReflectedStaticSamplerState {
    use metal2vulkan::reflect as native;
    use reims_vgpu_core as semantic;
    semantic::ReflectedStaticSamplerState {
        min_filter: match state.min_filter {
            native::SamplerFilter::Nearest => semantic::ReflectedSamplerFilter::Nearest,
            native::SamplerFilter::Linear => semantic::ReflectedSamplerFilter::Linear,
            native::SamplerFilter::Bicubic => semantic::ReflectedSamplerFilter::Bicubic,
        },
        mag_filter: match state.mag_filter {
            native::SamplerFilter::Nearest => semantic::ReflectedSamplerFilter::Nearest,
            native::SamplerFilter::Linear => semantic::ReflectedSamplerFilter::Linear,
            native::SamplerFilter::Bicubic => semantic::ReflectedSamplerFilter::Bicubic,
        },
        mip_filter: match state.mip_filter {
            native::SamplerMipFilter::None => semantic::ReflectedSamplerMipFilter::None,
            native::SamplerMipFilter::Nearest => semantic::ReflectedSamplerMipFilter::Nearest,
            native::SamplerMipFilter::Linear => semantic::ReflectedSamplerMipFilter::Linear,
        },
        address_mode_s: project_sampler_address(state.address_mode_s),
        address_mode_t: project_sampler_address(state.address_mode_t),
        address_mode_r: project_sampler_address(state.address_mode_r),
        coordinates: match state.coordinates {
            native::SamplerCoordinates::Normalized => {
                semantic::ReflectedSamplerCoordinates::Normalized
            }
            native::SamplerCoordinates::Pixel => semantic::ReflectedSamplerCoordinates::Pixel,
        },
        compare_function: match state.compare_function {
            native::SamplerCompareFunction::None => semantic::ReflectedSamplerCompareFunction::None,
            native::SamplerCompareFunction::Less => semantic::ReflectedSamplerCompareFunction::Less,
            native::SamplerCompareFunction::LessEqual => {
                semantic::ReflectedSamplerCompareFunction::LessEqual
            }
            native::SamplerCompareFunction::Greater => {
                semantic::ReflectedSamplerCompareFunction::Greater
            }
            native::SamplerCompareFunction::GreaterEqual => {
                semantic::ReflectedSamplerCompareFunction::GreaterEqual
            }
            native::SamplerCompareFunction::Equal => {
                semantic::ReflectedSamplerCompareFunction::Equal
            }
            native::SamplerCompareFunction::NotEqual => {
                semantic::ReflectedSamplerCompareFunction::NotEqual
            }
            native::SamplerCompareFunction::Always => {
                semantic::ReflectedSamplerCompareFunction::Always
            }
            native::SamplerCompareFunction::Never => {
                semantic::ReflectedSamplerCompareFunction::Never
            }
        },
        max_anisotropy: state.max_anisotropy,
        lod_min_clamp: state.lod_min_clamp,
        lod_max_clamp: state.lod_max_clamp,
        border_color: match state.border_color {
            native::SamplerBorderColor::TransparentBlack => {
                semantic::ReflectedSamplerBorderColor::TransparentBlack
            }
            native::SamplerBorderColor::OpaqueBlack => {
                semantic::ReflectedSamplerBorderColor::OpaqueBlack
            }
            native::SamplerBorderColor::OpaqueWhite => {
                semantic::ReflectedSamplerBorderColor::OpaqueWhite
            }
        },
        reduction: match state.reduction {
            native::SamplerReduction::WeightedAverage => {
                semantic::ReflectedSamplerReduction::WeightedAverage
            }
            native::SamplerReduction::Minimum => semantic::ReflectedSamplerReduction::Minimum,
            native::SamplerReduction::Maximum => semantic::ReflectedSamplerReduction::Maximum,
        },
        lod_bias: state.lod_bias,
        raw_words: state.raw_words,
    }
}

fn project_sampler_address(
    address: metal2vulkan::reflect::SamplerAddressMode,
) -> reims_vgpu_core::ReflectedSamplerAddressMode {
    match address {
        metal2vulkan::reflect::SamplerAddressMode::ClampToZero => {
            reims_vgpu_core::ReflectedSamplerAddressMode::ClampToZero
        }
        metal2vulkan::reflect::SamplerAddressMode::ClampToEdge => {
            reims_vgpu_core::ReflectedSamplerAddressMode::ClampToEdge
        }
        metal2vulkan::reflect::SamplerAddressMode::Repeat => {
            reims_vgpu_core::ReflectedSamplerAddressMode::Repeat
        }
        metal2vulkan::reflect::SamplerAddressMode::MirroredRepeat => {
            reims_vgpu_core::ReflectedSamplerAddressMode::MirroredRepeat
        }
        metal2vulkan::reflect::SamplerAddressMode::ClampToBorder => {
            reims_vgpu_core::ReflectedSamplerAddressMode::ClampToBorder
        }
    }
}

/// Every sampler descriptor declared by a reflected shader, transformed into
/// the numbering of the selected executable variant. The result is canonical
/// so it can be cached directly beside that variant's words.
pub fn reflected_sampler_descriptors(
    reflection: &ShaderReflection,
) -> Vec<ReflectedSamplerDescriptor> {
    let mut descriptors: Vec<ReflectedSamplerDescriptor> = reflection
        .bindings
        .iter()
        .filter_map(|resource| {
            reflected_sampler_binding(resource).map(|binding| ReflectedSamplerDescriptor {
                metal_index: resource.metal_index,
                binding,
                static_state: (resource.kind == ResourceKind::StaticSampler)
                    .then_some(resource.static_sampler)
                    .flatten()
                    .map(project_static_sampler),
            })
        })
        .collect();
    descriptors.sort_by_key(|descriptor| descriptor.binding);
    descriptors
}

/// Resolve a Metal texture-table index through the final reflection interface.
/// An exact scalar declaration wins over an enclosing array range.
pub fn reflected_texture_descriptor(
    reflection: &ShaderReflection,
    metal_index: u32,
) -> Option<ReflectedTextureDescriptor> {
    let exact = reflection
        .bindings
        .iter()
        .find(|binding| is_texture_kind(binding.kind) && binding.metal_index == metal_index);
    let reflected = exact.or_else(|| {
        reflection.bindings.iter().find(|binding| {
            if binding.kind != ResourceKind::TextureArray {
                return false;
            }
            let Some(descriptor) = binding.descriptor else {
                return false;
            };
            metal_index
                .checked_sub(binding.metal_index)
                .is_some_and(|element| element < descriptor.count)
        })
    })?;
    let descriptor = reflected.descriptor?;
    let array_element = if reflected.kind == ResourceKind::TextureArray {
        metal_index.checked_sub(reflected.metal_index)?
    } else {
        0
    };
    (descriptor.count > 0 && array_element < descriptor.count).then_some(
        ReflectedTextureDescriptor {
            binding: descriptor.binding,
            array_element,
            descriptor_count: descriptor.count,
            access: match reflected.access {
                Some(ResourceAccess::Sampled) => ReflectedTextureAccess::Sampled,
                Some(ResourceAccess::Storage) => ReflectedTextureAccess::Storage,
                _ => ReflectedTextureAccess::Unknown,
            },
        },
    )
}

/// First texture descriptor that cannot be exposed through a sampled-image
/// render binding. Compute has a storage-image request path; render does not.
pub fn first_non_sampled_texture_descriptor(
    reflection: &ShaderReflection,
) -> Option<(u32, ReflectedTextureDescriptor)> {
    reflection.bindings.iter().find_map(|binding| {
        if !is_texture_kind(binding.kind) || binding.access == Some(ResourceAccess::Sampled) {
            return None;
        }
        reflected_texture_descriptor(reflection, binding.metal_index)
            .map(|descriptor| (binding.metal_index, descriptor))
    })
}

/// Whether [`reims_vgpu_config::BUFFER_EXTENT`] is switched off, read once per process.
///
/// Latched because this sits on the per-bind path and `std::env::var_os` is a
/// lock and an allocation; the variable is read at the first bind of the boot
/// and cannot change under a running device. The refusal is named once, on the
/// off channel, so a boot whose gather volume is being compared says in its own
/// log which arm it ran rather than relying on the operator's shell history.
fn buffer_extent_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        let (state, value) = reims_vgpu_config::read(reims_vgpu_config::BUFFER_EXTENT);
        match state {
            reims_vgpu_config::Switch::Off => {
                reims_vgpu_observe::off("buffer_extent reason=buffer_extent_disabled_by_env");
                true
            }
            // An unrecognized spelling is named rather than silently read as the
            // default, which is the one way an operator concludes a switch does
            // not work. It still takes the default arm: this switch may only
            // turn a rail off, and a value nobody can parse is not that.
            reims_vgpu_config::Switch::Unrecognized => {
                reims_vgpu_observe::fail(format!(
                    "buffer_extent reason=buffer_extent_env_unrecognized value={}",
                    value.unwrap_or_default()
                ));
                false
            }
            reims_vgpu_config::Switch::On | reims_vgpu_config::Switch::Unset => false,
        }
    })
}

/// The byte extent reflection proves a `[[buffer(n)]]` bind cannot be read past,
/// or `None` when the bind must keep the whole window the guest declared.
///
/// A guest buffer bind names an allocation and an offset, never a length, so the
/// widest safe answer — and until this existed, the only answer — is "the rest of
/// the allocation". `try_buffer_zero_copy_resolved`'s doc records what that cost:
/// 67.5 % of a driven boot's gathered bytes are vertex-stage constant buffers no
/// vertex descriptor bounds, and narrowing them on `declared_size` alone was
/// closed because a `constant T&` (whose declared size *is* the bound) could not
/// be told from a `device T*` (whose declared size is one element the shader may
/// index far past). [`BufferExtent`] is the translator answering exactly that
/// question, so this is the reopening of that rail.
///
/// **Only [`BufferExtent::Object`] may narrow.** `Unbounded` says AIR carries an
/// element size but no length, `Unknown` says the metadata cannot separate the
/// two, and a binding this shader does not declare at all gets no answer here —
/// all three return `None` and keep the full window. The asymmetry is the whole
/// safety argument: an over-wide window costs bus bytes, and an under-wide one
/// hands the GPU a short buffer, which is a silent wrong-pixels bug with no error
/// anywhere. The translator states the same rule on `BufferExtent` itself.
///
/// The kind is checked as well as the index because `metal_index` is only unique
/// within a resource family — a `[[texture(2)]]` and a `[[buffer(2)]]` both
/// report `metal_index` 2 — and because a threadgroup `[[buffer(n)]]` consumes no
/// descriptor and binds no guest memory to narrow.
pub fn reflected_buffer_extent(reflection: &ShaderReflection, metal_index: u32) -> Option<u64> {
    if buffer_extent_disabled() {
        return None;
    }
    // Which answer this bind got, counted at the one place the field is read.
    //
    // # Why a census and not a comment
    //
    // Both narrowing counters — `zc_buffer_extent_narrowed` on the gather rail
    // and `cpu_buffer_extent_narrowed` on the staging one — read **zero for a
    // whole driven macos-13 boot**, across 2.57 million buffer binds. A rail
    // built to reopen 67.5 % of this device's gathered bytes is firing no times,
    // and nothing else in the boot says so: a narrowing counter at zero is the
    // same reading whether every bind is genuinely unbounded, the translator
    // reports nothing, or the lookup never finds the binding at all.
    //
    // These four separate those, and only the first is a workload fact:
    //
    // * `bext_unbounded` — AIR carries an element size and no length. Nothing to
    //   do; a `device T*` may be indexed anywhere.
    // * `bext_unknown` — the metadata cannot separate a bounded reference from a
    //   pointer. A translator gap, and the one that would be worth packaging.
    // * `bext_absent` — no `Buffer` binding at this index. Expected for a
    //   `[[stage_in]]` attribute source, which is bounded by the vertex
    //   descriptor's stride rather than by a declared argument.
    // * `bext_object` — the answer that may narrow, with the bytes it declares
    //   banded so a reader can tell "narrowable and tiny" from "narrowable and
    //   already the whole allocation".
    //
    // `bext_object` firing while both `*_narrowed` counters stay at zero would
    // mean the caps are being dropped between here and the rails, which is a
    // different bug from the translator not answering and would otherwise look
    // identical.
    let found = reflection.bindings.iter().find_map(|b| {
        (b.kind == ResourceKind::Buffer && b.metal_index == metal_index)
            .then_some(b.extent)
            .flatten()
    });
    let Some(extent) = found else {
        crate::telemetry::note_route("bext_absent");
        return None;
    };
    match extent {
        BufferExtent::Object { bytes } => {
            crate::telemetry::note_route("bext_object");
            crate::telemetry::note_route(band_declared_object(bytes));
            Some(u64::from(bytes))
        }
        BufferExtent::Unbounded => {
            crate::telemetry::note_route("bext_unbounded");
            None
        }
        BufferExtent::Unknown => {
            crate::telemetry::note_route("bext_unknown");
            None
        }
    }
}

/// Invocation bounds needed to turn a reflected render-buffer footprint into
/// one conservative byte extent.
///
/// These are maxima in the shader builtin's own value domain, not draw counts.
/// Keeping that distinction in the type prevents a caller from forgetting
/// `firstVertex` or `baseInstance`. `vertex_index` is absent for indexed draws:
/// the index buffer, rather than `index_count`, bounds the values observed by
/// the vertex shader, and this path deliberately does not read it merely to
/// make the gather smaller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderBufferIndexBounds {
    vertex_index: Option<u64>,
    instance_index: Option<u64>,
}

impl RenderBufferIndexBounds {
    pub fn new(
        first_vertex: u32,
        vertex_count: u32,
        base_instance: u32,
        instance_count: u32,
        indexed: bool,
    ) -> Self {
        let inclusive_max = |first: u32, count: u32| {
            count
                .checked_sub(1)
                .and_then(|last| first.checked_add(last))
                .map(u64::from)
        };
        Self {
            vertex_index: if indexed {
                None
            } else {
                inclusive_max(first_vertex, vertex_count)
            },
            instance_index: inclusive_max(base_instance, instance_count),
        }
    }

    fn maximum(self, source: metal2vulkan::reflect::BufferIndexSource) -> Option<u64> {
        use metal2vulkan::reflect::BufferIndexSource;
        match source {
            BufferIndexSource::VertexIndex => self.vertex_index,
            BufferIndexSource::InstanceIndex => self.instance_index,
            // A render stage cannot derive a useful bound for compute builtins.
            // Treating an unexpected one as unknown keeps the whole guest
            // window, which is the conservative answer.
            BufferIndexSource::GlobalInvocationIdX
            | BufferIndexSource::GlobalInvocationIdY
            | BufferIndexSource::GlobalInvocationIdZ
            | BufferIndexSource::LocalInvocationIdX
            | BufferIndexSource::LocalInvocationIdY
            | BufferIndexSource::LocalInvocationIdZ
            | BufferIndexSource::WorkgroupIdX
            | BufferIndexSource::WorkgroupIdY
            | BufferIndexSource::WorkgroupIdZ
            | BufferIndexSource::LocalInvocationIndex => None,
        }
    }

    fn maximum_interface(self, source: reims_vgpu_core::ShaderBufferIndexSource) -> Option<u64> {
        use reims_vgpu_core::ShaderBufferIndexSource;
        match source {
            ShaderBufferIndexSource::VertexIndex => self.vertex_index,
            ShaderBufferIndexSource::InstanceIndex => self.instance_index,
            ShaderBufferIndexSource::GlobalInvocationIdX
            | ShaderBufferIndexSource::GlobalInvocationIdY
            | ShaderBufferIndexSource::GlobalInvocationIdZ
            | ShaderBufferIndexSource::LocalInvocationIdX
            | ShaderBufferIndexSource::LocalInvocationIdY
            | ShaderBufferIndexSource::LocalInvocationIdZ
            | ShaderBufferIndexSource::WorkgroupIdX
            | ShaderBufferIndexSource::WorkgroupIdY
            | ShaderBufferIndexSource::WorkgroupIdZ
            | ShaderBufferIndexSource::LocalInvocationIndex => None,
        }
    }
}

/// Bound a render-stage `[[buffer(n)]]` by the final shader's actual byte
/// footprint for this draw.
///
/// The returned value is the largest exclusive byte offset any invocation may
/// touch, so it can be passed directly as a Vulkan buffer range or staging
/// length. Static ranges and affine vertex/instance accesses are both covered.
/// A data-dependent access, an arithmetic overflow, or an index source this
/// draw cannot bound returns the declared-object cap when one exists and
/// otherwise keeps the complete guest allocation.
///
/// The footprint and declared-object answers are independent conservative
/// upper bounds, so when both exist their minimum is still conservative. This
/// is also why this function consumes reflection once and exports one answer:
/// callers should not have to reconstruct which translator facts can safely be
/// combined.
pub fn reflected_render_buffer_extent(
    reflection: &ShaderReflection,
    metal_index: u32,
    bounds: RenderBufferIndexBounds,
) -> Option<u64> {
    reflected_buffer_footprint_extent(reflection, metal_index, |source| bounds.maximum(source))
}

/// [`reflected_render_buffer_extent`] for a compute dispatch.
///
/// `workgroups` is the exact triple passed to `vkCmdDispatch`; `local_size` is
/// the specialized shader's workgroup size. Their products bound the Vulkan
/// invocations that can execute, including padding invocations introduced when
/// a Metal `dispatchThreads` grid is rounded up to whole Vulkan workgroups.
/// Bounding the actual host invocations, rather than the guest's requested
/// thread count, keeps the staging valid even on that edge.
pub fn reflected_compute_buffer_extent(
    reflection: &ShaderReflection,
    metal_index: u32,
    workgroups: [u32; 3],
    local_size: [u32; 3],
) -> Option<u64> {
    use metal2vulkan::reflect::BufferIndexSource;

    let axis_max = |axis: usize| {
        u64::from(workgroups[axis])
            .checked_mul(u64::from(local_size[axis]))?
            .checked_sub(1)
    };
    let local_max = |axis: usize| u64::from(local_size[axis]).checked_sub(1);
    let workgroup_max = |axis: usize| u64::from(workgroups[axis]).checked_sub(1);
    let local_linear_max = || {
        local_size
            .into_iter()
            .try_fold(1u64, |product, axis| product.checked_mul(u64::from(axis)))?
            .checked_sub(1)
    };
    reflected_buffer_footprint_extent(reflection, metal_index, |source| match source {
        BufferIndexSource::GlobalInvocationIdX => axis_max(0),
        BufferIndexSource::GlobalInvocationIdY => axis_max(1),
        BufferIndexSource::GlobalInvocationIdZ => axis_max(2),
        BufferIndexSource::LocalInvocationIdX => local_max(0),
        BufferIndexSource::LocalInvocationIdY => local_max(1),
        BufferIndexSource::LocalInvocationIdZ => local_max(2),
        BufferIndexSource::WorkgroupIdX => workgroup_max(0),
        BufferIndexSource::WorkgroupIdY => workgroup_max(1),
        BufferIndexSource::WorkgroupIdZ => workgroup_max(2),
        BufferIndexSource::LocalInvocationIndex => local_linear_max(),
        BufferIndexSource::VertexIndex | BufferIndexSource::InstanceIndex => None,
    })
}

pub fn reflected_render_buffer_extent_interface(
    interface: &reims_vgpu_core::ShaderInterface,
    metal_index: u32,
    bounds: RenderBufferIndexBounds,
) -> Option<u64> {
    reflected_buffer_footprint_extent_interface(interface, metal_index, |source| {
        bounds.maximum_interface(source)
    })
}

pub fn vertex_buffer_extent_interface(
    interface: &reims_vgpu_core::ShaderInterface,
    metal_index: u32,
    feeds_stage_in: bool,
    bounds: RenderBufferIndexBounds,
) -> Option<u64> {
    (!feeds_stage_in)
        .then(|| reflected_render_buffer_extent_interface(interface, metal_index, bounds))
        .flatten()
}

pub fn reflected_compute_buffer_extent_interface(
    interface: &reims_vgpu_core::ShaderInterface,
    metal_index: u32,
    workgroups: [u32; 3],
    local_size: [u32; 3],
) -> Option<u64> {
    use reims_vgpu_core::ShaderBufferIndexSource;
    let axis_max = |axis: usize| {
        u64::from(workgroups[axis])
            .checked_mul(u64::from(local_size[axis]))?
            .checked_sub(1)
    };
    let local_max = |axis: usize| u64::from(local_size[axis]).checked_sub(1);
    let workgroup_max = |axis: usize| u64::from(workgroups[axis]).checked_sub(1);
    let local_linear_max = || {
        local_size
            .into_iter()
            .try_fold(1u64, |product, axis| product.checked_mul(u64::from(axis)))?
            .checked_sub(1)
    };
    reflected_buffer_footprint_extent_interface(interface, metal_index, |source| match source {
        ShaderBufferIndexSource::GlobalInvocationIdX => axis_max(0),
        ShaderBufferIndexSource::GlobalInvocationIdY => axis_max(1),
        ShaderBufferIndexSource::GlobalInvocationIdZ => axis_max(2),
        ShaderBufferIndexSource::LocalInvocationIdX => local_max(0),
        ShaderBufferIndexSource::LocalInvocationIdY => local_max(1),
        ShaderBufferIndexSource::LocalInvocationIdZ => local_max(2),
        ShaderBufferIndexSource::WorkgroupIdX => workgroup_max(0),
        ShaderBufferIndexSource::WorkgroupIdY => workgroup_max(1),
        ShaderBufferIndexSource::WorkgroupIdZ => workgroup_max(2),
        ShaderBufferIndexSource::LocalInvocationIndex => local_linear_max(),
        ShaderBufferIndexSource::VertexIndex | ShaderBufferIndexSource::InstanceIndex => None,
    })
}

fn reflected_buffer_footprint_extent_interface(
    interface: &reims_vgpu_core::ShaderInterface,
    metal_index: u32,
    maximum: impl Fn(reims_vgpu_core::ShaderBufferIndexSource) -> Option<u64>,
) -> Option<u64> {
    use reims_vgpu_core::{ShaderBufferExtent, ShaderResourceKind};
    if buffer_extent_disabled() {
        return None;
    }
    let resource = interface.bindings.iter().find(|resource| {
        resource.kind == ShaderResourceKind::Buffer && resource.metal_index == metal_index
    });
    let declared = match resource.and_then(|resource| resource.extent) {
        Some(ShaderBufferExtent::Object { bytes }) => {
            crate::telemetry::note_route("bext_object");
            crate::telemetry::note_route(band_declared_object(bytes));
            Some(u64::from(bytes))
        }
        Some(ShaderBufferExtent::Unbounded) => {
            crate::telemetry::note_route("bext_unbounded");
            None
        }
        Some(ShaderBufferExtent::Unknown) => {
            crate::telemetry::note_route("bext_unknown");
            None
        }
        None => {
            crate::telemetry::note_route("bext_absent");
            None
        }
    };
    let footprint = resource
        .and_then(|resource| resource.footprint.as_ref())
        .and_then(|footprint| {
            if footprint.has_unbounded_access {
                return None;
            }
            let mut end = 0u64;
            for range in &footprint.static_ranges {
                end = end.max(range.offset.checked_add(range.size)?);
            }
            for access in &footprint.strided_accesses {
                let mut access_end = access.base_offset.checked_add(access.access_size)?;
                for term in &access.terms {
                    access_end =
                        access_end.checked_add(maximum(term.source)?.checked_mul(term.stride)?)?;
                }
                end = end.max(access_end);
            }
            (end != 0).then_some(end)
        });
    match (declared, footprint) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(cap), None) | (None, Some(cap)) => Some(cap),
        (None, None) => None,
    }
}

fn reflected_buffer_footprint_extent(
    reflection: &ShaderReflection,
    metal_index: u32,
    maximum: impl Fn(metal2vulkan::reflect::BufferIndexSource) -> Option<u64>,
) -> Option<u64> {
    if buffer_extent_disabled() {
        return None;
    }
    let declared = reflected_buffer_extent(reflection, metal_index);
    let footprint = reflection
        .bindings
        .iter()
        .find(|binding| binding.kind == ResourceKind::Buffer && binding.metal_index == metal_index)
        .and_then(|binding| binding.footprint.as_ref())
        .and_then(|footprint| {
            if footprint.has_unbounded_access {
                return None;
            }
            let mut end = 0u64;
            for range in &footprint.static_ranges {
                end = end.max(range.offset.checked_add(range.size)?);
            }
            for access in &footprint.strided_accesses {
                let mut access_end = access.base_offset.checked_add(access.access_size)?;
                for term in &access.terms {
                    access_end =
                        access_end.checked_add(maximum(term.source)?.checked_mul(term.stride)?)?;
                }
                end = end.max(access_end);
            }
            // No reflected dereference is normally paired with
            // `ResourceAccess::Unused` and served by the neutral bind. If the
            // two facts ever disagree, an empty Vulkan range must not become
            // the accidental policy; retain the guest window instead.
            (end != 0).then_some(end)
        });
    match (declared, footprint) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(cap), None) | (None, Some(cap)) => Some(cap),
        (None, None) => None,
    }
}

/// Band a declared object size, because what decides whether narrowing is worth
/// anything is the order of magnitude and not the byte.
///
/// A survey of translated modules ran a median declared size of 64 bytes and a
/// maximum of 512, so the bands are placed around that rather than spread
/// evenly.
fn band_declared_object(bytes: u32) -> &'static str {
    match bytes {
        0..=64 => "bext_object_le64",
        65..=512 => "bext_object_le512",
        513..=4096 => "bext_object_le4k",
        4097..=65536 => "bext_object_le64k",
        _ => "bext_object_gt64k",
    }
}

/// [`reflected_buffer_extent`] for a **vertex** bind, carrying a stage-in
/// exclusion because reflected argument bounds do not describe vertex streams.
///
/// Metal's vertex-descriptor layouts and its buffer argument table share one
/// index space, so a vertex function may declare a bounded argument at an index
/// the pipeline also uses as a vertex layout — which is why Apple's guidance is
/// to place vertex buffers at high indices, and why the collision is legal
/// rather than a decode error. Reflection then reports an `Object` whose size
/// bounds the *argument*, and applying it truncates the *stream*: a few bytes
/// staged where kilobytes were asked for, an attribute walk that picks the
/// truncated content out for every attribute naming that index, and a draw that
/// rasterises nothing or garbage. The `StageInBytesMissing` guard downstream
/// fires only on an *empty* stream, so it does not catch a short one.
///
pub fn vertex_buffer_extent(
    reflection: &ShaderReflection,
    metal_index: u32,
    feeds_stage_in: bool,
    bounds: RenderBufferIndexBounds,
) -> Option<u64> {
    if feeds_stage_in {
        return None;
    }
    reflected_render_buffer_extent(reflection, metal_index, bounds)
}

/// How reflection describes a `[[buffer(n)]]` bind's use by this stage.
///
/// The asymmetry is the same one [`reflected_buffer_extent`] states, and for a
/// sharper reason. Reading `Unused` where the shader does dereference the buffer
/// hands the GPU stale or absent bytes — silent wrong pixels, no error anywhere
/// — so only an explicit [`ResourceAccess::Unused`] may answer
/// [`ReflectedBufferAccess::Unused`]. A bind reflection never mentions is
/// [`ReflectedBufferAccess::Absent`]; one it mentions without a usable access
/// answer is [`ReflectedBufferAccess::Unknown`]. Keeping those distinct lets a
/// compute dispatch skip an extra guest bind while failing closed on a declared
/// buffer whose access could not be classified.
///
/// Deliberately not gated on [`reims_vgpu_config::BUFFER_EXTENT`]. That switch governs
/// narrowing a bind's byte window, which is a different question from whether
/// the bind is read at all, and folding the two would make one switch silently
/// answer for two rails.
///
/// The kind is checked as well as the index for the reason
/// [`reflected_buffer_extent`] gives: `metal_index` is unique only within a
/// resource family, so a `[[texture(2)]]` would otherwise answer for
/// `[[buffer(2)]]`.
pub fn reflected_buffer_access(
    reflection: &ShaderReflection,
    metal_index: u32,
) -> ReflectedBufferAccess {
    // `find_map` here yields `Option<Option<ResourceAccess>>` on purpose: the
    // outer level says whether reflection declares a Buffer at this index at
    // all, the inner says whether that declaration carries an access. Flattening
    // them would merge "not declared" into "declared, access unrecorded", and
    // this function exists to keep those apart.
    let Some(access) = reflection.bindings.iter().find_map(|b| {
        (b.kind == ResourceKind::Buffer && b.metal_index == metal_index).then_some(b.access)
    }) else {
        return ReflectedBufferAccess::Absent;
    };
    match access {
        Some(ResourceAccess::Unused) => ReflectedBufferAccess::Unused,
        Some(ResourceAccess::ReadOnly) => ReflectedBufferAccess::ReadOnly,
        Some(ResourceAccess::WriteOnly | ResourceAccess::ReadWrite) => {
            ReflectedBufferAccess::Writable
        }
        Some(ResourceAccess::Sampled | ResourceAccess::Storage) | None => {
            ReflectedBufferAccess::Unknown
        }
    }
}

/// Classify descriptor `binding` for the sampled render path from reflection.
pub fn reflected_sampled_kind(reflection: &ShaderReflection, binding: u32) -> ReflectedSampledKind {
    match texture_shape_for_binding(reflection, binding) {
        None => ReflectedSampledKind::Absent,
        Some(shape) => match sampled_image_kind_from_shape(shape) {
            Some(kind) => ReflectedSampledKind::Kind(kind),
            None => ReflectedSampledKind::Unsupported,
        },
    }
}

/// Classify texture descriptor `binding` for the compute rail from the
/// translator's reflection.
///
/// Sampled-vs-storage comes from the declared Metal access qualifier
/// (`TextureShape.writable`), which is exact at translate time — there is no
/// `Unknown`. The shape axis comes from the same decoded `OpTypeImage`, and
/// the rail refuses anything it would otherwise stage as 2D behind the
/// shader's back: binding a `TYPE_2D` view to a SPIR-V image declared
/// `2DArray`/`3D`/`1D`/`Cube`/`Buffer`/multisampled is a descriptor-type
/// mismatch, not a degraded render.
pub fn reflected_compute_texture(
    reflection: &ShaderReflection,
    binding: u32,
) -> ReflectedComputeTexture {
    let Some(shape) = texture_shape_for_binding(reflection, binding) else {
        return ReflectedComputeTexture::Absent;
    };
    let axis = match shape.dimension {
        TextureDimension::D1 => Some("dim_1d"),
        TextureDimension::D3 => Some("dim_3d"),
        TextureDimension::Cube => Some("dim_cube"),
        TextureDimension::Buffer => Some("dim_buffer"),
        TextureDimension::D2 if shape.arrayed => Some("arrayed"),
        TextureDimension::D2 if shape.multisampled && shape.writable => {
            Some("multisampled_storage")
        }
        TextureDimension::D2 if shape.multisampled => {
            return ReflectedComputeTexture::Multisampled2d;
        }
        TextureDimension::D2 => None,
    };
    match axis {
        Some(axis) => ReflectedComputeTexture::UnstageableShape { axis },
        None => ReflectedComputeTexture::Plain2d(if shape.writable {
            ImageAccess::Storage
        } else {
            ImageAccess::Sampled
        }),
    }
}

/// Validate that the translator's reflection is internally well-formed, once per
/// translate (miss path). This is the always-on regression proxy that replaces
/// the former reflection-vs-SPIR-V cross-check: the hot path now reads texture
/// shape and sampled-vs-storage access solely from reflection, so the guard must
/// catch a reflection that is self-contradictory or emitted against a different
/// ABI — without a second walk of the SPIR-V. Checks:
///   - `reflection_version` matches the ABI this consumer was built against
///     (catches a translator/consumer version skew);
///   - each static sampler carries decoded state and a set-0 descriptor inside
///     the sampler ABI band;
///   - each texture-family binding carries a descriptor location;
///   - the redundant sampled-vs-storage encodings agree — `ResourceKind`
///     (`StorageImage`), `TextureShape.writable`, and `ResourceAccess`. The
///     translator derives all three from one decoded `TextureShape`, so any
///     disagreement is a translator regression the consumer must not trust.
///
/// Logs `m2v_reflect_malformed reason=<slug>` fail-visibly; quiet on a healthy
/// boot (returns the number of violations found, 0 when clean).
pub fn census_reflection_wellformed(reflection: &ShaderReflection, pipeline_ref: u32) -> usize {
    use std::collections::BTreeSet;

    let mut bad = 0;
    let mut sampler_bindings = BTreeSet::new();
    if reflection.reflection_version != REFLECTION_VERSION {
        bad += 1;
        reims_vgpu_observe::fail(format!(
            "m2v_reflect_malformed pipe={pipeline_ref} reason=reflection_version_mismatch \
             got={} want={REFLECTION_VERSION}",
            reflection.reflection_version
        ));
    }
    if let Err(error) = reflection.descriptor_layout.validate() {
        bad += 1;
        reims_vgpu_observe::fail(format!(
            "m2v_reflect_malformed pipe={pipeline_ref} reason=descriptor_layout_invalid \
             detail={error:?}"
        ));
    }
    match (reflection.stage, reflection.local_size) {
        (ShaderStage::Kernel, Some(size)) if size.into_iter().all(|axis| axis > 0) => {}
        (ShaderStage::Kernel, local_size) => {
            bad += 1;
            reims_vgpu_observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} reason=kernel_local_size \
                 local_size={local_size:?}"
            ));
        }
        (_, Some(local_size)) => {
            bad += 1;
            reims_vgpu_observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} reason=nonkernel_local_size \
                 stage={:?} local_size={local_size:?}",
                reflection.stage
            ));
        }
        (_, None) => {}
    }
    for b in &reflection.bindings {
        if matches!(b.kind, ResourceKind::Sampler | ResourceKind::StaticSampler) {
            if let Some(descriptor) = b.descriptor {
                if !sampler_bindings.insert(descriptor.binding) {
                    bad += 1;
                    reims_vgpu_observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=sampler_descriptor_duplicate bind={} kind={:?}",
                        descriptor.binding, b.kind
                    ));
                }
            }
        }
        if b.kind == ResourceKind::StaticSampler {
            match (b.descriptor, b.static_sampler) {
                (None, _) => {
                    bad += 1;
                    reims_vgpu_observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=static_sampler_no_descriptor metal_index={}",
                        b.metal_index
                    ));
                }
                (Some(descriptor), None) => {
                    bad += 1;
                    reims_vgpu_observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=static_sampler_no_state bind={}",
                        descriptor.binding
                    ));
                }
                (Some(descriptor), Some(_))
                    if descriptor.set != reflection.descriptor_layout.set
                        || descriptor.count != 1
                        || !reflection
                            .descriptor_layout
                            .samplers
                            .contains(descriptor.binding) =>
                {
                    bad += 1;
                    reims_vgpu_observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=static_sampler_descriptor_out_of_band set={} bind={} count={} \
                         expected_set={} \
                         expected_band={}..{}",
                        descriptor.set,
                        descriptor.binding,
                        descriptor.count,
                        reflection.descriptor_layout.set,
                        reflection.descriptor_layout.samplers.start,
                        reflection.descriptor_layout.samplers.end
                    ));
                }
                (Some(_), Some(_)) => {}
            }
            continue;
        }
        if b.kind == ResourceKind::Sampler {
            match b.descriptor {
                None => {
                    bad += 1;
                    reims_vgpu_observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=sampler_no_descriptor metal_index={}",
                        b.metal_index
                    ));
                }
                Some(descriptor)
                    if descriptor.set != reflection.descriptor_layout.set
                        || descriptor.count != 1
                        || descriptor.binding
                            != reflection
                                .descriptor_layout
                                .sampler_binding(b.metal_index)
                                .unwrap_or(u32::MAX) =>
                {
                    bad += 1;
                    reims_vgpu_observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=sampler_descriptor_out_of_band set={} bind={} count={} \
                         expected_set={} expected_bind={} \
                         expected_band={}..{}",
                        descriptor.set,
                        descriptor.binding,
                        descriptor.count,
                        reflection.descriptor_layout.set,
                        reflection
                            .descriptor_layout
                            .sampler_binding(b.metal_index)
                            .unwrap_or(u32::MAX),
                        reflection.descriptor_layout.samplers.start,
                        reflection.descriptor_layout.samplers.end
                    ));
                }
                Some(_) => {}
            }
            if b.static_sampler.is_some() {
                bad += 1;
                reims_vgpu_observe::fail(format!(
                    "m2v_reflect_malformed pipe={pipeline_ref} \
                     reason=static_sampler_state_on_nonstatic kind={:?} metal_index={}",
                    b.kind, b.metal_index
                ));
            }
            continue;
        }
        if b.static_sampler.is_some() {
            bad += 1;
            reims_vgpu_observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} \
                 reason=static_sampler_state_on_nonstatic kind={:?} metal_index={}",
                b.kind, b.metal_index
            ));
        }
        if let Some(descriptor) = b.descriptor {
            if descriptor.set != reflection.descriptor_layout.set {
                bad += 1;
                reims_vgpu_observe::fail(format!(
                    "m2v_reflect_malformed pipe={pipeline_ref} reason=descriptor_set \
                     kind={:?} metal_index={} set={} expected_set={}",
                    b.kind, b.metal_index, descriptor.set, reflection.descriptor_layout.set
                ));
            }
        }
        if b.kind == ResourceKind::Buffer
            && matches!(
                b.access,
                Some(ResourceAccess::Sampled | ResourceAccess::Storage)
            )
        {
            bad += 1;
            reims_vgpu_observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} reason=buffer_access_wrong_class \
                 metal_index={} access={:?}",
                b.metal_index, b.access
            ));
        }
        if b.kind == ResourceKind::Buffer
            && matches!(b.extent, Some(BufferExtent::Object { bytes: 0 }))
        {
            bad += 1;
            reims_vgpu_observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} reason=buffer_object_extent_zero \
                 metal_index={}",
                b.metal_index
            ));
        }
        let texture_family = matches!(
            b.kind,
            ResourceKind::Texture
                | ResourceKind::TextureArray
                | ResourceKind::StorageImage
                | ResourceKind::EmbeddedArgBufferTexture
        );
        if !texture_family {
            if b.descriptor.is_some_and(|descriptor| descriptor.count != 1) {
                bad += 1;
                reims_vgpu_observe::fail(format!(
                    "m2v_reflect_malformed pipe={pipeline_ref} reason=descriptor_count \
                     kind={:?} metal_index={} count={}",
                    b.kind,
                    b.metal_index,
                    b.descriptor.expect("checked Some").count
                ));
            }
            continue;
        }
        let binding = b.descriptor.map(|d| d.binding);
        if binding.is_none() {
            bad += 1;
            reims_vgpu_observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} reason=texture_binding_no_descriptor \
                 kind={:?} metal_index={}",
                b.kind, b.metal_index
            ));
        }
        if let Some(descriptor) = b.descriptor {
            let invalid_count = match b.kind {
                ResourceKind::TextureArray => descriptor.count <= 1,
                ResourceKind::EmbeddedArgBufferTexture => {
                    let reflected_length = b
                        .texture_shape
                        .as_ref()
                        .and_then(|shape| shape.array_length);
                    descriptor.count == 0 || reflected_length.unwrap_or(1) != descriptor.count
                }
                _ => descriptor.count != 1,
            };
            if invalid_count {
                bad += 1;
                reims_vgpu_observe::fail(format!(
                    "m2v_reflect_malformed pipe={pipeline_ref} reason=texture_descriptor_count \
                     kind={:?} metal_index={} count={}",
                    b.kind, b.metal_index, descriptor.count
                ));
            }
        }
        // Only the two malformed-reflection lines below read this, and a missing
        // descriptor has already emitted its own. Rendering that case as `0`
        // names a real binding index the reflection never carried, so the two
        // failures downstream would read as being about binding 0.
        let bind = match binding {
            Some(bind) => bind.to_string(),
            None => "none".to_string(),
        };
        // Storage-vs-sampled must agree across the three encodings the consumer
        // and the translator both derive from the one `TextureShape`.
        let access_storage = match b.access {
            Some(ResourceAccess::Storage) => Some(true),
            Some(ResourceAccess::Sampled) => Some(false),
            _ => None,
        };
        if access_storage.is_none() {
            bad += 1;
            reims_vgpu_observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} reason=texture_access_missing_or_wrong \
                 bind={bind} kind={:?} access={:?}",
                b.kind, b.access
            ));
        }
        let Some(shape) = b.texture_shape.as_ref() else {
            bad += 1;
            reims_vgpu_observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} reason=texture_shape_missing \
                 bind={bind} kind={:?}",
                b.kind
            ));
            continue;
        };
        // A writable image may deliberately be formatless: runtime
        // specialization emits `Unknown` only after proving the supplied
        // read/write-without-format capabilities cover the final operations.
        // An explicit storage format on a sampled image is contradictory.
        if !shape.writable && shape.storage_format.is_some() {
            bad += 1;
            reims_vgpu_observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} reason=texture_format_access_disagree \
                 bind={bind} kind={:?} writable={} format={:?}",
                b.kind, shape.writable, shape.storage_format
            ));
        }
        let kind_storage = match b.kind {
            ResourceKind::StorageImage => Some(true),
            ResourceKind::TextureArray | ResourceKind::EmbeddedArgBufferTexture => access_storage,
            _ => Some(false),
        };
        if let (Some(writable), Some(kind_storage)) = (Some(shape.writable), kind_storage) {
            if writable != kind_storage {
                bad += 1;
                reims_vgpu_observe::fail(format!(
                    "m2v_reflect_malformed pipe={pipeline_ref} reason=kind_writable_disagree \
                     bind={bind} kind={:?} writable={writable}",
                    b.kind
                ));
            }
        }
        if let (Some(access_storage), Some(kind_storage)) = (access_storage, kind_storage) {
            if access_storage != kind_storage {
                bad += 1;
                reims_vgpu_observe::fail(format!(
                    "m2v_reflect_malformed pipe={pipeline_ref} reason=kind_access_disagree \
                     bind={bind} kind={:?} access={:?}",
                    b.kind, b.access
                ));
            }
        }
    }
    bad
}

/// Report a shader's runtime `[[function_constant(N)]]` inventory when it declares any.
///
/// metal2vulkan exposes byte-exact runtime function-constant specialization and
/// reflects every index and ABI type needed to call it. The paravirt command
/// stream decoded here carries no `MTLFunctionConstantValues` to supply — the
/// pipeline/function descriptors contain references and the AIR blob only. A
/// shader declaring runtime constants therefore remains a visible integration
/// gap rather than being assigned values inferred from its contents.
///
/// This once-per-translate line makes that gap measurable:
/// which shaders (by Metal entry name) carry runtime FCs, so a future rendering
/// delta can be correlated with FC usage and the specialization gap sized before any
/// fix. It is diagnostic, not a per-draw failure, so it goes to the OFF-prefixed
/// analysis sink (not `fail`, which must read zero on a healthy boot).
///
/// The input is the reflection's `function_constants` — the translator's single
/// source of truth, scanned once from the AIR `air.fc_initializer` ABI globals — so
/// there is no SPIR-V re-walk. Silent for the common FC-free shader. Returns the
/// count reported (0 = silent) for tests.
pub fn log_unavailable_function_constants(reflection: &ShaderReflection) -> usize {
    if reflection.function_constants.is_empty() {
        return 0;
    }
    let stage = match reflection.stage {
        ShaderStage::Vertex => "v",
        ShaderStage::TessellationEvaluation => "te",
        ShaderStage::Fragment => "f",
        ShaderStage::Kernel => "k",
    };
    let entry = reflection.entry_point.as_deref().unwrap_or("?");
    let inventory: Vec<String> = reflection
        .function_constants
        .iter()
        .map(|fc| format!("{}:{}:{}", fc.index, fc.name, fc.type_name))
        .collect();
    reims_vgpu_observe::off(format!(
        "fc_values_unavailable stage={stage} entry={entry} count={} fcs=[{}]",
        inventory.len(),
        inventory.join(",")
    ));
    reflection.function_constants.len()
}

/// Module builders shared by this module's own tests and by the engine tests
/// that ask the same questions of a draw's two modules.
///
/// Here rather than copied into each test module: the answer
/// [`descriptor_static_use`] gives turns on the exact word layout below — an
/// `OpEntryPoint` interface list that names the variable without referencing it
/// is what separates `DeclaredUnused` from `Used` — so a second hand-built
/// module that drifted from this one would test a different question under the
/// same name.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_support {
    use super::*;

    /// Build a minimal module: header, `OpCapability Shader`, `OpMemoryModel`,
    /// then whatever body words are given.
    pub fn module_with(body: &[u32]) -> Vec<u32> {
        let mut words = vec![
            0x0723_0203,       // magic
            0x0001_0600,       // version
            0,                 // generator
            64,                // bound
            0,                 // schema
            (2u32 << 16) | 17, // OpCapability
            1,                 // Shader
            (3u32 << 16) | 14, // OpMemoryModel
            0,
            1,
        ];
        words.extend_from_slice(body);
        words
    }

    /// A fragment module declaring one `UniformConstant` variable on `binding`,
    /// named by `OpEntryPoint`'s interface list the way SPIR-V 1.4 requires, and
    /// referenced by an `OpLoad` only when `loaded`.
    ///
    /// The entry point is not decoration: from 1.4 its interface list carries
    /// every global variable whether the body touches it or not, so a module
    /// without it would not exercise the one exclusion that decides
    /// [`descriptor_static_use`]'s answer.
    pub fn module_with_descriptor(binding: u32, loaded: bool) -> Vec<u32> {
        module_with_descriptor_storage(binding, loaded, STORAGE_CLASS_UNIFORM_CONSTANT)
    }

    /// A buffer-descriptor counterpart of [`module_with_descriptor`].
    pub fn module_with_buffer_descriptor(binding: u32, loaded: bool) -> Vec<u32> {
        module_with_descriptor_storage(binding, loaded, STORAGE_CLASS_STORAGE_BUFFER)
    }

    fn module_with_descriptor_storage(binding: u32, loaded: bool, storage_class: u32) -> Vec<u32> {
        const VAR: u32 = 10;
        const FN: u32 = 11;
        let mut body = vec![
            // OpEntryPoint Fragment %FN "m" %VAR
            (5u32 << 16) | OP_ENTRY_POINT as u32,
            4,
            FN,
            u32::from_le_bytes([b'm', 0, 0, 0]),
            VAR,
            // OpName %VAR "t"
            (3u32 << 16) | OP_NAME as u32,
            VAR,
            u32::from_le_bytes([b't', 0, 0, 0]),
            // OpDecorate %VAR Binding <binding>
            (4u32 << 16) | OP_DECORATE as u32,
            VAR,
            DECORATION_BINDING,
            binding,
            // %VAR = OpVariable %2 UniformConstant
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            VAR,
            storage_class,
        ];
        if loaded {
            // %12 = OpLoad %2 %VAR
            body.extend_from_slice(&[(4u32 << 16) | OP_LOAD as u32, 2, 12, VAR]);
        }
        module_with(&body)
    }
}

#[cfg(test)]
mod more_tests {
    use super::test_support::module_with;
    use super::*;

    fn module_with_descriptor(binding: u32, loaded: bool) -> Vec<u32> {
        super::test_support::module_with_descriptor(binding, loaded)
    }

    /// Declaration and static use are different questions, and only the second
    /// one is what Vulkan requires of a pipeline layout.
    ///
    /// This is the discriminator the `frag_declared_descriptor_unbound` firings
    /// on macos-14, macos-15 and macos-26 were missing: the guard proved the
    /// module declares the binding, which is not the bar, and the fail line read
    /// identically for a real violation and for a variable nothing references.
    #[test]
    fn a_descriptor_is_used_only_when_something_references_it() {
        let unused = module_with_descriptor(96, false);
        let used = module_with_descriptor(96, true);

        assert!(declares_descriptor(&unused, 96));
        assert!(declares_descriptor(&used, 96));

        assert_eq!(
            descriptor_static_use(&unused, 96),
            DescriptorUse::DeclaredUnused,
            "OpEntryPoint's interface list and OpDecorate/OpName name the variable \
             without referencing it; counting them makes every descriptor read as used"
        );
        assert!(!descriptor_static_use(&unused, 96).is_violation());

        assert_eq!(descriptor_static_use(&used, 96), DescriptorUse::Used);
        assert!(descriptor_static_use(&used, 96).is_violation());

        assert_eq!(
            descriptor_static_use(&used, 97),
            DescriptorUse::NotDeclared,
            "a binding no variable carries is not a use of anything"
        );
    }

    #[test]
    fn buffer_descriptor_use_is_independent_of_source_reflection() {
        let unused = test_support::module_with_buffer_descriptor(7, false);
        let used = test_support::module_with_buffer_descriptor(7, true);

        assert_eq!(
            buffer_descriptor_static_use(&unused, 7),
            DescriptorUse::DeclaredUnused
        );
        assert_eq!(buffer_descriptor_static_use(&used, 7), DescriptorUse::Used);
        assert_eq!(
            descriptor_static_use(&unused, 7),
            DescriptorUse::DeclaredUnused
        );
        assert_eq!(descriptor_static_use(&used, 7), DescriptorUse::Used);
        assert_eq!(
            buffer_descriptor_static_use(&used, 8),
            DescriptorUse::NotDeclared
        );
    }

    /// Two variables on one binding is its own defect and must not be reported
    /// as a layout violation, which would name the wrong thing.
    #[test]
    fn two_variables_on_one_binding_fail_closed_rather_than_pick_one() {
        let mut body = vec![
            (4u32 << 16) | OP_DECORATE as u32,
            20,
            DECORATION_BINDING,
            5,
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            20,
            STORAGE_CLASS_UNIFORM_CONSTANT,
            (4u32 << 16) | OP_DECORATE as u32,
            21,
            DECORATION_BINDING,
            5,
        ];
        body.extend_from_slice(&[
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            21,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        let words = module_with(&body);
        assert_eq!(descriptor_static_use(&words, 5), DescriptorUse::Ambiguous);
        assert!(
            !descriptor_static_use(&words, 5).is_violation(),
            "an ambiguous binding is not evidence that the layout is missing a descriptor"
        );
    }

    /// Every verdict carries its own name, so a census cannot collapse two.
    #[test]
    fn every_descriptor_use_has_its_own_slug() {
        let all = [
            DescriptorUse::NotDeclared,
            DescriptorUse::DeclaredUnused,
            DescriptorUse::Used,
            DescriptorUse::Ambiguous,
        ];
        let slugs: std::collections::BTreeSet<&str> = all.iter().map(|u| u.slug()).collect();
        assert_eq!(slugs.len(), all.len());
    }

    // `module_with` is `test_support`'s, imported above: one builder, so a
    // module built here and one built by an engine test are the same module.

    /// `OpTypeImage %result %sampled_ty 2D 0 0 0 <sampled> <format>` (9 words).
    fn op_type_image(result: u32, sampled: u32, format: u32) -> [u32; 9] {
        [
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            result,
            2, // sampled type id (unused by the walk)
            1, // Dim2D
            0, // depth
            0, // arrayed
            0, // MS
            sampled,
            format,
        ]
    }

    /// A storage image declaring an extended format needs the capability, and
    /// this is the exact shape a macOS 11 guest's kernel was rejected for:
    /// `%1138 = OpTypeImage %float 2D 0 0 0 2 Rg16f`.
    #[test]
    fn an_extended_storage_format_requires_its_capability() {
        let words = module_with(&op_type_image(10, IMAGE_SAMPLED_STORAGE, 7));
        let need = required_image_capabilities(&words);
        assert!(need.extended_formats, "Rg16f is outside the core set");
        assert!(!need.write_without_format);
        assert!(!need.read_without_format);
    }

    /// Every core storage format is admitted without a capability.
    ///
    /// Swept rather than sampled because the core set is **not contiguous** in
    /// the enum — the two- and one-channel formats sit between the four-channel
    /// ones — so a `<=` style bound reads as thorough and would quietly demand
    /// the capability for half of them, or waive it for formats that need it.
    #[test]
    fn the_core_storage_formats_need_no_capability() {
        for raw in [0u32, 1, 2, 3, 4, 5, 21, 22, 23, 24, 30, 31, 32, 33] {
            assert!(
                !storage_format_is_extended(raw),
                "format {raw} is in SPIR-V's core storage set"
            );
        }
        // A representative span of the extended ones, including the neighbours
        // of core values, which is where an off-by-one would hide.
        for raw in [6u32, 7, 8, 9, 13, 15, 20, 25, 29, 34, 39] {
            assert!(
                storage_format_is_extended(raw),
                "format {raw} needs StorageImageExtendedFormats"
            );
        }
    }

    /// A *sampled* image is not a storage image, so its format asks for nothing.
    #[test]
    fn a_sampled_image_format_does_not_demand_the_storage_capability() {
        let words = module_with(&op_type_image(10, 1, 7));
        assert!(!required_image_capabilities(&words).extended_formats);
    }

    /// The candidate list is class-blind and the use test is not, which is the
    /// division the layout backstop is built on.
    ///
    /// [`declared_binding_numbers`] must return both variables — it is the
    /// cheap sweep that decides what gets asked about — while
    /// [`descriptor_static_use`] separates the one Vulkan requires a descriptor
    /// for from the one it is legal to omit. Asserting them together is what
    /// says the pair partitions; either alone reads correct while the caller
    /// built from them refuses legal work or admits the divide-by-zero.
    #[test]
    fn the_binding_sweep_is_class_blind_and_the_use_test_narrows_it() {
        let words = test_module_with_two_sampled_images(33, 34);
        assert_eq!(declared_binding_numbers(&words), vec![33, 34]);
        assert_eq!(descriptor_static_use(&words, 33), DescriptorUse::Used);
        assert_eq!(
            descriptor_static_use(&words, 34),
            DescriptorUse::DeclaredUnused
        );
        // A binding no variable carries is not invented.
        assert_eq!(
            descriptor_static_use(&words, 99),
            DescriptorUse::NotDeclared
        );
    }

    /// An `Unknown`-format storage image plus a write demands
    /// `StorageImageWriteWithoutFormat`.
    ///
    /// The pointer/variable/load words in the body are the shape the translator
    /// emits around such a write. They are deliberately *not* what the check
    /// depends on — see `required_image_capabilities` on why attributing the
    /// write to its image was tried and abandoned — but they belong in the
    /// fixture so it stays a realistic module rather than two instructions.
    #[test]
    fn a_write_to_an_unknown_format_image_requires_its_capability() {
        let mut body = Vec::new();
        body.extend_from_slice(&op_type_image(10, IMAGE_SAMPLED_STORAGE, 0));
        // OpTypePointer %11 UniformConstant %10
        body.extend_from_slice(&[(4u32 << 16) | OP_TYPE_POINTER as u32, 11, 0, 10]);
        // OpVariable %11 %12 UniformConstant
        body.extend_from_slice(&[(4u32 << 16) | OP_VARIABLE as u32, 11, 12, 0]);
        // OpLoad %10 %13 %12
        body.extend_from_slice(&[(4u32 << 16) | OP_LOAD as u32, 10, 13, 12]);
        // OpImageWrite %13 %20 %21
        body.extend_from_slice(&[(4u32 << 16) | OP_IMAGE_WRITE as u32, 13, 20, 21]);
        let words = module_with(&body);

        let need = required_image_capabilities(&words);
        assert!(
            need.write_without_format,
            "write to an Unknown-format image"
        );
        assert!(!need.extended_formats, "Unknown is in the core set");
    }

    /// A write to a *declared*-format image asks for nothing.
    #[test]
    fn a_write_to_a_declared_format_image_needs_no_capability() {
        let mut body = Vec::new();
        body.extend_from_slice(&op_type_image(10, IMAGE_SAMPLED_STORAGE, 4)); // Rgba8
        body.extend_from_slice(&[(4u32 << 16) | OP_TYPE_POINTER as u32, 11, 0, 10]);
        body.extend_from_slice(&[(4u32 << 16) | OP_VARIABLE as u32, 11, 12, 0]);
        body.extend_from_slice(&[(4u32 << 16) | OP_LOAD as u32, 10, 13, 12]);
        body.extend_from_slice(&[(4u32 << 16) | OP_IMAGE_WRITE as u32, 13, 20, 21]);
        let words = module_with(&body);
        assert!(!required_image_capabilities(&words).any());
    }

    use metal2vulkan::meta::{FunctionConstant, TextureComponent, TextureShape};
    use metal2vulkan::reflect::{
        BufferByteRange, BufferFootprint, BufferIndexSource, BufferStrideTerm, BufferStridedAccess,
        DescriptorLocation, ResourceBinding, ResourceKind, ShaderReflection, ShaderStage,
        REFLECTION_VERSION, RESOURCE_DESCRIPTOR_SET,
    };

    fn empty_reflection(stage: ShaderStage) -> ShaderReflection {
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            stage,
            entry_point: None,
            kernel_dispatch: None,
            bindings: vec![],
            argument_buffer_fields: vec![],
            vertex_attributes: vec![],
            varyings: vec![],
            render_targets: vec![],
            depth_members: vec![],
            depth_qualifier: None,
            stencil_members: vec![],
            local_size: (stage == ShaderStage::Kernel).then_some([1, 1, 1]),
            vertex_builtins: None,
            tessellation: None,
            imageblock_layouts: vec![],
            implicit_imageblock_attachments: vec![],
            fragment_imageblock: None,
            descriptor_layout: metal2vulkan::reflect::DescriptorLayout::default(),
            datalayout: None,
            runtime_sampler_specializations: vec![],
            runtime_storage_image_specializations: vec![],
            function_constants: vec![],
        }
    }

    fn buffer_binding(metal_index: u32, extent: Option<BufferExtent>) -> ResourceBinding {
        ResourceBinding {
            kind: ResourceKind::Buffer,
            metal_index,
            descriptor: Some(DescriptorLocation {
                set: RESOURCE_DESCRIPTOR_SET,
                binding: metal_index,
                count: 1,
            }),
            param_index: None,
            stage_input_location: None,
            address_space: None,
            declared_size: None,
            extent,
            footprint: None,
            type_layout: None,
            type_name: None,
            texture_shape: None,
            embedded_source: None,
            access: None,
            static_sampler: None,
        }
    }

    /// Only `Object` narrows a bind; every other answer keeps the whole window.
    ///
    /// The asymmetry is the safety argument for the whole narrowing rail, and it
    /// is the direction with no alarm behind it: too wide a window costs bus
    /// bytes and nothing else, too narrow a one hands the GPU a short buffer and
    /// draws whatever follows it. So each of the four ways to *not* know an
    /// extent is asserted separately rather than as one "not Object" case —
    /// `Unbounded`, `Unknown`, a binding this shader does not declare, and a
    /// binding present but carrying no extent at all.
    #[test]
    fn only_a_bounded_object_extent_narrows_a_buffer_bind() {
        let mut r = empty_reflection(ShaderStage::Vertex);
        r.bindings = vec![
            buffer_binding(0, Some(BufferExtent::Object { bytes: 288 })),
            buffer_binding(1, Some(BufferExtent::Unbounded)),
            buffer_binding(2, Some(BufferExtent::Unknown)),
            buffer_binding(3, None),
        ];

        assert_eq!(
            reflected_buffer_extent(&r, 0),
            Some(288),
            "a bounded object"
        );
        assert_eq!(reflected_buffer_extent(&r, 1), None, "an unbounded pointer");
        assert_eq!(reflected_buffer_extent(&r, 2), None, "an undecided extent");
        assert_eq!(reflected_buffer_extent(&r, 3), None, "no extent carried");
        assert_eq!(reflected_buffer_extent(&r, 9), None, "not declared at all");
    }

    /// A vertex index the pipeline also uses as a `[[stage_in]]` layout takes no
    /// extent, however confidently reflection describes the argument at it.
    ///
    /// One index space, two uses: reflection's `Object` bounds the declared
    /// argument, and the same index is the byte source for every attribute
    /// naming it. Applying the argument's bound to the stream stages a few bytes
    /// where kilobytes were asked for, and the only guard downstream fires on an
    /// *empty* stream rather than a short one.
    ///
    #[test]
    fn a_vertex_stream_index_is_not_bounded_by_the_argument_declared_at_it() {
        let mut r = empty_reflection(ShaderStage::Vertex);
        r.bindings = vec![buffer_binding(3, Some(BufferExtent::Object { bytes: 64 }))];

        assert_eq!(
            vertex_buffer_extent(&r, 3, false, RenderBufferIndexBounds::default()),
            Some(64),
            "a declared argument that feeds no attribute keeps its bound"
        );
        assert_eq!(
            vertex_buffer_extent(&r, 3, true, RenderBufferIndexBounds::default()),
            None,
            "the same index feeding stage-in must keep the whole stream — 64 \
             bytes of a vertex buffer rasterises garbage and declines nothing"
        );
    }

    #[test]
    fn render_footprint_bounds_static_vertex_and_instance_accesses() {
        let mut r = empty_reflection(ShaderStage::Vertex);
        let mut binding = buffer_binding(3, Some(BufferExtent::Unbounded));
        binding.footprint = Some(BufferFootprint {
            static_ranges: vec![BufferByteRange {
                offset: 24,
                size: 8,
            }],
            strided_accesses: vec![BufferStridedAccess {
                base_offset: 16,
                access_size: 12,
                terms: vec![
                    BufferStrideTerm {
                        source: BufferIndexSource::VertexIndex,
                        stride: 32,
                    },
                    BufferStrideTerm {
                        source: BufferIndexSource::InstanceIndex,
                        stride: 256,
                    },
                ],
            }],
            has_unbounded_access: false,
        });
        r.bindings.push(binding);

        let bounds = RenderBufferIndexBounds::new(4, 3, 2, 2, false);
        assert_eq!(
            reflected_render_buffer_extent(&r, 3, bounds),
            Some(16 + 12 + 6 * 32 + 3 * 256),
            "the cap includes firstVertex and baseInstance, not just the two counts"
        );
    }

    #[test]
    fn render_footprint_fails_closed_for_indexed_and_unbounded_access() {
        let mut r = empty_reflection(ShaderStage::Vertex);
        let mut binding = buffer_binding(3, Some(BufferExtent::Unbounded));
        binding.footprint = Some(BufferFootprint {
            static_ranges: vec![],
            strided_accesses: vec![BufferStridedAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![BufferStrideTerm {
                    source: BufferIndexSource::VertexIndex,
                    stride: 4,
                }],
            }],
            has_unbounded_access: false,
        });
        r.bindings.push(binding);

        assert_eq!(
            reflected_render_buffer_extent(&r, 3, RenderBufferIndexBounds::new(0, 200, 0, 1, true),),
            None,
            "index_count does not bound the values fetched from an index buffer"
        );
        r.bindings[0]
            .footprint
            .as_mut()
            .unwrap()
            .has_unbounded_access = true;
        assert_eq!(
            reflected_render_buffer_extent(
                &r,
                3,
                RenderBufferIndexBounds::new(0, 200, 0, 1, false),
            ),
            None,
            "one unmodelled pointer access keeps the complete guest window"
        );
    }

    #[test]
    fn declared_object_and_render_footprint_choose_the_tighter_proven_bound() {
        let mut r = empty_reflection(ShaderStage::Fragment);
        let mut binding = buffer_binding(5, Some(BufferExtent::Object { bytes: 512 }));
        binding.footprint = Some(BufferFootprint {
            static_ranges: vec![BufferByteRange {
                offset: 40,
                size: 8,
            }],
            strided_accesses: vec![],
            has_unbounded_access: false,
        });
        r.bindings.push(binding);
        assert_eq!(
            reflected_render_buffer_extent(&r, 5, RenderBufferIndexBounds::default()),
            Some(48)
        );
    }

    #[test]
    fn compute_footprint_bounds_every_dispatch_builtin_in_host_invocation_space() {
        let mut r = empty_reflection(ShaderStage::Kernel);
        let mut binding = buffer_binding(2, Some(BufferExtent::Unbounded));
        let terms = [
            BufferIndexSource::GlobalInvocationIdX,
            BufferIndexSource::GlobalInvocationIdY,
            BufferIndexSource::GlobalInvocationIdZ,
            BufferIndexSource::LocalInvocationIdX,
            BufferIndexSource::LocalInvocationIdY,
            BufferIndexSource::LocalInvocationIdZ,
            BufferIndexSource::WorkgroupIdX,
            BufferIndexSource::WorkgroupIdY,
            BufferIndexSource::WorkgroupIdZ,
            BufferIndexSource::LocalInvocationIndex,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, source)| BufferStrideTerm {
            source,
            stride: 1u64 << i,
        })
        .collect();
        binding.footprint = Some(BufferFootprint {
            static_ranges: vec![],
            strided_accesses: vec![BufferStridedAccess {
                base_offset: 5,
                access_size: 4,
                terms,
            }],
            has_unbounded_access: false,
        });
        r.bindings.push(binding);

        // 2x3x4 workgroups of 8x4x2 local invocations produce global maxima
        // 15,11,7; local maxima 7,3,1; workgroup maxima 1,2,3; and local linear
        // index maximum 63.
        let expected =
            5 + 4 + 15 + 11 * 2 + 7 * 4 + 7 * 8 + 3 * 16 + 32 + 64 + 2 * 128 + 3 * 256 + 63 * 512;
        assert_eq!(
            reflected_compute_buffer_extent(&r, 2, [2, 3, 4], [8, 4, 2]),
            Some(expected)
        );
    }

    #[test]
    fn compute_footprint_rejects_a_render_builtin_and_zero_dispatch_axis() {
        let mut r = empty_reflection(ShaderStage::Kernel);
        let mut binding = buffer_binding(2, Some(BufferExtent::Unbounded));
        binding.footprint = Some(BufferFootprint {
            static_ranges: vec![],
            strided_accesses: vec![BufferStridedAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![BufferStrideTerm {
                    source: BufferIndexSource::VertexIndex,
                    stride: 4,
                }],
            }],
            has_unbounded_access: false,
        });
        r.bindings.push(binding);
        assert_eq!(
            reflected_compute_buffer_extent(&r, 2, [2, 3, 4], [8, 4, 2]),
            None,
            "an index domain this dispatch does not own keeps the full window"
        );

        r.bindings[0].footprint.as_mut().unwrap().strided_accesses[0].terms[0].source =
            BufferIndexSource::GlobalInvocationIdX;
        assert_eq!(
            reflected_compute_buffer_extent(&r, 2, [0, 3, 4], [8, 4, 2]),
            None,
            "zero work has no inclusive invocation maximum"
        );
    }

    /// A `metal_index` is unique only within a resource family, so the lookup
    /// must not read a texture's or a threadgroup buffer's slot as a bind it may
    /// narrow. A `[[texture(0)]]` and a `[[buffer(0)]]` both report index 0, and
    /// a threadgroup `[[buffer(n)]]` binds no guest memory at all.
    #[test]
    fn a_buffer_extent_is_not_read_off_another_resource_family() {
        let mut r = empty_reflection(ShaderStage::Fragment);
        let mut threadgroup = buffer_binding(0, Some(BufferExtent::Object { bytes: 16 }));
        threadgroup.kind = ResourceKind::ThreadgroupBuffer;
        let mut texture = buffer_binding(1, Some(BufferExtent::Object { bytes: 32 }));
        texture.kind = ResourceKind::Texture;
        r.bindings = vec![threadgroup, texture];

        assert_eq!(reflected_buffer_extent(&r, 0), None, "threadgroup buffer");
        assert_eq!(reflected_buffer_extent(&r, 1), None, "texture");
    }

    /// Every foreign access answer maps to one total local class.
    ///
    /// Asserted case by case rather than as one "not Unused" arm, for the reason
    /// `only_a_bounded_object_extent_narrows_a_buffer_bind` gives about the
    /// extent: this is the direction with no alarm behind it. A bind wrongly
    /// read as unused would have its guest bytes withheld, and the shader would
    /// read whatever the descriptor happened to point at — wrong pixels, no
    /// error anywhere.
    #[test]
    fn only_an_explicit_unused_access_answers_unused() {
        use metal2vulkan::reflect::ResourceAccess;

        let mut r = empty_reflection(ShaderStage::Fragment);
        let with = |index: u32, access: Option<ResourceAccess>| {
            let mut b = buffer_binding(index, None);
            b.access = access;
            b
        };
        r.bindings = vec![
            with(0, Some(ResourceAccess::Unused)),
            with(1, Some(ResourceAccess::ReadOnly)),
            with(2, Some(ResourceAccess::WriteOnly)),
            with(3, Some(ResourceAccess::ReadWrite)),
            with(4, None),
        ];

        assert_eq!(
            reflected_buffer_access(&r, 0),
            ReflectedBufferAccess::Unused,
            "the one class that may withhold bytes"
        );
        assert_eq!(
            reflected_buffer_access(&r, 1),
            ReflectedBufferAccess::ReadOnly
        );
        assert_eq!(
            reflected_buffer_access(&r, 2),
            ReflectedBufferAccess::Writable
        );
        assert_eq!(
            reflected_buffer_access(&r, 3),
            ReflectedBufferAccess::Writable
        );
        assert_eq!(
            reflected_buffer_access(&r, 4),
            ReflectedBufferAccess::Unknown,
            "declared, but carrying no access"
        );
        assert_eq!(
            reflected_buffer_access(&r, 9),
            ReflectedBufferAccess::Absent,
            "not declared at all"
        );
    }

    /// The same family rule the extent lookup obeys. A `[[texture(0)]]` marked
    /// unused must not answer for `[[buffer(0)]]`, or a bind the shader does
    /// read would have its bytes withheld on a texture's say-so.
    #[test]
    fn a_buffer_access_is_not_read_off_another_resource_family() {
        use metal2vulkan::reflect::ResourceAccess;

        let mut r = empty_reflection(ShaderStage::Fragment);
        let mut threadgroup = buffer_binding(0, None);
        threadgroup.kind = ResourceKind::ThreadgroupBuffer;
        threadgroup.access = Some(ResourceAccess::Unused);
        let mut texture = buffer_binding(1, None);
        texture.kind = ResourceKind::Texture;
        texture.access = Some(ResourceAccess::Unused);
        r.bindings = vec![threadgroup, texture];

        assert_eq!(
            reflected_buffer_access(&r, 0),
            ReflectedBufferAccess::Absent,
            "threadgroup buffer"
        );
        assert_eq!(
            reflected_buffer_access(&r, 1),
            ReflectedBufferAccess::Absent,
            "texture"
        );
    }

    /// The classification must not answer differently because the *extent* rail
    /// was switched off. They are two questions about one binding and one switch
    /// must not silently answer for both.
    #[test]
    fn the_access_class_does_not_follow_the_extent_switch() {
        use metal2vulkan::reflect::ResourceAccess;

        let mut r = empty_reflection(ShaderStage::Vertex);
        let mut b = buffer_binding(0, Some(BufferExtent::Object { bytes: 288 }));
        b.access = Some(ResourceAccess::Unused);
        r.bindings = vec![b];

        // Whatever `buffer_extent_disabled()` reads from the environment on this
        // host, the access answer is the same one.
        assert_eq!(
            reflected_buffer_access(&r, 0),
            ReflectedBufferAccess::Unused,
            "access is not gated on the extent switch"
        );
    }

    fn texture_binding(binding: u32, shape: TextureShape) -> ResourceBinding {
        let access = if shape.writable {
            ResourceAccess::Storage
        } else {
            ResourceAccess::Sampled
        };
        ResourceBinding {
            kind: ResourceKind::Texture,
            metal_index: binding - TEXTURE_BINDING_BASE,
            descriptor: Some(DescriptorLocation {
                set: 0,
                binding,
                count: 1,
            }),
            param_index: None,
            stage_input_location: None,
            address_space: None,
            declared_size: None,
            extent: None,
            footprint: None,
            type_layout: None,
            type_name: None,
            texture_shape: Some(shape),
            embedded_source: None,
            access: Some(access),
            static_sampler: None,
        }
    }

    #[test]
    fn a_texture_handle_array_maps_consecutive_metal_slots_to_descriptor_elements() {
        let mut reflection = empty_reflection(ShaderStage::Kernel);
        let mut array = texture_binding(
            TEXTURE_BINDING_BASE + 4,
            TextureShape {
                dimension: metal2vulkan::meta::TextureDimension::D2,
                arrayed: false,
                multisampled: false,
                component: metal2vulkan::meta::TextureComponent::Float,
                writable: false,
                array_ref: true,
                array_length: None,
                storage_format: None,
            },
        );
        array.kind = ResourceKind::TextureArray;
        array.descriptor.as_mut().unwrap().count = 8;
        reflection.bindings.push(array);

        assert_eq!(
            reflected_texture_descriptor(&reflection, 4),
            Some(ReflectedTextureDescriptor {
                binding: TEXTURE_BINDING_BASE + 4,
                array_element: 0,
                descriptor_count: 8,
                access: ReflectedTextureAccess::Sampled,
            })
        );
        assert_eq!(
            reflected_texture_descriptor(&reflection, 11),
            Some(ReflectedTextureDescriptor {
                binding: TEXTURE_BINDING_BASE + 4,
                array_element: 7,
                descriptor_count: 8,
                access: ReflectedTextureAccess::Sampled,
            })
        );
        assert_eq!(reflected_texture_descriptor(&reflection, 12), None);

        assert_eq!(first_non_sampled_texture_descriptor(&reflection), None);
        reflection.bindings[0].access = Some(ResourceAccess::Storage);
        assert!(matches!(
            first_non_sampled_texture_descriptor(&reflection),
            Some((4, ReflectedTextureDescriptor {
                binding,
                descriptor_count: 8,
                access: ReflectedTextureAccess::Storage,
                ..
            })) if binding == TEXTURE_BINDING_BASE + 4
        ));
    }

    /// A reflected static sampler. `binding` is the translator's number, because
    /// reflection is the translator's own output and `census_reflection_wellformed`
    /// validates it against the translator's bands.
    fn static_sampler_binding(binding: u32) -> ResourceBinding {
        use metal2vulkan::reflect::{
            SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerCoordinates,
            SamplerFilter, SamplerMipFilter, SamplerReduction, StaticSamplerState,
        };

        ResourceBinding {
            kind: ResourceKind::StaticSampler,
            metal_index: binding - SAMPLER_BINDING_BASE,
            descriptor: Some(DescriptorLocation {
                set: RESOURCE_DESCRIPTOR_SET,
                binding,
                count: 1,
            }),
            param_index: None,
            stage_input_location: None,
            address_space: None,
            declared_size: None,
            extent: None,
            footprint: None,
            type_layout: None,
            type_name: None,
            texture_shape: None,
            embedded_source: None,
            access: None,
            static_sampler: Some(StaticSamplerState {
                min_filter: SamplerFilter::Linear,
                mag_filter: SamplerFilter::Linear,
                mip_filter: SamplerMipFilter::None,
                address_mode_s: SamplerAddressMode::ClampToEdge,
                address_mode_t: SamplerAddressMode::ClampToEdge,
                address_mode_r: SamplerAddressMode::ClampToEdge,
                coordinates: SamplerCoordinates::Normalized,
                compare_function: SamplerCompareFunction::Never,
                max_anisotropy: 1,
                lod_min_clamp: 0.0,
                lod_max_clamp: 65504.0,
                border_color: SamplerBorderColor::TransparentBlack,
                reduction: SamplerReduction::WeightedAverage,
                lod_bias: 0.0,
                raw_words: [0x807b_ff00_0008_0a49, 0],
            }),
        }
    }

    #[test]
    fn reflected_sampler_binding_uses_the_translators_effective_layout() {
        let sampler = static_sampler_binding(SAMPLER_BINDING_BASE + 5);
        assert_eq!(
            reflected_sampler_binding(&sampler),
            Some(SAMPLER_BINDING_BASE + 5)
        );
    }

    fn shape(dimension: TextureDimension, arrayed: bool, writable: bool) -> TextureShape {
        TextureShape {
            dimension,
            arrayed,
            multisampled: false,
            component: TextureComponent::Float,
            writable,
            array_ref: false,
            array_length: None,
            storage_format: writable.then_some(metal2vulkan::meta::TextureFormat::R32f),
        }
    }

    #[test]
    fn reflection_derived_kind_and_access_cover_every_shape() {
        // Dimensionality mapping matches the SPIR-V-walk `SampledImageKind`.
        let cases = [
            (TextureDimension::D1, false, Some(SampledImageKind::D1)),
            (TextureDimension::D1, true, Some(SampledImageKind::D1Array)),
            (TextureDimension::D2, false, Some(SampledImageKind::D2)),
            (TextureDimension::D2, true, Some(SampledImageKind::D2Array)),
            (TextureDimension::D3, false, Some(SampledImageKind::D3)),
            (TextureDimension::Cube, false, Some(SampledImageKind::Cube)),
            (
                TextureDimension::Cube,
                true,
                Some(SampledImageKind::CubeArray),
            ),
            // SPIR-V's SampledImageKind cannot express these — the walk returns
            // None for them too, so reflection must agree.
            (TextureDimension::D3, true, None),
            (TextureDimension::Buffer, false, None),
        ];
        for (dim, arrayed, want) in cases {
            let mut r = empty_reflection(ShaderStage::Fragment);
            let bind = TEXTURE_BINDING_BASE + 3;
            r.bindings
                .push(texture_binding(bind, shape(dim, arrayed, false)));
            assert_eq!(
                reflected_sampled_kind(&r, bind),
                want.map_or(
                    ReflectedSampledKind::Unsupported,
                    ReflectedSampledKind::Kind
                ),
                "dim={dim:?} arrayed={arrayed}"
            );
        }

        // Access mapping: writable => storage image, else sampled.
        let mut r = empty_reflection(ShaderStage::Kernel);
        let sampled = TEXTURE_BINDING_BASE;
        let storage = TEXTURE_BINDING_BASE + 1;
        r.bindings.push(texture_binding(
            sampled,
            shape(TextureDimension::D2, false, false),
        ));
        r.bindings.push(texture_binding(
            storage,
            shape(TextureDimension::D2, false, true),
        ));
        assert_eq!(
            reflected_compute_texture(&r, sampled),
            ReflectedComputeTexture::Plain2d(ImageAccess::Sampled)
        );
        assert_eq!(
            reflected_compute_texture(&r, storage),
            ReflectedComputeTexture::Plain2d(ImageAccess::Storage)
        );
        // A binding reflection does not carry => Absent (the walk's miss).
        assert_eq!(
            reflected_compute_texture(&r, TEXTURE_BINDING_BASE + 9),
            ReflectedComputeTexture::Absent
        );
        // Absent, not Unsupported: the binding is not in the reflection at all,
        // and only `reflected_sampled_kind` can tell those apart.
        assert_eq!(
            reflected_sampled_kind(&r, TEXTURE_BINDING_BASE + 9),
            ReflectedSampledKind::Absent
        );
    }

    /// The compute rail stages one flat `width × height` rectangle per texture
    /// binding, so every declared shape with a slice, depth, or sample axis must
    /// come back named rather than collapsing into the plain-2D arm — binding a
    /// `TYPE_2D` view to an image the shader declared otherwise is a
    /// descriptor-type mismatch, not a degraded render.
    #[test]
    fn every_unstageable_compute_texture_shape_names_its_axis() {
        let bind = TEXTURE_BINDING_BASE + 4;
        let unstageable = [
            (TextureDimension::D1, false, false, "dim_1d"),
            (TextureDimension::D1, true, false, "dim_1d"),
            (TextureDimension::D3, false, false, "dim_3d"),
            (TextureDimension::Cube, false, false, "dim_cube"),
            (TextureDimension::Cube, true, false, "dim_cube"),
            (TextureDimension::Buffer, false, false, "dim_buffer"),
            (TextureDimension::D2, true, false, "arrayed"),
        ];
        for (dimension, arrayed, multisampled, axis) in unstageable {
            for writable in [false, true] {
                let mut r = empty_reflection(ShaderStage::Kernel);
                let mut s = shape(dimension, arrayed, writable);
                s.multisampled = multisampled;
                r.bindings.push(texture_binding(bind, s));
                assert_eq!(
                    reflected_compute_texture(&r, bind),
                    ReflectedComputeTexture::UnstageableShape { axis },
                    "dim={dimension:?} arrayed={arrayed} ms={multisampled} writable={writable}"
                );
            }
        }

        let mut sampled_ms = shape(TextureDimension::D2, false, false);
        sampled_ms.multisampled = true;
        let mut r = empty_reflection(ShaderStage::Kernel);
        r.bindings.push(texture_binding(bind, sampled_ms));
        assert_eq!(
            reflected_compute_texture(&r, bind),
            ReflectedComputeTexture::Multisampled2d
        );

        let mut storage_ms = shape(TextureDimension::D2, false, true);
        storage_ms.multisampled = true;
        let mut r = empty_reflection(ShaderStage::Kernel);
        r.bindings.push(texture_binding(bind, storage_ms));
        assert_eq!(
            reflected_compute_texture(&r, bind),
            ReflectedComputeTexture::UnstageableShape {
                axis: "multisampled_storage"
            }
        );

        // The one stageable shape, both access classes, is not swept up by it.
        for (writable, want) in [(false, ImageAccess::Sampled), (true, ImageAccess::Storage)] {
            let mut r = empty_reflection(ShaderStage::Kernel);
            r.bindings.push(texture_binding(
                bind,
                shape(TextureDimension::D2, false, writable),
            ));
            assert_eq!(
                reflected_compute_texture(&r, bind),
                ReflectedComputeTexture::Plain2d(want)
            );
        }
    }

    #[test]
    fn wellformed_guard_passes_consistent_reflection_and_catches_desync() {
        use metal2vulkan::reflect::ResourceAccess;

        // A consistent sampled 2D texture: kind Texture, !writable, access Sampled.
        let bind = TEXTURE_BINDING_BASE + 2;
        let mut r = empty_reflection(ShaderStage::Fragment);
        let mut b = texture_binding(bind, shape(TextureDimension::D2, false, false));
        b.access = Some(ResourceAccess::Sampled);
        r.bindings.push(b);
        assert_eq!(census_reflection_wellformed(&r, 0), 0);
        let mut overlapping_layout = r.clone();
        overlapping_layout.descriptor_layout.sampled_textures =
            overlapping_layout.descriptor_layout.buffers;
        assert_eq!(census_reflection_wellformed(&overlapping_layout, 0), 1);
        let mut nonkernel_local_size = r.clone();
        nonkernel_local_size.local_size = Some([1, 1, 1]);
        assert_eq!(census_reflection_wellformed(&nonkernel_local_size, 0), 1);
        let mut missing_kernel_local_size = empty_reflection(ShaderStage::Kernel);
        missing_kernel_local_size.local_size = None;
        assert_eq!(
            census_reflection_wellformed(&missing_kernel_local_size, 0),
            1
        );

        // Static samplers must carry decoded state in set 0 inside [64,96).
        let mut static_reflection = empty_reflection(ShaderStage::Fragment);
        static_reflection
            .bindings
            .push(static_sampler_binding(SAMPLER_BINDING_BASE + 1));
        assert_eq!(census_reflection_wellformed(&static_reflection, 0), 0);
        let mut missing_state = static_reflection.clone();
        missing_state.bindings[0].static_sampler = None;
        assert_eq!(census_reflection_wellformed(&missing_state, 0), 1);
        let mut out_of_band = static_reflection.clone();
        out_of_band.bindings[0].descriptor.as_mut().unwrap().binding = COLOR_INPUT_BINDING_BASE;
        assert_eq!(census_reflection_wellformed(&out_of_band, 0), 1);

        // Dynamic samplers obey the same translator-band contract. This is the
        // premise that lets the runtime transform their reflection instead of
        // rediscovering the interface by walking SPIR-V.
        let mut dynamic_reflection = static_reflection.clone();
        dynamic_reflection.bindings[0].kind = ResourceKind::Sampler;
        dynamic_reflection.bindings[0].static_sampler = None;
        assert_eq!(census_reflection_wellformed(&dynamic_reflection, 0), 0);
        let mut duplicate_sampler = dynamic_reflection.clone();
        duplicate_sampler
            .bindings
            .push(duplicate_sampler.bindings[0].clone());
        assert_eq!(census_reflection_wellformed(&duplicate_sampler, 0), 1);
        let mut shifted_dynamic = dynamic_reflection.clone();
        shifted_dynamic.bindings[0]
            .descriptor
            .as_mut()
            .unwrap()
            .binding += 1;
        assert_eq!(census_reflection_wellformed(&shifted_dynamic, 0), 1);

        // A consistent storage image: kind StorageImage, writable, access Storage.
        let mut rs = empty_reflection(ShaderStage::Kernel);
        let mut sb = texture_binding(
            TEXTURE_BINDING_BASE + 1,
            shape(TextureDimension::D2, false, true),
        );
        sb.kind = ResourceKind::StorageImage;
        sb.access = Some(ResourceAccess::Storage);
        rs.bindings.push(sb);
        assert_eq!(census_reflection_wellformed(&rs, 0), 0);
        assert_eq!(
            crate::m2v_cache::project_shader_interface(&rs)
                .storage_image_format(TEXTURE_BINDING_BASE + 1),
            Some(reims_vgpu_protocol::StorageImageFormat::R32Float)
        );
        let mut missing_storage_format = rs.clone();
        missing_storage_format.bindings[0]
            .texture_shape
            .as_mut()
            .unwrap()
            .storage_format = None;
        assert_eq!(
            census_reflection_wellformed(&missing_storage_format, 0),
            0,
            "runtime specialization may prove a writable formatless image safe"
        );
        let mut sampled_with_storage_format = r.clone();
        sampled_with_storage_format.bindings[0]
            .texture_shape
            .as_mut()
            .unwrap()
            .storage_format = Some(metal2vulkan::meta::TextureFormat::R32f);
        assert_eq!(
            census_reflection_wellformed(&sampled_with_storage_format, 0),
            1
        );

        // Texture arrays use access, not kind, to choose sampled versus
        // storage descriptors, and their reflected descriptor width is part
        // of the ABI rather than an executor default.
        let mut storage_array = texture_binding(
            TEXTURE_BINDING_BASE + 3,
            TextureShape {
                array_ref: true,
                ..shape(TextureDimension::D2, false, true)
            },
        );
        storage_array.kind = ResourceKind::TextureArray;
        storage_array.descriptor.as_mut().unwrap().count = 8;
        let mut rsa = empty_reflection(ShaderStage::Kernel);
        rsa.bindings.push(storage_array);
        assert_eq!(census_reflection_wellformed(&rsa, 0), 0);
        let mut zero_count = rsa.clone();
        zero_count.bindings[0].descriptor.as_mut().unwrap().count = 0;
        assert_eq!(census_reflection_wellformed(&zero_count, 0), 1);
        let mut scalar_wide = r.clone();
        scalar_wide.bindings[0].descriptor.as_mut().unwrap().count = 8;
        assert_eq!(census_reflection_wellformed(&scalar_wide, 0), 1);
        let mut wrong_set = r.clone();
        wrong_set.bindings[0].descriptor.as_mut().unwrap().set = 1;
        assert_eq!(census_reflection_wellformed(&wrong_set, 0), 1);

        let mut embedded_storage_array = rsa.bindings[0].clone();
        embedded_storage_array.kind = ResourceKind::EmbeddedArgBufferTexture;
        embedded_storage_array
            .texture_shape
            .as_mut()
            .unwrap()
            .array_length = Some(8);
        embedded_storage_array.embedded_source = Some(metal2vulkan::reflect::EmbeddedArgBuffer {
            buffer_param_index: 0,
            buffer_index: 0,
            field_offset: 16,
            field_ordinal: 1,
            argument_index: 2,
            resource_buffer_index: None,
        });
        let mut embedded = empty_reflection(ShaderStage::Kernel);
        embedded.bindings.push(embedded_storage_array);
        assert_eq!(census_reflection_wellformed(&embedded, 0), 0);
        let embedded_resource = first_unsupported_vulkan_resource(&embedded).unwrap();
        assert_eq!(
            unsupported_vulkan_resource_kind_name(embedded_resource.kind),
            Some("embedded_texture")
        );
        assert_eq!(
            first_unsupported_vulkan_interface(&embedded, ShaderStage::Fragment),
            Some(UnsupportedVulkanInterface {
                feature: "shader_stage_kernel",
                count: 1,
            })
        );

        // A missing buffer access is the translator's documented conservative
        // answer and stays valid; an image access class on a buffer is not.
        let mut buffers = empty_reflection(ShaderStage::Kernel);
        let mut read = buffer_binding(0, Some(BufferExtent::Unbounded));
        read.access = Some(ResourceAccess::ReadOnly);
        buffers.bindings.push(read);
        assert_eq!(census_reflection_wellformed(&buffers, 0), 0);
        let mut missing_buffer_access = buffers.clone();
        missing_buffer_access.bindings[0].access = None;
        assert_eq!(census_reflection_wellformed(&missing_buffer_access, 0), 0);
        let mut wrong_buffer_access = buffers.clone();
        wrong_buffer_access.bindings[0].access = Some(ResourceAccess::Storage);
        assert_eq!(census_reflection_wellformed(&wrong_buffer_access, 0), 1);
        let mut zero_object_extent = buffers.clone();
        zero_object_extent.bindings[0].extent = Some(BufferExtent::Object { bytes: 0 });
        assert_eq!(census_reflection_wellformed(&zero_object_extent, 0), 1);
        let mut buffer_array = buffers.clone();
        buffer_array.bindings[0].descriptor.as_mut().unwrap().count = 2;
        assert_eq!(census_reflection_wellformed(&buffer_array, 0), 1);

        // Desync writable=true while kind stays Texture: one violation.
        let mut rbad = empty_reflection(ShaderStage::Fragment);
        let mut writable_desync = texture_binding(bind, shape(TextureDimension::D2, false, true));
        writable_desync.access = Some(ResourceAccess::Sampled);
        rbad.bindings.push(writable_desync);
        assert_eq!(census_reflection_wellformed(&rbad, 0), 1);

        // Desync access=Storage while kind stays Texture: one violation.
        let mut racc = empty_reflection(ShaderStage::Fragment);
        let mut ba = texture_binding(bind, shape(TextureDimension::D2, false, false));
        ba.access = Some(ResourceAccess::Storage);
        racc.bindings.push(ba);
        assert_eq!(census_reflection_wellformed(&racc, 0), 1);

        // A stale reflection ABI version is a violation on its own.
        let mut rver = empty_reflection(ShaderStage::Fragment);
        rver.reflection_version = REFLECTION_VERSION.wrapping_add(1);
        assert_eq!(census_reflection_wellformed(&rver, 0), 1);
    }

    #[test]
    fn unavailable_function_constants_reported_only_when_present() {
        // FC-free shader: silent (returns 0), no analysis line.
        let none = empty_reflection(ShaderStage::Fragment);
        assert_eq!(log_unavailable_function_constants(&none), 0);

        // Shader declaring runtime function constants: consumed straight from the
        // reflection (single source of truth), reported once. The count is the
        // inventory length — no SPIR-V re-walk.
        let mut r = empty_reflection(ShaderStage::Kernel);
        r.entry_point = Some("gaussian_blur".to_string());
        r.function_constants = vec![
            FunctionConstant {
                index: 0,
                name: "enable_tap".to_string(),
                type_name: "i1".to_string(),
                abi_type_encoding: "b".to_string(),
            },
            FunctionConstant {
                index: 3,
                name: "channel_count".to_string(),
                type_name: "i32".to_string(),
                abi_type_encoding: "i".to_string(),
            },
        ];
        assert_eq!(log_unavailable_function_constants(&r), 2);
    }

    #[test]
    fn reflects_storage_image_content_access_without_names() {
        // %1=image type, %2=pointer, %3=variable(binding 34), %4=loaded
        // image, %5=read result. The first module reads and writes; removing
        // OpImageRead proves write-only access without decorations or names.
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 6, 0];
        words.extend([
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            1,
            99,
            1,
            0,
            0,
            0,
            2,
            32,
        ]);
        words.extend([
            (4u32 << 16) | OP_TYPE_POINTER as u32,
            2,
            STORAGE_CLASS_UNIFORM_CONSTANT,
            1,
        ]);
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            3,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 34]);
        words.extend([(4u32 << 16) | OP_LOAD as u32, 1, 4, 3]);
        let write_at = words.len();
        words.extend([(4u32 << 16) | OP_IMAGE_WRITE as u32, 4, 90, 91]);
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::WriteOnly)
        );
        words.splice(
            write_at..write_at,
            [(5u32 << 16) | OP_IMAGE_READ as u32, 92, 5, 4, 90],
        );
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::ReadWrite)
        );

        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            5,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 5, DECORATION_BINDING, 34]);
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::AmbiguousBinding)
        );
    }

    /// The two reflectors share `propagate_derived` but seed it differently, and
    /// the difference is not cosmetic.
    ///
    /// `buffer_access` tracks pointers and seeds the descriptor variable itself;
    /// its escape scan enumerates the opcodes it cares about, so naming the
    /// variable elsewhere is harmless. `storage_image_access` tracks image
    /// *values* and must not seed the variable, because its escape scan ends in a
    /// catch-all that treats any instruction mentioning a tracked id as an
    /// escape. `OpDecorate` names the variable in every real module, so a seeded
    /// root would make every storage image reflect `Unknown` — silently, and in
    /// the direction that looks like caution.
    ///
    /// This module is the ordinary read/write case plus the `OpEntryPoint`
    /// interface list, which is the second place a variable id appears.
    #[test]
    fn storage_image_root_is_not_seeded_so_naming_the_variable_is_not_an_escape() {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 6, 0];
        words.extend([
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            1,
            99,
            1,
            0,
            0,
            0,
            2,
            32,
        ]);
        words.extend([
            (4u32 << 16) | OP_TYPE_POINTER as u32,
            2,
            STORAGE_CLASS_UNIFORM_CONSTANT,
            1,
        ]);
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            3,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 34]);
        // OpEntryPoint's interface list names every module-scope variable.
        words.extend([(5u32 << 16) | 15u32, 5, 100, 0, 3]);
        words.extend([(4u32 << 16) | OP_LOAD as u32, 1, 4, 3]);
        words.extend([(5u32 << 16) | OP_IMAGE_READ as u32, 92, 5, 4, 90]);
        words.extend([(4u32 << 16) | OP_IMAGE_WRITE as u32, 4, 90, 91]);
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::ReadWrite),
            "the variable being decorated and listed as an interface is not an escape"
        );
    }

    /// The smallest module a Vulkan validator accepts: an empty `GLCompute`
    /// entry point with a declared local size.
    fn minimal_compute_module() -> Vec<u32> {
        const OP_CAPABILITY: u32 = 17;
        const OP_MEMORY_MODEL: u32 = 14;
        const OP_ENTRY_POINT: u32 = 15;
        const OP_EXECUTION_MODE: u32 = 16;
        const OP_TYPE_VOID: u32 = 19;
        const OP_TYPE_FUNCTION: u32 = 33;
        const OP_FUNCTION: u32 = 54;
        const OP_FUNCTION_END: u32 = 56;
        const OP_LABEL: u32 = 248;
        const OP_RETURN: u32 = 253;
        let mut w = vec![0x0723_0203, 0x0001_0000, 0, 5, 0];
        w.extend([(2 << 16) | OP_CAPABILITY, 1]); // Shader
        w.extend([(3 << 16) | OP_MEMORY_MODEL, 0, 1]); // Logical GLSL450
        w.extend([(5 << 16) | OP_ENTRY_POINT, 5, 3, 0x6E69_616D, 0]); // GLCompute %3 "main"
        w.extend([(6 << 16) | OP_EXECUTION_MODE, 3, 17, 1, 1, 1]); // LocalSize 1 1 1
        w.extend([(2 << 16) | OP_TYPE_VOID, 1]);
        w.extend([(3 << 16) | OP_TYPE_FUNCTION, 2, 1]);
        w.extend([(5 << 16) | OP_FUNCTION, 1, 3, 0, 2]);
        w.extend([(2 << 16) | OP_LABEL, 4]);
        w.extend([1 << 16 | OP_RETURN]);
        w.extend([1 << 16 | OP_FUNCTION_END]);
        w
    }

    /// The gate that stands between a guest's shader and undefined behaviour
    /// inside a driver.
    ///
    /// A valid module passes and an invalid one is named, which is the whole
    /// contract — the device declines the dispatch rather than handing the
    /// driver something it is entitled to crash on, and one driver does.
    ///
    /// Skips where SPIRV-Tools is not installed, because there the function is
    /// specified to accept: an absent instrument is not a verdict.
    #[test]
    fn a_module_a_validator_rejects_never_reaches_a_driver() {
        let good = minimal_compute_module();
        if validate(&good) != SpirvValidation::Accepted {
            eprintln!("skip: no usable spirv-val, or it rejects the minimal module");
            return;
        }
        // Point the function's return type at the function id instead of at
        // %void, which is a type error no driver has to survive.
        let mut bad = good.clone();
        let at = bad
            .iter()
            .position(|w| *w == (5 << 16) | 54)
            .expect("OpFunction");
        bad[at + 1] = 3;
        match validate(&bad) {
            SpirvValidation::Rejected(why) => assert!(!why.is_empty(), "the refusal must say why"),
            SpirvValidation::Accepted => panic!("a type-broken module was accepted"),
        }
    }

    /// Concurrent validations do not answer for each other.
    ///
    /// The validator wrapper writes a fixed file name inside the directory it
    /// is handed, so a shared directory makes two callers race over one path
    /// and the loser gets a verdict about bytes it did not submit. That is not
    /// hypothetical: it rejected three good modules on a working macos-13 boot,
    /// where the concurrency comes from this device's drain thread and
    /// `metal2vulkan`'s own async translations sharing `/tmp`.
    ///
    /// Skips where SPIRV-Tools is absent, where every answer is `Accepted` and
    /// the test could not fail.
    #[test]
    fn concurrent_validations_do_not_collide() {
        let good = minimal_compute_module();
        if validate(&good) != SpirvValidation::Accepted {
            eprintln!("skip: no usable spirv-val");
            return;
        }
        let verdicts: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let words = good.clone();
                    scope.spawn(move || validate(&words))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for verdict in &verdicts {
            assert_eq!(*verdict, SpirvValidation::Accepted, "{verdicts:?}");
        }
    }

    #[test]
    fn semantic_shader_interface_preserves_texture_classification() {
        let shape = TextureShape {
            dimension: metal2vulkan::meta::TextureDimension::D2,
            arrayed: false,
            multisampled: false,
            component: TextureComponent::Float,
            writable: true,
            array_ref: false,
            array_length: None,
            storage_format: Some(metal2vulkan::meta::TextureFormat::R32f),
        };
        let mut reflection = empty_reflection(ShaderStage::Kernel);
        reflection.bindings.push(texture_binding(37, shape));
        let interface = crate::m2v_cache::project_shader_interface(&reflection);

        assert_eq!(
            reflected_texture_descriptor(&reflection, 5),
            interface.texture_descriptor(5)
        );
        assert_eq!(
            reflected_compute_texture(&reflection, 37),
            interface.compute_texture(37)
        );
        assert_eq!(
            interface.storage_image_format(37),
            Some(reims_vgpu_protocol::StorageImageFormat::R32Float)
        );
    }

    #[test]
    fn semantic_shader_interface_preserves_buffer_access_and_bounds() {
        let mut reflection = empty_reflection(ShaderStage::Kernel);
        let mut buffer = buffer_binding(3, Some(BufferExtent::Object { bytes: 512 }));
        buffer.access = Some(ResourceAccess::ReadOnly);
        buffer.footprint = Some(BufferFootprint {
            static_ranges: vec![BufferByteRange {
                offset: 8,
                size: 16,
            }],
            strided_accesses: vec![BufferStridedAccess {
                base_offset: 32,
                access_size: 4,
                terms: vec![BufferStrideTerm {
                    source: BufferIndexSource::GlobalInvocationIdX,
                    stride: 8,
                }],
            }],
            has_unbounded_access: false,
        });
        reflection.bindings.push(buffer);
        let interface = crate::m2v_cache::project_shader_interface(&reflection);

        assert_eq!(
            reflected_buffer_access(&reflection, 3),
            interface.buffer_access(3)
        );
        assert_eq!(
            reflected_compute_buffer_extent(&reflection, 3, [4, 1, 1], [2, 1, 1]),
            reflected_compute_buffer_extent_interface(&interface, 3, [4, 1, 1], [2, 1, 1])
        );
    }
}
