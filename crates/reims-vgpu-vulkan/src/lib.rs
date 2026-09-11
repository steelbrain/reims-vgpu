//! The Vulkan rail: what a host device offers, and what this device does with
//! that.
//!
//! # Why this is a crate
//!
//! The architecture names one layer as the sole executor of the semantic model
//! and the only place host capabilities become placement and transfer policy.
//! "Only place" is a claim a module boundary cannot hold — a placement decision
//! one `use` away from device state will eventually read some — so it is a
//! crate whose dependency list says what it may see: `ash`, the semantic model,
//! the refusal vocabulary, and the operator switches. No QEMU, no device model,
//! no guest-RAM ownership, no decode.
//!
//! # The rule every gate here follows
//!
//! Gate on the **capability**, never on a driver name, a vendor id, an API
//! version, or `VK_KHR_portability_subset`. Vulkan 1.2 is the baseline on every
//! supported host; anything newer needs a capability-gated fallback. An
//! operator switch may narrow what this rail does and may never widen it.
//!
//! # The parts
//!
//! - [`bindings`] — what a draw has to re-emit, and far more often what it
//!   does not.
//! - [`buffer`] — why every guest buffer here can be bound as every class,
//!   and the two things a device can still refuse about one.
//! - [`census`] — what this physical device offers, taken once, and the floor
//!   it has to clear to be used at all. Every capability gate below reads from
//!   it, and it holds no device name for one to branch on.
//! - [`device`] — the `VkDevice`, the set of features and extensions its
//!   census admitted, and the identity every handle made from it carries.
//! - [`descriptor`] — which mechanism carries a draw's descriptors on this
//!   host, and what one emission is therefore allowed to write.
//! - [`frames`] — swapchain frame slots, and the binary semaphores that a
//!   failed frame leaves with a signal outstanding on them.
//! - [`host`] — the instance, the physical device this rail bound, and the
//!   only Vulkan state the architecture allows to be process-global.
//! - [`image`] — what a checked texture declaration becomes here, and the
//!   device query that has to admit it before anything is allocated.
//! - [`layout`] — which layout each image subresource is in, and the explicit
//!   transitions and ownership moves that get it to the next one.
//! - [`memory`] — how host and device memory relate on the bound physical
//!   device, and which memory type an allocation gets. A misclassification here
//!   is a performance bug and never a correctness one: topology selects a
//!   preference order, the required flags are always the fallback, and nothing
//!   may branch on topology in a way the guest can observe.
//! - [`raster`] — the fixed-function state a guest sets, and the two pieces of
//!   it that are host capabilities rather than mappings.
//! - [`record`] — issuing the planned commands into a command buffer, and the
//!   one choice it makes: which spelling of a barrier this host takes.
//! - [`recording`] — everything one native recording owns, held as one value
//!   from the slots it takes to the pipelines it releases.
//! - [`queues`] — which queue family this rail submits to, and the value that
//!   makes a `VkQueue` have exactly one owner.
//! - [`barrier`] — what a declared barrier becomes on this host, and which of
//!   the guest's stages this host has no equivalent for.
//! - [`mipmap`] — the blit ladder that fills a texture's mip chain, and the
//!   layout each rung has to be in before the next one reads it.
//! - [`pass`] — what a render-pass descriptor becomes here: its attachments,
//!   their operations, and the clear values read the way the format says.
//! - [`placement`] — where a resource's bytes live, decided in one order from
//!   the guest's declaration and this host's measured capabilities.
//! - [`pools`] — one command pool per worker, and the rule that a command
//!   buffer is recordable again only when the timeline says the GPU is done
//!   with it.
//! - [`variant`] — the native pipelines one semantic pipeline turns into, who
//!   is allowed to compile each, and why none of them is ever evicted.
//! - [`resident`] — which native object a guest resource name resolves to,
//!   and what becomes of the previous one when the guest reuses the name.
//! - [`sampler`] — what a checked sampler declaration becomes here, and the
//!   three things about it that are host capabilities.
//! - [`staging`] — host-visible scratch memory, sub-allocated linearly and
//!   returned only by the timeline.
//! - [`submission`] — what happens to a timeline point between reserving it
//!   and the GPU reaching it, and why a refused one is signalled rather than
//!   forgotten.
//! - [`transfer`] — the native copies a resolved transfer becomes, and the
//!   byte-to-texel pitch conversion that is exact or is a refusal.
//! - [`view`] — the whole-texture view a shader samples, and the one view per
//!   attachable slice a render pass has no other way to select.
//! - [`timeline`] — which value a submission signals, and the bookkeeping that
//!   keeps "reserved" and "reached" from being confused for each other.

// Calling Vulkan is unsafe by construction; a `forbid` here would be a lie the
// first entry point breaks. `unsafe_op_in_unsafe_fn` is the rule that actually
// carries: an unsafe function's body still has to say where it is relying on a
// precondition.
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod barrier;
pub mod bindings;
pub mod blend;
pub mod buffer;
pub mod census;
pub mod depth_stencil;
pub mod descriptor;
pub mod device;
pub mod frames;
pub mod host;
pub mod image;
pub mod layout;
pub mod memory;
pub mod mipmap;
pub mod pass;
pub mod pipeline;
pub mod pixel;
pub mod placement;
pub mod pools;
pub mod queues;
pub mod raster;
pub mod record;
pub mod recording;
pub mod renderpass;
pub mod resident;
pub mod sampler;
pub mod staging;
pub mod submission;
pub mod swapchain;
pub mod timeline;
pub mod topology;
pub mod transfer;
pub mod variant;
pub mod vertex;
pub mod view;

/// The dependency list, read back and checked against the claim above it.
///
/// # Why a test and not a comment
///
/// This crate's doc says "only place" is a claim a module boundary cannot hold,
/// and that the dependency list is what holds it instead. A list held only by a
/// comment is one a convenient import silently retires — and the import this
/// rail is most likely to reach for is the protocol crate, because every wire
/// vocabulary it wants is *in* there and also re-exported through the semantic
/// model. Reaching it directly compiles, passes every test here, and quietly
/// makes this rail a second decoder.
///
/// So the manifest is parsed and compared. The parser is a dozen lines here
/// rather than a shared helper elsewhere: a boundary test that had to depend on
/// something to run would be adding an edge to the graph it exists to bound.
#[cfg(test)]
mod boundary {
    /// Every crate named in one section of a manifest, in the order it lists
    /// them. Not just the in-workspace ones — `ash` is as much a part of this
    /// crate's claim as `reims-vgpu-core` is.
    fn deps(manifest: &str, section: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut inside = false;
        for line in manifest.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                inside = trimmed == section;
                continue;
            }
            if !inside || trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some(name) = trimmed.split('=').next().map(str::trim) {
                if !name.is_empty() {
                    out.push(name.to_string());
                }
            }
        }
        out
    }

    fn manifest() -> String {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
            .expect("a crate can read its own manifest")
    }

    /// `ash`, the semantic model, the refusal vocabulary, the operator
    /// switches. That is the whole list, and it is asserted as a list rather
    /// than as a set of prohibitions because the edge that breaks this is the
    /// one nobody thought to prohibit.
    #[test]
    fn the_rail_sees_ash_the_model_the_refusals_and_the_switches() {
        assert_eq!(
            deps(&manifest(), "[dependencies]"),
            vec![
                "ash",
                "reims-vgpu-core",
                "reims-vgpu-observe",
                "reims-vgpu-config",
            ],
            "this crate's dependency list is what makes \"the only place host \
             capabilities become policy\" true"
        );
    }

    /// And in particular not the protocol crate, which is the one edge that
    /// would be easy to justify and would make this rail a second decoder.
    ///
    /// Its own test because the list above would catch it, and because a reader
    /// who changes that list needs to meet this sentence rather than a diff.
    #[test]
    fn the_rail_never_reaches_the_decode_vocabulary_directly() {
        let m = manifest();
        for section in [
            "[dependencies]",
            "[dev-dependencies]",
            "[build-dependencies]",
        ] {
            for d in deps(&m, section) {
                assert!(
                    !matches!(
                        d.as_str(),
                        "reims-vgpu-protocol" | "reims-vgpu-wire" | "reims-vgpu-memory"
                    ),
                    "{section} names {d}: a wire tag's meaning reaches this rail \
                     through the semantic model or not at all, and guest-RAM \
                     ownership does not reach it"
                );
            }
        }
    }
}

/// How a guest enumerant's spelling becomes the Vulkan enumerant's.
///
/// For the tests, and only for them: two eight-arm mapping tables --- the
/// comparison functions and the stencil operations --- are written out by hand,
/// and what a hand-written table of same-named values gets wrong is a *swap*.
/// Injectivity cannot see one, and a spot-checked arm catches only itself. The
/// derivation here shares nothing with either table; it reads the guest name.
#[cfg(test)]
pub(crate) mod naming {
    /// The guest name a Vulkan enumerant's own name, under two vocabulary
    /// rules and nothing else.
    ///
    /// `LessEqual` is `LESS_OR_EQUAL` and `IncrementClamp` is
    /// `INCREMENT_AND_CLAMP`; every other word survives the split unchanged.
    /// Shared with [`crate::depth_stencil`], which maps the other
    /// eight-arm table.
    #[must_use]
    pub(crate) fn vulkan_spelling(camel: &str) -> String {
        let mut words: Vec<String> = Vec::new();
        for ch in camel.chars() {
            if ch.is_uppercase()
                && !words.is_empty()
                && !words.last().expect("non-empty").is_empty()
            {
                words.push(String::new());
            }
            if words.is_empty() {
                words.push(String::new());
            }
            words
                .last_mut()
                .expect("non-empty")
                .push(ch.to_ascii_uppercase());
        }
        // Word-for-word vocabulary, which is what makes this a derivation
        // rather than a second table: each substitution is a fact about how
        // the two APIs spell one concept, and every arm containing that word
        // takes it. Nothing here knows which arm it is looking at.
        for word in &mut words {
            *word = match word.as_str() {
                "SOURCE" => "SRC".to_string(),
                "SOURCE1" => "SRC1".to_string(),
                "DESTINATION" => "DST".to_string(),
                // Metal names the blend constant after the blend; Vulkan names
                // it after what it is.
                "BLEND" => "CONSTANT".to_string(),
                "SATURATED" => "SATURATE".to_string(),
                other => other.to_string(),
            };
        }
        // Metal writes the comparison as one word and Vulkan spells the
        // conjunction; the same for the stencil operations' saturation.
        // `NotEqual` is `NOT_EQUAL` in both, which is why the first rule asks
        // what the `EQUAL` follows rather than only that it is last.
        if words.len() > 1 {
            let last = words[words.len() - 1].clone();
            let before = words[words.len() - 2].clone();
            if last == "EQUAL" && (before == "LESS" || before == "GREATER") {
                words.insert(words.len() - 1, "OR".to_string());
            } else if last == "CLAMP" || last == "WRAP" {
                words.insert(words.len() - 1, "AND".to_string());
            }
        }
        words.join("_")
    }

    /// The same, for the topology table, whose one difference is a suffix
    /// rather than a word.
    ///
    /// Metal names the primitive and leaves the grouping implicit; Vulkan
    /// names both. So a guest type that does not say `Strip` is a list, and
    /// that is the whole rule --- which is exactly the distinction a pipeline
    /// once lost by declaring a stand-in topology, so it is worth a check that
    /// does not read the table.
    #[must_use]
    pub(crate) fn vulkan_topology_spelling(camel: &str) -> String {
        let spelled = vulkan_spelling(camel);
        if spelled.ends_with("_STRIP") {
            spelled
        } else {
            format!("{spelled}_LIST")
        }
    }
}

pub mod dmabuf;
