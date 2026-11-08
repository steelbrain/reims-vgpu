//! Compiles the library's own shaders, the metal crate's `src/gpu/shaders.metal`, into
//! `$OUT_DIR/shaders.metallib` with Apple's tools, which `src/gpu/shaders.rs` includes, and the
//! command predication's step kernel, `src/gpu/predication.metal`, into
//! `$OUT_DIR/predication.metallib`, which `src/gpu/predication.rs` includes, the vertex function
//! of the pipeline that ends a pass over a mutable rate map, `src/gpu/no_draw.metal`, into
//! `$OUT_DIR/no_draw.metallib`, which `src/gpu/rate_map.rs` includes, and the tests' kernel
//! that encodes indirect commands, `src/gpu/icb_encode.metal`, into `$OUT_DIR/icb_encode.metallib`,
//! the tests' render functions that read barycentric weights, `src/gpu/barycentric.metal`, into
//! `$OUT_DIR/barycentric.metallib`, those that read the vertex ID, `src/gpu/vertex_ids.metal`, into
//! `$OUT_DIR/vertex_ids.metallib`, the kernel that reaches a buffer through an address it loads,
//! `src/gpu/reach.metal`, into `$OUT_DIR/reach.metallib`, and those that write a sweep of values
//! through textures, `src/gpu/rounding.metal`, into `$OUT_DIR/rounding.metallib`, and again
//! with fast math off (`-fno-fast-math`) into `$OUT_DIR/rounding_safe.metallib`, the tests'
//! kernels that read YCbCr textures, `src/gpu/ycbcr_probe.metal`, into
//! `$OUT_DIR/ycbcr_probe.metallib`, the tests' functions Metal's compiler refuses,
//! `src/gpu/refused.metal`, into `$OUT_DIR/refused.metallib`, those that use SIMD-group
//! operations in each stage, `src/gpu/simd_stages.metal`, into `$OUT_DIR/simd_stages.metallib`,
//! the depth-denormal test's render functions, `src/gpu/depth_denormals.metal`, into
//! `$OUT_DIR/depth_denormals.metallib`, the tests' kernels that write a slice of an imageblock into a
//! texture, `src/gpu/slice_write_repro.metal`, into `$OUT_DIR/slice_write_repro.metallib`, the
//! tests' vertex function that returns nothing, `src/gpu/unrasterized.metal`, into
//! `$OUT_DIR/unrasterized.metallib`, the tests' fragment function that writes colour 1 alone,
//! `src/gpu/icb_color1.metal`, into `$OUT_DIR/icb_color1.metallib`, the tests' functions whose
//! pipeline states take sampler IDs, `src/gpu/numbered_samplers.metal`, into
//! `$OUT_DIR/numbered_samplers.metallib`, the tests' kernel whose pipeline state numbers five
//! constexpr samplers, `src/gpu/five_samplers.metal`, into `$OUT_DIR/five_samplers.metallib`,
//! the tests' kernel that keeps the GPU busy, `src/gpu/spin.metal`, into
//! `$OUT_DIR/spin.metallib`, the tests' kernels that read, sample and write textures of Apple's
//! RGBX formats, `src/gpu/rgbx.metal`, into `$OUT_DIR/rgbx.metallib`, the tests' Core Image
//! kernel, `src/gpu/ci_kernel.metal`,
//! compiled as Core Image compiles one (`-fcikernel`, then `metallib -cikernel`) into
//! `$OUT_DIR/ci_kernel.metallib`, the tests' mesh render pipelines' functions,
//! `src/gpu/mesh.metal`, into `$OUT_DIR/mesh.metallib`, the tests' kernel of a function constant
//! it reads only where defined, `src/gpu/optional_constant.metal`, into
//! `$OUT_DIR/optional_constant.metallib`, and those that import a symbol from a
//! dynamic library, `src/gpu/refused_import.metal`, linked into three metallibs ([`imports`]), and
//! the shell tests' stitchable functions, `guest/bundle/tests/stitchable.metal`, into
//! `$OUT_DIR/stitchable.metallib`, with the library Metal's stitcher makes of them into
//! `$OUT_DIR/stitched.metallib` ([`stitched`]): the runtime makes libraries of metallibs, not of
//! source. The two commands compile with the language Metal 4.0
//! and fast math, which are what Metal's own `newLibraryWithSource:options:error:` compiles the
//! same source with: a new `MTLCompileOptions` reads `languageVersion` Metal 4.0 and `mathMode`
//! Fast (measured on an M3 Max, macOS 26.7, 25G229). A Mac needs Xcode's Metal toolchain
//! (`xcrun -sdk macosx metal`); a `MACOSX_DEPLOYMENT_TARGET` changes no byte of the output.
//! **Measured** with Metal 32023.922 on an M3 Max (2026-11-07): a compile in another directory
//! writes the same metallib (the AIR names the source's path, the metallib does not), and so does
//! a second compile, the stitched library's included. The bytes follow the toolchain: a metallib
//! records the compiler's version and its AIR version and target (`air64_v29-apple-macosx27.2.0`
//! there, `air64_v28-apple-macosx26.0.0` from Metal 32023.921), so a Mac with another Xcode
//! writes each file again with the same sources.
//!
//! Any other host (Linux) takes the metallibs a Mac compiled instead: those committed in
//! `metallibs/` beside this file ([`COMMITTED`]), or those in the directory `RVG_METALLIB_DIR`
//! names. Each file is copied once the directory's `metallibs.toml` names it with the sha256 of
//! the commands and sources this tree would compile it from ([`digest`]); a file it names with
//! another fails the build, which `cargo xtask metallibs` on a Mac mends. That command builds with
//! `RVG_METALLIB_EXPORT=<dir>` (`metallibs/` unless it is given one), which compiles as above and
//! then copies every portable file ([`Step::portable`]) and the record into `<dir>`. Every build
//! writes the record of this tree's sources into `$OUT_DIR/metallibs.toml` too, which a test
//! holds `metallibs/`'s equal to (`gpu::shaders`), so a source changed without its metallib fails
//! `cargo xtask check` on a Mac. The record names no Xcode, so a Mac always compiles its own with
//! its Xcode, and ignores `RVG_METALLIB_DIR`, with a warning.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

/// The commands that compile a metallib of one source: to AIR, then the library.
const COMPILE: [&str; 3] = ["metal", "-std=metal4.0", "-c"];

/// The commands that compile a metallib of one source with fast math off.
const COMPILE_SAFE: [&str; 4] = ["metal", "-std=metal4.0", "-fno-fast-math", "-c"];

/// The commands that compile a dynamic library, before its install name.
const DYNAMIC: [&str; 4] = ["metal", "-std=metal4.0", "-dynamiclib", "-install_name"];

/// The commands that link a source against a dynamic library, before the search path and name.
const LINK: [&str; 2] = ["metal", "-std=metal4.0"];

/// The commands that compile a Core Image kernel library of one source, as Core Image's own,
/// read from standard input: of a file, `-fcikernel` writes the file's absolute path into the
/// metallib, and of standard input no path (measured with Apple metal 32023.922).
const CI_KERNEL: [&str; 6] = ["metal", "-fcikernel", "-x", "metal", "-c", "-"];

/// The commands that make the metallib of a Core Image kernel.
const CI_KERNEL_LIBRARY: [&str; 2] = ["metallib", "-cikernel"];

/// The commands that build `src/gpu/stitch.m`, the stitcher [`stitched`] runs.
const STITCH_TOOL: [&str; 6] = [
    "clang",
    "-fobjc-arc",
    "-framework",
    "Metal",
    "-framework",
    "Foundation",
];

/// The tests' metallibs of one source each, `src/gpu/<name>.metal`.
const TEST_LIBRARIES: [&str; 23] = [
    "predication",
    "no_draw",
    "icb_encode",
    "barycentric",
    "vertex_ids",
    "reach",
    "rounding",
    "ycbcr_probe",
    "refused",
    "simd_stages",
    "depth_denormals",
    "slice_write_repro",
    "unrasterized",
    "icb_color1",
    "numbered_samplers",
    "five_samplers",
    "spin",
    "rgbx",
    "argument_probe",
    "linked_caller",
    "linked_callee",
    "mesh",
    "optional_constant",
];

/// The files [`imports`] makes.
const IMPORT_FILES: [&str; 7] = [
    "refused_other.metallib",
    "librefused_link_ext.metallib",
    "librefused_link_other.metallib",
    "librefused_link_absent.metallib",
    "refused_import_ext.metallib",
    "refused_import_other.metallib",
    "refused_import_absent.metallib",
];

/// One step of the build: the files it makes in `OUT_DIR`, what makes them (the commands, as
/// recorded), and the sources they are made of.
struct Step {
    files: Vec<String>,
    commands: String,
    sources: Vec<PathBuf>,
    /// Whether its files are the same bytes wherever they are built, so that a build elsewhere
    /// can take them. The import step's ([`imports`]) are not: their install names are paths in
    /// the building `OUT_DIR`. Only the macOS tests read them (`states::tests::metal_tests`), so a
    /// build elsewhere goes without them.
    portable: bool,
}

/// Every step, in order.
fn steps(manifest: &Path) -> Vec<Step> {
    let gpu = manifest.join("src/gpu");
    let one = |name: &str, source: PathBuf, compile: &[&str]| Step {
        files: vec![format!("{name}.metallib")],
        commands: format!("{} | metallib", compile.join(" ")),
        sources: vec![source],
        portable: true,
    };
    let mut steps = vec![one(
        "shaders",
        manifest.join("../reims-vgpu-metal/src/gpu/shaders.metal"),
        &COMPILE,
    )];
    for name in TEST_LIBRARIES {
        steps.push(one(name, gpu.join(format!("{name}.metal")), &COMPILE));
    }
    steps.push(one(
        "rounding_safe",
        gpu.join("rounding.metal"),
        &COMPILE_SAFE,
    ));
    steps.push(Step {
        files: vec!["ci_kernel.metallib".to_owned()],
        commands: format!("{} | {}", CI_KERNEL.join(" "), CI_KERNEL_LIBRARY.join(" ")),
        sources: vec![gpu.join("ci_kernel.metal")],
        portable: true,
    });
    steps.push(Step {
        files: vec![
            "stitchable.metallib".to_owned(),
            "stitched.metallib".to_owned(),
        ],
        commands: format!(
            "{} | metallib | {} | stitch",
            COMPILE.join(" "),
            STITCH_TOOL.join(" ")
        ),
        sources: vec![
            manifest.join("../../guest/bundle/tests/stitchable.metal"),
            gpu.join("stitch.m"),
        ],
        portable: true,
    });
    steps.push(Step {
        files: IMPORT_FILES.map(str::to_owned).to_vec(),
        commands: format!("{} | {} -L -l", DYNAMIC.join(" "), LINK.join(" ")),
        sources: ["refused_ext", "refused_other", "refused_import"]
            .map(|name| gpu.join(format!("{name}.metal")))
            .to_vec(),
        portable: false,
    });
    steps
}

/// The sha256 of a step's commands and sources, in hex: what `metallibs.toml` records of each of
/// its files.
fn digest(step: &Step) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(step.commands.as_bytes());
    for source in &step.sources {
        let bytes = std::fs::read(source).map_err(|e| format!("{}: {e}", source.display()))?;
        let name = source
            .file_name()
            .map(|n| n.as_encoded_bytes())
            .unwrap_or_default();
        hash.update([0]);
        hash.update(name);
        hash.update([0]);
        hash.update(&bytes);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The record of a directory of metallibs: `metallibs.toml`'s name.
const RECORD: &str = "metallibs.toml";

/// The directory beside this file that holds the metallibs a Mac compiled, which a build on any
/// other host takes when `RVG_METALLIB_DIR` names none.
const COMMITTED: &str = "metallibs";

fn main() -> Result<(), String> {
    let manifest = PathBuf::from(var("CARGO_MANIFEST_DIR")?);
    let out = PathBuf::from(var("OUT_DIR")?);
    println!("cargo:rerun-if-env-changed=RVG_METALLIB_DIR");
    println!("cargo:rerun-if-env-changed=RVG_METALLIB_EXPORT");
    let steps = steps(&manifest);
    for source in steps.iter().flat_map(|s| &s.sources) {
        println!("cargo:rerun-if-changed={}", source.display());
    }
    write_record(&steps, &out.join(RECORD))?;
    let dir = std::env::var_os("RVG_METALLIB_DIR");
    if !cfg!(target_os = "macos") {
        let dir = dir.map_or_else(|| manifest.join(COMMITTED), PathBuf::from);
        return copy_checked(&steps, &dir, &out);
    }
    if dir.is_some() {
        println!(
            "cargo:warning=RVG_METALLIB_DIR is ignored on a Mac, which compiles its metallibs \
             with its own Xcode"
        );
    }
    compile(&manifest, &out)?;
    match std::env::var_os("RVG_METALLIB_EXPORT") {
        Some(dir) => export(&steps, &out, Path::new(&dir)),
        None => Ok(()),
    }
}

/// Compiles every metallib into `out` with Apple's tools (module docs).
fn compile(manifest: &Path, out: &Path) -> Result<(), String> {
    let source = manifest.join("../reims-vgpu-metal/src/gpu/shaders.metal");
    xcrun(&COMPILE, &source, &out.join("shaders.air"))?;
    xcrun(
        &["metallib"],
        &out.join("shaders.air"),
        &out.join("shaders.metallib"),
    )?;
    for name in TEST_LIBRARIES {
        let source = manifest.join(format!("src/gpu/{name}.metal"));
        let air = out.join(format!("{name}.air"));
        let metallib = out.join(format!("{name}.metallib"));
        xcrun(&COMPILE, &source, &air)?;
        xcrun(&["metallib"], &air, &metallib)?;
    }
    // The same functions with fast math off, for the tests that compare a draw's blend in both
    // math modes.
    let source = manifest.join("src/gpu/rounding.metal");
    let (air, metallib) = (
        out.join("rounding_safe.air"),
        out.join("rounding_safe.metallib"),
    );
    xcrun(&COMPILE_SAFE, &source, &air)?;
    xcrun(&["metallib"], &air, &metallib)?;
    // The tests' Core Image kernel library.
    let source = manifest.join("src/gpu/ci_kernel.metal");
    let (air, metallib) = (out.join("ci_kernel.air"), out.join("ci_kernel.metallib"));
    xcrun_stdin(&CI_KERNEL, &source, &air)?;
    xcrun(&CI_KERNEL_LIBRARY, &air, &metallib)?;
    stitched(manifest, out)?;
    imports(&manifest.join("src/gpu"), out)
}

/// Compiles the shell tests' stitchable functions into `stitchable.metallib`, and writes into
/// `stitched.metallib` the library Metal's stitcher makes of its graph `fused` over them, as the
/// bytes Metal answers for it: what a guest sends of a stitched library, which only Metal's
/// stitcher makes. `src/gpu/stitch.m` stitches, on the CPU, on the Mac's default device.
fn stitched(manifest: &Path, out: &Path) -> Result<(), String> {
    let source = manifest.join("../../guest/bundle/tests/stitchable.metal");
    let (air, stitchable, stitched) = (
        out.join("stitchable.air"),
        out.join("stitchable.metallib"),
        out.join("stitched.metallib"),
    );
    xcrun(&COMPILE, &source, &air)?;
    xcrun(&["metallib"], &air, &stitchable)?;
    let tool = out.join("stitch");
    xcrun(&STITCH_TOOL, &manifest.join("src/gpu/stitch.m"), &tool)?;
    let mut cmd = Command::new(&tool);
    cmd.arg(&stitchable).arg(&stitched);
    let status = cmd.status().map_err(|e| format!("{cmd:?}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{cmd:?}: {status}"))
    }
}

/// Copies the metallibs of every portable step from `dir` into `out`, once `dir`'s record names
/// each with the hash of this tree's commands and sources.
fn copy_checked(steps: &[Step], dir: &Path, out: &Path) -> Result<(), String> {
    let record_path = dir.join(RECORD);
    println!("cargo:rerun-if-changed={}", record_path.display());
    let text = std::fs::read_to_string(&record_path).map_err(|e| {
        format!(
            "{}: {e} (run `cargo xtask metallibs {}` on a Mac)",
            record_path.display(),
            dir.display()
        )
    })?;
    let record: BTreeMap<String, String> = text
        .lines()
        .filter_map(|line| {
            let (file, hash) = line.split_once(" = ")?;
            Some((
                file.trim_matches('"').to_owned(),
                hash.trim_matches('"').to_owned(),
            ))
        })
        .collect();
    for step in steps.iter().filter(|s| s.portable) {
        let hash = digest(step)?;
        for file in &step.files {
            if record.get(file) != Some(&hash) {
                return Err(format!(
                    "{file} in {} was not built from this tree's sources: re-run `cargo xtask \
                     metallibs {}` on a Mac",
                    dir.display(),
                    dir.display()
                ));
            }
            let from = dir.join(file);
            println!("cargo:rerun-if-changed={}", from.display());
            std::fs::copy(&from, out.join(file)).map_err(|e| format!("{}: {e}", from.display()))?;
        }
    }
    Ok(())
}

/// Copies the metallibs of every portable step from `out` into `dir`, with the record of the hash
/// each was built from.
fn export(steps: &[Step], out: &Path, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for step in steps.iter().filter(|s| s.portable) {
        for file in &step.files {
            let from = out.join(file);
            std::fs::copy(&from, dir.join(file)).map_err(|e| format!("{}: {e}", from.display()))?;
        }
    }
    let record_path = dir.join(RECORD);
    write_record(steps, &record_path)?;
    println!("cargo:rerun-if-changed={}", record_path.display());
    Ok(())
}

/// Writes at `path` the record of every portable step's files: each with the hash of the commands
/// and sources it is built from.
fn write_record(steps: &[Step], path: &Path) -> Result<(), String> {
    let mut record = String::from(
        "# The metallibs beside this file and the sha256 of the commands and sources each was\n\
         # built from (cargo xtask metallibs; reims-vgpu-vulkan/build.rs).\n",
    );
    for step in steps.iter().filter(|s| s.portable) {
        let hash = digest(step)?;
        for file in &step.files {
            record.push_str(&format!("\"{file}\" = \"{hash}\"\n"));
        }
    }
    std::fs::write(path, record).map_err(|e| format!("{}: {e}", path.display()))
}

/// Links `refused_import.metal`, whose functions import `ext`, against three dynamic libraries
/// in `out`, each of an install name in `out`: `refused_import_ext.metallib` against one that
/// exports it, at its own file; `refused_import_other.metallib` against one whose install name's
/// file, `refused_other.metallib`, exports `other` instead; `refused_import_absent.metallib`
/// against one whose install name has no file. Metal loads an install name's library as it makes
/// a pipeline, and so does the runtime (`states::tests`).
fn imports(dir: &Path, out: &Path) -> Result<(), String> {
    let source = |name: &str| {
        let path = dir.join(format!("{name}.metal"));
        println!("cargo:rerun-if-changed={}", path.display());
        path
    };
    let (ext, other, import) = (
        source("refused_ext"),
        source("refused_other"),
        source("refused_import"),
    );
    let at = |name: &str| out.join(format!("{name}.metallib"));
    // Each library linked against: its file, and the install name the importer records.
    let linked = [
        (
            "ext",
            &ext,
            at("librefused_link_ext"),
            at("librefused_link_ext"),
        ),
        (
            "other",
            &ext,
            at("librefused_link_other"),
            at("refused_other"),
        ),
        (
            "absent",
            &ext,
            at("librefused_link_absent"),
            at("refused_absent"),
        ),
    ];
    let other_name = at("refused_other");
    dynamic_library(&other, &other_name, &other_name)?;
    let _ = std::fs::remove_file(at("refused_absent"));
    for (what, library, file, install_name) in &linked {
        dynamic_library(library, file, install_name)?;
        let link = format!("-lrefused_link_{what}");
        let search = format!("-L{}", out.display());
        let [metal, std] = LINK;
        xcrun(
            &[metal, std, &search, &link],
            &import,
            &at(&format!("refused_import_{what}")),
        )?;
    }
    Ok(())
}

/// Compiles `source` into the dynamic library `output`, of the install name `install_name`.
fn dynamic_library(source: &Path, output: &Path, install_name: &Path) -> Result<(), String> {
    let install_name = install_name
        .to_str()
        .ok_or_else(|| format!("{}: not UTF-8", install_name.display()))?;
    let [metal, std, dynamic, name] = DYNAMIC;
    xcrun(&[metal, std, dynamic, name, install_name], source, output)
}

/// What a build that compiles the metallibs, a Mac's, needs.
const NEEDS: &str = "on a Mac the shaders need Xcode's Metal toolchain (`xcodebuild \
                     -downloadComponent MetalToolchain`)";

/// The build script's environment variable `name`.
fn var(name: &str) -> Result<String, String> {
    std::env::var(name).map_err(|e| format!("{name}: {e}"))
}

/// Runs `xcrun -sdk macosx <tool...> -o <output>` with `input` its standard input, failing on a
/// non-zero exit.
fn xcrun_stdin(tool: &[&str], input: &Path, output: &Path) -> Result<(), String> {
    let stdin = std::fs::File::open(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx"])
        .args(tool)
        .arg("-o")
        .arg(output)
        .stdin(stdin);
    let status = cmd
        .status()
        .map_err(|e| format!("{cmd:?}: {e} ({NEEDS})"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{cmd:?}: {status} ({NEEDS})"))
    }
}

/// Runs `xcrun -sdk macosx <tool...> <input> -o <output>`, failing on a non-zero exit.
fn xcrun(tool: &[&str], input: &Path, output: &Path) -> Result<(), String> {
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx"])
        .args(tool)
        .arg(input)
        .arg("-o")
        .arg(output);
    let status = cmd
        .status()
        .map_err(|e| format!("{cmd:?}: {e} ({NEEDS})"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{cmd:?}: {status} ({NEEDS})"))
    }
}
