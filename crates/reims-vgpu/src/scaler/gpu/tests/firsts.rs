//! A process's first transforms on each kernel, as a guest's first transforms after its device
//! opened meet them: with nothing made ahead, as the making ahead starts, and once it is done
//! (`make_ahead`). Each prints, of each transform, its first wall time and the next, the
//! dispatches' kernels, and on the Vulkan platform what the runtime counted meanwhile: functions
//! translated and found translated, `VkPipeline`s made, and the driver's hits and misses in the
//! caches the makings were handed (`ScalerGpu::cache_counts`), which tell a cold making from a
//! cached one; as the making ahead starts, its own lines follow, which say when it made each
//! kernel. Run each alone, in its own process, for the record in docs/video/scaler.md, "Making
//! the kernels" (`--run-ignored only --no-capture`, in release), with the caches as the record
//! says: a cold first transform wants the runtime's shader cache new, which lies under `HOME`, and
//! the driver's off.

use super::*;

/// A first transform: its name, its source's format and size, its destination's, and its flags.
type First = (
    &'static str,
    ScalerFormat,
    (u32, u32),
    ScalerFormat,
    (u32, u32),
    u64,
);

/// One transform for each kernel and threadgroup size the transforms dispatch
/// ([`params::DISPATCHED`]).
const FIRSTS: [First; 6] = [
    (
        "2160p 420v to 2160p BGRA",
        ScalerFormat::Nv12VideoRange,
        (3840, 2160),
        ScalerFormat::Bgra,
        (3840, 2160),
        0x1000,
    ),
    (
        "1080p 420v to 720p BGRA",
        ScalerFormat::Nv12VideoRange,
        (1920, 1080),
        ScalerFormat::Bgra,
        (1280, 720),
        0x1000,
    ),
    (
        "2160p 420v to 1080p 420v",
        ScalerFormat::Nv12VideoRange,
        (3840, 2160),
        ScalerFormat::Nv12VideoRange,
        (1920, 1080),
        0x1000,
    ),
    (
        "720p BGRA to 180p BGRA",
        ScalerFormat::Bgra,
        (1280, 720),
        ScalerFormat::Bgra,
        (320, 180),
        0x1000,
    ),
    (
        "1080p BGRA turned 90",
        ScalerFormat::Bgra,
        (1920, 1080),
        ScalerFormat::Bgra,
        (1080, 1920),
        0x1004,
    ),
    (
        "240p BGRA turned 90 to 160p",
        ScalerFormat::Bgra,
        (320, 240),
        ScalerFormat::Bgra,
        (240, 160),
        0x1004,
    ),
];

/// How the probe meets the making ahead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ahead {
    /// None runs.
    None,
    /// The transforms start as it starts.
    Starting,
    /// The transforms start once it is done.
    Done,
}

/// The runtime's counts (module docs), where the GPU is the Vulkan platform's.
type Counts<'a> = &'a dyn Fn() -> Option<[u64; 5]>;

/// A first transform's status, the kernels and threadgroups its dispatches take, and the
/// `VkPipeline`s the runtime made while it ran, where it counts them.
type Taken = (IoReturn, Vec<(Kernel, (u32, u32))>, Option<u64>);

/// What the runtime counted from `then` to `now`, in words; nothing where it counts nothing.
fn counted(then: Option<[u64; 5]>, now: Option<[u64; 5]>) -> String {
    let (Some(then), Some(now)) = (then, now) else {
        return String::new();
    };
    let d: Vec<u64> = now.iter().zip(then).map(|(n, t)| n - t).collect();
    format!(
        "; translated +{}, found translated +{}, pipelines made +{}, driver hits +{}, misses +{}",
        d[0], d[1], d[2], d[3], d[4]
    )
}

/// The first transforms on `gpu` (module docs), with the runtime's `counts`: what each took.
fn firsts<G: Gpu>(name: &str, gpu: Arc<G>, ahead: Ahead, counts: Counts<'_>) -> Vec<Taken> {
    let mut out = Vec::new();
    let runtime = StdRuntime::shared();
    let opened = runtime.now();
    let before = counts();
    // As it starts, the making ahead's own lines say when it made each kernel, beside the
    // transforms'.
    let logs = (ahead == Ahead::Starting).then(|| {
        let logs = reims_vgpu_core::testkit::logs::capture();
        log::set_max_level(log::LevelFilter::Debug);
        logs
    });
    let answer = (ahead != Ahead::None).then(|| make_ahead(&gpu, &runtime).unwrap());
    if ahead == Ahead::Done {
        let deadline = runtime.deadline_in(Duration::from_secs(300));
        answer
            .as_ref()
            .unwrap()
            .wait_until(&*runtime, deadline)
            .unwrap();
        println!(
            "first transforms, {ahead:?}: {name}: made ahead in {:.0} ms{}",
            runtime.elapsed_since(opened).as_secs_f64() * 1000.0,
            counted(before, counts())
        );
    }
    let backend = GpuScalerBackend::new(gpu, Arc::clone(&runtime));
    let mut accelerator = backend.create_accelerator().unwrap();
    for (label, from, (sw, sh), to, (dw, dh), flags) in FIRSTS {
        let (sl, dl) = (layout(from, sw, sh, 64, 0), layout(to, dw, dh, 64, 0));
        let source = platform_surface(sl.clone(), &picture_bytes(&sl, None), None);
        let grey = picture_bytes(&dl, Some([0.5, 0.5, 0.5, 1.0]));
        let destination = platform_surface(dl.clone(), &grey, None);
        let transform = Transform {
            flags: TransformFlags(flags),
            ..Transform::default()
        };
        let plan = Plan::new(&transform, &sl, &dl).unwrap();
        let kernels: Vec<(Kernel, (u32, u32))> = params::work(
            &plan,
            &at_base(&plan.source_format.planes, 0).unwrap(),
            &at_base(&plan.destination_format.planes, 0).unwrap(),
        )
        .unwrap()
        .passes
        .iter()
        .map(|(kernel, _, _, group)| (*kernel, *group))
        .collect();
        let named: Vec<String> = (kernels.iter())
            .map(|(kernel, group)| format!("{kernel:?} {group:?}"))
            .collect();
        let mut times = Vec::new();
        let before = counts();
        for _ in 0..2 {
            let mut job = TransformJob {
                transform: &transform,
                plan: plan.clone(),
                source: &source,
                destination: &destination,
                hold: GuestHold::nothing(),
                inputs: Default::default(),
                outputs: Default::default(),
            };
            let started = runtime.now();
            let status = accelerator.transform(&mut job);
            times.push((
                runtime.elapsed_since(started).as_secs_f64() * 1000.0,
                status,
            ));
        }
        println!(
            "first transforms, {ahead:?}: {name}: {label}: first {:.1} ms (status {:#x}), then \
             {:.1} ms; at {:.0} ms after the opening; {}{}",
            times[0].0,
            times[0].1,
            times[1].0,
            runtime.elapsed_since(opened).as_secs_f64() * 1000.0,
            named.join(", "),
            counted(before, counts())
        );
        let made = counts().zip(before).map(|(now, then)| now[2] - then[2]);
        out.push((times[0].1, kernels, made));
    }
    if let (Some(logs), Some(answer)) = (logs, answer) {
        let deadline = runtime.deadline_in(Duration::from_secs(300));
        answer.wait_until(&*runtime, deadline).unwrap();
        for (_, target, line) in logs.take() {
            if target.starts_with("reims_vgpu::scaler") {
                println!("first transforms, {ahead:?}: {name}: {line}");
            }
        }
    }
    out
}

/// The first transforms on each GPU, `ahead` as it says.
fn each_gpu(ahead: Ahead) {
    #[cfg(target_os = "macos")]
    {
        let runtime = StdRuntime::shared();
        let started = runtime.now();
        let gpu = metal::MetalGpu::process().expect("Metal opens the kernel");
        let name = format!("Metal, {}", gpu.name());
        println!(
            "first transforms, {ahead:?}: {name}: opened in {:.0} ms",
            runtime.elapsed_since(started).as_secs_f64() * 1000.0
        );
        firsts(&name, gpu, ahead, &|| None);
    }
    #[cfg(vulkan_platform)]
    {
        let runtime = StdRuntime::shared();
        let started = runtime.now();
        match vulkan::VulkanGpu::process(&runtime) {
            Ok(gpu) => {
                let name = format!("Vulkan, {}", gpu.name());
                let (opening, directory) = gpu.cache_counts();
                println!(
                    "first transforms, {ahead:?}: {name}: opened in {:.0} ms, its shader cache in \
                     {}{}",
                    runtime.elapsed_since(started).as_secs_f64() * 1000.0,
                    directory.display(),
                    counted(Some([0; 5]), Some(opening))
                );
                let of = Arc::clone(&gpu);
                firsts(&name, gpu, ahead, &move || Some(of.cache_counts().0));
            }
            Err(why) => eprintln!("no Vulkan GPU: {why}; skipped"),
        }
    }
}

/// Once the making ahead is done, a process's first transform on each kernel and threadgroup the
/// transforms dispatch makes no `VkPipeline`: the making ahead made the one it takes, for the same
/// bindings. On the Vulkan platform, which makes them at first dispatches; Metal makes them where
/// it makes the pipeline states
#[test]
fn the_making_ahead_makes_what_the_first_transforms_take() {
    #[cfg(vulkan_platform)]
    match vulkan::VulkanGpu::process(&StdRuntime::shared()) {
        Ok(gpu) => {
            let name = format!("Vulkan, {}", gpu.name());
            let of = Arc::clone(&gpu);
            let firsts = firsts(&name, gpu, Ahead::Done, &move || Some(of.cache_counts().0));
            let mut taken: Vec<(Kernel, (u32, u32))> = Vec::new();
            for (status, kernels, made) in firsts {
                assert_eq!(status, io_return::SUCCESS, "{name}");
                assert_eq!(made, Some(0), "{name}: {kernels:?}");
                taken.extend(kernels);
            }
            for dispatched in params::DISPATCHED {
                assert!(taken.contains(&dispatched), "{dispatched:?} untried");
            }
        }
        Err(why) => eprintln!("no Vulkan GPU: {why}; skipped"),
    }
}

/// A process's first transforms with nothing made ahead (module docs)
#[test]
#[ignore = "a probe of a process's first transforms: run alone, for the record"]
fn first_transforms_with_nothing_made_ahead() {
    each_gpu(Ahead::None);
}

/// A process's first transforms as the making ahead starts (module docs)
#[test]
#[ignore = "a probe of a process's first transforms: run alone, for the record"]
fn first_transforms_as_the_making_ahead_starts() {
    each_gpu(Ahead::Starting);
}

/// A process's first transforms once the making ahead is done (module docs)
#[test]
#[ignore = "a probe of a process's first transforms: run alone, for the record"]
fn first_transforms_once_made_ahead() {
    each_gpu(Ahead::Done);
}
