//! Windows pools on the GPU (`super::shared_surfaces`): their surfaces are shared textures that
//! the driver imports, and a stand-in's copy (`Surface::copy_to`) gives a surface what another
//! holds, the GPU's copy between their textures. Run on the Windows host, on the adapter
//! libva-win32 opens first, holding its GPU.

use crate::core::runtime::StdRuntime;
use crate::vaapi::va::display::Display;

use super::{Rect, Surface, SurfaceFormat, SurfacePool, create_image};

const SIZE: (u32, u32) = (1280, 720);

/// What `surface` reads back as, its two planes one after the other, each row `SIZE.0`
/// samples.
fn read(surface: &Surface, format: SurfaceFormat) -> Vec<u8> {
    let sample = usize::try_from(format.bits_per_pixel() * 2 / 3 / 8).expect("a size");
    let (width, height) = (SIZE.0 as usize, SIZE.1 as usize);
    let stride = width * sample;
    let mut y = vec![0; stride * height];
    let mut uv = vec![0; stride * height / 2];
    let whole = Rect {
        x: 0,
        y: 0,
        width: SIZE.0,
        height: SIZE.1,
    };
    (surface.read_back().expect("a read back"))
        .copy_planes_to(whole, &mut y, stride, &mut uv, stride)
        .expect("the planes");
    y.extend(uv);
    y
}

/// Puts frame `n` of a pattern into `surface` (`vaPutImage`), whatever the driver makes of it.
fn put(display: &Display, surface: &Surface, format: SurfaceFormat, n: u8) {
    let mut image = create_image(display, format, SIZE.0, SIZE.1).expect("an image");
    {
        let mut mapping = image.map().expect("a map");
        for (i, byte) in mapping.bytes_mut().iter_mut().enumerate() {
            *byte = (i.wrapping_mul(31) as u8) ^ n;
        }
    }
    image
        .put_into(surface.id(), SIZE.0, SIZE.1)
        .expect("vaPutImage");
}

/// Each surface of an NV12 and a P010 pool is a shared texture, and a stand-in's copy gives
/// `to` what `from` reads back as.
#[test]
#[ignore = "needs vaon12 on a Direct3D 12 GPU: run on the Windows host"]
fn a_stand_in_copy_gives_the_surface_what_the_other_holds() {
    let display = Display::open_adapter(None, StdRuntime::shared()).expect("a VA display");
    for format in [SurfaceFormat::Nv12, SurfaceFormat::P010] {
        let pool = SurfacePool::new(&display, format, SIZE.0, SIZE.1, 2).expect("a pool");
        let from = pool.get().expect("a surface");
        let to = pool.get().expect("a surface");
        assert!(from.texture().is_some(), "{}: not shared", format.name());
        assert!(to.texture().is_some(), "{}: not shared", format.name());
        let more = pool.get().expect("a surface made after the pool");
        assert!(
            more.texture().is_some(),
            "{}: a later surface not shared",
            format.name()
        );
        // A pool made with none, as a decoder's is, makes its first surface shared too.
        let empty = SurfacePool::new(&display, format, SIZE.0, SIZE.1, 0).expect("a pool");
        assert!(
            empty.get().expect("a surface").texture().is_some(),
            "{}: an empty pool's first surface not shared",
            format.name()
        );
        put(&display, &from, format, 0x5a);
        put(&display, &to, format, 0xa5);
        let held = read(&from, format);
        assert_ne!(
            read(&to, format),
            held,
            "{}: the same before the copy",
            format.name()
        );
        assert!(
            from.copy_to(&to).expect("a copy"),
            "{}: no copy",
            format.name()
        );
        assert!(
            read(&to, format) == held,
            "{}: the copy differs",
            format.name()
        );
    }
}

/// A decoder's pool gives back the GPU memory it held once let go, as Windows charges the process
/// for it (`super::super::d3d12`'s module doc): three pools in turn of six surfaces of 3840x2160
/// NV12 each, lent and let go with the pool, as a session's pool is at its end; the charge after
/// the third is up by less than a tenth of a pool's textures on the charge after the first. A pool
/// whose textures the driver imported by NT handle stayed charged for good, 72 MiB a pool, which
/// this held two of (**measured** on the UHD 630, 2026-11-26). Run on the Windows host, holding
/// its GPU.
#[test]
#[ignore = "needs vaon12 on a Direct3D 12 GPU: run on the Windows host"]
fn pools_let_go_give_back_their_memory() {
    const BIG: (u32, u32) = (3840, 2160);
    const SURFACES: usize = 6;
    let textures = SURFACES as u64 * u64::from(BIG.0 * BIG.1 * 3 / 2);
    let display = Display::open_adapter(None, StdRuntime::shared()).expect("a VA display");
    let mut charged = Vec::new();
    for pool in 1..=3 {
        {
            let made =
                SurfacePool::new(&display, SurfaceFormat::Nv12, BIG.0, BIG.1, 0).expect("a pool");
            let lent: Vec<Surface> = (0..SURFACES)
                .map(|_| made.get().expect("a surface"))
                .collect();
            assert!(
                lent.iter().all(|surface| surface.texture().is_some()),
                "pool {pool}: a surface of no texture of ours"
            );
        }
        charged.push(super::super::d3d12::process_gpu_memory_settled());
    }
    let grown = charged[2].saturating_sub(charged[0]);
    println!("charged after each pool: {charged:?} bytes");
    assert!(
        grown < textures / 10,
        "{grown} bytes more charged after the third pool than after the first, of the {textures} \
         that a pool's textures hold"
    );
}
