//! The tests of `rate_map` that compare with Metal on this host: macOS only.

use reims_vgpu_core::platform::resource::Buffer as _;

use super::*;

/// A layer descriptor is nil for a count of 0 on either axis, else of zeroed samples and
/// that count, depth 0; a map descriptor counts its layers from 0 to the first index with
/// none and answers the layer set at an index, nil where none is, as on Metal
#[test]
fn layers_and_descriptors_answer_as_on_metal() {
    fn answers<P: Platform>(p: &P) -> Vec<u64> {
        let mut out = Vec::new();
        for (w, h) in [(0, 1), (1, 0), (0, 0), (1, 1), (3, 2)] {
            match p
                .new_rate_map_layer(OnFault::Die, MtlSize::new(w, h, 7))
                .unwrap()
            {
                None => out.push(u64::MAX),
                Some(l) => {
                    let n = l.sample_count();
                    out.extend([n.width, n.height, n.depth]);
                    // SAFETY: the storage holds the counts' samples.
                    let first = unsafe {
                        (
                            *l.horizontal_sample_storage().as_ptr::<f32>(),
                            *l.vertical_sample_storage().as_ptr::<f32>(),
                        )
                    };
                    out.extend([u64::from(first.0.to_bits()), u64::from(first.1.to_bits())]);
                }
            }
        }
        let d = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(64, 32, 1),
            ..RateMapDesc::default()
        });
        let (a, b) = (layer(p, &[1.0], &[1.0]), layer(p, &[0.5], &[0.5]));
        d.set_layer(OnFault::Die, &b, 1).unwrap();
        let at = |i| u64::from(d.layer(OnFault::Die, i).unwrap().is_some());
        out.extend([d.layer_count(), at(0), at(1)]);
        d.set_layer(OnFault::Die, &a, 0).unwrap();
        out.extend([d.layer_count(), at(0), at(1), at(2)]);
        out
    }
    let vulkan = answers(&platform());
    assert_eq!(vulkan[..3], [u64::MAX; 3]);
    reims_vgpu_core::metal_check!("answers", vulkan);
}

/// A map's answers: an immutable one's compiled at creation, a mutable one's of no
/// compile (no physical size, (0, 0) for every screen coordinate) until a reset compiles its
/// descriptor, its mutability kept; Metal's parameter size and alignment, and granularity,
/// for both, as on Metal, the coordinates bit for bit
#[test]
fn maps_answer_as_on_metal() {
    fn answers<P: Platform>(p: &P, d: &P::Device) -> Vec<Vec<f64>> {
        let read = |m: &P::RateMap, compiled: bool| {
            let size = m.physical_size(0).unwrap();
            let granularity = m.physical_granularity();
            let sa = m.parameter_buffer_size_and_align();
            let s2p = m
                .screen_to_physical(MtlCoordinate2D { x: 100.0, y: 100.0 }, 0)
                .unwrap();
            let mut v = vec![
                m.layer_count() as f64,
                m.mutability() as f64,
                size.width as f64,
                size.height as f64,
                size.depth as f64,
                granularity.width as f64,
                granularity.height as f64,
                granularity.depth as f64,
                sa.size as f64,
                sa.align as f64,
                f64::from(s2p.x),
                f64::from(s2p.y),
            ];
            if compiled {
                let p2s = m
                    .physical_to_screen(MtlCoordinate2D { x: 10.0, y: 10.0 }, 0)
                    .unwrap();
                v.extend([f64::from(p2s.x), f64::from(p2s.y)]);
            }
            v
        };
        let mut out = Vec::new();
        for quality in [1.0, 0.1] {
            let m = d
                .new_rate_map(&descriptor(p, quality, false))
                .unwrap()
                .unwrap();
            out.push(read(&m, true));
        }
        let m = d.new_rate_map(&descriptor(p, 0.1, true)).unwrap().unwrap();
        out.push(read(&m, false));
        m.reset(&descriptor(p, 0.1, false));
        out.push(read(&m, true));
        out
    }
    let vulkan = answers(&platform(), &dev());
    assert_eq!(
        vulkan[2][..12],
        [
            1.0, 1.0, 0.0, 0.0, 1.0, 32.0, 32.0, 1.0, 24624.0, 4.0, 0.0, 0.0
        ]
    );
    let bits = |answers: &[Vec<f64>]| -> Vec<Vec<u64>> {
        answers
            .iter()
            .map(|a| a.iter().map(|x| x.to_bits()).collect())
            .collect()
    };
    reims_vgpu_core::metal_check!("bits", bits(&vulkan));
}

/// The parameter copy writes Metal's 24,624 bytes at its offset, byte for byte as Metal's:
/// a compiled map's, Apple's layout, of quality 1 and 0.1, and zeros for
/// a mutable map not yet reset; nothing else of the buffer changes
#[test]
fn the_parameter_copy_is_metals() {
    /// The bytes of a buffer of 30000 bytes of 0xee after the copy at offset 4 of the
    /// parameter data of a map of `quality`, mutable when asked.
    fn copied<P: Platform>(p: &P, d: &P::Device, quality: f32, mutable: bool) -> Vec<u8> {
        let m = d
            .new_rate_map(&descriptor(p, quality, mutable))
            .unwrap()
            .unwrap();
        let b = d
            .new_buffer_of_length(OnFault::Die, 30000, 0)
            .unwrap()
            .unwrap();
        // SAFETY: the Shared buffer's 30000 bytes, which no command uses.
        let bytes = unsafe { std::slice::from_raw_parts_mut(b.contents().as_ptr::<u8>(), 30000) };
        bytes.fill(0xee);
        m.copy_parameter_data(&b, 4).unwrap();
        bytes.to_vec()
    }
    let p = platform();
    let d = dev();
    let unset = copied(&p, &d, 0.1, true);
    let around = |bytes: &[u8]| {
        bytes[..4]
            .iter()
            .chain(&bytes[4 + 24624..])
            .all(|&x| x == 0xee)
    };
    assert!(around(&unset) && unset[4..4 + 24624].iter().all(|&x| x == 0));
    let compiled = [1.0, 0.1].map(|quality| copied(&p, &d, quality, false));
    for bytes in &compiled {
        assert!(around(bytes));
    }
    reims_vgpu_core::metal_check_bytes!("unset", &unset);
    for (quality, ours) in [1.0, 0.1].into_iter().zip(&compiled) {
        reims_vgpu_core::metal_check_bytes!(format!("quality {quality}"), ours);
    }
}

/// A map of no layers is made, from an immutable descriptor, by a mutable one's reset and by a
/// compiled map's reset, which compiles over the tables the map held, as Metal makes it: its
/// layer count is 0, its parameter data Metal's byte for byte, and its physical sizes at
/// layers 0 and 1 Metal's; its coordinate maps die at their selectors, where Metal ends the
/// process
#[test]
fn maps_of_no_layers_are_made_as_metals() {
    /// What `P` answers of a map of no layers made from an immutable descriptor, of a mutable
    /// one reset to none and of an immutable one of quality 0.1 reset to none: the layer count,
    /// the physical sizes at layers 0 and 1, and the 30000 bytes of a buffer of 0xee after the
    /// parameter copy at offset 4; then the maps.
    fn made<P: Platform>(p: &P, d: &P::Device) -> (Vec<Vec<u64>>, Vec<P::RateMap>) {
        let none = |mutable: bool| {
            p.rate_map_descriptor(&RateMapDesc {
                screen_size: MtlSize::new(64, 32, 1),
                mutability: mutable.then_some(MUTABLE),
                ..RateMapDesc::default()
            })
        };
        let immutable = d.new_rate_map(&none(false)).unwrap().unwrap();
        let reset = d.new_rate_map(&descriptor(p, 0.1, true)).unwrap().unwrap();
        reset.reset(&none(false));
        let compiled = d.new_rate_map(&descriptor(p, 0.1, false)).unwrap().unwrap();
        compiled.reset(&none(false));
        let answers = [&immutable, &reset, &compiled]
            .map(|m| {
                let mut out = vec![m.layer_count()];
                for layer in [0, 1] {
                    let size = m.physical_size(layer).unwrap();
                    out.extend([size.width, size.height, size.depth]);
                }
                let b = d
                    .new_buffer_of_length(OnFault::Die, 30000, 0)
                    .unwrap()
                    .unwrap();
                // SAFETY: the Shared buffer's 30000 bytes, which no command uses.
                let bytes =
                    unsafe { std::slice::from_raw_parts_mut(b.contents().as_ptr::<u8>(), 30000) };
                bytes.fill(0xee);
                m.copy_parameter_data(&b, 4).unwrap();
                out.extend(bytes.iter().map(|&x| u64::from(x)));
                out
            })
            .to_vec();
        (answers, vec![immutable, reset, compiled])
    }
    let p = platform();
    let d = dev();
    let (ours, maps) = made(&p, &d);
    for answers in &ours {
        assert_eq!(answers[0], 0, "no layers");
    }
    for m in &maps {
        let site = catch_death(|| m.screen_to_physical(MtlCoordinate2D { x: 1.0, y: 1.0 }, 0))
            .unwrap_err();
        assert_eq!(site.file, "mapScreenToPhysicalCoordinates:forLayer:");
        let site = catch_death(|| m.physical_to_screen(MtlCoordinate2D { x: 1.0, y: 1.0 }, 0))
            .unwrap_err();
        assert_eq!(site.file, "mapPhysicalToScreenCoordinates:forLayer:");
    }
    let which = ["immutable", "reset", "compiled and reset"];
    for (which, o) in which.iter().zip(&ours) {
        reims_vgpu_core::metal_check!(format!("{which}: sizes"), &o[..7]);
        reims_vgpu_core::metal_check_raw!(format!("{which}: copy"), o[7..]);
    }
}
