//! The way back the examples' decodes take: `RVGV_VAAPI_READBACK`
//! (`staged`, `getimage`, `derive` or `mapped`) has every pool read its
//! frames back that way, to compare the ways on a driver
//! (tools/perf-check.sh passes it through). The library reads no
//! environment: an example sets it through
//! `reims_vgpu_video::vaapi::read_back_by`.

/// Sets the way back `RVGV_VAAPI_READBACK` names, where it names one. A
/// name of no way ends the process, so that a comparison of the ways does
/// not run the default under another's name.
pub fn read_back_as_asked() {
    let Ok(name) = std::env::var("RVGV_VAAPI_READBACK") else {
        return;
    };
    match reims_vgpu_video::vaapi::ReadbackWay::from_name(&name) {
        Some(way) => reims_vgpu_video::vaapi::read_back_by(Some(way)),
        None => {
            eprintln!("RVGV_VAAPI_READBACK={name} is none of staged, getimage, derive, mapped");
            std::process::exit(2);
        }
    }
}
