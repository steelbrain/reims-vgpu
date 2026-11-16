//! The runtime's check of textures over host pages (`Device::settle_in_place_checks`) in the
//! host's log: its report (`Device::in_place_check_report`), a line per class of pages, access and
//! row end with its verdict, `Pending` where none is known yet, then the check's notes (what
//! changed where a verdict failed, why one is unchecked), at info under [`TARGET`]. A boot's log
//! then says which textures over the guest's pages its device takes in place and which it
//! shadows, and why. A diagnostic: no Metal call answers the report, and which lines the host
//! writes is free under the contract.
//!
//! - **Twice per device:** as it is adopted ([`crate::gpu::device::adopt`]), with what is known
//!   then (the verdicts a record of an earlier process's check kept are known at once, the rest
//!   pending while its check runs), and as it is let go ([`crate::let_go`]), by when every check
//!   that ran has its verdict. Each is a state the device reaches once, not a per-event line
//!   (docs/conventions.md).
//! - **One call each:** no wait and no thread.

use metal2vulkan_runtime as rt;

/// The log target of the report's lines, at info.
pub(crate) const TARGET: &str = "reims_vgpu_vulkan::in_place";

/// Logs `device`'s report (the module docs), `when` naming the moment (`as it is adopted`), where
/// the log writes [`TARGET`] at info.
pub(crate) fn log_report(device: &rt::Device, when: &str) {
    if !log::log_enabled!(target: TARGET, log::Level::Info) {
        return;
    }
    for line in lines(device.name(), when, device.in_place_check_report()) {
        log::info!(target: TARGET, "{line}");
    }
}

/// The lines [`log_report`] logs: a header naming the device, the moment and how many of the
/// report's verdicts are pending, then `report`'s lines, indented and otherwise as the runtime
/// wrote them.
pub(crate) fn lines(name: &str, when: &str, report: Vec<String>) -> Vec<String> {
    let pending = report.iter().filter(|l| l.ends_with(": Pending")).count();
    let header = format!(
        "{name}: the check of textures over host pages, {when}: {pending} verdicts pending"
    );
    std::iter::once(header)
        .chain(report.into_iter().map(|line| format!("  {line}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The report's lines follow a header that names the device, the moment and how many
    /// verdicts are pending, each indented and otherwise as the runtime wrote it
    #[test]
    fn the_report_follows_its_header() {
        let report = vec![
            "Pointer: Passed".to_string(),
            "DmaBuf renders: Pending".to_string(),
            "DmaBuf copies: Failed".to_string(),
        ];
        assert_eq!(
            lines("Gpu", "as it is let go", report),
            [
                "Gpu: the check of textures over host pages, as it is let go: 1 verdicts pending",
                "  Pointer: Passed",
                "  DmaBuf renders: Pending",
                "  DmaBuf copies: Failed",
            ]
        );
    }
}
