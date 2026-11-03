// The lane gate's summary of a crash report (guest-run.sh, the final phase): its exception and
// the faulting thread's top frames, one line.
//   osascript -l JavaScript ips-summary.js FILE.ips
ObjC.import("Foundation");
function run(argv) {
	var t = $.NSString.stringWithContentsOfFileEncodingError(argv[0], $.NSUTF8StringEncoding, null);
	if (!t || t.isNil()) return "unreadable";
	t = t.js;
	var b = JSON.parse(t.slice(t.indexOf("\n") + 1));
	var e = b.exception || {}, ft = b.faultingThread || 0, th = (b.threads || [])[ft] || {}, im = b.usedImages || [];
	var fr = (th.frames || []).slice(0, 10).map(function (f) {
		return (f.symbol || "?") + " (" + ((im[f.imageIndex] || {}).name || "?") + ")";
	});
	return (e.type || "?") + " " + (e.signal || "") + " " + (e.subtype || "") + "; thread " + ft + ": " + fr.join(" < ");
}
