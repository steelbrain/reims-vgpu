// Read the guest Metal device's OpenGL capability without Python or Xcode.
// Run inside macOS with: osascript -l JavaScript guest_opengl_caps.js
//
// A stock macOS installation may ship /usr/bin/python3 as an xcode-select shim,
// so the Python/ctypes probe cannot be assumed to run on a clean guest. JXA's
// Objective-C bridge is part of macOS and asks the same MTLDevice directly.
ObjC.import("Foundation");
ObjC.import("Metal");

const device = $.MTLCreateSystemDefaultDevice();
if (!device) {
  "MTLCreateSystemDefaultDevice=nil";
} else {
  const selector = $.NSSelectorFromString("supportsOpenGL");
  const available = Boolean(device.respondsToSelector(selector));
  const familySelector = $.NSSelectorFromString("supportsFamily:");
  const familySelectorAvailable = Boolean(
    device.respondsToSelector(familySelector),
  );
  const featureSetSelector = $.NSSelectorFromString("supportsFeatureSet:");
  const featureSetSelectorAvailable = Boolean(
    device.respondsToSelector(featureSetSelector),
  );
  const families = {
    Apple1: 1001,
    Apple2: 1002,
    Apple3: 1003,
    Apple4: 1004,
    Apple5: 1005,
    Apple6: 1006,
    Apple7: 1007,
    Apple8: 1008,
    Mac1: 2001,
    Mac2: 2002,
    Common1: 3001,
    Common2: 3002,
    Common3: 3003,
    MacCatalyst1: 4001,
    MacCatalyst2: 4002,
  };
  const supportedFamilies = {};
  if (familySelectorAvailable) {
    Object.keys(families).forEach((name) => {
      supportedFamilies[name] = Boolean(device.supportsFamily(families[name]));
    });
  }
  const featureSets = {
    macOS_GPUFamily1_v1: 10000,
    macOS_GPUFamily1_v2: 10001,
    macOS_GPUFamily1_v3: 10003,
    macOS_GPUFamily1_v4: 10004,
    macOS_GPUFamily2_v1: 10005,
  };
  const supportedFeatureSets = {};
  if (featureSetSelectorAvailable) {
    Object.keys(featureSets).forEach((name) => {
      supportedFeatureSets[name] = Boolean(
        device.supportsFeatureSet(featureSets[name]),
      );
    });
  }
  const result = {
    device: ObjC.unwrap(device.name),
    supportsOpenGLSelectorAvailable: available,
    supportsOpenGL: available ? Boolean(device.supportsOpenGL) : null,
    supportsFamilySelectorAvailable: familySelectorAvailable,
    supportedFamilies: familySelectorAvailable ? supportedFamilies : null,
    supportsFeatureSetSelectorAvailable: featureSetSelectorAvailable,
    supportedFeatureSets: featureSetSelectorAvailable
      ? supportedFeatureSets
      : null,
  };
  JSON.stringify(result);
}
