//! BuildManifest.plist: the IPSW's build identities, and the one for Apple's virtual Mac.

use anyhow::{Context, Result, bail};
use plist::{Dictionary, Value};

/// The `DeviceClass` of the virtual Mac's identities (`VirtualMac2,1`, board 0x20, chip 0xFE00).
pub const DEVICE_CLASS: &str = "vma2macosap";
/// The variant macOS's own restore installs from: its components are what the boot manifest
/// covers.
pub const VARIANT_INSTALL: &str = "macOS Customer";
/// The erase install's variant, whose components the SFR manifest covers: the installed ones and
/// those of the restore OS and firmware stages.
pub const VARIANT_ERASE: &str = "Customer Erase Install (IPSW)";

/// A parsed BuildManifest.plist.
pub struct BuildManifest {
    dict: Dictionary,
}

impl BuildManifest {
    /// Parses a BuildManifest.plist.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let v =
            Value::from_reader(std::io::Cursor::new(bytes)).context("parse BuildManifest.plist")?;
        let dict = v
            .into_dictionary()
            .context("BuildManifest.plist is not a dictionary")?;
        if !matches!(dict.get("BuildIdentities"), Some(Value::Array(_))) {
            bail!("BuildManifest.plist has no BuildIdentities");
        }
        Ok(Self { dict })
    }

    /// `ProductVersion`, such as `27.2.1`.
    pub fn product_version(&self) -> Result<&str> {
        self.dict
            .get("ProductVersion")
            .and_then(Value::as_string)
            .context("BuildManifest.plist has no ProductVersion")
    }

    /// A top-level value, such as `ManifestVersion`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.dict.get(key)
    }

    /// `ProductBuildVersion`, such as `26B131`.
    pub fn build_version(&self) -> Result<&str> {
        self.dict
            .get("ProductBuildVersion")
            .and_then(Value::as_string)
            .context("BuildManifest.plist has no ProductBuildVersion")
    }

    /// Every build identity.
    pub fn identities(&self) -> impl Iterator<Item = Identity<'_>> {
        self.dict
            .get("BuildIdentities")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_dictionary)
            .map(|dict| Identity { dict })
    }

    /// The virtual Mac's identity of `variant`; fails if there is none, or more than one.
    pub fn vma2_identity(&self, variant: &str) -> Result<Identity<'_>> {
        let mut found = self.identities().filter(|i| {
            i.info_str("DeviceClass") == Some(DEVICE_CLASS)
                && i.info_str("Variant") == Some(variant)
        });
        let first = found.next().with_context(|| {
            format!(
                "this IPSW has no {DEVICE_CLASS} identity of variant {variant:?}: it is not a restore image for \
                 Apple's virtual Mac (VirtualMac2,1)"
            )
        })?;
        if found.next().is_some() {
            bail!("this IPSW has more than one {DEVICE_CLASS} identity of variant {variant:?}");
        }
        Ok(first)
    }
}

/// One build identity.
#[derive(Clone, Copy)]
pub struct Identity<'a> {
    dict: &'a Dictionary,
}

impl<'a> Identity<'a> {
    /// The identity's dictionary.
    pub fn dict(&self) -> &'a Dictionary {
        self.dict
    }

    /// A string of its `Info`.
    pub fn info_str(&self, key: &str) -> Option<&'a str> {
        self.dict
            .get("Info")?
            .as_dictionary()?
            .get(key)?
            .as_string()
    }

    /// A value of its `Info`.
    pub fn info(&self, key: &str) -> Option<&'a Value> {
        self.dict.get("Info")?.as_dictionary()?.get(key)
    }

    /// A top-level integer given as a hex string or an integer, such as `ApChipID` (`0xFE00`).
    pub fn int(&self, key: &str) -> Result<u64> {
        parse_int(
            self.dict
                .get(key)
                .with_context(|| format!("the identity has no {key}"))?,
        )
        .with_context(|| format!("the identity's {key} is not a number"))
    }

    /// Its components: `Manifest`'s entries.
    pub fn components(&self) -> impl Iterator<Item = (&'a str, Component<'a>)> {
        self.dict
            .get("Manifest")
            .and_then(Value::as_dictionary)
            .into_iter()
            .flat_map(|m| m.iter())
            .filter_map(|(k, v)| {
                Some((
                    k.as_str(),
                    Component {
                        dict: v.as_dictionary()?,
                    },
                ))
            })
    }

    /// The component called `name`.
    pub fn component(&self, name: &str) -> Result<Component<'a>> {
        self.dict
            .get("Manifest")
            .and_then(Value::as_dictionary)
            .and_then(|m| m.get(name))
            .and_then(Value::as_dictionary)
            .map(|dict| Component { dict })
            .with_context(|| format!("the build identity has no {name}"))
    }
}

/// One component of a build identity.
#[derive(Clone, Copy)]
pub struct Component<'a> {
    dict: &'a Dictionary,
}

impl<'a> Component<'a> {
    /// The component's dictionary.
    pub fn dict(&self) -> &'a Dictionary {
        self.dict
    }

    /// Its path in the IPSW.
    pub fn path(&self) -> Option<&'a str> {
        self.info()?.get("Path")?.as_string()
    }

    /// Its `Info`.
    pub fn info(&self) -> Option<&'a Dictionary> {
        self.dict.get("Info")?.as_dictionary()
    }

    /// A boolean of its `Info`, false where absent.
    pub fn info_bool(&self, key: &str) -> bool {
        self.info()
            .and_then(|i| i.get(key))
            .and_then(Value::as_boolean)
            .unwrap_or(false)
    }

    /// Its digest.
    pub fn digest(&self) -> Option<&'a [u8]> {
        self.dict.get("Digest")?.as_data()
    }

    /// Whether it is `Trusted`.
    pub fn trusted(&self) -> bool {
        self.dict
            .get("Trusted")
            .and_then(Value::as_boolean)
            .unwrap_or(false)
    }
}

/// An integer as a manifest gives one: an integer, or a string in hex (`0x…`) or decimal.
pub fn parse_int(v: &Value) -> Option<u64> {
    match v {
        Value::Integer(i) => i.as_unsigned(),
        Value::String(s) => {
            let s = s.trim();
            match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                Some(h) => u64::from_str_radix(h, 16).ok(),
                None => s.parse().ok(),
            }
        }
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A BuildManifest with the three identities of a virtual Mac, written from scratch.
    pub fn synthetic() -> Vec<u8> {
        let comp = |path: &str, digest: u8| {
            let mut info = Dictionary::new();
            info.insert("Path".into(), Value::String(path.into()));
            info.insert("Personalize".into(), Value::Boolean(true));
            let mut d = Dictionary::new();
            d.insert("Digest".into(), Value::Data(vec![digest; 48]));
            d.insert("Trusted".into(), Value::Boolean(true));
            d.insert("Info".into(), Value::Dictionary(info));
            Value::Dictionary(d)
        };
        let ident = |variant: &str, behavior: &str| {
            let mut info = Dictionary::new();
            info.insert("DeviceClass".into(), Value::String(DEVICE_CLASS.into()));
            info.insert("Variant".into(), Value::String(variant.into()));
            info.insert("RestoreBehavior".into(), Value::String(behavior.into()));
            let mut m = Dictionary::new();
            m.insert("KernelCache".into(), comp("kernelcache.test", 1));
            m.insert("DeviceTree".into(), comp("Firmware/dt.im4p", 2));
            let mut d = Dictionary::new();
            d.insert("ApChipID".into(), Value::String("0xFE00".into()));
            d.insert("ApBoardID".into(), Value::String("0x20".into()));
            d.insert("ApSecurityDomain".into(), Value::String("0x01".into()));
            d.insert("Info".into(), Value::Dictionary(info));
            d.insert("Manifest".into(), Value::Dictionary(m));
            Value::Dictionary(d)
        };
        let mut root = Dictionary::new();
        root.insert("ProductVersion".into(), Value::String("99.1".into()));
        root.insert("ProductBuildVersion".into(), Value::String("99A1".into()));
        root.insert(
            "BuildIdentities".into(),
            Value::Array(vec![
                ident(VARIANT_ERASE, "Erase"),
                ident("Customer Upgrade Install (IPSW)", "Update"),
                ident(VARIANT_INSTALL, "Erase"),
            ]),
        );
        let mut out = Vec::new();
        Value::Dictionary(root).to_writer_xml(&mut out).unwrap();
        out
    }

    #[test]
    fn picks_the_virtual_macs_identities() {
        let bm = BuildManifest::parse(&synthetic()).unwrap();
        assert_eq!(bm.product_version().unwrap(), "99.1");
        let id = bm.vma2_identity(VARIANT_INSTALL).unwrap();
        assert_eq!(id.int("ApChipID").unwrap(), 0xfe00);
        assert_eq!(id.int("ApBoardID").unwrap(), 0x20);
        assert_eq!(
            id.component("KernelCache").unwrap().path(),
            Some("kernelcache.test")
        );
        assert!(bm.vma2_identity(VARIANT_ERASE).is_ok());
        assert!(bm.vma2_identity("Nope").is_err());
    }

    #[test]
    fn integers_parse_as_manifests_give_them() {
        assert_eq!(parse_int(&Value::String("0x01".into())), Some(1));
        assert_eq!(parse_int(&Value::String("32".into())), Some(32));
        assert_eq!(parse_int(&Value::Integer(7.into())), Some(7));
        assert_eq!(parse_int(&Value::Boolean(true)), None);
    }
}
