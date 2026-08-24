//! Semantic render-pass load and store actions decoded from guest ordinals.

pub const MTL_LOAD_ACTION_DONT_CARE: u16 = 0;
pub const MTL_LOAD_ACTION_LOAD: u16 = 1;
pub const MTL_LOAD_ACTION_CLEAR: u16 = 2;
pub const MTL_STORE_ACTION_DONT_CARE: u16 = 0;
pub const MTL_STORE_ACTION_STORE: u16 = 1;
pub const MTL_STORE_ACTION_MULTISAMPLE_RESOLVE: u16 = 2;
pub const MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE: u16 = 3;

#[must_use]
pub fn is_declared_load_action(raw: u16) -> bool {
    raw <= MTL_LOAD_ACTION_CLEAR
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadAction {
    #[default]
    DontCare,
    Load,
    Clear,
}

impl LoadAction {
    #[must_use]
    pub const fn guest_ordinal(self) -> u16 {
        match self {
            Self::DontCare => MTL_LOAD_ACTION_DONT_CARE,
            Self::Load => MTL_LOAD_ACTION_LOAD,
            Self::Clear => MTL_LOAD_ACTION_CLEAR,
        }
    }

    #[must_use]
    pub fn census_routes(self) -> (&'static str, &'static str) {
        match self {
            Self::DontCare => ("color0_declared_dontcare", "color0_declared_dontcare_px"),
            Self::Load => ("color0_declared_load", "color0_declared_load_px"),
            Self::Clear => ("color0_declared_clear", "color0_declared_clear_px"),
        }
    }
}

impl core::fmt::Display for LoadAction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.guest_ordinal().fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StoreAction {
    #[default]
    DontCare,
    Store,
    MultisampleResolve,
    StoreAndMultisampleResolve,
}

impl StoreAction {
    #[must_use]
    pub const fn guest_ordinal(self) -> u16 {
        match self {
            Self::DontCare => MTL_STORE_ACTION_DONT_CARE,
            Self::Store => MTL_STORE_ACTION_STORE,
            Self::MultisampleResolve => MTL_STORE_ACTION_MULTISAMPLE_RESOLVE,
            Self::StoreAndMultisampleResolve => MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE,
        }
    }

    #[must_use]
    pub const fn publishes_single_sample(self) -> bool {
        !matches!(self, Self::DontCare)
    }
}

impl core::fmt::Display for StoreAction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.guest_ordinal().fmt(f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassActionDecodeError {
    Load(u16),
    Store(u16),
}

impl PassActionDecodeError {
    #[must_use]
    pub const fn raw(self) -> u16 {
        match self {
            Self::Load(raw) | Self::Store(raw) => raw,
        }
    }

    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Load(_) => "load_action_unmapped",
            Self::Store(_) => "store_action_unmapped",
        }
    }
}

pub const fn load_action(raw: u16) -> Result<LoadAction, PassActionDecodeError> {
    match raw {
        MTL_LOAD_ACTION_DONT_CARE => Ok(LoadAction::DontCare),
        MTL_LOAD_ACTION_LOAD => Ok(LoadAction::Load),
        MTL_LOAD_ACTION_CLEAR => Ok(LoadAction::Clear),
        other => Err(PassActionDecodeError::Load(other)),
    }
}

pub const fn store_action(raw: u16) -> Result<StoreAction, PassActionDecodeError> {
    match raw {
        MTL_STORE_ACTION_DONT_CARE => Ok(StoreAction::DontCare),
        MTL_STORE_ACTION_STORE => Ok(StoreAction::Store),
        MTL_STORE_ACTION_MULTISAMPLE_RESOLVE => Ok(StoreAction::MultisampleResolve),
        MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE => {
            Ok(StoreAction::StoreAndMultisampleResolve)
        }
        other => Err(PassActionDecodeError::Store(other)),
    }
}

#[must_use]
pub fn is_declared_store_action(raw: u16) -> bool {
    raw <= MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE
}

#[must_use]
pub fn store_action_publishes_single_sample(raw: u16) -> bool {
    matches!(
        raw,
        MTL_STORE_ACTION_STORE
            | MTL_STORE_ACTION_MULTISAMPLE_RESOLVE
            | MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_sets_are_closed() {
        assert_eq!(
            (0..=64)
                .filter(|&raw| is_declared_load_action(raw))
                .collect::<alloc::vec::Vec<_>>(),
            alloc::vec![0, 1, 2]
        );
        assert_eq!(
            (0..=64)
                .filter(|&raw| is_declared_store_action(raw))
                .collect::<alloc::vec::Vec<_>>(),
            alloc::vec![0, 1, 2, 3]
        );
        assert!(!is_declared_load_action(u16::MAX));
        assert!(!is_declared_store_action(u16::MAX));
    }

    #[test]
    fn load_fold_and_store_publication_are_total() {
        assert_eq!(load_action(0), Ok(LoadAction::DontCare));
        assert_eq!(load_action(1), Ok(LoadAction::Load));
        assert_eq!(load_action(2), Ok(LoadAction::Clear));
        assert_eq!(
            load_action(u16::MAX),
            Err(PassActionDecodeError::Load(u16::MAX))
        );
        assert_eq!(store_action(1), Ok(StoreAction::Store));
        assert_eq!(
            store_action(u16::MAX),
            Err(PassActionDecodeError::Store(u16::MAX))
        );
        assert!(!store_action_publishes_single_sample(0));
        assert!((1..=3).all(store_action_publishes_single_sample));
        assert!(!store_action_publishes_single_sample(4));
    }

    #[test]
    fn load_census_routes_are_distinct_pairs() {
        let actions = [LoadAction::DontCare, LoadAction::Load, LoadAction::Clear];
        let routes = actions.map(LoadAction::census_routes);
        assert_eq!(routes[0].1, "color0_declared_dontcare_px");
        assert_eq!(routes[1].1, "color0_declared_load_px");
        assert_eq!(routes[2].1, "color0_declared_clear_px");
        assert_ne!(routes[0], routes[1]);
        assert_ne!(routes[1], routes[2]);
        assert_ne!(routes[0], routes[2]);
    }
}
