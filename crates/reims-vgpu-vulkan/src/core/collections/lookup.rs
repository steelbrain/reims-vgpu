//! [`LookupMap`]: a hash map that is only looked up.
//!
//! Its hasher is fixed ([`super::KeyHasher`]), so the order it drops its values in is a function
//! of the keys and the insert and remove history, the same in every run. It has no iteration
//! API: nothing depends on its order.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::ops::Index;

use super::hasher::KeyBuild;

/// A hash map with a fixed hasher and no iteration.
pub struct LookupMap<K, V> {
    map: HashMap<K, V, KeyBuild>,
}

impl<K: Hash + Eq, V> LookupMap<K, V> {
    /// An empty map.
    pub fn new() -> Self {
        LookupMap {
            map: HashMap::with_hasher(KeyBuild::default()),
        }
    }

    /// The value under `key`.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(key)
    }

    /// The value under `key`, to change in place.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get_mut(key)
    }

    /// Whether `key` has a value.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.contains_key(key)
    }

    /// Stores `value` under `key` and returns the value it replaced.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.map.insert(key, value)
    }

    /// Removes `key` and returns its value.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.remove(key)
    }

    /// Drops every value, in the fixed hasher's table order.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// The number of keys.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the map holds no key.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<K: Hash + Eq, V> Default for LookupMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> fmt::Debug for LookupMap<K, V> {
    /// The length only: the entries' order is not to be read.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LookupMap")
            .field("len", &self.map.len())
            .finish_non_exhaustive()
    }
}

impl<K, Q, V> Index<&Q> for LookupMap<K, V>
where
    K: Hash + Eq + Borrow<Q>,
    Q: Hash + Eq + ?Sized,
{
    type Output = V;

    /// The value under `key`.
    ///
    /// # Panics
    ///
    /// When `key` has no value, as `HashMap`'s index does.
    #[expect(
        clippy::indexing_slicing,
        reason = "`HashMap`'s index, which this one is"
    )]
    fn index(&self, key: &Q) -> &V {
        &self.map[key]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct Tok(u64, Rc<RefCell<Vec<u64>>>);

    impl Drop for Tok {
        fn drop(&mut self) {
            self.1.borrow_mut().push(self.0);
        }
    }

    #[test]
    fn looks_up_inserts_and_removes() {
        let mut m: LookupMap<u64, &str> = LookupMap::new();
        assert!(m.is_empty());
        assert_eq!(m.insert(1, "a"), None);
        assert_eq!(m.insert(1, "b"), Some("a"));
        assert_eq!(m.get(&1), Some(&"b"));
        assert_eq!(m[&1], "b");
        *m.get_mut(&1).unwrap() = "c";
        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&2));
        assert_eq!(m.len(), 1);
        assert_eq!(m.remove(&1), Some("c"));
        assert_eq!(m.remove(&1), None);
        assert_eq!(format!("{m:?}"), "LookupMap { len: 0, .. }");
    }

    /// A history of mapping-id-like keys, one removed and inserted again, then the map drops.
    /// The removed value drops first, at its remove.
    fn drop_log() -> Vec<u64> {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut m = LookupMap::new();
        for k in [1u64, 2, 3, 4, 5, 6, 7, 8, 99, 100, 4096] {
            m.insert(k, Tok(k, log.clone()));
        }
        drop(m.remove(&2));
        m.insert(2, Tok(2, log.clone()));
        drop(m);
        log.take()
    }

    #[test]
    fn drop_order_is_fixed_by_the_keys_and_history() {
        let first = drop_log();
        assert_eq!(first, drop_log());
        assert_eq!(first.first(), Some(&2));
        let mut rest = first[1..].to_vec();
        rest.sort_unstable();
        assert_eq!(rest, [1, 2, 3, 4, 5, 6, 7, 8, 99, 100, 4096]);
    }

    #[test]
    fn clear_drops_in_the_same_order_as_drop() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut m = LookupMap::new();
        for k in [1u64, 3, 4, 5, 6, 7, 8, 99, 100, 4096, 2] {
            m.insert(k, Tok(k, log.clone()));
        }
        m.clear();
        let cleared = log.take();
        assert_eq!(cleared.len(), 11);
        for k in [1u64, 3, 4, 5, 6, 7, 8, 99, 100, 4096, 2] {
            m.insert(k, Tok(k, log.clone()));
        }
        drop(m);
        assert_eq!(log.take(), cleared);
    }
}
