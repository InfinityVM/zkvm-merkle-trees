use core::fmt::{Debug, Display};
use std::{rc::Rc, sync::Arc};

use kairos_trie::{PortableHash, PortableHasher};

use crate::node::{NodeHash, NodeOrLeafSnapshotRef};

pub type Idx = u32;

pub trait Store {
    type Error: Display + Debug;
    type Key: Ord + Clone + PortableHash + Debug;
    type Value: Clone + PortableHash + Debug;

    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, Self::Error>;

    // TODO: Remove the Arc, there's no way to reuse the allocation.
    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<NodeOrLeafSnapshotRef<'_, Self::Key, Self::Value>, Self::Error>;
}

impl<S: Store> Store for &S {
    type Error = S::Error;
    type Key = S::Key;
    type Value = S::Value;

    #[inline(always)]
    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, Self::Error> {
        (**self).calc_subtree_hash(hasher, hash_idx)
    }

    #[inline(always)]
    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<NodeOrLeafSnapshotRef<'_, Self::Key, Self::Value>, Self::Error> {
        (**self).get(hash_idx)
    }
}

impl<S: Store> Store for Rc<S> {
    type Error = S::Error;
    type Key = S::Key;
    type Value = S::Value;

    #[inline(always)]
    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, Self::Error> {
        (**self).calc_subtree_hash(hasher, hash_idx)
    }

    #[inline(always)]
    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<NodeOrLeafSnapshotRef<'_, Self::Key, Self::Value>, Self::Error> {
        (**self).get(hash_idx)
    }
}

impl<S: Store> Store for Arc<S> {
    type Error = S::Error;
    type Key = S::Key;
    type Value = S::Value;

    #[inline(always)]
    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, Self::Error> {
        (**self).calc_subtree_hash(hasher, hash_idx)
    }

    #[inline(always)]
    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<NodeOrLeafSnapshotRef<'_, Self::Key, Self::Value>, Self::Error> {
        (**self).get(hash_idx)
    }
}
