use alloc::{rc::Rc, sync::Arc};
use core::fmt::{Debug, Display};
use std::ops::{Deref, Index};

use kairos_trie::{PortableHash, PortableHasher};

use crate::node::{InnerOuterSnapshotRef, NodeHash};

pub type Idx = u32;

pub trait Store {
    type Error: Display + Debug;
    type Key: Ord + Clone + PortableHash;
    type Value: Clone + PortableHash;

    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, Self::Error>;

    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<InnerOuterSnapshotRef<'_, Self::Key, Self::Value>, Self::Error>;
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
    ) -> Result<InnerOuterSnapshotRef<'_, Self::Key, Self::Value>, Self::Error> {
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
    ) -> Result<InnerOuterSnapshotRef<'_, Self::Key, Self::Value>, Self::Error> {
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
    ) -> Result<InnerOuterSnapshotRef<'_, Self::Key, Self::Value>, Self::Error> {
        (**self).get(hash_idx)
    }
}

pub trait StoreProperties {
    type NodePtr<T>: NodePtr<T>;
    type Vector<T>: Index<usize> + Clone;
}

pub trait NodePtr<T>: Clone + AsRef<T> + Deref<Target = T> {
    fn new(v: T) -> Self;
    fn make_mut(&mut self) -> &mut T;
}

impl<T: Clone> NodePtr<T> for Box<T> {
    fn new(v: T) -> Self {
        Box::new(v)
    }

    fn make_mut(&mut self) -> &mut T {
        self
    }
}

impl<T: Clone> NodePtr<T> for Rc<T> {
    fn new(v: T) -> Self {
        Rc::new(v)
    }

    fn make_mut(&mut self) -> &mut T {
        Rc::make_mut(self)
    }
}

impl<T: Clone> NodePtr<T> for Arc<T> {
    fn new(v: T) -> Self {
        Arc::new(v)
    }

    fn make_mut(&mut self) -> &mut T {
        Arc::make_mut(self)
    }
}
