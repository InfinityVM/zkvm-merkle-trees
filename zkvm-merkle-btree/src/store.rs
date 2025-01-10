use alloc::{rc::Rc, sync::Arc};
use std::ops::{Deref, Index};

use kairos_trie::{PortableHash, PortableHasher};

use crate::{
    errors::BTreeError,
    node::{InnerOuterSnapshotRef, NodeHash},
};

pub type Idx = u32;

pub trait Store {
    type Key: Ord + Clone + PortableHash;
    type Value: Clone + PortableHash;

    /// Get the root node idx of the merkle b+tree in the store.
    /// This is the root of the merkle b+tree prior to the current transaction.
    /// It's unlikely you'll need this method, you probably want `get_store_root_hash` instead.
    ///
    /// None means the tree is empty.
    fn get_store_root_idx(&self) -> Option<Idx>;

    /// Get the root hash of the merkle b+tree in the store.
    /// Note that this is not the merkle root of the current transaction.
    /// This is the root of the merkle b+tree prior to the current transaction.
    fn get_store_root_hash(&self) -> NodeHash;

    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, BTreeError>;

    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<InnerOuterSnapshotRef<'_, Self::Key, Self::Value>, BTreeError>;
}

impl<S: Store> Store for &S {
    type Key = S::Key;
    type Value = S::Value;

    #[inline(always)]
    fn get_store_root_idx(&self) -> Option<Idx> {
        (**self).get_store_root_idx()
    }

    #[inline(always)]
    fn get_store_root_hash(&self) -> NodeHash {
        (**self).get_store_root_hash()
    }

    #[inline(always)]
    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, BTreeError> {
        (**self).calc_subtree_hash(hasher, hash_idx)
    }

    #[inline(always)]
    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<InnerOuterSnapshotRef<'_, Self::Key, Self::Value>, BTreeError> {
        (**self).get(hash_idx)
    }
}

impl<S: Store> Store for Rc<S> {
    type Key = S::Key;
    type Value = S::Value;

    #[inline(always)]
    fn get_store_root_idx(&self) -> Option<Idx> {
        (**self).get_store_root_idx()
    }

    #[inline(always)]
    fn get_store_root_hash(&self) -> NodeHash {
        (**self).get_store_root_hash()
    }

    #[inline(always)]
    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, BTreeError> {
        (**self).calc_subtree_hash(hasher, hash_idx)
    }

    #[inline(always)]
    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<InnerOuterSnapshotRef<'_, Self::Key, Self::Value>, BTreeError> {
        (**self).get(hash_idx)
    }
}

impl<S: Store> Store for Arc<S> {
    type Key = S::Key;
    type Value = S::Value;

    #[inline(always)]
    fn get_store_root_idx(&self) -> Option<Idx> {
        (**self).get_store_root_idx()
    }

    #[inline(always)]
    fn get_store_root_hash(&self) -> NodeHash {
        (**self).get_store_root_hash()
    }

    #[inline(always)]
    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, BTreeError> {
        (**self).calc_subtree_hash(hasher, hash_idx)
    }

    #[inline(always)]
    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<InnerOuterSnapshotRef<'_, Self::Key, Self::Value>, BTreeError> {
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
