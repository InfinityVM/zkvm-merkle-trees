use alloc::{rc::Rc, sync::Arc};
use core::fmt::{Debug, Display};
use std::ops::{Deref, Index};

use kairos_trie::{PortableHash, PortableHasher};

use crate::{
    error::BTreeErr,
    node::{InnerOuterSnapshotRef, NodeHash},
};

pub type Idx = u32;

pub trait Store {
    type DbGetError: Display + Debug;
    type DbSetError: Display + Debug;
    type Key: Ord + Clone + PortableHash;
    type Value: Clone + PortableHash;

    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, BTreeErr<Self::DbGetError, Self::DbSetError>>;

    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<
        InnerOuterSnapshotRef<'_, Self::Key, Self::Value>,
        BTreeErr<Self::DbGetError, Self::DbSetError>,
    >;
}

impl<S: Store> Store for &S {
    type DbGetError = S::DbGetError;
    type DbSetError = S::DbSetError;
    type Key = S::Key;
    type Value = S::Value;

    #[inline(always)]
    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, BTreeErr<Self::DbGetError, Self::DbSetError>> {
        (**self).calc_subtree_hash(hasher, hash_idx)
    }

    #[inline(always)]
    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<
        InnerOuterSnapshotRef<'_, Self::Key, Self::Value>,
        BTreeErr<Self::DbGetError, Self::DbSetError>,
    > {
        (**self).get(hash_idx)
    }
}

impl<S: Store> Store for Rc<S> {
    type DbGetError = S::DbGetError;
    type DbSetError = S::DbSetError;
    type Key = S::Key;
    type Value = S::Value;

    #[inline(always)]
    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, BTreeErr<Self::DbGetError, Self::DbSetError>> {
        (**self).calc_subtree_hash(hasher, hash_idx)
    }

    #[inline(always)]
    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<
        InnerOuterSnapshotRef<'_, Self::Key, Self::Value>,
        BTreeErr<Self::DbGetError, Self::DbSetError>,
    > {
        (**self).get(hash_idx)
    }
}

impl<S: Store> Store for Arc<S> {
    type DbGetError = S::DbGetError;
    type DbSetError = S::DbSetError;
    type Key = S::Key;
    type Value = S::Value;

    #[inline(always)]
    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, BTreeErr<Self::DbGetError, Self::DbSetError>> {
        (**self).calc_subtree_hash(hasher, hash_idx)
    }

    #[inline(always)]
    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<
        InnerOuterSnapshotRef<'_, Self::Key, Self::Value>,
        BTreeErr<Self::DbGetError, Self::DbSetError>,
    > {
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
