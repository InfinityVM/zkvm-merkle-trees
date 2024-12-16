use core::fmt::Display;

use alloc::{rc::Rc, sync::Arc};

use crate::node::{NodeHash, NodeOrLeafDb};

pub trait DatabaseGet<K, V> {
    type GetError: Display;

    fn get(&self, hash: &NodeHash) -> Result<NodeOrLeafDb<K, V>, Self::GetError>;
}

impl<K, V, D: DatabaseGet<K, V>> DatabaseGet<K, V> for &D {
    type GetError = D::GetError;

    #[inline]
    fn get(&self, hash: &NodeHash) -> Result<NodeOrLeafDb<K, V>, Self::GetError> {
        (**self).get(hash)
    }
}

pub trait DatabaseSet<K, V>: DatabaseGet<K, V> {
    type SetError: Display;

    fn set(&self, hash: &NodeHash, node: NodeOrLeafDb<K, V>) -> Result<(), Self::SetError>;
}

impl<K, V, D: DatabaseSet<K, V>> DatabaseSet<K, V> for &D {
    type SetError = D::SetError;

    #[inline]
    fn set(&self, hash: &NodeHash, node: NodeOrLeafDb<K, V>) -> Result<(), Self::SetError> {
        (**self).set(hash, node)
    }
}

impl<K, V, D: DatabaseGet<K, V>> DatabaseGet<K, V> for Rc<D> {
    type GetError = D::GetError;

    #[inline]
    fn get(&self, hash: &NodeHash) -> Result<NodeOrLeafDb<K, V>, Self::GetError> {
        (**self).get(hash)
    }
}

impl<K, V, D: DatabaseSet<K, V>> DatabaseSet<K, V> for Rc<D> {
    type SetError = D::SetError;

    #[inline]
    fn set(&self, hash: &NodeHash, node: NodeOrLeafDb<K, V>) -> Result<(), Self::SetError> {
        (**self).set(hash, node)
    }
}

impl<K, V, D: DatabaseGet<K, V>> DatabaseGet<K, V> for Arc<D> {
    type GetError = D::GetError;

    #[inline]
    fn get(&self, hash: &NodeHash) -> Result<NodeOrLeafDb<K, V>, Self::GetError> {
        (**self).get(hash)
    }
}

impl<K, V, D: DatabaseSet<K, V>> DatabaseSet<K, V> for Arc<D> {
    type SetError = D::SetError;

    #[inline]
    fn set(&self, hash: &NodeHash, node: NodeOrLeafDb<K, V>) -> Result<(), Self::SetError> {
        (**self).set(hash, node)
    }
}
