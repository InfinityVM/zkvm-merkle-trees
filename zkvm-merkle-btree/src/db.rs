use alloc::{collections::BTreeMap, rc::Rc, sync::Arc};
use core::{
    cell::RefCell,
    fmt::{Debug, Display},
};

use crate::node::{NodeHash, NodeOrLeafDb};

pub trait DatabaseGet<K, V> {
    type GetError: Display + Debug;

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
    type SetError: Display + Debug;

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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct MemoryDb<K, V> {
    leaves: RefCell<BTreeMap<NodeHash, NodeOrLeafDb<K, V>>>,
}

impl<K: Clone, V: Clone> DatabaseGet<K, V> for MemoryDb<K, V> {
    type GetError = String;

    #[inline]
    fn get(&self, hash: &NodeHash) -> Result<NodeOrLeafDb<K, V>, Self::GetError> {
        self.leaves
            .borrow()
            .get(hash)
            .cloned()
            .ok_or_else(|| format!("Hash: `{:?}` not found", hash))
    }
}

impl<K: Clone, V: Clone> DatabaseSet<K, V> for MemoryDb<K, V> {
    type SetError = String;

    #[inline]
    fn set(&self, hash: &NodeHash, node: NodeOrLeafDb<K, V>) -> Result<(), Self::SetError> {
        self.leaves.borrow_mut().insert(*hash, node);
        Ok(())
    }
}
