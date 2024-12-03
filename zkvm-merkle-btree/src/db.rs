use core::fmt::Display;

use alloc::{rc::Rc, sync::Arc};

use crate::node::{Node, NodeHash};

pub trait DatabaseGet<K, V> {
    type GetError: Display;

    fn get_node(&self, hash: &NodeHash) -> Result<Arc<Node<K, V>>, Self::GetError>;

    fn get_value(&self, hash: &NodeHash) -> Result<V, Self::GetError>;
}

impl<K, V, D: DatabaseGet<K, V>> DatabaseGet<K, V> for &D {
    type GetError = D::GetError;

    #[inline]
    fn get_node(&self, hash: &NodeHash) -> Result<Arc<Node<K, V>>, Self::GetError> {
        (**self).get_node(hash)
    }

    #[inline]
    fn get_value(&self, hash: &NodeHash) -> Result<V, Self::GetError> {
        (**self).get_value(hash)
    }
}

pub trait DatabaseSet<K, V>: DatabaseGet<K, V> {
    type SetError: Display;

    fn set_node(&self, hash: NodeHash, node: Arc<Node<K, V>>) -> Result<(), Self::SetError>;

    fn set_value(&self, hash: NodeHash, value: V) -> Result<(), Self::SetError>;
}

impl<K, V, D: DatabaseSet<K, V>> DatabaseSet<K, V> for &D {
    type SetError = D::SetError;

    #[inline]
    fn set_node(&self, hash: NodeHash, node: Arc<Node<K, V>>) -> Result<(), Self::SetError> {
        (**self).set_node(hash, node)
    }

    #[inline]
    fn set_value(&self, hash: NodeHash, value: V) -> Result<(), Self::SetError> {
        (**self).set_value(hash, value)
    }
}

impl<K, V, D: DatabaseGet<K, V>> DatabaseGet<K, V> for Rc<D> {
    type GetError = D::GetError;

    #[inline]
    fn get_node(&self, hash: &NodeHash) -> Result<Arc<Node<K, V>>, Self::GetError> {
        (**self).get_node(hash)
    }

    #[inline]
    fn get_value(&self, hash: &NodeHash) -> Result<V, Self::GetError> {
        (**self).get_value(hash)
    }
}

impl<K, V, D: DatabaseSet<K, V>> DatabaseSet<K, V> for Rc<D> {
    type SetError = D::SetError;

    #[inline]
    fn set_node(&self, hash: NodeHash, node: Arc<Node<K, V>>) -> Result<(), Self::SetError> {
        (**self).set_node(hash, node)
    }

    #[inline]
    fn set_value(&self, hash: NodeHash, value: V) -> Result<(), Self::SetError> {
        (**self).set_value(hash, value)
    }
}

impl<K, V, D: DatabaseGet<K, V>> DatabaseGet<K, V> for Arc<D> {
    type GetError = D::GetError;

    #[inline]
    fn get_node(&self, hash: &NodeHash) -> Result<Arc<Node<K, V>>, Self::GetError> {
        (**self).get_node(hash)
    }

    #[inline]
    fn get_value(&self, hash: &NodeHash) -> Result<V, Self::GetError> {
        (**self).get_value(hash)
    }
}

impl<K, V, D: DatabaseSet<K, V>> DatabaseSet<K, V> for Arc<D> {
    type SetError = D::SetError;

    #[inline]
    fn set_node(&self, hash: NodeHash, node: Arc<Node<K, V>>) -> Result<(), Self::SetError> {
        (**self).set_node(hash, node)
    }

    #[inline]
    fn set_value(&self, hash: NodeHash, value: V) -> Result<(), Self::SetError> {
        (**self).set_value(hash, value)
    }
}
