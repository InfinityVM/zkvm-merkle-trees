use core::cell::RefCell;

use alloc::{collections::BTreeMap, format, string::String};

use crate::{Branch, Leaf};

use super::{DatabaseGet, DatabaseSet, Node, NodeHash};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct MemoryDb<V> {
    leaves: RefCell<BTreeMap<NodeHash, Node<Branch<NodeHash>, Leaf<V>>>>,
}

impl<V> MemoryDb<V> {
    pub fn empty() -> Self {
        Self {
            leaves: RefCell::default(),
        }
    }
}

impl<V: Clone> DatabaseGet<V> for MemoryDb<V> {
    type GetError = String;

    fn get(&self, hash: &NodeHash) -> Result<Node<Branch<NodeHash>, Leaf<V>>, Self::GetError> {
        self.leaves
            .borrow()
            .get(hash)
            .cloned()
            .ok_or_else(|| format!("Hash: `{}` not found", hash))
    }
}

impl<V: Clone> DatabaseSet<V> for MemoryDb<V> {
    type SetError = String;

    fn set(
        &self,
        hash: NodeHash,
        node: Node<Branch<NodeHash>, Leaf<V>>,
    ) -> Result<(), Self::SetError> {
        self.leaves.borrow_mut().insert(hash, node);
        Ok(())
    }
}