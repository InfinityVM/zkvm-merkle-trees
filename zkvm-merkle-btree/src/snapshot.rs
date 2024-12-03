use alloc::string::String;
use core::cell::RefCell;
use core::fmt::Debug;

use imbl::Vector;
use kairos_trie::{PortableHash, PortableHasher};

use crate::{
    node::{NodeHash, NodeOrLeaf},
    store::{Idx, Store},
};

#[derive(Clone)]
pub struct SnapshotBuilder<K: Ord + Clone + PortableHash, V: Clone + PortableHash, Db> {
    inner: RefCell<SnapshotBuilderInner<K, V, Db>>,
}

#[derive(Clone)]
pub struct SnapshotBuilderInner<K: Ord + Clone + PortableHash, V: Clone + PortableHash, Db> {
    pub db: Db,
    nodes: Vector<(NodeHash, Option<NodeOrLeaf<K, V>>)>,
}

impl<K: Ord + Clone + PortableHash, V: Clone + PortableHash, Db> SnapshotBuilder<K, V, Db> {
    #[inline]
    pub fn new(root: NodeHash, db: Db) -> Self {
        debug_assert!(crate::node::EMPTY_TREE_ROOT_HASH == NodeHash::default());

        if root == crate::node::EMPTY_TREE_ROOT_HASH {
            Self {
                inner: RefCell::new(SnapshotBuilderInner {
                    db,
                    nodes: Vector::new(),
                }),
            }
        } else {
            Self {
                inner: RefCell::new(SnapshotBuilderInner {
                    db,
                    nodes: Vector::from_iter([(root, None)]),
                }),
            }
        }
    }
}

impl<K: Ord + Clone + PortableHash + Debug, V: Clone + PortableHash + Debug, Db> Store
    for SnapshotBuilder<K, V, Db>
{
    type Error = String;
    type Key = K;
    type Value = V;

    fn calc_subtree_hash(
        &self,
        _hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, Self::Error> {
        let inner = self.inner.borrow();
        let (node_hash, _) = inner
            .nodes
            .get(hash_idx as usize)
            .ok_or("Hash Index out of bounds")?;

        Ok(*node_hash)
    }

    fn get(&self, hash_idx: Idx) -> Result<NodeOrLeaf<Self::Key, Self::Value>, Self::Error> {
        let inner = self.inner.borrow();
        let (_, node_opt) = inner
            .nodes
            .get(hash_idx as usize)
            .ok_or("Index out of bounds")?;
        node_opt.clone().ok_or("Node not found".into())
    }
}
