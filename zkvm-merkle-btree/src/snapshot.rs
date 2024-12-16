use alloc::string::String;
use core::cell::RefCell;
use core::fmt::Debug;

use imbl::Vector;
use kairos_trie::{PortableHash, PortableHasher};

use crate::{
    node::{NodeHash, NodeOrLeaf, NodeOrLeafOwned, NodeOrLeafRef, EMPTY_TREE_ROOT_HASH},
    store::{Idx, Store},
};

// /// A snapshot of the merkle B+Tree.
// ///
// /// The Snapshot contains visited nodes and the merkle hashes of every unvisited node.
// /// The Snapshot acts as a serializable representation of a tree.
// /// The Snapshot contains the minimum information required to perform a given `Transaction` on the tree in a verifiable manner.
// #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
// #[derive(Clone, Debug, PartialEq, Eq)]
// pub struct Snapshot<K, V> {
//     /// The last branch is the root of the trie if it exists.
//     branches: Box<[Arc<Node<K, V>>]>,
//     leaves: Box<[Arc<Leaf<K, V>>]>,

//     // we only store the hashes of the nodes that have not been visited.
//     unvisited_nodes: Box<[NodeHash]>,
// }

#[derive(Clone)]
pub struct SnapshotBuilder<K: Ord + Clone + PortableHash, V: Clone + PortableHash, Db> {
    pub db: Db,
    inner: RefCell<SnapshotBuilderInner<K, V>>,
}

#[derive(Clone)]
pub struct SnapshotBuilderInner<K: Ord + Clone + PortableHash, V: Clone + PortableHash> {
    nodes: Vector<(NodeHash, Option<NodeOrLeafOwned<K, V>>)>,
}

impl<K: Ord + Clone + PortableHash, V: Clone + PortableHash, Db> SnapshotBuilder<K, V, Db> {
    #[inline]
    pub fn new(root: NodeHash, db: Db) -> Self {
        debug_assert!(crate::node::EMPTY_TREE_ROOT_HASH == NodeHash::default());

        if root == crate::node::EMPTY_TREE_ROOT_HASH {
            Self {
                db,
                inner: RefCell::new(SnapshotBuilderInner {
                    nodes: Vector::new(),
                }),
            }
        } else {
            Self {
                db,
                inner: RefCell::new(SnapshotBuilderInner {
                    nodes: Vector::from_iter([(root, None)]),
                }),
            }
        }
    }

    /// Returns the Merkle root hash of the tree represented by this snapshot.
    /// If the tree is empty, the root hash will be all zeroes.
    pub fn root_hash(&self) -> NodeHash {
        let inner = self.inner.borrow();
        inner
            .nodes
            .get(0)
            .map(|(hash, _)| *hash)
            .unwrap_or(EMPTY_TREE_ROOT_HASH)
    }
}

impl<K: Ord + Clone + PortableHash + Debug, V: Clone + PortableHash + Debug, Db> Store
    for SnapshotBuilder<K, V, Db>
{
    type Error = String;
    type Key = K;
    type Value = V;

    /// Calculate the merkle root hash of the snapshot.
    /// This computation can be thought of as verifying a Snapshot has a particular Merkle root hash.
    /// However, in reality, it is calculating the root hash of the snapshot
    /// by visiting all nodes touched by the transaction.
    ///
    /// Always check that the snapshot is of the merkle tree you expect.
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

    fn get(&self, hash_idx: Idx) -> Result<NodeOrLeafRef<'_, Self::Key, Self::Value>, Self::Error> {
        let inner = self.inner.borrow();
        let (_, node_opt) = inner
            .nodes
            .get(hash_idx as usize)
            .ok_or("Index out of bounds")?;

        let node_or_leaf = node_opt.as_ref().ok_or("Node not found")?;
        // Safety: This is safe because the SnapshotBuilder is garanteed to outlive 'l the lifetime
        // The SnapshotBuilder hold one copy of each Arc<Node> and Arc<Leaf> in the nodes Vector until it is dropped.
        // Hence, the reference to the Arc<Node> or Arc<Leaf> is valid for the lifetime of the SnapshotBuilder.
        unsafe {
            match node_or_leaf {
                NodeOrLeaf::Node(node) => Ok(NodeOrLeafRef::Node(&*(node as *const _))),
                NodeOrLeaf::Leaf(leaf) => Ok(NodeOrLeafRef::Leaf(&*(leaf as *const _))),
            }
        }
    }
}
