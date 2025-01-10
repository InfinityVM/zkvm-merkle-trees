use alloc::{string::String, sync::Arc};
use core::cell::RefCell;

use imbl::Vector;
use kairos_trie::{PortableHash, PortableHasher};

use crate::errors::BTreeError;
use crate::node::{LeafNode, Node, NodeRef};
use crate::{
    db::DatabaseGet,
    node::{
        InnerNodeSnapshot, InnerOuter, InnerOuterSnapshotArc, InnerOuterSnapshotRef, NodeHash,
        EMPTY_TREE_ROOT_HASH,
    },
    store::{Idx, Store},
};

/// A snapshot of the merkle B+Tree verified
///
/// Contains visited nodes and unvisited nodes with pre-computed hashes
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedSnapshot<S: Store> {
    snapshot: S,

    /// The root hash of the snapshot is the last hash in the slice.
    /// The indexes of each hash match the indexes of nodes in the snapshot.
    node_hashes: Box<[NodeHash]>,
    leaf_hashes: Box<[NodeHash]>,
}

impl<S: Store + Default> Default for VerifiedSnapshot<S> {
    #[inline]
    fn default() -> Self {
        Self {
            snapshot: S::default(),
            node_hashes: Box::new([]),
            leaf_hashes: Box::new([]),
        }
    }
}

impl<S: Store + AsRef<Snapshot<S::Key, S::Value>>> VerifiedSnapshot<S> {
    #[inline]
    pub fn empty() -> Self
    where
        S: Default,
    {
        Self::default()
    }

    /// Verify the snapshot by checking that it is well formed and calculating the merkle hashes of all nodes.
    /// The merkle hashes are cached such that `calc_subtree_hash` is an O(1) operation for all nodes in the snapshot.
    #[inline]
    pub fn verify_snapshot(
        snapshot: S,
        hasher: &mut impl PortableHasher<32>,
    ) -> Result<Self, String> {
        let snapshot_ref = snapshot.as_ref();

        // Check that the snapshot is well formed
        snapshot_ref.root_node_ref()?;

        let mut node_hashes = Vec::with_capacity(snapshot_ref.branches.len());
        let mut leaf_hashes = Vec::with_capacity(snapshot_ref.leaves.len());

        for leaf in snapshot_ref.leaves.iter() {
            leaf.portable_hash(hasher);
            leaf_hashes.push(hasher.finalize_reset());
        }

        let leaf_offset = snapshot_ref.branches.len();
        let unvisited_offset = leaf_offset + snapshot_ref.leaves.len();

        for (idx, node) in snapshot_ref.branches.iter().enumerate() {
            let hash_of_child = |child| {
                if child < leaf_offset {
                    node_hashes.get(child).ok_or_else(|| {
                        format!(
                            "Invalid snapshot: node {} has child {},\
                            child node index must be less than parent",
                            idx, child
                        )
                    })
                } else if child < unvisited_offset {
                    leaf_hashes.get(child - leaf_offset).ok_or_else(|| {
                        format!(
                            "Invalid snapshot: node {} has child {},\
                            child leaf does not exist",
                            idx, child
                        )
                    })
                } else {
                    snapshot_ref
                        .unvisited_nodes
                        .get(child - unvisited_offset)
                        .ok_or_else(|| {
                            format!(
                                "Invalid snapshot: node {} has child {},\
                                child unvisited node does not exist",
                                idx, child
                            )
                        })
                }
            };

            let child_hashes: Result<Vec<_>, _> = node
                .children
                .iter()
                .map(|&c| hash_of_child(c as usize))
                .collect();
            let child_hashes = child_hashes?;

            node.portable_hash_iter(hasher, child_hashes.into_iter());
            node_hashes.push(hasher.finalize_reset());
        }

        Ok(VerifiedSnapshot {
            snapshot,
            node_hashes: node_hashes.into_boxed_slice(),
            leaf_hashes: leaf_hashes.into_boxed_slice(),
        })
    }

    #[inline]
    pub fn root_hash(&self) -> NodeHash {
        self.node_hashes
            .last()
            .copied()
            .or_else(|| self.leaf_hashes.first().copied())
            .or_else(|| self.snapshot.as_ref().unvisited_nodes.first().copied())
            .unwrap_or(EMPTY_TREE_ROOT_HASH)
    }

    #[inline]
    pub(crate) fn root_node_ref(&self) -> Option<NodeRef<S::Key, S::Value>> {
        let snapshot = self.snapshot.as_ref();
        snapshot
            .root_node_ref()
            .expect("ill-formed verified snapshot")
    }
}

impl<S: Store + AsRef<Snapshot<S::Key, S::Value>>> Store for VerifiedSnapshot<S> {
    type Key = S::Key;
    type Value = S::Value;

    #[inline]
    fn get_store_root_idx(&self) -> Option<Idx> {
        // Safety: We know the snapshot is well-formed, so the root node must be stored
        self.root_node_ref().map(|n| n.stored().unwrap())
    }

    #[inline]
    fn get_store_root_hash(&self) -> NodeHash {
        self.root_hash()
    }

    #[inline]
    fn calc_subtree_hash(
        &self,
        _: &mut impl PortableHasher<32>,
        node: Idx,
    ) -> Result<NodeHash, BTreeError> {
        let snapshot = self.snapshot.as_ref();

        let idx = node as usize;
        let leaf_offset = snapshot.branches.len();
        let unvisited_offset = leaf_offset + snapshot.leaves.len();

        if let Some(node) = self.node_hashes.get(idx) {
            Ok(*node)
        } else if let Some(leaf) = self.leaf_hashes.get(idx - leaf_offset) {
            Ok(*leaf)
        } else if let Some(hash) = snapshot.unvisited_nodes.get(idx - unvisited_offset) {
            Ok(*hash)
        } else {
            Err(format!(
                "Invalid arg: node {} does not exist\n\
                Snapshot has {} nodes",
                idx,
                snapshot.branches.len() + snapshot.leaves.len() + snapshot.unvisited_nodes.len(),
            )
            .into())
        }
    }

    #[inline]
    fn get(&self, idx: Idx) -> Result<InnerOuterSnapshotRef<Self::Key, Self::Value>, BTreeError> {
        let snapshot = self.snapshot.as_ref();

        let idx = idx as usize;
        let leaf_offset = snapshot.branches.len();
        let unvisited_offset = leaf_offset + snapshot.leaves.len();

        if let Some(node) = snapshot.branches.get(idx) {
            Ok(InnerOuterSnapshotRef::Inner(node))
        } else if let Some(leaf) = snapshot.leaves.get(idx - leaf_offset) {
            Ok(InnerOuterSnapshotRef::Outer(leaf))
        } else if snapshot
            .unvisited_nodes
            .get(idx - unvisited_offset)
            .is_some()
        {
            Err(format!(
                "Invalid arg: node {idx} is unvisited\n\
                get can only return visited nodes"
            )
            .into())
        } else {
            Err(format!(
                "Invalid arg: node {} does not exist\n\
                Snapshot has {} nodes",
                idx,
                snapshot.branches.len() + snapshot.leaves.len() + snapshot.unvisited_nodes.len(),
            )
            .into())
        }
    }
}

/// A snapshot of the merkle B+Tree.
///
/// The Snapshot contains visited nodes and the merkle hashes of every unvisited node.
/// The Snapshot acts as a serializable representation of a tree.
/// The Snapshot contains the minimum information required to perform a given `Transaction` on the tree in a verifiable manner.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot<K, V> {
    /// The last branch is the root of the trie if it exists.
    branches: Box<[InnerNodeSnapshot<K>]>,
    leaves: Box<[LeafNode<K, V>]>,

    // we only store the hashes of the nodes that have not been visited.
    unvisited_nodes: Box<[NodeHash]>,
}

impl<K, V> Default for Snapshot<K, V> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<K, V> Snapshot<K, V> {
    #[inline]
    pub fn empty() -> Self {
        Self {
            branches: Box::new([]),
            leaves: Box::new([]),
            unvisited_nodes: Box::new([]),
        }
    }

    /// Checks that the number of branches, leaves, and unvisited nodes could be a valid snapshot.
    /// This does not check that the snapshot represents a valid merkle tree.
    fn root_node_ref(&self) -> Result<Option<NodeRef<K, V>>, &'static str> {
        match (
            self.branches.as_ref(),
            self.leaves.as_ref(),
            self.unvisited_nodes.as_ref(),
        ) {
            ([], [], []) => Ok(None),
            ([.., _], _, _) => Ok(Some(NodeRef::Stored(self.branches.len() as Idx - 1))),
            ([], [_], []) | ([], [], [_]) => Ok(Some(NodeRef::Stored(0))),
            _ => Err("ill-formed snapshot"),
        }
    }
}

impl<K, V> AsRef<Snapshot<K, V>> for Snapshot<K, V> {
    fn as_ref(&self) -> &Snapshot<K, V> {
        self
    }
}

impl<K: Clone + Ord + PortableHash, V: Clone + PortableHash> Store for Snapshot<K, V> {
    type Key = K;
    type Value = V;

    fn get_store_root_idx(&self) -> Option<Idx> {
        unimplemented!("Use VerifiedSnapshot to get the root index of a snapshot")
    }

    fn get_store_root_hash(&self) -> NodeHash {
        // TODO: Don't implement Store for Snapshot
        unimplemented!("Use VerifiedSnapshot to get the root hash of a snapshot")
    }

    fn calc_subtree_hash(
        &self,
        _hasher: &mut impl PortableHasher<32>,
        _hash_idx: Idx,
    ) -> Result<NodeHash, BTreeError> {
        // TODO: split calc and get into two traits
        unimplemented!("Use VerifiedSnapshot to calculate the hash of a snapshot")
    }

    fn get(
        &self,
        hash_idx: Idx,
    ) -> Result<InnerOuterSnapshotRef<'_, Self::Key, Self::Value>, BTreeError> {
        let idx = hash_idx as usize;
        if let Some(node) = self.branches.get(idx) {
            Ok(InnerOuterSnapshotRef::Inner(node))
        } else if let Some(leaf) = self.leaves.get(idx - self.branches.len()) {
            Ok(InnerOuterSnapshotRef::Outer(leaf))
        } else if self
            .unvisited_nodes
            .get(idx - self.branches.len())
            .is_some()
        {
            Err(format!(
                "Invalid arg: node {idx} is unvisited\n\
                get can only return visited nodes"
            )
            .into())
        } else {
            Err(format!(
                "Invalid arg: node {} does not exist\n\
                Snapshot has {} nodes",
                idx,
                self.branches.len() + self.leaves.len() + self.unvisited_nodes.len(),
            )
            .into())
        }
    }
}

#[derive(Clone)]
pub struct SnapshotBuilder<K: Ord + Clone + PortableHash, V: Clone + PortableHash, Db> {
    pub db: Db,
    inner: RefCell<SnapshotBuilderInner<K, V>>,
}

#[derive(Clone)]
pub struct SnapshotBuilderInner<K: Ord + Clone + PortableHash, V: Clone + PortableHash> {
    nodes: Vector<(NodeHash, Option<InnerOuterSnapshotArc<K, V>>)>,
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

    pub fn build_initial_snapshot(&self) -> Snapshot<K, V> {
        let nodes = &self.inner.borrow().nodes;
        if nodes.is_empty() {
            Snapshot {
                branches: Box::new([]),
                leaves: Box::new([]),
                unvisited_nodes: Box::new([]),
            }
        } else {
            let mut state = SnapshotBuilderFold::new(nodes.clone());
            let root_idx = state.fold(0);

            debug_assert!(state.branches.is_empty() || root_idx == state.branches.len() as Idx - 1);
            debug_assert_eq!(state.branch_count, state.branches.len() as u32);
            debug_assert_eq!(state.leaf_count, state.leaves.len() as u32);
            debug_assert_eq!(state.unvisited_count, state.unvisited_nodes.len() as u32);

            state.build()
        }
    }
}

impl<K: Ord + Clone + PortableHash, V: Clone + PortableHash, Db: DatabaseGet<K, V>> Store
    for SnapshotBuilder<K, V, Db>
{
    type Key = K;
    type Value = V;

    #[inline]
    fn get_store_root_idx(&self) -> Option<Idx> {
        if self.inner.borrow().nodes.is_empty() {
            None
        } else {
            Some(0)
        }
    }

    #[inline]
    fn get_store_root_hash(&self) -> NodeHash {
        self.root_hash()
    }

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
    ) -> Result<NodeHash, BTreeError> {
        let inner = self.inner.borrow();
        let (node_hash, _) = inner
            .nodes
            .get(hash_idx as usize)
            .ok_or("Hash Index out of bounds")?;

        Ok(*node_hash)
    }

    fn get(&self, hash_idx: Idx) -> Result<InnerOuterSnapshotRef<K, V>, BTreeError> {
        let mut inner = self.inner.borrow_mut();
        let (hash, unread) = inner
            .nodes
            .get(hash_idx as usize)
            .ok_or("Index out of bounds")
            .map(|(hash, node_or_leaf_opt)| (*hash, node_or_leaf_opt.is_none()))?;

        if unread {
            let newly_read = self.db.get(&hash).map_err(|e| e.to_string())?;
            match newly_read {
                InnerOuter::Inner(node) => {
                    let children = node
                        .children
                        .iter()
                        .map(|child_hash| {
                            let child_hash_idx = inner.nodes.len() as Idx;
                            inner.nodes.push_back((*child_hash, None));
                            child_hash_idx
                        })
                        .collect();

                    inner.nodes[hash_idx as usize].1 =
                        Some(InnerOuterSnapshotArc::Inner(Arc::new(Node {
                            keys: node.keys,
                            children,
                        })));
                }
                InnerOuter::Outer(leaf) => {
                    inner.nodes[hash_idx as usize].1 =
                        Some(InnerOuterSnapshotArc::Outer(Arc::new(leaf.clone())));
                }
            }
        }

        // unwrap is safe because we just set the value to Some
        let node_or_leaf = inner.nodes[hash_idx as usize].1.as_ref().unwrap();

        // Safety: This is safe because the SnapshotBuilder is garanteed to outlive 'l the lifetime
        // The SnapshotBuilder hold one copy of each Arc<Node> and Arc<Leaf> in the nodes Vector until it is dropped.
        // Hence, the reference to the Arc<Node> or Arc<Leaf> is valid for the lifetime of the SnapshotBuilder.
        unsafe {
            match node_or_leaf {
                InnerOuter::Inner(node) => {
                    Ok(InnerOuterSnapshotRef::Inner(&*(node.as_ref() as *const _)))
                }
                InnerOuter::Outer(leaf) => {
                    Ok(InnerOuterSnapshotRef::Outer(&*(leaf.as_ref() as *const _)))
                }
            }
        }
    }
}

struct SnapshotBuilderFold<K, V> {
    nodes: Vector<(NodeHash, Option<InnerOuterSnapshotArc<K, V>>)>,
    /// The count of branches that will be in the snapshot
    branch_count: u32,
    /// The count of leaves that will be in the snapshot
    leaf_count: u32,
    /// The count of unvisited nodes that will be in the snapshot
    unvisited_count: u32,
    branches: Vec<Node<K, Idx>>,
    leaves: Vec<LeafNode<K, V>>,
    unvisited_nodes: Vec<NodeHash>,
}

impl<K: Clone, V: Clone> SnapshotBuilderFold<K, V> {
    #[inline]
    fn new(nodes: Vector<(NodeHash, Option<InnerOuterSnapshotArc<K, V>>)>) -> Self {
        let mut branch_count = 0;
        let mut leaf_count = 0;
        let mut unvisited_count = 0;

        for (_, node) in nodes.iter() {
            match node {
                Some(InnerOuter::Inner(_)) => branch_count += 1,
                Some(InnerOuter::Outer(_)) => leaf_count += 1,
                None => unvisited_count += 1,
            }
        }

        SnapshotBuilderFold {
            nodes,
            branch_count,
            leaf_count,
            unvisited_count,
            branches: Vec::with_capacity(branch_count as usize),
            leaves: Vec::with_capacity(leaf_count as usize),
            unvisited_nodes: Vec::with_capacity(unvisited_count as usize),
        }
    }

    #[inline]
    fn push_branch(&mut self, branch: Node<K, Idx>) -> Idx {
        let idx = self.branches.len() as Idx;
        self.branches.push(branch);
        idx
    }

    #[inline]
    fn push_leaf(&mut self, leaf: LeafNode<K, V>) -> Idx {
        let idx = self.leaves.len() as Idx;
        self.leaves.push(leaf);
        self.branch_count + idx
    }

    #[inline]
    fn push_unvisited(&mut self, hash: NodeHash) -> Idx {
        let idx = self.unvisited_nodes.len() as Idx;
        self.unvisited_nodes.push(hash);
        self.branch_count + self.leaf_count + idx
    }

    #[inline]
    fn fold(&mut self, node_idx: Idx) -> Idx {
        // TODO remove this clone
        match self.nodes[node_idx as usize].clone() {
            (_, Some(InnerOuter::Inner(branch))) => {
                let keys = branch.keys.clone();

                let children = branch
                    .children
                    .iter()
                    .map(|child_idx| self.fold(*child_idx))
                    .collect();

                self.push_branch(Node { keys, children })
            }
            // We could remove the clone by taking ownership of the SnapshotBuilder.
            // However, given this only runs on the server we can afford the clone.
            (_, Some(InnerOuter::Outer(leaf))) => self.push_leaf(leaf.as_ref().clone()),
            (hash, None) => self.push_unvisited(hash),
        }
    }

    #[inline]
    fn build(self) -> Snapshot<K, V> {
        Snapshot {
            branches: self.branches.into_boxed_slice(),
            leaves: self.leaves.into_boxed_slice(),
            unvisited_nodes: self.unvisited_nodes.into_boxed_slice(),
        }
    }
}
