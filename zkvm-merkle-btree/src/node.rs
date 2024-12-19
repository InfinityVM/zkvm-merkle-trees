use core::iter;

use alloc::sync::Arc;

use arrayvec::ArrayVec;
use kairos_trie::{PortableHash, PortableUpdate};

use crate::store::Idx;

/// TODO make it configurable
pub const BTREE_ORDER: usize = 10;
pub const EMPTY_TREE_ROOT_HASH: [u8; 32] = [0; 32];

pub type NodeHash = [u8; 32];

/// A Node of a B+Tree.
///
/// The keys and values should be small cheaply movable types.
/// Consider using an `Arc` if the value is larger than u64.
///
/// A key may appear in the nodes at multiple levels since this is a B+Tree.
/// Note that same list of keys and values may not result in the same merkle root hash,
/// if the keys are inserted in a different order.
///
/// In other words the order of the operations performed on the tree is significant.
///
/// The keys are sorted in ascending order.
/// The keys partition the children greater or equal to the key go right.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeRep<K, NR> {
    // TODO consider using a sparse array to avoid needless copying
    pub keys: ArrayVec<K, { BTREE_ORDER * 2 - 1 }>,
    pub children: ArrayVec<NR, { BTREE_ORDER * 2 }>,
}

pub type Node<K, V> = NodeRep<K, NodeRef<K, V>>;
pub type NodeSnapshot<K> = NodeRep<K, Idx>;

impl<K: Ord + Clone, V: Clone> Node<K, V> {
    pub fn is_full(&self) -> bool {
        self.keys.len() == BTREE_ORDER * 2 - 1
    }

    pub fn is_to_small(&self) -> bool {
        self.keys.len() < Self::min_keys()
    }

    pub const fn max_children() -> usize {
        BTREE_ORDER * 2
    }

    pub const fn min_children() -> usize {
        BTREE_ORDER
    }

    pub const fn max_keys() -> usize {
        BTREE_ORDER * 2 - 1
    }

    pub const fn min_keys() -> usize {
        BTREE_ORDER - 1
    }

    /// Try to merge or balance the node with a sibling.
    /// Returns Ok(()) if the node was merged or balanced.
    /// Returns Err(()) if the both siblings are stored nodes.
    ///
    /// This method is left biased, it will try to merge or balance with the left sibling first.
    /// This method will skip a stored sibling to avoid bloating the snapshot.
    /// The order of attempts is left merge, right merge, left balance, right balance.
    #[allow(clippy::result_unit_err)]
    pub fn merge_or_balance(&mut self, underflow_idx: usize) -> Result<(), ()> {
        let children_keys_len = self.children[underflow_idx].node().unwrap().keys.len();
        debug_assert!(children_keys_len < Self::min_keys());

        let (left, underflowed, right) = if underflow_idx == 0 {
            let Some([underflow_idx, right, ..]) = self.children.get_mut(underflow_idx..) else {
                return Ok(());
            };
            (&mut NodeRef::Null, underflow_idx, right)
        } else if underflow_idx + 1 == self.children.len() {
            let Some([left, underflow_idx, ..]) = self.children.get_mut(underflow_idx - 1..) else {
                return Ok(());
            };
            (left, underflow_idx, &mut NodeRef::Null)
        } else {
            let Some([left, underflow_idx, right, ..]) = self.children.get_mut(underflow_idx - 1..)
            else {
                return Ok(());
            };

            (left, underflow_idx, right)
        };

        match (left, underflowed, right) {
            (NodeRef::Node(left_arc), NodeRef::Node(underflow_arc), _)
                if left_arc.can_be_merged(underflow_arc) =>
            {
                let left_node = Arc::make_mut(left_arc);
                let underflow_node = Arc::make_mut(underflow_arc);
                let middle_key = self.keys.remove(underflow_idx - 1);
                left_node.merge(middle_key, underflow_node);
                self.children.remove(underflow_idx);
                Ok(())
            }

            (_, NodeRef::Node(underflow_arc), NodeRef::Node(right_arc))
                if underflow_arc.can_be_merged(right_arc) =>
            {
                let underflow_node = Arc::make_mut(underflow_arc);
                let right_node = Arc::make_mut(right_arc);
                let middle_key = self.keys.remove(underflow_idx);
                underflow_node.merge(middle_key, right_node);
                self.children.remove(underflow_idx + 1);
                Ok(())
            }

            (NodeRef::Node(left_arc), NodeRef::Node(underflow_arc), _) => {
                debug_assert!(left_arc.keys.len() > Self::min_keys());

                let left_node = Arc::make_mut(left_arc);
                let underflow_node = Arc::make_mut(underflow_arc);

                let middle_key = self.keys.get_mut(underflow_idx - 1).unwrap();
                left_node.balance(middle_key, underflow_node);

                Ok(())
            }

            (_, NodeRef::Node(underflow_arc), NodeRef::Node(right_arc)) => {
                debug_assert!(right_arc.keys.len() > Self::min_keys());

                let underflow_node = Arc::make_mut(underflow_arc);
                let right_node = Arc::make_mut(right_arc);

                let middle_key = self.keys.get_mut(underflow_idx).unwrap();
                underflow_node.balance(middle_key, right_node);

                Ok(())
            }

            (NodeRef::Stored(_) | NodeRef::Null, _, NodeRef::Stored(_))
            | (NodeRef::Stored(_), _, NodeRef::Null) => Err(()),

            _ => {
                unreachable!("merge_or_balance called on node with one child");
            }
        }
    }

    pub fn balance(&mut self, middle_key: &mut K, right_node: &mut Self) {
        // assert that we could not merge
        // one is the middle key from the parent
        debug_assert!(self.keys.len() + 1 + right_node.keys.len() > Self::max_keys());
        debug_assert!(self.children.len() + right_node.children.len() > Self::max_children());

        let child_count = self.children.len() + right_node.children.len();
        let left_count = child_count / 2;
        // This is not the most efficient way of doing this, but we need to replace the ArrayVec anyway.
        let mut children = self.children.drain(..).chain(right_node.children.drain(..));
        let mut keys = self
            .keys
            .drain(..)
            .chain(iter::once(middle_key.clone()))
            .chain(right_node.keys.drain(..));

        let left_keys = (&mut keys).take(left_count - 1).collect();
        *middle_key = keys.next().unwrap();
        let right_key = keys.collect();
        let left_children = (&mut children).take(left_count).collect();
        let right_children = children.collect();

        self.keys = left_keys;
        self.children = left_children;

        right_node.keys = right_key;
        right_node.children = right_children;

        self.assert_invariants();
        right_node.assert_invariants();
    }

    pub fn merge(&mut self, middle_key: K, right_node: &mut Self) {
        let key_count = self.keys.len() + 1 + right_node.keys.len();
        let child_count = self.children.len() + right_node.children.len();
        debug_assert!(key_count <= Self::max_keys());
        debug_assert!(child_count <= Self::max_children());
        debug_assert!(key_count == child_count - 1);

        self.keys.push(middle_key);
        self.keys.extend(right_node.keys.drain(..));

        self.children.extend(right_node.children.drain(..));

        self.assert_invariants();
    }

    #[inline(always)]
    fn can_be_merged(&self, right_node: &Self) -> bool {
        #[cfg(debug_assertions)]
        {
            match (self.keys.as_slice(), right_node.keys.as_slice()) {
                ([.., left_last], [right_first, ..]) => assert!(left_last < right_first),
                ([], _) => assert!(self.children.len() == 1),
                (_, []) => assert!(right_node.children.len() == 1),
            };
        }

        // We use less than because the middle key from the parent also needs to be added.
        self.keys.len() + right_node.keys.len() < Self::max_keys()
    }

    #[track_caller]
    pub fn assert_keys_sorted(&self) {
        for i in 1..self.keys.len() {
            assert!(self.keys[i - 1] < self.keys[i],);
        }
    }

    #[track_caller]
    pub fn assert_children_sorted(&self) {
        let mut prior_child_key = None;
        for (i, child) in self.children.iter().enumerate() {
            match child {
                NodeRef::Leaf(leaf) => {
                    let _ = self.keys.get(i).map(|key| assert!(*key >= leaf.key,));

                    if let Some(prior_key) = prior_child_key {
                        assert!(prior_key < &leaf.key);
                    }
                    prior_child_key = Some(&leaf.key);
                }
                NodeRef::Node(node) => {
                    let _ = self
                        .keys
                        .get(i)
                        .map(|key| assert!(key >= node.keys.last().unwrap()));

                    if let Some(child_key) = node.keys.last() {
                        if let Some(prior_key) = prior_child_key {
                            assert!(prior_key < child_key);
                        }
                        prior_child_key = Some(child_key);
                    }
                }
                _ => {}
            }
        }
    }

    #[track_caller]
    pub fn assert_invariants(&self) {
        let all_children_are_leafs = self
            .children
            .iter()
            .all(|child| matches!(child, NodeRef::Leaf(_)));

        if !all_children_are_leafs {
            // Fix with root check
            // assert!(self.keys.len() >= Self::min_keys());
            // assert!(self.children.len() >= Self::min_children());
        }

        assert!(self.keys.len() <= Self::max_keys());
        assert!(self.children.len() <= Self::max_children());

        assert_eq!(self.keys.len() + 1, self.children.len());

        self.assert_keys_sorted();
        self.assert_children_sorted();
    }
}

impl<K> PortableHash for NodeRep<K, NodeHash>
where
    K: PortableHash,
{
    fn portable_hash<H: PortableUpdate>(&self, hasher: &mut H) {
        // TODO: Add size to hash
        self.keys.iter().for_each(|key| key.portable_hash(hasher));
        self.children
            .iter()
            .for_each(|child_hash| child_hash.portable_hash(hasher));
    }
}

impl<K: PortableHash, NR> NodeRep<K, NR> {
    pub fn portable_hash_iter<'l>(
        &self,
        hasher: &mut impl PortableUpdate,
        child_hashes: impl Iterator<Item = &'l NodeHash>,
    ) {
        // TODO: Add size to hash
        self.keys.iter().for_each(|key| key.portable_hash(hasher));
        child_hashes.for_each(|child_hash| child_hash.portable_hash(hasher));
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Leaf<K, V> {
    pub key: K,
    pub value: V,
}

impl<K: PortableHash, V: PortableHash> PortableHash for Leaf<K, V> {
    fn portable_hash<H: PortableUpdate>(&self, hasher: &mut H) {
        self.key.portable_hash(hasher);
        self.value.portable_hash(hasher);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeOrLeaf<N, L> {
    Node(N),
    Leaf(L),
}

pub type NodeOrLeafRef<'a, K, V> = NodeOrLeaf<&'a Arc<Node<K, V>>, &'a Arc<Leaf<K, V>>>;
pub type NodeOrLeafOwned<K, V> = NodeOrLeaf<Arc<Node<K, V>>, Arc<Leaf<K, V>>>;
pub type NodeOrLeafDb<K, V> = NodeOrLeaf<NodeRep<K, NodeHash>, Leaf<K, V>>;
pub type NodeOrLeafSnapshotRef<'a, K, V> = NodeOrLeaf<&'a NodeSnapshot<K>, &'a Leaf<K, V>>;
pub type NodeOrLeafSnapshotArc<K, V> = NodeOrLeaf<Arc<NodeSnapshot<K>>, Arc<Leaf<K, V>>>;

impl<N, L> NodeOrLeaf<N, L> {
    pub fn node(&self) -> Option<&N> {
        match self {
            NodeOrLeaf::Node(node) => Some(node),
            _ => None,
        }
    }

    pub fn leaf(&self) -> Option<&L> {
        match self {
            NodeOrLeaf::Leaf(leaf) => Some(leaf),
            _ => None,
        }
    }
}

impl<K, V> NodeOrLeafOwned<K, V> {
    pub fn as_ref(&self) -> NodeOrLeaf<&Arc<Node<K, V>>, &Arc<Leaf<K, V>>> {
        match self {
            NodeOrLeaf::Node(node) => NodeOrLeaf::Node(node),
            NodeOrLeaf::Leaf(leaf) => NodeOrLeaf::Leaf(leaf),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NodeRef<K, V> {
    Node(Arc<Node<K, V>>),
    // TODO it should be a hash if the value is very large
    // TODO Fix leaf
    Leaf(Arc<Leaf<K, V>>),
    Stored(Idx),
    Null,
}

impl<K, V> NodeRef<K, V> {
    pub fn node(&self) -> Option<&Arc<Node<K, V>>> {
        match self {
            NodeRef::Node(node) => Some(node),
            _ => None,
        }
    }

    pub fn leaf(&self) -> Option<&Arc<Leaf<K, V>>> {
        match self {
            NodeRef::Leaf(leaf) => Some(leaf),
            _ => None,
        }
    }

    pub fn stored(&self) -> Option<Idx> {
        match self {
            NodeRef::Stored(idx) => Some(*idx),
            _ => None,
        }
    }
}

impl<K, V> NodeRef<K, V> {
    pub fn new_leaf(key: K, value: V) -> Self {
        NodeRef::Leaf(Arc::new(Leaf { key, value }))
    }
}

impl<K: Clone, V: Clone> From<NodeOrLeafSnapshotRef<'_, K, V>> for NodeRef<K, V> {
    fn from(node_or_leaf: NodeOrLeafSnapshotRef<K, V>) -> Self {
        match node_or_leaf {
            NodeOrLeaf::Node(node) => NodeRef::Node(Arc::new(NodeRep {
                keys: node.keys.clone(),
                children: node
                    .children
                    .iter()
                    .map(|idx| NodeRef::Stored(*idx))
                    .collect(),
            })),
            NodeOrLeaf::Leaf(leaf) => NodeRef::Leaf(Arc::new(leaf.clone())),
        }
    }
}

impl<K, V> From<Arc<Node<K, V>>> for NodeRef<K, V> {
    fn from(node: Arc<Node<K, V>>) -> Self {
        NodeRef::Node(node)
    }
}

impl<K, V> From<Arc<Leaf<K, V>>> for NodeRef<K, V> {
    fn from(leaf: Arc<Leaf<K, V>>) -> Self {
        NodeRef::Leaf(leaf)
    }
}

impl<K, V> From<NodeOrLeafOwned<K, V>> for NodeRef<K, V> {
    fn from(node_or_leaf: NodeOrLeafOwned<K, V>) -> Self {
        match node_or_leaf {
            NodeOrLeaf::Node(node) => NodeRef::Node(node),
            NodeOrLeaf::Leaf(leaf) => NodeRef::Leaf(leaf),
        }
    }
}

impl<K, V> From<NodeOrLeafRef<'_, K, V>> for NodeRef<K, V> {
    fn from(node_or_leaf: NodeOrLeafRef<'_, K, V>) -> Self {
        match node_or_leaf {
            NodeOrLeaf::Node(node) => NodeRef::Node(node.clone()),
            NodeOrLeaf::Leaf(leaf) => NodeRef::Leaf(leaf.clone()),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use kairos_trie::{DigestHasher, PortableHasher};
    use proptest::prelude::*;
    use sha2::Sha256;

    proptest! {
        #[test]
        fn test_portable_hash_matches(
            // This test does not ensure the Node is valid
            // the number of children and keys may not match
            keys in proptest::collection::vec(0u32..100, 1..(BTREE_ORDER * 2 - 1)),
            child_hashes in proptest::collection::vec(any::<[u8; 32]>(), 2..(BTREE_ORDER * 2)),
        ) {
            let node_with_hashes = NodeRep {
                keys: ArrayVec::from_iter(keys.clone()),
                children: ArrayVec::from_iter(child_hashes.clone()),
            };

            let hasher1 = &mut DigestHasher::<Sha256>::default();
            let hasher2 = &mut DigestHasher::<Sha256>::default();

            node_with_hashes.portable_hash(hasher1);
            node_with_hashes.portable_hash_iter(hasher2, child_hashes.iter());

            assert_eq!(hasher1.finalize_reset(), hasher2.finalize_reset());
        }
    }
}
