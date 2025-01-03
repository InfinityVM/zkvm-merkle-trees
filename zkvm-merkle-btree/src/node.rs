use alloc::sync::Arc;
use core::iter;

use arrayvec::ArrayVec;
use kairos_trie::{PortableHash, PortableUpdate};

use crate::{store::Idx, transaction::Insert_};

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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeRep<K, NR> {
    // TODO consider using a sparse array to avoid needless copying
    /// An Inner Node has at most `BTREE_ORDER * 2 - 1` keys.
    /// A node with the same number of keys and children is a leaf node.
    pub keys: ArrayVec<K, { BTREE_ORDER * 2 }>,
    pub children: ArrayVec<NR, { BTREE_ORDER * 2 }>,
}

pub type InnerNode<K, V> = NodeRep<K, NodeRefType<K, V>>;
pub type LeafNode<K, V> = NodeRep<K, V>;

pub type InnerNodeSnapshot<K> = NodeRep<K, Idx>;

// TODO consider making these specialized for inner and outer nodes
impl<K, V> NodeRep<K, V> {
    pub const fn is_full(&self) -> bool {
        self.children.len() == Self::max_children()
    }

    pub fn is_too_small(&self) -> bool {
        self.children.len() < Self::min_children()
    }

    pub const fn max_children() -> usize {
        BTREE_ORDER * 2
    }

    pub const fn min_children() -> usize {
        BTREE_ORDER
    }

    pub const fn max_inner_node_keys() -> usize {
        BTREE_ORDER * 2 - 1
    }

    pub const fn min_keys() -> usize {
        BTREE_ORDER - 1
    }

    #[track_caller]
    pub fn assert_keys_sorted(&self)
    where
        K: Ord,
    {
        for i in 1..self.keys.len() {
            assert!(self.keys[i - 1] < self.keys[i],);
        }
    }
}

impl<K: Clone + Ord, V: Clone> InnerNode<K, V> {
    /// Splits an inner node, If the parent node (`self`) is full this function will return a `SplitNode`.
    /// The split node contains the middle key and the right sibling node of the parent node.
    /// The parent becomes the left sibling node.
    /// The caller must propagate the split up the tree.
    pub(crate) fn handle_split(
        &mut self,
        idx: usize,
        middle_key: K,
        new_right: InnerOuter<Arc<InnerNode<K, V>>, Arc<LeafNode<K, V>>>,
    ) -> Insert_<K, V> {
        if !self.is_full() {
            self.keys.insert(idx, middle_key);
            self.children.insert(idx + 1, new_right.into());

            Insert_::Inserted
        } else {
            let mut new_right_inner = NodeRep {
                keys: self.keys.drain((Self::min_keys() + 1)..).collect(),
                children: self.children.drain(Self::min_children()..).collect(),
            };
            let new_middle_key = self.keys.pop().unwrap();

            if idx < Self::min_children() {
                self.keys.insert(idx, middle_key);
                self.children.insert(idx + 1, new_right.into());
            } else {
                let adjusted_idx = idx - Self::min_children();
                new_right_inner.keys.insert(adjusted_idx, middle_key);
                new_right_inner
                    .children
                    .insert(adjusted_idx + 1, new_right.into());
            }

            #[cfg(debug_assertions)]
            {
                self.assert_inner_invariants();
                new_right_inner.assert_inner_invariants();
            }

            Insert_::SplitNode {
                middle_key: new_middle_key,
                right_node: Arc::new(new_right_inner),
            }
        }
    }

    /// Try to merge or balance the node with a sibling.
    /// Returns Ok(()) if the node was merged or balanced.
    /// Returns Err(()) if the both siblings are stored nodes.
    ///
    /// This method is left biased, it will try to merge or balance with the left sibling first.
    /// This method will skip a stored sibling to avoid bloating the snapshot.
    /// The order of attempts is left merge, right merge, left balance, right balance.
    #[allow(clippy::result_unit_err)]
    pub(crate) fn merge_or_balance(&mut self, underflow_idx: usize) -> Result<(), ()> {
        debug_assert!(underflow_idx < self.children.len());
        // By definition the underflowed node has one less than the minimum number of keys.
        let (left, underflowed, right) = if underflow_idx == 0 {
            let Some([underflow_idx, right, ..]) = self.children.get_mut(underflow_idx..) else {
                unreachable!();
            };
            (None, underflow_idx, Some(right))
        } else if underflow_idx + 1 == self.children.len() {
            let Some([left, underflow_idx, ..]) = self.children.get_mut(underflow_idx - 1..) else {
                unreachable!();
            };
            (Some(left), underflow_idx, None)
        } else {
            debug_assert!(underflow_idx > 0);
            debug_assert!(underflow_idx + 1 < self.children.len());

            let Some([left, underflow_idx, right, ..]) = self.children.get_mut(underflow_idx - 1..)
            else {
                unreachable!();
            };

            (Some(left), underflow_idx, Some(right))
        };

        match (left, underflowed, right) {
            // Try to merge left then right
            (Some(NodeRefType::Leaf(left)), NodeRefType::Leaf(underflowed), _)
                if left.can_leafs_be_merged(underflowed) =>
            {
                let left = Arc::make_mut(left);
                let underflowed = Arc::make_mut(underflowed);
                left.merge_leafs(underflowed);
                self.keys.remove(underflow_idx - 1);
                self.children.remove(underflow_idx);
                Ok(())
            }

            (_, NodeRefType::Leaf(underflowed), Some(NodeRefType::Leaf(right)))
                if underflowed.can_leafs_be_merged(right) =>
            {
                let underflowed = Arc::make_mut(underflowed);
                let right = Arc::make_mut(right);
                underflowed.merge_leafs(right);
                self.keys.remove(underflow_idx);
                self.children.remove(underflow_idx + 1);
                Ok(())
            }

            // Try to balance left then right
            (Some(NodeRefType::Leaf(left)), NodeRefType::Leaf(underflowed), _) => {
                let left = Arc::make_mut(left);
                let underflowed = Arc::make_mut(underflowed);
                let middle_key = left.balance_leafs(underflowed);
                self.keys[underflow_idx - 1] = middle_key;
                Ok(())
            }

            (_, NodeRefType::Leaf(underflowed), Some(NodeRefType::Leaf(right))) => {
                let underflowed = Arc::make_mut(underflowed);
                let right = Arc::make_mut(right);
                let middle_key = underflowed.balance_leafs(right);
                self.keys[underflow_idx] = middle_key;
                Ok(())
            }

            // Try to merge InnerNodes left then right
            (Some(NodeRefType::Inner(left)), NodeRefType::Inner(underflowed), _)
                if left.can_inner_nodes_be_merged(underflowed) =>
            {
                let left = Arc::make_mut(left);
                let underflowed = Arc::make_mut(underflowed);
                let middle_key = self.keys.remove(underflow_idx - 1);
                left.merge_inner_nodes(middle_key, underflowed);
                self.children.remove(underflow_idx);
                Ok(())
            }

            (_, NodeRefType::Inner(underflowed), Some(NodeRefType::Inner(right)))
                if underflowed.can_inner_nodes_be_merged(right) =>
            {
                let underflowed = Arc::make_mut(underflowed);
                let right = Arc::make_mut(right);
                let middle_key = self.keys.remove(underflow_idx);
                underflowed.merge_inner_nodes(middle_key, right);
                self.children.remove(underflow_idx + 1);
                Ok(())
            }

            // Try to balance InnerNodes left then right
            (Some(NodeRefType::Inner(left)), NodeRefType::Inner(underflowed), _) => {
                let left = Arc::make_mut(left);
                let underflowed = Arc::make_mut(underflowed);
                left.balance_inner_nodes(&mut self.keys[underflow_idx - 1], underflowed);
                Ok(())
            }

            (_, NodeRefType::Inner(underflowed), Some(NodeRefType::Inner(right))) => {
                let underflowed = Arc::make_mut(underflowed);
                let right = Arc::make_mut(right);
                underflowed.balance_inner_nodes(&mut self.keys[underflow_idx], right);
                Ok(())
            }

            // We can't merge or balance with stored nodes
            (Some(NodeRefType::Stored(_)) | None, _, Some(NodeRefType::Stored(_)))
            | (Some(NodeRefType::Stored(_)), _, None) => Err(()),

            _ => unreachable!("An InnerNode should only have one type of children"),
        }
    }

    fn can_inner_nodes_be_merged(&self, right_node: &Self) -> bool {
        #[cfg(debug_assertions)]
        {
            match (self.keys.as_slice(), right_node.keys.as_slice()) {
                ([.., left_last], [right_first, ..]) => assert!(left_last < right_first),
                ([], _) => assert!(self.children.len() == 1),
                (_, []) => assert!(right_node.children.len() == 1),
            };
        }

        // We use less than because the middle key from the parent also needs to be added.
        self.keys.len() + right_node.keys.len() < Self::max_inner_node_keys()
    }

    fn merge_inner_nodes(&mut self, middle_key: K, right_node: &mut Self) {
        #[cfg(debug_assertions)]
        {
            self.assert_inner_invariants();
            right_node.assert_inner_invariants();
        }

        let key_count = self.keys.len() + 1 + right_node.keys.len();
        let child_count = self.children.len() + right_node.children.len();
        debug_assert_eq!(key_count, child_count - 1);
        debug_assert!(key_count <= Self::max_inner_node_keys());
        debug_assert!(child_count <= Self::max_children());

        self.keys.push(middle_key);
        self.keys.extend(right_node.keys.drain(..));

        self.children.extend(right_node.children.drain(..));

        #[cfg(debug_assertions)]
        self.assert_inner_invariants();
    }

    fn balance_inner_nodes(&mut self, middle_key: &mut K, right_node: &mut Self) {
        #[cfg(debug_assertions)]
        {
            self.assert_inner_invariants();
            right_node.assert_inner_invariants();
        }
        // assert that we could not merge
        // one is the middle key from the parent
        debug_assert!(self.keys.len() + 1 + right_node.keys.len() > Self::max_inner_node_keys());
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

        // The last key that would be in the left node is the middle key.
        let left_keys = (&mut keys).take(left_count - 1).collect();

        // The middle key partitions the left and right in the parent node.
        *middle_key = keys.next().unwrap();

        let right_key = keys.collect();
        let left_children = (&mut children).take(left_count).collect();
        let right_children = children.collect();

        self.keys = left_keys;
        self.children = left_children;

        right_node.keys = right_key;
        right_node.children = right_children;

        #[cfg(debug_assertions)]
        {
            self.assert_inner_invariants();
            assert!(self.keys.len() >= Self::min_keys());

            right_node.assert_inner_invariants();
            assert!(right_node.keys.len() >= Self::min_keys());
        }
    }

    #[track_caller]
    pub fn assert_inner_invariants(&self) {
        self.assert_keys_sorted();
        assert_eq!(self.keys.len(), self.children.len() - 1);
    }
}

impl<K: Clone + Ord, V> LeafNode<K, V> {
    /// Splits a leaf node, returning the middle key and the newly created right sibling node.
    ///
    /// This function takes the current node, the index where the key should be inserted, the key, and the value.
    pub(crate) fn insert_split(&mut self, idx: usize, key: K, value: V) -> (K, Arc<Self>) {
        #[cfg(debug_assertions)]
        self.assert_leaf_invariants();

        let mut new_right_leaf = NodeRep {
            keys: self.keys.drain((Self::min_children())..).collect(),
            children: self.children.drain(Self::min_children()..).collect(),
        };

        if idx < Self::min_children() {
            self.keys.insert(idx, key);
            self.children.insert(idx, value);
        } else {
            let adjusted_idx = idx - Self::min_children();
            new_right_leaf.keys.insert(adjusted_idx, key);
            new_right_leaf.children.insert(adjusted_idx, value);
        }

        let middle_key = self.keys.last().unwrap().clone();

        #[cfg(debug_assertions)]
        {
            self.assert_leaf_invariants();
            new_right_leaf.assert_leaf_invariants();
        }

        (middle_key.clone(), Arc::new(new_right_leaf))
    }

    /// Merge the leafs into the left leaf (self).
    /// Caller must remove the right leaf from the parent node.
    pub fn merge_leafs(&mut self, right_node: &mut Self) {
        let key_count = self.keys.len() + right_node.keys.len();
        let child_count = self.children.len() + right_node.children.len();
        debug_assert_eq!(key_count, child_count);
        debug_assert!(child_count <= Self::max_children());

        self.keys.extend(right_node.keys.drain(..));

        self.children.extend(right_node.children.drain(..));

        self.assert_leaf_invariants();
    }

    #[inline(always)]
    fn can_leafs_be_merged(&self, right_node: &Self) -> bool {
        #[cfg(debug_assertions)]
        if let ([.., left_last], [right_first, ..]) =
            (self.keys.as_slice(), right_node.keys.as_slice())
        {
            assert!(left_last < right_first)
        };

        // We use less than because the middle key from the parent also needs to be added.
        self.children.len() + right_node.children.len() <= Self::max_children()
    }

    // Balance the leafs, returning the new middle key.
    pub fn balance_leafs(&mut self, right_node: &mut Self) -> K {
        #[cfg(debug_assertions)]
        {
            self.assert_leaf_invariants();
            right_node.assert_leaf_invariants();
        }
        // assert that we could not merge
        // one is the middle key from the parent
        debug_assert!(self.children.len() + right_node.children.len() > Self::max_children());

        let child_count = self.children.len() + right_node.children.len();
        let left_count = child_count / 2;
        // This is not the most efficient way of doing this, but we need to replace the ArrayVec anyway.
        let mut children = self.children.drain(..).chain(right_node.children.drain(..));
        let mut keys = self.keys.drain(..).chain(right_node.keys.drain(..));

        let left_keys = (&mut keys).take(left_count).collect();
        let right_key = keys.collect();
        let left_children = (&mut children).take(left_count).collect();
        let right_children = children.collect();

        self.keys = left_keys;
        self.children = left_children;

        right_node.keys = right_key;
        right_node.children = right_children;

        #[cfg(debug_assertions)]
        {
            self.assert_leaf_invariants();
            assert!(self.keys.len() >= Self::min_keys());

            right_node.assert_leaf_invariants();
            assert!(right_node.keys.len() >= Self::min_keys());
        }

        // Return the new middle key
        self.keys.last().unwrap().clone()
    }

    #[track_caller]
    pub fn assert_leaf_invariants(&self) {
        self.assert_keys_sorted();
        assert_eq!(self.keys.len(), self.children.len());
    }
}

impl<K: PortableHash, V: PortableHash> PortableHash for NodeRep<K, V> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InnerOuter<N, L> {
    Inner(N),
    Outer(L),
}

impl<K: Clone, V: Clone> From<InnerOuterSnapshotRef<'_, K, V>> for NodeRefType<K, V> {
    fn from(node_or_leaf: InnerOuterSnapshotRef<K, V>) -> Self {
        match node_or_leaf {
            InnerOuter::Outer(leaf) => NodeRefType::Leaf(Arc::new(leaf.clone())),
            InnerOuter::Inner(node) => NodeRefType::Inner(Arc::new(NodeRep {
                keys: node.keys.clone(),
                children: node
                    .children
                    .iter()
                    .map(|idx| NodeRefType::Stored(*idx))
                    .collect(),
            })),
        }
    }
}

pub type NodeOrLeafDb<K, V> = InnerOuter<NodeRep<K, NodeHash>, LeafNode<K, V>>;
pub type InnerOuterSnapshotRef<'a, K, V> = InnerOuter<&'a InnerNodeSnapshot<K>, &'a LeafNode<K, V>>;
pub type InnerOuterSnapshotArc<K, V> = InnerOuter<Arc<InnerNodeSnapshot<K>>, Arc<LeafNode<K, V>>>;

impl<N, L> InnerOuter<N, L> {
    pub fn node(&self) -> Option<&N> {
        match self {
            InnerOuter::Inner(node) => Some(node),
            _ => None,
        }
    }

    pub fn leaf(&self) -> Option<&L> {
        match self {
            InnerOuter::Outer(leaf) => Some(leaf),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum NodeRefType<K, V> {
    Inner(Arc<InnerNode<K, V>>),
    Leaf(Arc<LeafNode<K, V>>),
    Stored(Idx),
}

impl<K, V> NodeRefType<K, V> {
    pub fn inner(&self) -> Option<&Arc<InnerNode<K, V>>> {
        match self {
            NodeRefType::Inner(node) => Some(node),
            _ => None,
        }
    }

    pub fn leaf(&self) -> Option<&Arc<LeafNode<K, V>>> {
        match self {
            NodeRefType::Leaf(leaf) => Some(leaf),
            _ => None,
        }
    }

    pub fn stored(&self) -> Option<Idx> {
        match self {
            NodeRefType::Stored(idx) => Some(*idx),
            _ => None,
        }
    }
}

impl<K, V> From<InnerOuter<Arc<InnerNode<K, V>>, Arc<LeafNode<K, V>>>> for NodeRefType<K, V> {
    fn from(node_or_leaf: InnerOuter<Arc<InnerNode<K, V>>, Arc<LeafNode<K, V>>>) -> Self {
        match node_or_leaf {
            InnerOuter::Inner(node) => NodeRefType::Inner(node),
            InnerOuter::Outer(leaf) => NodeRefType::Leaf(leaf),
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
