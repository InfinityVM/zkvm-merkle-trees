use alloc::sync::Arc;

use arrayvec::ArrayVec;

use crate::store::Idx;

/// TODO make it configurable
const BTREE_ORDER: usize = 6;
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
#[derive(Debug, Clone)]
pub struct Node<K, V> {
    // TODO consider using a sparse array to avoid needless copying
    pub keys: ArrayVec<K, { BTREE_ORDER * 2 - 1 }>,
    pub children: ArrayVec<NodeRef<K, V>, { BTREE_ORDER * 2 }>,
}

impl<K, V> Node<K, V> {
    pub fn is_full(&self) -> bool {
        self.keys.len() == BTREE_ORDER * 2 - 1
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
}

impl<K: Clone + Ord + std::fmt::Debug, V: std::fmt::Debug + Clone> Node<K, V> {
    #[track_caller]
    pub fn insert(&mut self, idx: usize, key: K, child: NodeRef<K, V>) {
        self.keys.insert(idx, key);
        self.children.insert(idx + 1, child);

        if cfg!(debug_assertions) {
            self.assert_invariants();
        }
    }

    #[track_caller]
    pub fn assert_keys_sorted(&self) {
        for i in 1..self.keys.len() {
            assert!(
                self.keys[i - 1] < self.keys[i],
                "keys are not sorted: {:?}",
                self.keys
            );
        }
    }

    #[track_caller]
    pub fn assert_children_sorted(&self) {
        let mut prior_child_key = None;
        for (i, child) in self.children.iter().enumerate() {
            match child {
                NodeRef::Leaf(leaf) => {
                    let _ = self.keys.get(i).map(|key| {
                        assert!(
                            *key >= leaf.key,
                            "Assertion failed: key: {:?} >= leaf.key: {:?} in node {:?}",
                            key,
                            leaf.key,
                            self
                        )
                    });

                    if let Some(prior_key) = prior_child_key {
                        assert!(
                            prior_key < leaf.key,
                            "prior_key: {:?}, leaf: {:?}, children: {:?}",
                            prior_key,
                            leaf.key,
                            self.children
                        );
                    }
                    prior_child_key = Some(leaf.key.clone());
                }
                NodeRef::Node(node) => {
                    let _ = self
                        .keys
                        .get(i)
                        .map(|key| assert!(key >= node.keys.last().unwrap()));

                    let child_key = node.keys.last().unwrap().clone();
                    if let Some(prior_key) = prior_child_key {
                        assert!(prior_key < child_key)
                    }
                    prior_child_key = Some(child_key);
                }
                _ => {}
            }
        }
    }

    #[track_caller]
    pub fn assert_invariants(&self)
    where
        K: Ord + Clone + std::fmt::Debug,
        V: Clone + std::fmt::Debug,
    {
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

#[derive(Debug, Clone)]
pub struct Leaf<K, V> {
    pub key: K,
    pub value: V,
}

#[derive(Debug, Clone)]
pub enum NodeOrLeaf<K, V> {
    Node(Arc<Node<K, V>>),
    Leaf(Arc<Leaf<K, V>>),
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

impl<K, V> From<NodeOrLeaf<K, V>> for NodeRef<K, V> {
    fn from(node_or_leaf: NodeOrLeaf<K, V>) -> Self {
        match node_or_leaf {
            NodeOrLeaf::Node(node) => NodeRef::Node(node),
            NodeOrLeaf::Leaf(leaf) => NodeRef::Leaf(leaf),
        }
    }
}
