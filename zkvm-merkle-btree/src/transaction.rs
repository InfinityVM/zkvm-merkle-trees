use alloc::sync::Arc;
use core::fmt::Debug;
use core::{cmp::Ordering, mem};
use std::cmp;

use arrayvec::ArrayVec;
use kairos_trie::PortableHash;

use crate::{
    node::{Leaf, Node, NodeHash, NodeOrLeaf, NodeRef, EMPTY_TREE_ROOT_HASH},
    snapshot::SnapshotBuilder,
    store::{Idx, Store},
};

/// A transaction against a merkle b+tree.
pub struct MerkleBTreeTxn<S: Store> {
    pub data_store: S,
    current_root: NodeRef<S::Key, S::Value>,
}

impl<K: Ord + Clone + PortableHash + Debug, V: Clone + PortableHash + Debug, Db>
    MerkleBTreeTxn<SnapshotBuilder<K, V, Db>>
{
    pub fn new_snapshot_builder_txn(root: NodeHash, db: Db) -> Self {
        debug_assert!(EMPTY_TREE_ROOT_HASH == NodeHash::default());

        if root == EMPTY_TREE_ROOT_HASH {
            Self {
                data_store: SnapshotBuilder::new(root, db),
                current_root: NodeRef::Null,
            }
        } else {
            Self {
                data_store: SnapshotBuilder::new(root, db),
                current_root: NodeRef::Stored(0),
            }
        }
    }
}

impl<S: Store> MerkleBTreeTxn<S> {
    pub fn get(&self, key: &S::Key) -> Result<Option<S::Value>, S::Error> {
        let mut node_ref = &self.current_root;

        loop {
            match node_ref {
                NodeRef::Node(node) => match node.keys.binary_search(key) {
                    Ok(equal_key_idx) => node_ref = &node.children[equal_key_idx + 1],
                    Err(idx) => node_ref = &node.children[idx],
                },
                NodeRef::Leaf(leaf) => {
                    if &leaf.key == key {
                        return Ok(Some(leaf.value.clone()));
                    } else {
                        return Ok(None);
                    }
                }
                NodeRef::Stored(idx) => {
                    self.get_stored(*idx, key)?;
                }
                NodeRef::Null => return Ok(None),
            }
        }
    }

    fn get_stored(&self, mut stored_idx: Idx, key: &S::Key) -> Result<Option<S::Value>, S::Error> {
        loop {
            // TODO consider making Store::get return &Arc not Arc
            let node = self.data_store.get(stored_idx)?;

            match node {
                NodeOrLeaf::Node(node) => {
                    let next_node_ref = match node.keys.binary_search(key) {
                        Ok(equal_key_idx) => &node.children[equal_key_idx + 1],
                        Err(idx) => &node.children[idx],
                    };

                    match next_node_ref {
                        NodeRef::Node(_) => {
                            unreachable!("A stored node cannot have a modified child")
                        }
                        NodeRef::Leaf(leaf) => {
                            if &leaf.key == key {
                                return Ok(Some(leaf.value.clone()));
                            } else {
                                return Ok(None);
                            }
                        }
                        NodeRef::Stored(next_stored) => {
                            stored_idx = *next_stored;
                        }
                        NodeRef::Null => return Ok(None),
                    }
                }
                NodeOrLeaf::Leaf(leaf) => {
                    if &leaf.key == key {
                        return Ok(Some(leaf.value.clone()));
                    } else {
                        return Ok(None);
                    }
                }
            }
        }
    }

    pub fn insert(&mut self, key: S::Key, value: S::Value) -> Result<Option<S::Value>, S::Error> {
        match Self::insert_inner(&self.data_store, &mut self.current_root, key, value)? {
            Insert::Inserted => Ok(None),
            Insert::Replaced(old_value) => Ok(Some(old_value)),
            Insert::SplitNode {
                right_node,
                middle_key,
            } => {
                let new_root = Node {
                    keys: ArrayVec::from_iter([middle_key]),
                    children: ArrayVec::from_iter([
                        self.current_root.clone(),
                        NodeRef::Node(right_node),
                    ]),
                };
                self.current_root = NodeRef::Node(Arc::new(new_root));
                Ok(None)
            }
            Insert::SplitLeaf { new_leaf } => {
                if new_leaf.key > self.current_root.leaf().unwrap().key {
                    let new_root = Node {
                        keys: ArrayVec::from_iter([new_leaf.key.clone()]),
                        children: ArrayVec::from_iter([
                            self.current_root.clone(),
                            NodeRef::Leaf(new_leaf),
                        ]),
                    };
                    self.current_root = NodeRef::Node(Arc::new(new_root));
                } else {
                    let new_root = Node {
                        keys: ArrayVec::from_iter([self.current_root.leaf().unwrap().key.clone()]),
                        children: ArrayVec::from_iter([
                            NodeRef::Leaf(new_leaf),
                            self.current_root.clone(),
                        ]),
                    };
                    self.current_root = NodeRef::Node(Arc::new(new_root));
                }

                Ok(None)
            }
        }
    }

    fn insert_inner(
        data_store: &S,
        node: &mut NodeRef<S::Key, S::Value>,
        key: S::Key,
        value: S::Value,
    ) -> Result<Insert<S>, S::Error> {
        match node {
            NodeRef::Stored(idx) => {
                *node = NodeRef::from(data_store.get(*idx)?);

                // TODO use loop and break
                Self::insert_inner(data_store, node, key, value)
            }
            NodeRef::Node(node) => {
                Node::assert_invariants(node);
                let node = Arc::make_mut(node);

                let idx = match node.keys.binary_search(&key) {
                    Ok(equal_key_idx) => equal_key_idx + 1,
                    Err(idx) => idx,
                };

                match Self::insert_inner(data_store, &mut node.children[idx], key, value)? {
                    Insert::Inserted => Ok(Insert::Inserted),
                    Insert::Replaced(v) => Ok(Insert::Replaced(v)),
                    Insert::SplitNode {
                        right_node,
                        middle_key,
                    } => {
                        if node.is_full() {
                            let mut new_right_node = Node {
                                keys: node
                                    .keys
                                    .drain((Node::<S::Key, S::Value>::min_keys() + 1)..)
                                    .collect(),
                                children: node
                                    .children
                                    .drain(Node::<S::Key, S::Value>::min_children()..)
                                    .collect(),
                            };
                            let new_middle_key = node.keys.pop().unwrap();

                            if idx < Node::<S::Key, S::Value>::min_children() {
                                node.keys.insert(idx, middle_key);
                                node.children.insert(idx + 1, NodeRef::Node(right_node));
                            } else {
                                let adjusted_idx = idx - Node::<S::Key, S::Value>::min_children();
                                new_right_node.keys.insert(adjusted_idx, middle_key);
                                new_right_node
                                    .children
                                    .insert(adjusted_idx + 1, NodeRef::Node(right_node));
                            }

                            node.assert_invariants();
                            Ok(Insert::SplitNode {
                                middle_key: new_middle_key,
                                right_node: Arc::new(new_right_node),
                            })
                        } else {
                            node.keys.insert(idx, middle_key);
                            node.children.insert(idx + 1, NodeRef::Node(right_node));

                            Node::assert_invariants(node);
                            Ok(Insert::Inserted)
                        }
                    }
                    Insert::SplitLeaf { new_leaf } => {
                        let old_key = &node
                            .children
                            .get(idx)
                            .unwrap_or_else(|| node.children.last().unwrap())
                            .leaf()
                            .expect("SplitLeaf was returned, but existing child is not a leaf")
                            .key;

                        if node.is_full() {
                            debug_assert!(node.keys.len() == Node::<S::Key, S::Value>::max_keys());
                            debug_assert!(
                                node.children.len() == Node::<S::Key, S::Value>::max_children()
                            );

                            let old_key = old_key.clone();

                            let new_in_left_node = idx < Node::<S::Key, S::Value>::min_children();

                            let mut right_node = Node {
                                keys: node
                                    .keys
                                    .drain((Node::<S::Key, S::Value>::min_keys() + 1)..)
                                    .collect(),
                                children: node
                                    .children
                                    .drain(Node::<S::Key, S::Value>::min_children()..)
                                    .collect(),
                            };

                            let middle_key = node.keys.pop().unwrap();

                            if new_in_left_node {
                                if new_leaf.key < old_key {
                                    node.keys.insert(idx, old_key);
                                    node.children.insert(idx, NodeRef::Leaf(new_leaf));
                                } else {
                                    debug_assert!(new_leaf.key > old_key);
                                    node.keys.insert(idx, new_leaf.key.clone());
                                    node.children.insert(idx + 1, NodeRef::Leaf(new_leaf));
                                }
                            } else if new_leaf.key < old_key {
                                right_node.keys.insert(
                                    idx - Node::<S::Key, S::Value>::min_children(),
                                    old_key,
                                );
                                right_node.children.insert(
                                    idx - Node::<S::Key, S::Value>::min_children(),
                                    NodeRef::Leaf(new_leaf),
                                );
                            } else {
                                debug_assert!(new_leaf.key > old_key);
                                right_node.keys.insert(
                                    idx - Node::<S::Key, S::Value>::min_children(),
                                    new_leaf.key.clone(),
                                );
                                right_node.children.insert(
                                    idx + 1 - Node::<S::Key, S::Value>::min_children(),
                                    NodeRef::Leaf(new_leaf),
                                );
                            }

                            Node::assert_invariants(node);
                            Ok(Insert::SplitNode {
                                middle_key,
                                right_node: Arc::new(right_node),
                            })
                        } else {
                            node.keys
                                .insert(idx, cmp::max(&new_leaf.key, old_key).clone());

                            debug_assert!(new_leaf.key != *old_key);
                            let child_idx = if new_leaf.key < *old_key {
                                idx
                            } else {
                                idx + 1
                            };

                            node.children.insert(child_idx, NodeRef::Leaf(new_leaf));
                            Node::assert_invariants(node);
                            Ok(Insert::Inserted)
                        }
                    }
                }
            }
            NodeRef::Leaf(leaf) => match key.cmp(&leaf.key) {
                Ordering::Equal => {
                    let mut value = value;
                    let leaf = Arc::make_mut(leaf);
                    mem::swap(&mut leaf.value, &mut value);

                    Ok(Insert::Replaced(value))
                }
                // split with the higher of the two values being the middle key
                Ordering::Less => Ok(Insert::SplitLeaf {
                    new_leaf: Arc::new(Leaf { key, value }),
                }),
                Ordering::Greater => Ok(Insert::SplitLeaf {
                    new_leaf: Arc::new(Leaf { key, value }),
                }),
            },
            NodeRef::Null => {
                *node = NodeRef::Leaf(Arc::new(Leaf { key, value }));
                Ok(Insert::Inserted)
            }
        }
    }

    pub fn remove(&mut self, key: &S::Key) -> Result<Option<S::Value>, S::Error> {
        match Self::remove_inner(&self.data_store, &mut self.current_root, key)? {
            Remove::NotPresent => Ok(None),
            Remove::Removed(value) => Ok(Some(value)),
            Remove::Underflow(value) => {
                match self.current_root {
                    // remove_inner removed the single leaf node replacing it with null
                    NodeRef::Null => Ok(Some(value)),
                    NodeRef::Leaf(_) => {
                        unreachable!("Removing a leaf replaces it with null")
                    }
                    NodeRef::Stored(_) => {
                        unreachable!("Stored node would have been replaced with a node or leaf")
                    }
                    NodeRef::Node(ref mut node) => {
                        if node.keys.is_empty() {
                            debug_assert!(node.children.len() == 1);
                            self.current_root = Arc::make_mut(node).children.pop().unwrap();
                        }

                        Ok(Some(value))
                    }
                }
            }
        }
    }

    fn remove_inner(
        data_store: &S,
        node: &mut NodeRef<S::Key, S::Value>,
        key: &S::Key,
    ) -> Result<Remove<S>, S::Error> {
        match node {
            NodeRef::Stored(idx) => {
                *node = NodeRef::from(data_store.get(*idx)?);

                // TODO use loop and break
                Self::remove_inner(data_store, node, key)
            }
            NodeRef::Node(node) => {
                Node::assert_invariants(node);
                let node = Arc::make_mut(node);

                let idx = match node.keys.binary_search(key) {
                    Ok(equal_key_idx) => equal_key_idx + 1,
                    Err(idx) => idx,
                };

                match Self::remove_inner(data_store, &mut node.children[idx], key)? {
                    Remove::NotPresent => Ok(Remove::NotPresent),
                    Remove::Removed(value) => Ok(Remove::Removed(value)),
                    Remove::Underflow(value) => {
                        Self::handle_underflow(data_store, node, idx, value)
                    }
                }
            }
            NodeRef::Leaf(leaf) => {
                if &leaf.key == key {
                    // TODO maybe return whole leaf or do somthing more advanced to enforce good usage
                    let value = leaf.value.clone();
                    *node = NodeRef::Null;
                    Ok(Remove::Underflow(value))
                } else {
                    Ok(Remove::NotPresent)
                }
            }
            NodeRef::Null => Ok(Remove::NotPresent),
        }
    }

    fn handle_underflow(
        data_store: &S,
        node: &mut Node<S::Key, S::Value>,
        idx: usize,
        value: S::Value,
    ) -> Result<Remove<S>, S::Error> {
        match node.children[idx] {
            NodeRef::Node(_) => {
                if let Err(()) = node.merge_or_balance(idx) {
                    if idx == 0 {
                        let hash_idx = node.children[1].stored().unwrap();
                        node.children[1] = NodeRef::Node(
                            data_store
                                .get(hash_idx)?
                                .try_into()
                                .expect("unbalanced node"),
                        );
                    } else {
                        let hash_idx = node.children[idx - 1].stored().unwrap();
                        node.children[idx - 1] = NodeRef::Node(
                            data_store
                                .get(hash_idx)?
                                .try_into()
                                .expect("unbalanced node"),
                        );
                        // We just added a sibling node to the tree
                        node.merge_or_balance(idx).unwrap()
                    }
                };

                if node.is_to_small() {
                    Ok(Remove::Underflow(value))
                } else {
                    Ok(Remove::Removed(value))
                }
            }
            // A leaf was just removed leaving null
            NodeRef::Null => {
                if idx == 0 {
                    node.keys.remove(idx);
                } else {
                    node.keys.remove(idx - 1);
                }

                node.children.remove(idx);

                if node.keys.len() < Node::<S::Key, S::Value>::min_keys() {
                    Ok(Remove::Underflow(value))
                } else {
                    Ok(Remove::Removed(value))
                }
            }
            _ => {
                unreachable!("Underflow is only returned from visiting a node or leaf")
            }
        }
    }

    /// Prints the whole tree in a pretty format.
    pub fn pretty_print(&self) {
        Self::pretty_print_node(&self.data_store, &self.current_root, 0).unwrap();
    }

    fn pretty_print_node(
        data_store: &S,
        node_ref: &NodeRef<S::Key, S::Value>,
        indent: usize,
    ) -> Result<(), S::Error> {
        let indent_str = "  ".repeat(indent);
        match node_ref {
            NodeRef::Node(node) => {
                println!("{}Node(keys: {:?})", indent_str, node.keys);
                for child in &node.children {
                    Self::pretty_print_node(data_store, child, indent + 1)?;
                }
            }
            NodeRef::Leaf(leaf) => {
                println!(
                    "{}Leaf(key: {:?}, value: {:?})",
                    indent_str, leaf.key, leaf.value
                );
            }
            NodeRef::Stored(idx) => {
                let node_or_leaf = data_store.get(*idx)?;
                let node_ref = NodeRef::from(node_or_leaf);
                Self::pretty_print_node(data_store, &node_ref, indent)?;
            }
            NodeRef::Null => {
                println!("{}Null", indent_str);
            }
        }
        Ok(())
    }
}

enum Insert<S: Store> {
    Inserted,
    Replaced(S::Value),
    /// The key should be the right node's first key.
    SplitNode {
        middle_key: S::Key,
        right_node: Arc<Node<S::Key, S::Value>>,
    },
    SplitLeaf {
        new_leaf: Arc<Leaf<S::Key, S::Value>>,
    },
}

pub enum Remove<S: Store> {
    NotPresent,
    Removed(S::Value),
    /// Removing caused a node to be smaller than the minimum size.
    Underflow(S::Value),
}

impl<S: Store> Debug for Insert<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Insert::Inserted => write!(f, "Inserted"),
            Insert::Replaced(v) => write!(f, "Replaced({:?})", v),
            Insert::SplitNode {
                right_node,
                middle_key,
            } => {
                write!(
                    f,
                    "SplitNode {{ right_node: {:?}, middle_key: {:?} }}",
                    right_node, middle_key
                )
            }
            Insert::SplitLeaf { new_leaf } => write!(f, "SplitLeaf {{ new_leaf: {:?} }}", new_leaf),
        }
    }
}

#[cfg(test)]
mod test {
    use alloc::collections::btree_map::BTreeMap;

    use proptest::prelude::*;

    use crate::transaction::MerkleBTreeTxn;

    #[derive(Debug)]
    enum Op {
        Insert(u32, u32),
        Get(u32),
        Delete(u32),
    }

    fn run_operations(operations: Vec<Op>) {
        let mut txn_btree = MerkleBTreeTxn::new_snapshot_builder_txn(Default::default(), ());
        let mut std_btree = BTreeMap::new();

        for op in operations {
            match op {
                Op::Insert(k, v) => {
                    let res_txn = txn_btree.insert(k, v).unwrap();
                    let res_std = std_btree.insert(k, v);
                    assert_eq!(res_txn, res_std,
                        "insert failed for key: {}, value: {}, merkle btree returned {:?}, btreemap returned {:?}",
                        k, v, res_txn, res_std);
                }
                Op::Get(k) => {
                    let res_txn = txn_btree.get(&k).unwrap();
                    let res_std = std_btree.get(&k).cloned();
                    assert_eq!(res_txn, res_std,
                        "get failed for key: {}, merkle btree returned {:?}, btreemap returned {:?}",
                        k, res_txn, res_std);
                }
                Op::Delete(k) => {
                    let res_txn = txn_btree.remove(&k).unwrap();
                    let res_std = std_btree.remove(&k);
                    assert_eq!(res_txn, res_std,
                        "delete failed for key: {}, merkle btree returned {:?}, btreemap returned {:?}",
                        k, res_txn, res_std);
                }
            }
        }
    }

    #[test]
    fn test_insert_1() {
        let operations = vec![Op::Insert(0, 0), Op::Insert(1, 0), Op::Insert(2, 0)];
        run_operations(operations);
    }

    #[test]
    fn test_insert_2() {
        let operations = vec![
            Op::Insert(15, 0),
            Op::Insert(16, 0),
            Op::Insert(17, 0),
            Op::Insert(18, 0),
            Op::Insert(19, 0),
            Op::Insert(20, 0),
            Op::Insert(0, 0),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_insert_3() {
        let operations = vec![
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(1551649896, 0),
            Op::Insert(2, 0),
            Op::Insert(1551649897, 0),
            Op::Insert(1551649898, 0),
            Op::Insert(3, 0),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_insert_duplicate_1() {
        let operations = vec![
            Op::Insert(1551649896, 0),
            Op::Insert(1551649897, 0),
            Op::Insert(0, 0),
            Op::Insert(0, 0), // Duplicate insert
        ];
        run_operations(operations);
    }

    #[test]
    fn test_insert_delete_1() {
        let operations = vec![Op::Insert(0, 0), Op::Insert(1, 0), Op::Delete(1)];
        run_operations(operations);
    }

    #[test]
    fn test_insert_delete_get_1() {
        let operations = vec![
            Op::Insert(1, 10),
            Op::Insert(2, 20),
            Op::Insert(3, 30),
            Op::Delete(2),
            Op::Get(2),
            Op::Delete(1),
            Op::Delete(3),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_insert_delete_2() {
        let operations = vec![
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(2, 0),
            Op::Insert(3, 0),
            Op::Insert(257, 0),
            Op::Insert(63, 0),
            Op::Insert(4, 0),
            Op::Insert(5, 0),
            Op::Insert(64, 0),
            Op::Insert(65, 0),
            Op::Insert(66, 0),
            Op::Insert(67, 0),
            Op::Insert(6, 0),
            Op::Delete(257),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_insert_delete_get_2() {
        let operations = vec![
            Op::Insert(100, 1000),
            Op::Insert(200, 2000),
            Op::Insert(300, 3000),
            Op::Delete(200),
            Op::Get(200),
            Op::Insert(400, 4000),
            Op::Insert(500, 5000),
            Op::Delete(100),
            Op::Get(100),
            Op::Delete(300),
            Op::Get(300),
            Op::Delete(400),
            Op::Get(400),
            Op::Delete(500),
            Op::Get(500),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_minimal_failing_input() {
        let operations = vec![
            Op::Insert(887, 0),
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(119, 0),
            Op::Insert(2, 0),
            Op::Delete(887),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_minimal_failing_input_2() {
        let operations = vec![
            Op::Insert(0, 0),
            Op::Insert(0, 0),
            Op::Insert(422, 0),
            Op::Insert(422, 0),
            Op::Insert(423, 0),
            Op::Insert(424, 0),
            Op::Insert(425, 0),
            Op::Delete(422),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_minimal_failing_input_3() {
        let operations = vec![
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(2, 0),
            Op::Insert(3, 0),
            Op::Insert(4, 0),
            Op::Insert(580, 0),
            Op::Insert(581, 0),
            Op::Insert(582, 0),
            Op::Insert(5, 0),
            Op::Insert(527, 0),
            Op::Insert(6, 0),
            Op::Delete(527),
            Op::Insert(527, 0),
            Op::Insert(6, 0),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_minimal_failing_input_4() {
        let operations = vec![
            Op::Insert(0, 0),
            Op::Insert(0, 0),
            Op::Insert(513, 0),
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(2, 0),
            Op::Insert(617, 0),
            Op::Insert(618, 0),
            Op::Insert(0, 0),
            Op::Insert(0, 0),
            Op::Insert(514, 0),
            Op::Insert(467, 0),
            Op::Insert(3, 0),
            Op::Insert(4, 0),
            Op::Insert(467, 0),
            Op::Insert(512, 0),
            Op::Insert(505, 0),
            Op::Insert(5, 0),
            Op::Delete(512),
            Op::Delete(505),
        ];
        run_operations(operations);
    }

    proptest! {
        #[test]
        fn test_merkle_btree_txn_against_btreemap(operations in proptest::collection::vec(
            prop_oneof![
                (0..1000u32, 0..10u32).prop_map(|(k, v)| Op::Insert(k, v)),
                (0..1000u32).prop_map(Op::Get),
                (0..1000u32).prop_map(Op::Delete),
            ],
            1..10_000
        )) {
            run_operations(operations);
        }
    }
}
